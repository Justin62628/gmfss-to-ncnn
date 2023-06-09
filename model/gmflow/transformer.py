import torch
import torch.nn as nn
import torch.nn.functional as F

from model.log import print_mat

from .utils import split_feature, merge_splits


def single_head_full_attention(q, k, v):
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)  # [B, L, L]
    attn = torch.softmax(scores, dim=2)  # [B, L, L]
    out = torch.matmul(attn, v)  # [B, L, C]

    return out


class generate_shift_window_attn_mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.split_feature = split_feature()

    def forward(self, input_resolution, window_size_h, window_size_w,
                shift_size_h, shift_size_w, device=torch.device('cuda')):
        h, w = input_resolution
        # Create masks for each block
        mask1 = torch.ones((h - window_size_h, w - window_size_w)).to(device)  # mask for block 1
        mask2 = torch.ones((h - window_size_h, window_size_w - shift_size_w)).to(device) * 2  # mask for block 2
        mask3 = torch.ones((h - window_size_h, shift_size_w)).to(device) * 3  # mask for block 3
        mask4 = torch.ones((window_size_h - shift_size_h, w - window_size_w)).to(device) * 4  # mask for block 4
        mask5 = torch.ones((window_size_h - shift_size_h, window_size_w - shift_size_w)).to(
            device) * 5  # mask for block 5
        mask6 = torch.ones((window_size_h - shift_size_h, shift_size_w)).to(device) * 6  # mask for block 6
        mask7 = torch.ones((shift_size_h, w - window_size_w)).to(device) * 7  # mask for block 7
        mask8 = torch.ones((shift_size_h, window_size_w - shift_size_w)).to(device) * 8  # mask for block 8
        mask9 = torch.ones((shift_size_h, shift_size_w)).to(device) * 9  # mask for block 9

        # Concatenate the masks to create the full mask
        upper_mask = torch.cat([mask1, mask2, mask3], dim=1)
        middle_mask = torch.cat([mask4, mask5, mask6], dim=1)
        lower_mask = torch.cat([mask7, mask8, mask9], dim=1)
        full_mask = torch.cat([upper_mask, middle_mask, lower_mask], dim=0).unsqueeze(0).unsqueeze(
            -1)  # Add extra dimensions for batch size and channels
        full_mask = full_mask.permute(0, 3, 1, 2)  # [B, 1, H, W]
        mask_windows = self.split_feature(full_mask, num_splits=input_resolution[-1] // window_size_w, )  # [B, 9, 1, H, W]
        mask_windows = mask_windows.permute(0, 1, 3, 4, 2)  # [B, 9, H, W, 1]
        mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask


class single_head_split_window_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.split_feature = split_feature()
        self.merge_splits = merge_splits()

    def forward(self, q, k, v,
                num_splits=1,
                with_shift=False,
                h=None,
                w=None,
                attn_mask=None,
                ):
        # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # q, k, v: [B, L, C]
        assert q.dim() == k.dim() == v.dim() == 3

        assert h is not None and w is not None
        assert q.size(1) == h * w

        b, _, c = q.size()

        b_new = num_splits * num_splits  # B0

        window_size_h = h // num_splits
        window_size_w = w // num_splits

        q = q.view(b, h, w, c)  # [B, H, W, C], [2, 20, 32, 128]
        k = k.view(b, h, w, c)
        v = v.view(b, h, w, c)

        scale_factor = c ** 0.5

        if with_shift:
            assert attn_mask is not None  # compute once
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))

        q = q.permute(0, 3, 1, 2)  # [B, C, H, W]
        k = k.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        q = self.split_feature(q, num_splits=num_splits)  # [B, B0*K*K, C, H/K, W/K]
        k = self.split_feature(k, num_splits=num_splits)
        v = self.split_feature(v, num_splits=num_splits)
        q = q.permute(0, 1, 3, 4, 2)  # [B, B0*K*K, H/K, W/K, C]
        k = k.permute(0, 1, 3, 4, 2)
        v = v.permute(0, 1, 3, 4, 2)

        scores = torch.matmul(q.view(b, b_new, -1, c), k.view(b, b_new, -1, c).permute(0, 1, 3, 2)
                              ) / scale_factor  # [B, K*K, H/K*W/K, H/K*W/K]

        if with_shift:
            scores += attn_mask.repeat(b, 1, 1)  # TODO: check

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.view(b, b_new, -1, c))  # [B, K*K, H/K*W/K, C]
        out = out.view(b, b_new, h // num_splits, w // num_splits, c)
        out = out.permute(0, 1, 4, 2, 3)  # [B, K*K, C, H/K, W/K]
        out = self.merge_splits(out, num_splits=num_splits)  # [B, C, H, W]
        out = out.permute(0, 2, 3, 1)  # [B, H, W, C]

        # shift back
        if with_shift:
            out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(2, 3))

        out = out.view(b, -1, c)

        return out


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=1,
                 attention_type='swin',
                 no_ffn=False,
                 ffn_dim_expansion=4,
                 with_shift=False,
                 **kwargs,
                 ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.attention_type = attention_type
        self.no_ffn = no_ffn

        self.with_shift = with_shift

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)
        self.single_head_split_window_attention = single_head_split_window_attention()

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                attn_num_splits=None,
                **kwargs,
                ):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        if self.attention_type == 'swin' and attn_num_splits > 1:
            if self.nhead > 1:
                # we observe that multihead attention slows down the speed and increases the memory consumption
                # without bringing obvious performance gains and thus the implementation is removed
                raise NotImplementedError
            else:
                message = self.single_head_split_window_attention(query, key, value,
                                                                  num_splits=attn_num_splits,
                                                                  with_shift=self.with_shift,
                                                                  h=height,
                                                                  w=width,
                                                                  attn_mask=shifted_window_attn_mask,
                                                                  )
        else:
            message = single_head_full_attention(query, key, value)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(self,
                 d_model=256,
                 nhead=1,
                 attention_type='swin',
                 ffn_dim_expansion=4,
                 with_shift=False,
                 **kwargs,
                 ):
        super(TransformerBlock, self).__init__()

        self.self_attn = TransformerLayer(d_model=d_model,
                                          nhead=nhead,
                                          attention_type=attention_type,
                                          no_ffn=True,
                                          ffn_dim_expansion=ffn_dim_expansion,
                                          with_shift=with_shift,
                                          )

        self.cross_attn_ffn = TransformerLayer(d_model=d_model,
                                               nhead=nhead,
                                               attention_type=attention_type,
                                               ffn_dim_expansion=ffn_dim_expansion,
                                               with_shift=with_shift,
                                               )

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                attn_num_splits=None,
                **kwargs,
                ):
        # source, target: [B, L, C]

        # self attention
        source = self.self_attn(source, source,
                                height=height,
                                width=width,
                                shifted_window_attn_mask=shifted_window_attn_mask,
                                attn_num_splits=attn_num_splits,
                                )

        # cross attention and ffn
        source = self.cross_attn_ffn(source, target,
                                     height=height,
                                     width=width,
                                     shifted_window_attn_mask=shifted_window_attn_mask,
                                     attn_num_splits=attn_num_splits,
                                     )

        return source


class FeatureTransformer(nn.Module):
    def __init__(self,
                 num_layers=6,
                 d_model=128,
                 nhead=1,
                 attention_type='swin',
                 ffn_dim_expansion=4,
                 **kwargs,
                 ):
        super(FeatureTransformer, self).__init__()

        self.attention_type = attention_type

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             nhead=nhead,
                             attention_type=attention_type,
                             ffn_dim_expansion=ffn_dim_expansion,
                            #  with_shift=True if attention_type == 'swin' and i % 2 == 1 else False,
                             with_shift=False
                             )
            for i in range(num_layers)])

        self.generate_shift_window_attn_mask = generate_shift_window_attn_mask()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1,
                attn_num_splits=None,
                **kwargs,
                ):

        b, c, h, w = feature0.shape
        assert self.d_model == c

        feature0 = feature0.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        feature1 = feature1.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        if self.attention_type == 'swin' and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = self.generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=feature0.device,
            )  # [K*K, H/K*W/K, H/K*W/K] 4,160,160
        else:
            shifted_window_attn_mask = None

        for layer in self.layers:
            feature0_ = layer(feature0, feature1,
                             height=h,
                             width=w,
                             shifted_window_attn_mask=shifted_window_attn_mask,
                             attn_num_splits=attn_num_splits,
                             )
            feature1_ = layer(feature1, feature0,
                             height=h,
                             width=w,
                             shifted_window_attn_mask=shifted_window_attn_mask,
                             attn_num_splits=attn_num_splits,
                             )
            feature0, feature1 = feature0_, feature1_

        # reshape back
        feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        feature1 = feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return feature0, feature1


class FeatureFlowAttention(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels,
                 **kwargs,
                 ):
        super(FeatureFlowAttention, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow,
                local_window_attn=False,
                local_window_radius=1,
                **kwargs,
                ):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow,
                                                  local_window_radius=local_window_radius)

        b, c, h, w = feature0.size()

        query = feature0.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        # a note: the ``correct'' implementation should be:
        # ``query = self.q_proj(query), key = self.k_proj(query)''
        # this problem is observed while cleaning up the code
        # however, this doesn't affect the performance since the projection is a linear operation,
        # thus the two projection matrices for key can be merged
        # so I just leave it as is in order to not re-train all models :)
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, H*W, 2]
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]

        return out

    def forward_local_window_attn(self, feature0, flow,
                                  local_window_radius=1,
                                  ):
        assert flow.size(1) == 2
        assert local_window_radius > 0
        # This method is not reached, checked
        b, c, h, w = feature0.size()

        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)
                                       ).reshape(b, h * w, 1, c)  # [B, H*W, 1, C]

        kernel_size = 2 * local_window_radius + 1

        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)

        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size,
                                   padding=local_window_radius)  # [B, C*(2R+1)^2), H*W]

        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(
            0, 3, 4, 1, 2).reshape(b, h * w, c, kernel_size ** 2)  # [B, H*W, C, (2R+1)^2]

        flow_window = F.unfold(flow, kernel_size=kernel_size,
                               padding=local_window_radius)  # [B, 2*(2R+1)^2), H*W]

        flow_window = flow_window.view(b, 2, kernel_size ** 2, h, w).permute(
            0, 3, 4, 2, 1).reshape(b, h * w, kernel_size ** 2, 2)  # [B, H*W, (2R+1)^2, 2]

        # print_mat(feature0_reshape, 'flow_local_attn_b0')
        # print_mat(feature0_window, 'flow_local_attn_b1')
        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)  # [B, H*W, 1, (2R+1)^2]
        # print_mat(scores, 'flow_local_attn_t0')
        
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, flow_window)
        
        out = out.view(b, h, w, 2).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

        return out
