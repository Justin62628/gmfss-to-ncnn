import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .position import PositionEmbeddingSine
from .transformer import FeatureTransformer, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import normalize_img, merge_splits, split_feature, convex_upsampling


class GMFlow(nn.Module):
    def __init__(self,
                 num_scales=2,
                 upsample_factor=4,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 **kwargs,
                 ):
        super(GMFlow, self).__init__()

        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers

        # CNN backbone
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              attention_type=attention_type,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
        self.convex_upsampling = convex_upsampling(self.upsample_factor)
        self.global_correlation_softmax = global_correlation_softmax()
        self.local_correlation_softmax = local_correlation_softmax()
        self.split_feature = split_feature()
        self.merge_splits = merge_splits()

    def extract_feature(self, img0, img1):
        feature0 = self.backbone(img0)[::-1]  # resolution from low to high
        feature1 = self.backbone(img1)[::-1]
        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)  # (1, 144, 40, 64)

            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, -1)
            mask = torch.softmax(mask, dim=2)
            mask = mask.repeat(1, flow_channel, 1, 1)  # [B, 2, 9, H*W]
            mask = mask.view(b, flow_channel * 9, self.upsample_factor * self.upsample_factor, h, w)
            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)  # [1, 18, 8640]
            up_flow = up_flow.view(b, flow_channel * 9, 1, h * w)  # [B, 2 * 9, 1, H, W]
            up_flow = up_flow.repeat(1, 1, self.upsample_factor * self.upsample_factor, 1)  # [B, 2 * 9, K* K, H, W]
            up_flow = up_flow.view(b, flow_channel * 9, self.upsample_factor * self.upsample_factor, h, w)  # [B, 2, 9, K, K, H, W]
            up_flow = mask * up_flow
            # mask = mask.view(b, 1, 9, self.upsample_factor * self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            # mask = torch.softmax(mask, dim=2)

            # up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)  # [1, 18, 8640]
            # up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]
            up_flow = up_flow.view(b, flow_channel, 9, -1)  # [B, 2, 9, H * W]
            #
            up_flow = torch.sum(up_flow, dim=2)  # [B, 2, -1], remove 9

            # core
            up_flow = up_flow.view(b, flow_channel, self.upsample_factor*self.upsample_factor, h*w)  # [B, 2, K, K, H, W]
            up_flow = self.convex_upsampling(flow, up_flow)  # flow: (1,2,40,64)
        return up_flow

    def feature_add_position(self, feature0, feature1, attn_splits, feature_channels):
        pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

        if attn_splits > 1:  # add position in splited window
            feature0_splits = self.split_feature(feature0, num_splits=attn_splits)  # (4, 128, 10, 16)
            feature1_splits = self.split_feature(feature1, num_splits=attn_splits)

            position = pos_enc(feature0_splits)

            feature0_splits = feature0_splits + position
            feature1_splits = feature1_splits + position

            feature0 = self.merge_splits(feature0_splits, num_splits=attn_splits)
            feature1 = self.merge_splits(feature1_splits, num_splits=attn_splits)
        else:
            position = pos_enc(feature0)

            feature0 = feature0 + position
            feature1 = feature1 + position

        return feature0, feature1

    def forward(self, img0, img1,
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                pred_bidir_flow=False,
                **kwargs,
                ):

        # img0 = imgs[:, 0:3, :, :]  # [B, 3, H, W]
        # img1 = imgs[:, 3:6, :, :]  # [B, 3, H, W]
        # img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features, f0: [1, 128,20,32] [1,128,40,64]

        flow = None

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = self.feature_add_position(feature0, feature1, attn_splits, self.feature_channels)  # 6dim, f0: [1, 128,20,32]

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = self.global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]  # 1,2,20,32
            else:  # local matching
                flow_pred = self.local_correlation_softmax(feature0, feature1, corr_radius)[0]  # 1,2,40,64

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison
            if self.training:  # only need to upsample intermediate flow predictions at training time
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)

            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius)  # 1,2,20,32

            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)

            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
        flow_up = flow_up + 0.0001
        return flow_up
