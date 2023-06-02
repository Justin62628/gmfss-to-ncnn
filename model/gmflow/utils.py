import torch
from torch import nn
import torch.nn.functional as F
from .position import PositionEmbeddingSine


class convex_upsampling(nn.Module):
    def __init__(self, upsample_factor):
        super(convex_upsampling, self).__init__()
        self.upsample_factor = upsample_factor

    def forward(self, flow, up_flow):
        b, flow_channel, h, w = flow.shape
        # mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
        # # mask = torch.softmax(mask, dim=2)
        #
        # # up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)  # [1, 18, 8640]
        # up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]
        #
        # up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W], remove 9
        up_flow = up_flow.view(b, flow_channel, self.upsample_factor, self.upsample_factor, h, w)  # [B, 2, K, K, H, W]
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
        up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                  self.upsample_factor * w)  # [B, 2, K*H, K*W]
        return up_flow


class split_feature(nn.Module):
    def forward(self, feature,
                num_splits=2,
                channel_last=False,
                ):
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        # pixel_unshuffle = nn.PixelUnshuffle(num_splits)
        # feature = pixel_unshuffle(feature).contiguous()  # [B, C*K*K, H/K, W/K]
        # feature = feature.reshape(b_new, c, h_new, w_new)

        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits
                               ).permute(0, 2, 4, 1, 3, 5).reshape(b, b_new, c, h_new, w_new)  # [B*K*K, C, H/K, W/K]

        return feature


# class split_feature_c_last(nn.Module):
#     def forward(self, feature,
#                 num_splits=2,
#                 channel_last=False,
#                 ):
#         b, h, w, c = feature.size()
#         assert h % num_splits == 0 and w % num_splits == 0
#
#         b_new = num_splits * num_splits
#         h_new = h // num_splits
#         w_new = w // num_splits
#
#         feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c
#                                ).permute(0, 1, 3, 2, 4, 5).reshape(b, b_new, h_new, w_new, c)  # [B*K*K, H/K, W/K, C]
#
#         return feature


class merge_splits(nn.Module):
    def forward(self, splits,
                num_splits=2,
                channel_last=False,
                ):
        b0, b, c, h, w = splits.size()
        new_b = (b0 * b) // num_splits // num_splits

        # pixel_shuffle = nn.PixelShuffle(num_splits)
        # merge = pixel_shuffle(splits).contiguous()
        # merge = merge.reshape(new_b, c, num_splits * h, num_splits * w)
        splits = splits.view(new_b, num_splits, num_splits, c, h, w)  # 1,2,2,128,10,16
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            new_b, c, num_splits * h, num_splits * w)  # [B, C, H, W]

        return merge


# class merge_splits_c_last(nn.Module):
#     def forward(self, splits,
#                 num_splits=2,
#                 channel_last=False,
#                 ):
#         b0, b, h, w, c = splits.size()
#         new_b = (b0 * b) // num_splits // num_splits
#
#         splits = splits.view(new_b, num_splits, num_splits, h, w, c)
#         merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
#             new_b, num_splits * h, num_splits * w, c)  # [B, H, W, C]
#         return merge


def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    b, c, h, w = img0.shape
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, c, 1, 1).expand(b, c, h, w).to(img0.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, c, 1, 1).expand(b, c, h, w).to(img0.device)
    img0 = (img0 - mean) / std
    img1 = (img1 - mean) / std

    return img0, img1
