import torch
import torch.nn as nn
import torch.nn.functional as F

from model.log import print_mat

from .geometry import coords_grid, generate_window_grid, normalize_coords

class global_correlation_softmax(nn.Module):
    def forward(self, feature0, feature1, pred_bidir_flow=False):
        # global correlation
        b, c, h, w = feature0.shape
        feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
        feature1 = feature1.view(b, c, -1)  # [B, C, H*W]

        # print_mat(feature0, 'global_corr_b0')
        # print_mat(feature1, 'global_corr_b1')
        correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)  # [B, H, W, H, W]
        # print_mat(correlation, 'global_corr_t0')

        # flow from softmax
        init_grid = coords_grid(b, h, w).to(correlation.device)  # [B, 2, H, W]
        grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

        correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]

        prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]

        correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
        flow = correspondence - init_grid

        return flow, prob

class local_correlation_softmax(nn.Module):
    def forward(self, feature0, feature1, local_radius, padding_mode='zeros', ):
        b, c, h, w = feature0.size()
        coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
        coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

        local_h = 2 * local_radius + 1
        local_w = 2 * local_radius + 1

        window_grid = generate_window_grid(-local_radius, local_radius,
                                        -local_radius, local_radius,
                                        local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
        window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
        sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

        sample_coords_softmax = sample_coords

        # exclude coords that are out of image space
        # valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
        # valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]

        # valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

        # normalize coordinates to [-1, 1]
        cx = (w - 1) / 2.
        cy = (h - 1) / 2.
        sample_coords_x = (sample_coords[:, :, :, 0] - cx) / cx  # [B, H*W, (2R+1)^2]
        sample_coords_y = (sample_coords[:, :, :, 1] - cy) / cy  # [B, H*W, (2R+1)^2]
        sample_coords_norm = torch.stack([sample_coords_x, sample_coords_y], dim=-1)  # [B, H*W, (2R+1)^2, 2]
        # sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
        # sample_coords_norm = (sample_coords - c0) / c0  # [-1, 1]

        # print_mat(feature1, 'grid_sample_bottom0')
        # print_mat(sample_coords_norm, 'grid_sample_bottom1')
        window_feature = F.grid_sample(feature1, sample_coords_norm,
                                    padding_mode=padding_mode, align_corners=True
                                    ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
        # print_mat(window_feature, 'grid_sample_top0')
        
        feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

        # print_mat(feature0_view, 'local_corr_b0')
        # print_mat(window_feature, 'local_corr_b1')
        corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]
        # print_mat(corr, 'local_corr_t0')

        # mask invalid locations
        # corr[~valid] = -1e9

        # print_mat(corr, 'sm_b0')
        prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)^2]
        # print_mat(prob, 'sm_t0')


        correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(
            b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # print_mat(correspondence, 'subbbbbb_b0')
        # print_mat(coords_init, 'subbbbbb_b1')
        flow = correspondence - coords_init
        # print_mat(flow, 'subbbbbb_t0')

        match_prob = prob

        return flow, match_prob
