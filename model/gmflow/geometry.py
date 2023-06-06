import torch
import torch.nn.functional as F

from model.log import print_mat


def coords_grid(b, h, w, homogeneous=False, device=None):
    x = torch.arange(w, dtype=torch.float32).view(1, 1, 1, -1)
    y = torch.arange(h, dtype=torch.float32).view(1, 1, -1, 1)

    grid = torch.cat((x.expand(b, 1, h, w), y.expand(b, 1, h, w)), dim=1).to(device)

    if homogeneous:
        ones = torch.ones((b, 1, h, w), dtype=torch.float32, device=device)
        grid = torch.cat((grid, ones), dim=1)

    return grid


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x_values = torch.arange(w_min, w_max + 1, device=device)
    y_values = torch.arange(h_min, h_max + 1, device=device)

    x = x_values.view(1, -1).expand(len_h, -1)
    y = y_values.view(-1, 1).expand(-1, len_w)

    grid = torch.stack((x, y), 2).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    x_grid = x_grid.unsqueeze(3)  # [B, H, W, 1]
    y_grid = y_grid.unsqueeze(3)  # [B, H, W, 1]
    grid = torch.cat([x_grid, y_grid], dim=3)  # [B, H, W, 2]
    # grid = torch.stack([x_grid, y_grid], 3)  # [B, H, W, 2]
    # print_mat(img, 'grid_sample_bottom0')
    # print_mat(grid, 'grid_sample_bottom1')
    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    # print_mat(img, 'grid_sample_top0')

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]
    # print_mat(coords_grid(b, h, w), 'add_66_bottom0')
    # print_mat(flow, 'add_66_bottom1')
    # print_mat(grid, 'add_66_top0')

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta
    # print_mat(diff_fwd, 'gt_b0')
    # print_mat(threshold, 'gt_b1')
    # print_mat(diff_bwd, 'gt_b0')
    # print_mat(threshold, 'gt_b1')
    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()
    # print_mat(diff_fwd, 'gt_t0')
    # print_mat(diff_bwd, 'gt_t0')

    return fwd_occ, bwd_occ
