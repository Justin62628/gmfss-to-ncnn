import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
except:
    pass


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.backbone_conv1 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=3, kernel_size=(7, 7),
                                        out_channels=64, padding=(3, 3), padding_mode='zeros', stride=(2, 2))
        self.backbone_norm1 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.backbone_relu1 = nn.ReLU()
        self.backbone_layer1_0_conv1 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=64,
                                                 kernel_size=(3, 3), out_channels=64, padding=(1, 1),
                                                 padding_mode='zeros', stride=(1, 1))
        self.backbone_layer1_0_norm1 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.backbone_layer1_0_relu = nn.ReLU()
        self.backbone_layer1_0_conv2 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=64,
                                                 kernel_size=(3, 3), out_channels=64, padding=(1, 1),
                                                 padding_mode='zeros', stride=(1, 1))
        self.backbone_layer1_0_norm2 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.pnnx_unique_0 = nn.ReLU()
        self.pnnx_unique_1 = nn.ReLU()
        self.backbone_layer1_1_conv1 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=64,
                                                 kernel_size=(3, 3), out_channels=64, padding=(1, 1),
                                                 padding_mode='zeros', stride=(1, 1))
        self.backbone_layer1_1_norm1 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.backbone_layer1_1_relu = nn.ReLU()
        self.backbone_layer1_1_conv2 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=64,
                                                 kernel_size=(3, 3), out_channels=64, padding=(1, 1),
                                                 padding_mode='zeros', stride=(1, 1))
        self.backbone_layer1_1_norm2 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.pnnx_unique_2 = nn.ReLU()
        self.pnnx_unique_3 = nn.ReLU()
        self.backbone_layer2_0_conv1 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=64,
                                                 kernel_size=(3, 3), out_channels=96, padding=(1, 1),
                                                 padding_mode='zeros', stride=(2, 2))
        self.backbone_layer2_0_norm1 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.backbone_layer2_0_relu = nn.ReLU()
        self.backbone_layer2_0_conv2 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=96,
                                                 kernel_size=(3, 3), out_channels=96, padding=(1, 1),
                                                 padding_mode='zeros', stride=(1, 1))
        self.backbone_layer2_0_norm2 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.pnnx_unique_4 = nn.ReLU()
        self.backbone_layer2_0_downsample_0 = nn.Conv2d(bias=True, dilation=(1, 1), groups=1, in_channels=64,
                                                        kernel_size=(1, 1), out_channels=96, padding=(0, 0),
                                                        padding_mode='zeros', stride=(2, 2))
        self.backbone_layer2_0_downsample_1 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010,
                                                                    track_running_stats=False)
        self.pnnx_unique_5 = nn.ReLU()
        self.backbone_layer2_1_conv1 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=96,
                                                 kernel_size=(3, 3), out_channels=96, padding=(1, 1),
                                                 padding_mode='zeros', stride=(1, 1))
        self.backbone_layer2_1_norm1 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.backbone_layer2_1_relu = nn.ReLU()
        self.backbone_layer2_1_conv2 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=96,
                                                 kernel_size=(3, 3), out_channels=96, padding=(1, 1),
                                                 padding_mode='zeros', stride=(1, 1))
        self.backbone_layer2_1_norm2 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.pnnx_unique_6 = nn.ReLU()
        self.pnnx_unique_7 = nn.ReLU()
        self.backbone_layer3_0_conv1 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=96,
                                                 kernel_size=(3, 3), out_channels=128, padding=(1, 1),
                                                 padding_mode='zeros', stride=(1, 1))
        self.backbone_layer3_0_norm1 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.backbone_layer3_0_relu = nn.ReLU()
        self.backbone_layer3_0_conv2 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=128,
                                                 kernel_size=(3, 3), out_channels=128, padding=(1, 1),
                                                 padding_mode='zeros', stride=(1, 1))
        self.backbone_layer3_0_norm2 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.pnnx_unique_8 = nn.ReLU()
        self.backbone_layer3_0_downsample_0 = nn.Conv2d(bias=True, dilation=(1, 1), groups=1, in_channels=96,
                                                        kernel_size=(1, 1), out_channels=128, padding=(0, 0),
                                                        padding_mode='zeros', stride=(1, 1))
        self.backbone_layer3_0_downsample_1 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010,
                                                                    track_running_stats=False)
        self.pnnx_unique_9 = nn.ReLU()
        self.backbone_layer3_1_conv1 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=128,
                                                 kernel_size=(3, 3), out_channels=128, padding=(1, 1),
                                                 padding_mode='zeros', stride=(1, 1))
        self.backbone_layer3_1_norm1 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.backbone_layer3_1_relu = nn.ReLU()
        self.backbone_layer3_1_conv2 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=128,
                                                 kernel_size=(3, 3), out_channels=128, padding=(1, 1),
                                                 padding_mode='zeros', stride=(1, 1))
        self.backbone_layer3_1_norm2 = nn.LazyInstanceNorm2d(affine=False, eps=0.000010, track_running_stats=False)
        self.pnnx_unique_10 = nn.ReLU()
        self.pnnx_unique_11 = nn.ReLU()
        self.backbone_conv2 = nn.Conv2d(bias=True, dilation=(1, 1), groups=1, in_channels=128, kernel_size=(1, 1),
                                        out_channels=128, padding=(0, 0), padding_mode='zeros', stride=(1, 1))
        self.conv2d_0 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=128, kernel_size=(3, 3),
                                  out_channels=128, padding=(1, 1), padding_mode='zeros', stride=(2, 2))
        self.conv2d_1 = nn.Conv2d(bias=False, dilation=(1, 1), groups=1, in_channels=128, kernel_size=(3, 3),
                                  out_channels=128, padding=(1, 1), padding_mode='zeros', stride=(1, 1))
        self.transformer_layers_0_self_attn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_0_self_attn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_0_self_attn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_0_self_attn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_0_self_attn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                 normalized_shape=(128,))
        self.transformer_layers_0_cross_attn_ffn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_0_cross_attn_ffn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_0_cross_attn_ffn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_0_cross_attn_ffn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_0_cross_attn_ffn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.transformer_layers_0_cross_attn_ffn_mlp_0 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.transformer_layers_0_cross_attn_ffn_mlp_1 = nn.GELU()
        self.transformer_layers_0_cross_attn_ffn_mlp_2 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.transformer_layers_0_cross_attn_ffn_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.transformer_layers_1_self_attn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_1_self_attn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_1_self_attn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_1_self_attn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_1_self_attn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                 normalized_shape=(128,))
        self.transformer_layers_1_cross_attn_ffn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_1_cross_attn_ffn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_1_cross_attn_ffn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_1_cross_attn_ffn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_1_cross_attn_ffn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.transformer_layers_1_cross_attn_ffn_mlp_0 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.transformer_layers_1_cross_attn_ffn_mlp_1 = nn.GELU()
        self.transformer_layers_1_cross_attn_ffn_mlp_2 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.transformer_layers_1_cross_attn_ffn_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.transformer_layers_2_self_attn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_2_self_attn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_2_self_attn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_2_self_attn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_2_self_attn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                 normalized_shape=(128,))
        self.transformer_layers_2_cross_attn_ffn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_2_cross_attn_ffn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_2_cross_attn_ffn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_2_cross_attn_ffn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_2_cross_attn_ffn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.transformer_layers_2_cross_attn_ffn_mlp_0 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.transformer_layers_2_cross_attn_ffn_mlp_1 = nn.GELU()
        self.transformer_layers_2_cross_attn_ffn_mlp_2 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.transformer_layers_2_cross_attn_ffn_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.transformer_layers_3_self_attn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_3_self_attn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_3_self_attn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_3_self_attn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_3_self_attn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                 normalized_shape=(128,))
        self.transformer_layers_3_cross_attn_ffn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_3_cross_attn_ffn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_3_cross_attn_ffn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_3_cross_attn_ffn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_3_cross_attn_ffn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.transformer_layers_3_cross_attn_ffn_mlp_0 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.transformer_layers_3_cross_attn_ffn_mlp_1 = nn.GELU()
        self.transformer_layers_3_cross_attn_ffn_mlp_2 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.transformer_layers_3_cross_attn_ffn_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.transformer_layers_4_self_attn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_4_self_attn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_4_self_attn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_4_self_attn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_4_self_attn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                 normalized_shape=(128,))
        self.transformer_layers_4_cross_attn_ffn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_4_cross_attn_ffn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_4_cross_attn_ffn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_4_cross_attn_ffn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_4_cross_attn_ffn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.transformer_layers_4_cross_attn_ffn_mlp_0 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.transformer_layers_4_cross_attn_ffn_mlp_1 = nn.GELU()
        self.transformer_layers_4_cross_attn_ffn_mlp_2 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.transformer_layers_4_cross_attn_ffn_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.transformer_layers_5_self_attn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_5_self_attn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_5_self_attn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_5_self_attn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_5_self_attn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                 normalized_shape=(128,))
        self.transformer_layers_5_cross_attn_ffn_q_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_5_cross_attn_ffn_k_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_5_cross_attn_ffn_v_proj = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_5_cross_attn_ffn_merge = nn.Linear(bias=False, in_features=128, out_features=128)
        self.transformer_layers_5_cross_attn_ffn_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.transformer_layers_5_cross_attn_ffn_mlp_0 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.transformer_layers_5_cross_attn_ffn_mlp_1 = nn.GELU()
        self.transformer_layers_5_cross_attn_ffn_mlp_2 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.transformer_layers_5_cross_attn_ffn_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010,
                                                                      normalized_shape=(128,))
        self.feature_flow_attn_q_proj = nn.Linear(bias=True, in_features=128, out_features=128)
        self.feature_flow_attn_k_proj = nn.Linear(bias=True, in_features=128, out_features=128)
        self.pnnx_unique_12 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_13 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_14 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_15 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_16 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_17 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_18 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_19 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_20 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_21 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_22 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.pnnx_unique_23 = nn.GELU()
        self.pnnx_unique_24 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.pnnx_unique_25 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_26 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_27 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_28 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_29 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_30 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_31 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_32 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_33 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_34 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_35 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_36 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.pnnx_unique_37 = nn.GELU()
        self.pnnx_unique_38 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.pnnx_unique_39 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_40 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_41 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_42 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_43 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_44 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_45 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_46 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_47 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_48 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_49 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_50 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.pnnx_unique_51 = nn.GELU()
        self.pnnx_unique_52 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.pnnx_unique_53 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_54 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_55 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_56 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_57 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_58 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_59 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_60 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_61 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_62 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_63 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_64 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.pnnx_unique_65 = nn.GELU()
        self.pnnx_unique_66 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.pnnx_unique_67 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_68 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_69 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_70 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_71 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_72 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_73 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_74 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_75 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_76 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_77 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_78 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.pnnx_unique_79 = nn.GELU()
        self.pnnx_unique_80 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.pnnx_unique_81 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_82 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_83 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_84 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_85 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_86 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_87 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_88 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_89 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_90 = nn.Linear(bias=False, in_features=128, out_features=128)
        self.pnnx_unique_91 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_92 = nn.Linear(bias=False, in_features=256, out_features=1024)
        self.pnnx_unique_93 = nn.GELU()
        self.pnnx_unique_94 = nn.Linear(bias=False, in_features=1024, out_features=128)
        self.pnnx_unique_95 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(128,))
        self.pnnx_unique_96 = nn.Linear(bias=True, in_features=128, out_features=128)
        self.pnnx_unique_97 = nn.Linear(bias=True, in_features=128, out_features=128)
        self.upsampler_0 = nn.Conv2d(bias=True, dilation=(1, 1), groups=1, in_channels=130, kernel_size=(3, 3),
                                     out_channels=256, padding=(1, 1), padding_mode='zeros', stride=(1, 1))
        self.upsampler_1 = nn.ReLU()
        self.upsampler_2 = nn.Conv2d(bias=True, dilation=(1, 1), groups=1, in_channels=256, kernel_size=(1, 1),
                                     out_channels=144, padding=(0, 0), padding_mode='zeros', stride=(1, 1))

        archive = zipfile.ZipFile('D:/60-fps-Project/VFI/GMFSS2NCNN/flownet_288.pnnx.bin', 'r')
        self.backbone_conv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.conv1.weight', (64, 3, 7, 7),
                                                                     'float32')
        self.backbone_layer1_0_conv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer1.0.conv1.weight',
                                                                              (64, 64, 3, 3), 'float32')
        self.backbone_layer1_0_conv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer1.0.conv2.weight',
                                                                              (64, 64, 3, 3), 'float32')
        self.backbone_layer1_1_conv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer1.1.conv1.weight',
                                                                              (64, 64, 3, 3), 'float32')
        self.backbone_layer1_1_conv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer1.1.conv2.weight',
                                                                              (64, 64, 3, 3), 'float32')
        self.backbone_layer2_0_conv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer2.0.conv1.weight',
                                                                              (96, 64, 3, 3), 'float32')
        self.backbone_layer2_0_conv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer2.0.conv2.weight',
                                                                              (96, 96, 3, 3), 'float32')
        self.backbone_layer2_0_downsample_0.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                   'backbone.layer2.0.downsample.0.bias',
                                                                                   (96), 'float32')
        self.backbone_layer2_0_downsample_0.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                     'backbone.layer2.0.downsample.0.weight',
                                                                                     (96, 64, 1, 1), 'float32')
        self.backbone_layer2_1_conv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer2.1.conv1.weight',
                                                                              (96, 96, 3, 3), 'float32')
        self.backbone_layer2_1_conv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer2.1.conv2.weight',
                                                                              (96, 96, 3, 3), 'float32')
        self.backbone_layer3_0_conv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer3.0.conv1.weight',
                                                                              (128, 96, 3, 3), 'float32')
        self.backbone_layer3_0_conv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer3.0.conv2.weight',
                                                                              (128, 128, 3, 3), 'float32')
        self.backbone_layer3_0_downsample_0.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                   'backbone.layer3.0.downsample.0.bias',
                                                                                   (128), 'float32')
        self.backbone_layer3_0_downsample_0.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                     'backbone.layer3.0.downsample.0.weight',
                                                                                     (128, 96, 1, 1), 'float32')
        self.backbone_layer3_1_conv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer3.1.conv1.weight',
                                                                              (128, 128, 3, 3), 'float32')
        self.backbone_layer3_1_conv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.layer3.1.conv2.weight',
                                                                              (128, 128, 3, 3), 'float32')
        self.backbone_conv2.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.conv2.bias', (128), 'float32')
        self.backbone_conv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.conv2.weight', (128, 128, 1, 1),
                                                                     'float32')
        self.conv2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_0.weight', (128, 128, 3, 3), 'float32')
        self.conv2d_1.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_1.weight', (128, 128, 3, 3), 'float32')
        self.transformer_layers_0_self_attn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.0.self_attn.q_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_0_self_attn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.0.self_attn.k_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_0_self_attn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.0.self_attn.v_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_0_self_attn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.0.self_attn.merge.weight',
                                                                                           (128, 128), 'float32')
        self.transformer_layers_0_self_attn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                         'transformer.layers.0.self_attn.norm1.bias',
                                                                                         (128), 'float32')
        self.transformer_layers_0_self_attn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.0.self_attn.norm1.weight',
                                                                                           (128), 'float32')
        self.transformer_layers_0_cross_attn_ffn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.0.cross_attn_ffn.q_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_0_cross_attn_ffn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.0.cross_attn_ffn.k_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_0_cross_attn_ffn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.0.cross_attn_ffn.v_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_0_cross_attn_ffn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.0.cross_attn_ffn.merge.weight',
                                                                                                (128, 128), 'float32')
        self.transformer_layers_0_cross_attn_ffn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.0.cross_attn_ffn.norm1.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_0_cross_attn_ffn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.0.cross_attn_ffn.norm1.weight',
                                                                                                (128), 'float32')
        self.transformer_layers_0_cross_attn_ffn_mlp_0.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.0.cross_attn_ffn.mlp.0.weight',
                                                                                                (1024, 256), 'float32')
        self.transformer_layers_0_cross_attn_ffn_mlp_2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.0.cross_attn_ffn.mlp.2.weight',
                                                                                                (128, 1024), 'float32')
        self.transformer_layers_0_cross_attn_ffn_norm2.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.0.cross_attn_ffn.norm2.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_0_cross_attn_ffn_norm2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.0.cross_attn_ffn.norm2.weight',
                                                                                                (128), 'float32')
        self.transformer_layers_1_self_attn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.1.self_attn.q_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_1_self_attn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.1.self_attn.k_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_1_self_attn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.1.self_attn.v_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_1_self_attn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.1.self_attn.merge.weight',
                                                                                           (128, 128), 'float32')
        self.transformer_layers_1_self_attn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                         'transformer.layers.1.self_attn.norm1.bias',
                                                                                         (128), 'float32')
        self.transformer_layers_1_self_attn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.1.self_attn.norm1.weight',
                                                                                           (128), 'float32')
        self.transformer_layers_1_cross_attn_ffn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.1.cross_attn_ffn.q_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_1_cross_attn_ffn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.1.cross_attn_ffn.k_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_1_cross_attn_ffn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.1.cross_attn_ffn.v_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_1_cross_attn_ffn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.1.cross_attn_ffn.merge.weight',
                                                                                                (128, 128), 'float32')
        self.transformer_layers_1_cross_attn_ffn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.1.cross_attn_ffn.norm1.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_1_cross_attn_ffn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.1.cross_attn_ffn.norm1.weight',
                                                                                                (128), 'float32')
        self.transformer_layers_1_cross_attn_ffn_mlp_0.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.1.cross_attn_ffn.mlp.0.weight',
                                                                                                (1024, 256), 'float32')
        self.transformer_layers_1_cross_attn_ffn_mlp_2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.1.cross_attn_ffn.mlp.2.weight',
                                                                                                (128, 1024), 'float32')
        self.transformer_layers_1_cross_attn_ffn_norm2.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.1.cross_attn_ffn.norm2.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_1_cross_attn_ffn_norm2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.1.cross_attn_ffn.norm2.weight',
                                                                                                (128), 'float32')
        self.transformer_layers_2_self_attn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.2.self_attn.q_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_2_self_attn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.2.self_attn.k_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_2_self_attn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.2.self_attn.v_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_2_self_attn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.2.self_attn.merge.weight',
                                                                                           (128, 128), 'float32')
        self.transformer_layers_2_self_attn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                         'transformer.layers.2.self_attn.norm1.bias',
                                                                                         (128), 'float32')
        self.transformer_layers_2_self_attn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.2.self_attn.norm1.weight',
                                                                                           (128), 'float32')
        self.transformer_layers_2_cross_attn_ffn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.2.cross_attn_ffn.q_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_2_cross_attn_ffn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.2.cross_attn_ffn.k_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_2_cross_attn_ffn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.2.cross_attn_ffn.v_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_2_cross_attn_ffn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.2.cross_attn_ffn.merge.weight',
                                                                                                (128, 128), 'float32')
        self.transformer_layers_2_cross_attn_ffn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.2.cross_attn_ffn.norm1.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_2_cross_attn_ffn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.2.cross_attn_ffn.norm1.weight',
                                                                                                (128), 'float32')
        self.transformer_layers_2_cross_attn_ffn_mlp_0.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.2.cross_attn_ffn.mlp.0.weight',
                                                                                                (1024, 256), 'float32')
        self.transformer_layers_2_cross_attn_ffn_mlp_2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.2.cross_attn_ffn.mlp.2.weight',
                                                                                                (128, 1024), 'float32')
        self.transformer_layers_2_cross_attn_ffn_norm2.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.2.cross_attn_ffn.norm2.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_2_cross_attn_ffn_norm2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.2.cross_attn_ffn.norm2.weight',
                                                                                                (128), 'float32')
        self.transformer_layers_3_self_attn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.3.self_attn.q_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_3_self_attn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.3.self_attn.k_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_3_self_attn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.3.self_attn.v_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_3_self_attn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.3.self_attn.merge.weight',
                                                                                           (128, 128), 'float32')
        self.transformer_layers_3_self_attn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                         'transformer.layers.3.self_attn.norm1.bias',
                                                                                         (128), 'float32')
        self.transformer_layers_3_self_attn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.3.self_attn.norm1.weight',
                                                                                           (128), 'float32')
        self.transformer_layers_3_cross_attn_ffn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.3.cross_attn_ffn.q_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_3_cross_attn_ffn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.3.cross_attn_ffn.k_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_3_cross_attn_ffn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.3.cross_attn_ffn.v_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_3_cross_attn_ffn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.3.cross_attn_ffn.merge.weight',
                                                                                                (128, 128), 'float32')
        self.transformer_layers_3_cross_attn_ffn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.3.cross_attn_ffn.norm1.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_3_cross_attn_ffn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.3.cross_attn_ffn.norm1.weight',
                                                                                                (128), 'float32')
        self.transformer_layers_3_cross_attn_ffn_mlp_0.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.3.cross_attn_ffn.mlp.0.weight',
                                                                                                (1024, 256), 'float32')
        self.transformer_layers_3_cross_attn_ffn_mlp_2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.3.cross_attn_ffn.mlp.2.weight',
                                                                                                (128, 1024), 'float32')
        self.transformer_layers_3_cross_attn_ffn_norm2.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.3.cross_attn_ffn.norm2.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_3_cross_attn_ffn_norm2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.3.cross_attn_ffn.norm2.weight',
                                                                                                (128), 'float32')
        self.transformer_layers_4_self_attn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.4.self_attn.q_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_4_self_attn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.4.self_attn.k_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_4_self_attn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.4.self_attn.v_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_4_self_attn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.4.self_attn.merge.weight',
                                                                                           (128, 128), 'float32')
        self.transformer_layers_4_self_attn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                         'transformer.layers.4.self_attn.norm1.bias',
                                                                                         (128), 'float32')
        self.transformer_layers_4_self_attn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.4.self_attn.norm1.weight',
                                                                                           (128), 'float32')
        self.transformer_layers_4_cross_attn_ffn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.4.cross_attn_ffn.q_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_4_cross_attn_ffn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.4.cross_attn_ffn.k_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_4_cross_attn_ffn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.4.cross_attn_ffn.v_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_4_cross_attn_ffn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.4.cross_attn_ffn.merge.weight',
                                                                                                (128, 128), 'float32')
        self.transformer_layers_4_cross_attn_ffn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.4.cross_attn_ffn.norm1.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_4_cross_attn_ffn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.4.cross_attn_ffn.norm1.weight',
                                                                                                (128), 'float32')
        self.transformer_layers_4_cross_attn_ffn_mlp_0.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.4.cross_attn_ffn.mlp.0.weight',
                                                                                                (1024, 256), 'float32')
        self.transformer_layers_4_cross_attn_ffn_mlp_2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.4.cross_attn_ffn.mlp.2.weight',
                                                                                                (128, 1024), 'float32')
        self.transformer_layers_4_cross_attn_ffn_norm2.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.4.cross_attn_ffn.norm2.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_4_cross_attn_ffn_norm2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.4.cross_attn_ffn.norm2.weight',
                                                                                                (128), 'float32')
        self.transformer_layers_5_self_attn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.5.self_attn.q_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_5_self_attn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.5.self_attn.k_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_5_self_attn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                            'transformer.layers.5.self_attn.v_proj.weight',
                                                                                            (128, 128), 'float32')
        self.transformer_layers_5_self_attn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.5.self_attn.merge.weight',
                                                                                           (128, 128), 'float32')
        self.transformer_layers_5_self_attn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                         'transformer.layers.5.self_attn.norm1.bias',
                                                                                         (128), 'float32')
        self.transformer_layers_5_self_attn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'transformer.layers.5.self_attn.norm1.weight',
                                                                                           (128), 'float32')
        self.transformer_layers_5_cross_attn_ffn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.5.cross_attn_ffn.q_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_5_cross_attn_ffn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.5.cross_attn_ffn.k_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_5_cross_attn_ffn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                 'transformer.layers.5.cross_attn_ffn.v_proj.weight',
                                                                                                 (128, 128), 'float32')
        self.transformer_layers_5_cross_attn_ffn_merge.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.5.cross_attn_ffn.merge.weight',
                                                                                                (128, 128), 'float32')
        self.transformer_layers_5_cross_attn_ffn_norm1.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.5.cross_attn_ffn.norm1.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_5_cross_attn_ffn_norm1.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.5.cross_attn_ffn.norm1.weight',
                                                                                                (128), 'float32')
        self.transformer_layers_5_cross_attn_ffn_mlp_0.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.5.cross_attn_ffn.mlp.0.weight',
                                                                                                (1024, 256), 'float32')
        self.transformer_layers_5_cross_attn_ffn_mlp_2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.5.cross_attn_ffn.mlp.2.weight',
                                                                                                (128, 1024), 'float32')
        self.transformer_layers_5_cross_attn_ffn_norm2.bias = self.load_pnnx_bin_as_parameter(archive,
                                                                                              'transformer.layers.5.cross_attn_ffn.norm2.bias',
                                                                                              (128), 'float32')
        self.transformer_layers_5_cross_attn_ffn_norm2.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                                                'transformer.layers.5.cross_attn_ffn.norm2.weight',
                                                                                                (128), 'float32')
        self.feature_flow_attn_q_proj.bias = self.load_pnnx_bin_as_parameter(archive, 'feature_flow_attn.q_proj.bias',
                                                                             (128), 'float32')
        self.feature_flow_attn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                               'feature_flow_attn.q_proj.weight',
                                                                               (128, 128), 'float32')
        self.feature_flow_attn_k_proj.bias = self.load_pnnx_bin_as_parameter(archive, 'feature_flow_attn.k_proj.bias',
                                                                             (128), 'float32')
        self.feature_flow_attn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive,
                                                                               'feature_flow_attn.k_proj.weight',
                                                                               (128, 128), 'float32')
        self.pnnx_unique_12.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_12.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_13.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_13.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_14.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_14.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_15.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_15.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_16.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_16.bias', (128), 'float32')
        self.pnnx_unique_16.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_16.weight', (128), 'float32')
        self.pnnx_unique_17.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_17.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_18.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_18.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_19.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_19.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_20.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_20.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_21.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_21.bias', (128), 'float32')
        self.pnnx_unique_21.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_21.weight', (128), 'float32')
        self.pnnx_unique_22.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_22.weight', (1024, 256),
                                                                     'float32')
        self.pnnx_unique_24.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_24.weight', (128, 1024),
                                                                     'float32')
        self.pnnx_unique_25.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_25.bias', (128), 'float32')
        self.pnnx_unique_25.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_25.weight', (128), 'float32')
        self.pnnx_unique_26.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_26.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_27.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_27.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_28.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_28.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_29.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_29.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_30.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_30.bias', (128), 'float32')
        self.pnnx_unique_30.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_30.weight', (128), 'float32')
        self.pnnx_unique_31.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_31.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_32.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_32.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_33.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_33.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_34.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_34.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_35.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_35.bias', (128), 'float32')
        self.pnnx_unique_35.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_35.weight', (128), 'float32')
        self.pnnx_unique_36.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_36.weight', (1024, 256),
                                                                     'float32')
        self.pnnx_unique_38.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_38.weight', (128, 1024),
                                                                     'float32')
        self.pnnx_unique_39.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_39.bias', (128), 'float32')
        self.pnnx_unique_39.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_39.weight', (128), 'float32')
        self.pnnx_unique_40.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_40.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_41.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_41.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_42.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_42.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_43.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_43.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_44.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_44.bias', (128), 'float32')
        self.pnnx_unique_44.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_44.weight', (128), 'float32')
        self.pnnx_unique_45.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_45.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_46.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_46.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_47.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_47.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_48.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_48.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_49.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_49.bias', (128), 'float32')
        self.pnnx_unique_49.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_49.weight', (128), 'float32')
        self.pnnx_unique_50.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_50.weight', (1024, 256),
                                                                     'float32')
        self.pnnx_unique_52.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_52.weight', (128, 1024),
                                                                     'float32')
        self.pnnx_unique_53.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_53.bias', (128), 'float32')
        self.pnnx_unique_53.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_53.weight', (128), 'float32')
        self.pnnx_unique_54.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_54.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_55.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_55.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_56.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_56.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_57.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_57.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_58.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_58.bias', (128), 'float32')
        self.pnnx_unique_58.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_58.weight', (128), 'float32')
        self.pnnx_unique_59.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_59.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_60.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_60.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_61.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_61.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_62.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_62.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_63.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_63.bias', (128), 'float32')
        self.pnnx_unique_63.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_63.weight', (128), 'float32')
        self.pnnx_unique_64.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_64.weight', (1024, 256),
                                                                     'float32')
        self.pnnx_unique_66.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_66.weight', (128, 1024),
                                                                     'float32')
        self.pnnx_unique_67.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_67.bias', (128), 'float32')
        self.pnnx_unique_67.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_67.weight', (128), 'float32')
        self.pnnx_unique_68.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_68.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_69.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_69.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_70.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_70.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_71.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_71.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_72.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_72.bias', (128), 'float32')
        self.pnnx_unique_72.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_72.weight', (128), 'float32')
        self.pnnx_unique_73.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_73.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_74.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_74.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_75.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_75.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_76.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_76.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_77.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_77.bias', (128), 'float32')
        self.pnnx_unique_77.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_77.weight', (128), 'float32')
        self.pnnx_unique_78.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_78.weight', (1024, 256),
                                                                     'float32')
        self.pnnx_unique_80.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_80.weight', (128, 1024),
                                                                     'float32')
        self.pnnx_unique_81.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_81.bias', (128), 'float32')
        self.pnnx_unique_81.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_81.weight', (128), 'float32')
        self.pnnx_unique_82.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_82.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_83.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_83.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_84.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_84.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_85.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_85.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_86.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_86.bias', (128), 'float32')
        self.pnnx_unique_86.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_86.weight', (128), 'float32')
        self.pnnx_unique_87.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_87.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_88.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_88.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_89.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_89.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_90.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_90.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_91.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_91.bias', (128), 'float32')
        self.pnnx_unique_91.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_91.weight', (128), 'float32')
        self.pnnx_unique_92.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_92.weight', (1024, 256),
                                                                     'float32')
        self.pnnx_unique_94.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_94.weight', (128, 1024),
                                                                     'float32')
        self.pnnx_unique_95.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_95.bias', (128), 'float32')
        self.pnnx_unique_95.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_95.weight', (128), 'float32')
        self.pnnx_unique_96.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_96.bias', (128), 'float32')
        self.pnnx_unique_96.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_96.weight', (128, 128),
                                                                     'float32')
        self.pnnx_unique_97.bias = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_97.bias', (128), 'float32')
        self.pnnx_unique_97.weight = self.load_pnnx_bin_as_parameter(archive, 'pnnx_unique_97.weight', (128, 128),
                                                                     'float32')
        self.upsampler_0.bias = self.load_pnnx_bin_as_parameter(archive, 'upsampler.0.bias', (256), 'float32')
        self.upsampler_0.weight = self.load_pnnx_bin_as_parameter(archive, 'upsampler.0.weight', (256, 130, 3, 3),
                                                                  'float32')
        self.upsampler_2.bias = self.load_pnnx_bin_as_parameter(archive, 'upsampler.2.bias', (144), 'float32')
        self.upsampler_2.weight = self.load_pnnx_bin_as_parameter(archive, 'upsampler.2.weight', (144, 256, 1, 1),
                                                                  'float32')
        self.pnnx_fold_mean_1_pnnx_fold_mean_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                                 'pnnx_fold_mean.1.pnnx_fold_mean.1',
                                                                                 (1, 3, 288, 480), 'float32')
        self.pnnx_fold_mean_1_1_pnnx_fold_mean_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                                   'pnnx_fold_mean.1_1.pnnx_fold_mean.1',
                                                                                   (1, 3, 288, 480), 'float32')
        self.pnnx_fold_std_1_pnnx_fold_std_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                               'pnnx_fold_std.1.pnnx_fold_std.1',
                                                                               (1, 3, 288, 480), 'float32')
        self.pnnx_fold_std_1_1_pnnx_fold_std_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                                 'pnnx_fold_std.1_1.pnnx_fold_std.1',
                                                                                 (1, 3, 288, 480), 'float32')
        self.pnnx_fold_position_1_pnnx_fold_position_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                                         'pnnx_fold_position.1.pnnx_fold_position.1',
                                                                                         (4, 128, 18, 30), 'float32')
        self.pnnx_fold_position_1_1_pnnx_fold_position_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'pnnx_fold_position.1_1.pnnx_fold_position.1',
                                                                                           (4, 128, 18, 30), 'float32')
        self.pnnx_fold_2325_pnnx_fold_2325 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_2325.pnnx_fold_2325',
                                                                             (8, 540, 540), 'float32')
        self.pnnx_fold_2542_pnnx_fold_2542 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_2542.pnnx_fold_2542',
                                                                             (8, 540, 540), 'float32')
        self.pnnx_fold_3153_pnnx_fold_3153 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_3153.pnnx_fold_3153',
                                                                             (8, 540, 540), 'float32')
        self.pnnx_fold_3370_pnnx_fold_3370 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_3370.pnnx_fold_3370',
                                                                             (8, 540, 540), 'float32')
        self.pnnx_fold_3981_pnnx_fold_3981 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_3981.pnnx_fold_3981',
                                                                             (8, 540, 540), 'float32')
        self.pnnx_fold_4198_pnnx_fold_4198 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_4198.pnnx_fold_4198',
                                                                             (8, 540, 540), 'float32')
        self.pnnx_fold_init_grid_1_pnnx_fold_init_grid_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'pnnx_fold_init_grid.1.pnnx_fold_init_grid.1',
                                                                                           (1, 2, 36, 60), 'float32')
        self.pnnx_fold_grid_5_pnnx_fold_grid_5 = self.load_pnnx_bin_as_parameter(archive,
                                                                                 'pnnx_fold_grid.5.pnnx_fold_grid.5',
                                                                                 (1, 2160, 2), 'float32')
        self.pnnx_fold_732_pnnx_fold_732 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_732.pnnx_fold_732',
                                                                           (1, 2, 72, 120), 'float32')
        self.pnnx_fold_position0_1_pnnx_fold_position0_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                                           'pnnx_fold_position0.1.pnnx_fold_position0.1',
                                                                                           (64, 128, 9, 15), 'float32')
        self.pnnx_fold_position0_1_1_pnnx_fold_position0_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                                             'pnnx_fold_position0.1_1.pnnx_fold_position0.1',
                                                                                             (64, 128, 9, 15),
                                                                                             'float32')
        self.pnnx_fold_5210_pnnx_fold_5210 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_5210.pnnx_fold_5210',
                                                                             (128, 135, 135), 'float32')
        self.pnnx_fold_5429_pnnx_fold_5429 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_5429.pnnx_fold_5429',
                                                                             (128, 135, 135), 'float32')
        self.pnnx_fold_6044_pnnx_fold_6044 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_6044.pnnx_fold_6044',
                                                                             (128, 135, 135), 'float32')
        self.pnnx_fold_6263_pnnx_fold_6263 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_6263.pnnx_fold_6263',
                                                                             (128, 135, 135), 'float32')
        self.pnnx_fold_6878_pnnx_fold_6878 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_6878.pnnx_fold_6878',
                                                                             (128, 135, 135), 'float32')
        self.pnnx_fold_7097_pnnx_fold_7097 = self.load_pnnx_bin_as_parameter(archive, 'pnnx_fold_7097.pnnx_fold_7097',
                                                                             (128, 135, 135), 'float32')
        self.pnnx_fold_coords_init_1_pnnx_fold_coords_init_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                                               'pnnx_fold_coords_init.1.pnnx_fold_coords_init.1',
                                                                                               (1, 2, 72, 120),
                                                                                               'float32')
        self.pnnx_fold_coords0_1_pnnx_fold_coords0_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                                       'pnnx_fold_coords0.1.pnnx_fold_coords0.1',
                                                                                       (1, 8640, 81, 2), 'float32')
        self.pnnx_fold_grid_1_pnnx_fold_grid_1 = self.load_pnnx_bin_as_parameter(archive,
                                                                                 'pnnx_fold_grid.1.pnnx_fold_grid.1',
                                                                                 (1, 8640, 81, 2), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        _, tmppath = tempfile.mkstemp(dir='tmp')
        tmpf = open(tmppath, 'wb')
        with archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        tmpf.close()
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        # os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0, v_1):
        v_2 = self.pnnx_fold_mean_1_pnnx_fold_mean_1
        v_3 = self.pnnx_fold_mean_1_1_pnnx_fold_mean_1
        v_4 = self.pnnx_fold_std_1_pnnx_fold_std_1
        v_5 = self.pnnx_fold_std_1_1_pnnx_fold_std_1
        v_6 = ((v_0 - v_2) / v_4)
        v_7 = ((v_1 - v_3) / v_5)
        # region CNNEncoder
        v_8 = torch.cat((v_6, v_7), dim=0)
        v_9 = self.backbone_conv1(v_8)
        v_10 = self.backbone_norm1(v_9)
        v_11 = self.backbone_relu1(v_10)
        v_12 = self.backbone_layer1_0_conv1(v_11)
        v_13 = self.backbone_layer1_0_norm1(v_12)
        v_14 = self.backbone_layer1_0_relu(v_13)
        v_15 = self.backbone_layer1_0_conv2(v_14)
        v_16 = self.backbone_layer1_0_norm2(v_15)
        v_17 = self.pnnx_unique_0(v_16)
        v_18 = (v_11 + v_17)
        v_19 = self.pnnx_unique_1(v_18)
        v_20 = self.backbone_layer1_1_conv1(v_19)
        v_21 = self.backbone_layer1_1_norm1(v_20)
        v_22 = self.backbone_layer1_1_relu(v_21)
        v_23 = self.backbone_layer1_1_conv2(v_22)
        v_24 = self.backbone_layer1_1_norm2(v_23)
        v_25 = self.pnnx_unique_2(v_24)
        v_26 = (v_19 + v_25)
        v_27 = self.pnnx_unique_3(v_26)
        v_28 = self.backbone_layer2_0_conv1(v_27)
        v_29 = self.backbone_layer2_0_norm1(v_28)
        v_30 = self.backbone_layer2_0_relu(v_29)
        v_31 = self.backbone_layer2_0_conv2(v_30)
        v_32 = self.backbone_layer2_0_norm2(v_31)
        v_33 = self.pnnx_unique_4(v_32)
        v_34 = self.backbone_layer2_0_downsample_0(v_27)
        v_35 = self.backbone_layer2_0_downsample_1(v_34)
        v_36 = (v_35 + v_33)
        v_37 = self.pnnx_unique_5(v_36)
        v_38 = self.backbone_layer2_1_conv1(v_37)
        v_39 = self.backbone_layer2_1_norm1(v_38)
        v_40 = self.backbone_layer2_1_relu(v_39)
        v_41 = self.backbone_layer2_1_conv2(v_40)
        v_42 = self.backbone_layer2_1_norm2(v_41)
        v_43 = self.pnnx_unique_6(v_42)
        v_44 = (v_37 + v_43)
        v_45 = self.pnnx_unique_7(v_44)
        v_46 = self.backbone_layer3_0_conv1(v_45)
        v_47 = self.backbone_layer3_0_norm1(v_46)
        v_48 = self.backbone_layer3_0_relu(v_47)
        v_49 = self.backbone_layer3_0_conv2(v_48)
        v_50 = self.backbone_layer3_0_norm2(v_49)
        v_51 = self.pnnx_unique_8(v_50)
        v_52 = self.backbone_layer3_0_downsample_0(v_45)
        v_53 = self.backbone_layer3_0_downsample_1(v_52)
        v_54 = (v_53 + v_51)
        v_55 = self.pnnx_unique_9(v_54)
        v_56 = self.backbone_layer3_1_conv1(v_55)
        v_57 = self.backbone_layer3_1_norm1(v_56)
        v_58 = self.backbone_layer3_1_relu(v_57)
        v_59 = self.backbone_layer3_1_conv2(v_58)
        v_60 = self.backbone_layer3_1_norm2(v_59)
        v_61 = self.pnnx_unique_10(v_60)
        v_62 = (v_55 + v_61)
        v_63 = self.pnnx_unique_11(v_62)
        v_64 = self.backbone_conv2(v_63)
        # endregion
        v_65 = self.conv2d_0(v_64)
        v_66, v_67 = torch.chunk(input=v_65, chunks=2, dim=0)  # v_66: 1,128,36,60
        v_68 = self.conv2d_1(v_64)
        v_69, v_70 = torch.chunk(input=v_68, chunks=2, dim=0)  # v_69: 1,128,72,120

        pixel_unshuffle = nn.PixelUnshuffle(2)  # The upscale factor is 2 because you're splitting each dimension by 2
        v_71 = pixel_unshuffle(v_66)
        v_72 = pixel_unshuffle(v_67)

        # v_71 = v_66.view(1, 128, 2, 18, 2, 30)
        # v_72 = v_67.view(1, 128, 2, 18, 2, 30)
        # v_73 = torch.permute(input=v_71, dims=(0, 2, 4, 1, 3, 5))   # split feature
        v_74 = v_71.reshape(4, 128, 18, 30)  # split feature
        v_75 = self.pnnx_fold_position_1_pnnx_fold_position_1  # pos_enc position
        v_76 = self.pnnx_fold_position_1_1_pnnx_fold_position_1
        v_77 = (v_74 + v_75)
        # v_78 = torch.permute(input=v_72, dims=(0, 2, 4, 1, 3, 5))
        v_79 = v_72.reshape(4, 128, 18, 30)
        v_80 = (v_79 + v_76)
        v_81 = v_77.view(1, 2, 2, 128, 18, 30)
        v_82 = torch.permute(input=v_81, dims=(0, 3, 1, 4, 2, 5))
        v_83 = v_80.view(1, 2, 2, 128, 18, 30)
        v_84 = torch.permute(input=v_83, dims=(0, 3, 1, 4, 2, 5))
        v_85 = v_82.reshape(1, 128, 36, 60)
        v_86 = torch.flatten(input=v_85, end_dim=-1, start_dim=-2)  # transformer, FeatureTransformer
        v_87 = v_84.reshape(1, 128, 36, 60)
        v_88 = torch.flatten(input=v_87, end_dim=-1, start_dim=-2)
        v_89 = torch.permute(input=v_88, dims=(0, 2, 1))  # global_correlation_softmax
        v_90 = torch.permute(input=v_86, dims=(0, 2, 1))
        v_91 = torch.cat((v_90, v_89), dim=0)
        v_92 = self.transformer_layers_0_self_attn_q_proj(v_91)
        v_93 = self.transformer_layers_0_self_attn_k_proj(v_91)
        v_94 = self.transformer_layers_0_self_attn_v_proj(v_91)
        v_95 = v_92.reshape(2, 2, 18, 2, 30, 128)
        v_96 = v_93.reshape(2, 2, 18, 2, 30, 128)
        v_97 = v_94.reshape(2, 2, 18, 2, 30, 128)
        v_98 = torch.permute(input=v_95, dims=(0, 1, 3, 2, 4, 5))
        v_99 = torch.permute(input=v_96, dims=(0, 1, 3, 2, 4, 5))
        v_100 = v_99.reshape(8, 540, 128)
        v_101 = v_98.reshape(8, 540, 128)
        v_102 = torch.permute(input=v_100, dims=(0, 2, 1))
        v_103 = torch.matmul(input=v_101, other=v_102)
        v_104 = (v_103 / 11.313708)
        v_105 = torch.permute(input=v_97, dims=(0, 1, 3, 2, 4, 5))
        v_106 = F.softmax(input=v_104, dim=-1)
        v_107 = v_105.reshape(8, 540, 128)
        v_108 = torch.matmul(input=v_106, other=v_107)
        v_109 = v_108.reshape(2, 2, 2, 18, 30, 128)
        v_110 = torch.permute(input=v_109, dims=(0, 1, 3, 2, 4, 5))
        v_111 = v_110.reshape(2, 2160, 128)
        v_112 = self.transformer_layers_0_self_attn_merge(v_111)
        v_113 = self.transformer_layers_0_self_attn_norm1(v_112)
        v_114 = (v_91 + v_113)
        v_115 = self.transformer_layers_0_cross_attn_ffn_q_proj(v_114)
        v_116 = torch.cat((v_89, v_90), dim=0)
        v_117 = self.transformer_layers_0_cross_attn_ffn_k_proj(v_116)
        v_118 = self.transformer_layers_0_cross_attn_ffn_v_proj(v_116)
        v_119 = v_115.reshape(2, 2, 18, 2, 30, 128)
        v_120 = v_117.reshape(2, 2, 18, 2, 30, 128)
        v_121 = v_118.reshape(2, 2, 18, 2, 30, 128)
        v_122 = torch.permute(input=v_119, dims=(0, 1, 3, 2, 4, 5))
        v_123 = torch.permute(input=v_120, dims=(0, 1, 3, 2, 4, 5))
        v_124 = v_123.reshape(8, 540, 128)
        v_125 = v_122.reshape(8, 540, 128)
        v_126 = torch.permute(input=v_124, dims=(0, 2, 1))
        v_127 = torch.matmul(input=v_125, other=v_126)
        v_128 = (v_127 / 11.313708)
        v_129 = torch.permute(input=v_121, dims=(0, 1, 3, 2, 4, 5))
        v_130 = F.softmax(input=v_128, dim=-1)
        v_131 = v_129.reshape(8, 540, 128)
        v_132 = torch.matmul(input=v_130, other=v_131)
        v_133 = v_132.reshape(2, 2, 2, 18, 30, 128)
        v_134 = torch.permute(input=v_133, dims=(0, 1, 3, 2, 4, 5))
        v_135 = v_134.reshape(2, 2160, 128)
        v_136 = self.transformer_layers_0_cross_attn_ffn_merge(v_135)
        v_137 = self.transformer_layers_0_cross_attn_ffn_norm1(v_136)
        v_138 = torch.cat((v_114, v_137), dim=-1)
        v_139 = self.transformer_layers_0_cross_attn_ffn_mlp_0(v_138)
        v_140 = self.transformer_layers_0_cross_attn_ffn_mlp_1(v_139)
        v_141 = self.transformer_layers_0_cross_attn_ffn_mlp_2(v_140)
        v_142 = self.transformer_layers_0_cross_attn_ffn_norm2(v_141)
        v_143 = (v_114 + v_142)
        v_144, v_145 = torch.chunk(input=v_143, chunks=2, dim=0)
        v_146 = self.transformer_layers_1_self_attn_q_proj(v_143)
        v_147 = self.transformer_layers_1_self_attn_k_proj(v_143)
        v_148 = self.transformer_layers_1_self_attn_v_proj(v_143)
        v_149 = v_146.view(2, 36, 60, 128)
        v_150 = torch.roll(input=v_149, dims=(1, 2), shifts=(-9, -15))
        v_151 = v_150.view(2, 2, 18, 2, 30, 128)
        v_152 = v_147.view(2, 36, 60, 128)
        v_153 = torch.roll(input=v_152, dims=(1, 2), shifts=(-9, -15))
        v_154 = v_153.view(2, 2, 18, 2, 30, 128)
        v_155 = v_148.view(2, 36, 60, 128)
        v_156 = torch.roll(input=v_155, dims=(1, 2), shifts=(-9, -15))
        v_157 = v_156.view(2, 2, 18, 2, 30, 128)
        v_158 = torch.permute(input=v_151, dims=(0, 1, 3, 2, 4, 5))
        v_159 = torch.permute(input=v_154, dims=(0, 1, 3, 2, 4, 5))
        v_160 = v_159.reshape(8, 540, 128)
        v_161 = v_158.reshape(8, 540, 128)
        v_162 = torch.permute(input=v_160, dims=(0, 2, 1))
        v_163 = torch.matmul(input=v_161, other=v_162)
        v_164 = self.pnnx_fold_2325_pnnx_fold_2325
        v_165 = ((v_163 / 11.313708) + v_164)
        v_166 = torch.permute(input=v_157, dims=(0, 1, 3, 2, 4, 5))
        v_167 = F.softmax(input=v_165, dim=-1)
        v_168 = v_166.reshape(8, 540, 128)
        v_169 = torch.matmul(input=v_167, other=v_168)
        v_170 = v_169.reshape(2, 2, 2, 18, 30, 128)
        v_171 = torch.permute(input=v_170, dims=(0, 1, 3, 2, 4, 5))
        v_172 = v_171.reshape(2, 36, 60, 128)
        v_173 = torch.roll(input=v_172, dims=(1, 2), shifts=(9, 15))
        v_174 = v_173.view(2, -1, 128)
        v_175 = self.transformer_layers_1_self_attn_merge(v_174)
        v_176 = self.transformer_layers_1_self_attn_norm1(v_175)
        v_177 = (v_143 + v_176)
        v_178 = self.transformer_layers_1_cross_attn_ffn_q_proj(v_177)
        v_179 = torch.cat((v_145, v_144), dim=0)
        v_180 = self.transformer_layers_1_cross_attn_ffn_k_proj(v_179)
        v_181 = self.transformer_layers_1_cross_attn_ffn_v_proj(v_179)
        v_182 = v_178.view(2, 36, 60, 128)
        v_183 = torch.roll(input=v_182, dims=(1, 2), shifts=(-9, -15))
        v_184 = v_183.view(2, 2, 18, 2, 30, 128)
        v_185 = v_180.view(2, 36, 60, 128)
        v_186 = torch.roll(input=v_185, dims=(1, 2), shifts=(-9, -15))
        v_187 = v_186.view(2, 2, 18, 2, 30, 128)
        v_188 = v_181.view(2, 36, 60, 128)
        v_189 = torch.roll(input=v_188, dims=(1, 2), shifts=(-9, -15))
        v_190 = v_189.view(2, 2, 18, 2, 30, 128)
        v_191 = torch.permute(input=v_184, dims=(0, 1, 3, 2, 4, 5))
        v_192 = torch.permute(input=v_187, dims=(0, 1, 3, 2, 4, 5))
        v_193 = v_192.reshape(8, 540, 128)
        v_194 = v_191.reshape(8, 540, 128)
        v_195 = torch.permute(input=v_193, dims=(0, 2, 1))
        v_196 = torch.matmul(input=v_194, other=v_195)
        v_197 = self.pnnx_fold_2542_pnnx_fold_2542
        v_198 = ((v_196 / 11.313708) + v_197)
        v_199 = torch.permute(input=v_190, dims=(0, 1, 3, 2, 4, 5))
        v_200 = F.softmax(input=v_198, dim=-1)
        v_201 = v_199.reshape(8, 540, 128)
        v_202 = torch.matmul(input=v_200, other=v_201)
        v_203 = v_202.reshape(2, 2, 2, 18, 30, 128)
        v_204 = torch.permute(input=v_203, dims=(0, 1, 3, 2, 4, 5))
        v_205 = v_204.reshape(2, 36, 60, 128)
        v_206 = torch.roll(input=v_205, dims=(1, 2), shifts=(9, 15))
        v_207 = v_206.view(2, -1, 128)
        v_208 = self.transformer_layers_1_cross_attn_ffn_merge(v_207)
        v_209 = self.transformer_layers_1_cross_attn_ffn_norm1(v_208)
        v_210 = torch.cat((v_177, v_209), dim=-1)
        v_211 = self.transformer_layers_1_cross_attn_ffn_mlp_0(v_210)
        v_212 = self.transformer_layers_1_cross_attn_ffn_mlp_1(v_211)
        v_213 = self.transformer_layers_1_cross_attn_ffn_mlp_2(v_212)
        v_214 = self.transformer_layers_1_cross_attn_ffn_norm2(v_213)
        v_215 = (v_177 + v_214)
        v_216, v_217 = torch.chunk(input=v_215, chunks=2, dim=0)
        v_218 = self.transformer_layers_2_self_attn_q_proj(v_215)
        v_219 = self.transformer_layers_2_self_attn_k_proj(v_215)
        v_220 = self.transformer_layers_2_self_attn_v_proj(v_215)
        v_221 = v_218.reshape(2, 2, 18, 2, 30, 128)
        v_222 = v_219.reshape(2, 2, 18, 2, 30, 128)
        v_223 = v_220.reshape(2, 2, 18, 2, 30, 128)
        v_224 = torch.permute(input=v_221, dims=(0, 1, 3, 2, 4, 5))
        v_225 = torch.permute(input=v_222, dims=(0, 1, 3, 2, 4, 5))
        v_226 = v_225.reshape(8, 540, 128)
        v_227 = v_224.reshape(8, 540, 128)
        v_228 = torch.permute(input=v_226, dims=(0, 2, 1))
        v_229 = torch.matmul(input=v_227, other=v_228)
        v_230 = (v_229 / 11.313708)
        v_231 = torch.permute(input=v_223, dims=(0, 1, 3, 2, 4, 5))
        v_232 = F.softmax(input=v_230, dim=-1)
        v_233 = v_231.reshape(8, 540, 128)
        v_234 = torch.matmul(input=v_232, other=v_233)
        v_235 = v_234.reshape(2, 2, 2, 18, 30, 128)
        v_236 = torch.permute(input=v_235, dims=(0, 1, 3, 2, 4, 5))
        v_237 = v_236.reshape(2, 2160, 128)
        v_238 = self.transformer_layers_2_self_attn_merge(v_237)
        v_239 = self.transformer_layers_2_self_attn_norm1(v_238)
        v_240 = (v_215 + v_239)
        v_241 = self.transformer_layers_2_cross_attn_ffn_q_proj(v_240)
        v_242 = torch.cat((v_217, v_216), dim=0)
        v_243 = self.transformer_layers_2_cross_attn_ffn_k_proj(v_242)
        v_244 = self.transformer_layers_2_cross_attn_ffn_v_proj(v_242)
        v_245 = v_241.reshape(2, 2, 18, 2, 30, 128)
        v_246 = v_243.reshape(2, 2, 18, 2, 30, 128)
        v_247 = v_244.reshape(2, 2, 18, 2, 30, 128)
        v_248 = torch.permute(input=v_245, dims=(0, 1, 3, 2, 4, 5))
        v_249 = torch.permute(input=v_246, dims=(0, 1, 3, 2, 4, 5))
        v_250 = v_249.reshape(8, 540, 128)
        v_251 = v_248.reshape(8, 540, 128)
        v_252 = torch.permute(input=v_250, dims=(0, 2, 1))
        v_253 = torch.matmul(input=v_251, other=v_252)
        v_254 = (v_253 / 11.313708)
        v_255 = torch.permute(input=v_247, dims=(0, 1, 3, 2, 4, 5))
        v_256 = F.softmax(input=v_254, dim=-1)
        v_257 = v_255.reshape(8, 540, 128)
        v_258 = torch.matmul(input=v_256, other=v_257)
        v_259 = v_258.reshape(2, 2, 2, 18, 30, 128)
        v_260 = torch.permute(input=v_259, dims=(0, 1, 3, 2, 4, 5))
        v_261 = v_260.reshape(2, 2160, 128)
        v_262 = self.transformer_layers_2_cross_attn_ffn_merge(v_261)
        v_263 = self.transformer_layers_2_cross_attn_ffn_norm1(v_262)
        v_264 = torch.cat((v_240, v_263), dim=-1)
        v_265 = self.transformer_layers_2_cross_attn_ffn_mlp_0(v_264)
        v_266 = self.transformer_layers_2_cross_attn_ffn_mlp_1(v_265)
        v_267 = self.transformer_layers_2_cross_attn_ffn_mlp_2(v_266)
        v_268 = self.transformer_layers_2_cross_attn_ffn_norm2(v_267)
        v_269 = (v_240 + v_268)
        v_270, v_271 = torch.chunk(input=v_269, chunks=2, dim=0)
        v_272 = self.transformer_layers_3_self_attn_q_proj(v_269)
        v_273 = self.transformer_layers_3_self_attn_k_proj(v_269)
        v_274 = self.transformer_layers_3_self_attn_v_proj(v_269)
        v_275 = v_272.view(2, 36, 60, 128)
        v_276 = torch.roll(input=v_275, dims=(1, 2), shifts=(-9, -15))
        v_277 = v_276.view(2, 2, 18, 2, 30, 128)
        v_278 = v_273.view(2, 36, 60, 128)
        v_279 = torch.roll(input=v_278, dims=(1, 2), shifts=(-9, -15))
        v_280 = v_279.view(2, 2, 18, 2, 30, 128)
        v_281 = v_274.view(2, 36, 60, 128)
        v_282 = torch.roll(input=v_281, dims=(1, 2), shifts=(-9, -15))
        v_283 = v_282.view(2, 2, 18, 2, 30, 128)
        v_284 = torch.permute(input=v_277, dims=(0, 1, 3, 2, 4, 5))
        v_285 = torch.permute(input=v_280, dims=(0, 1, 3, 2, 4, 5))
        v_286 = v_285.reshape(8, 540, 128)
        v_287 = v_284.reshape(8, 540, 128)
        v_288 = torch.permute(input=v_286, dims=(0, 2, 1))
        v_289 = torch.matmul(input=v_287, other=v_288)
        v_290 = self.pnnx_fold_3153_pnnx_fold_3153
        v_291 = ((v_289 / 11.313708) + v_290)
        v_292 = torch.permute(input=v_283, dims=(0, 1, 3, 2, 4, 5))
        v_293 = F.softmax(input=v_291, dim=-1)
        v_294 = v_292.reshape(8, 540, 128)
        v_295 = torch.matmul(input=v_293, other=v_294)
        v_296 = v_295.reshape(2, 2, 2, 18, 30, 128)
        v_297 = torch.permute(input=v_296, dims=(0, 1, 3, 2, 4, 5))
        v_298 = v_297.reshape(2, 36, 60, 128)
        v_299 = torch.roll(input=v_298, dims=(1, 2), shifts=(9, 15))
        v_300 = v_299.view(2, -1, 128)
        v_301 = self.transformer_layers_3_self_attn_merge(v_300)
        v_302 = self.transformer_layers_3_self_attn_norm1(v_301)
        v_303 = (v_269 + v_302)
        v_304 = self.transformer_layers_3_cross_attn_ffn_q_proj(v_303)
        v_305 = torch.cat((v_271, v_270), dim=0)
        v_306 = self.transformer_layers_3_cross_attn_ffn_k_proj(v_305)
        v_307 = self.transformer_layers_3_cross_attn_ffn_v_proj(v_305)
        v_308 = v_304.view(2, 36, 60, 128)
        v_309 = torch.roll(input=v_308, dims=(1, 2), shifts=(-9, -15))
        v_310 = v_309.view(2, 2, 18, 2, 30, 128)
        v_311 = v_306.view(2, 36, 60, 128)
        v_312 = torch.roll(input=v_311, dims=(1, 2), shifts=(-9, -15))
        v_313 = v_312.view(2, 2, 18, 2, 30, 128)
        v_314 = v_307.view(2, 36, 60, 128)
        v_315 = torch.roll(input=v_314, dims=(1, 2), shifts=(-9, -15))
        v_316 = v_315.view(2, 2, 18, 2, 30, 128)
        v_317 = torch.permute(input=v_310, dims=(0, 1, 3, 2, 4, 5))
        v_318 = torch.permute(input=v_313, dims=(0, 1, 3, 2, 4, 5))
        v_319 = v_318.reshape(8, 540, 128)
        v_320 = v_317.reshape(8, 540, 128)
        v_321 = torch.permute(input=v_319, dims=(0, 2, 1))
        v_322 = torch.matmul(input=v_320, other=v_321)
        v_323 = self.pnnx_fold_3370_pnnx_fold_3370
        v_324 = ((v_322 / 11.313708) + v_323)
        v_325 = torch.permute(input=v_316, dims=(0, 1, 3, 2, 4, 5))
        v_326 = F.softmax(input=v_324, dim=-1)
        v_327 = v_325.reshape(8, 540, 128)
        v_328 = torch.matmul(input=v_326, other=v_327)
        v_329 = v_328.reshape(2, 2, 2, 18, 30, 128)
        v_330 = torch.permute(input=v_329, dims=(0, 1, 3, 2, 4, 5))
        v_331 = v_330.reshape(2, 36, 60, 128)
        v_332 = torch.roll(input=v_331, dims=(1, 2), shifts=(9, 15))
        v_333 = v_332.view(2, -1, 128)
        v_334 = self.transformer_layers_3_cross_attn_ffn_merge(v_333)
        v_335 = self.transformer_layers_3_cross_attn_ffn_norm1(v_334)
        v_336 = torch.cat((v_303, v_335), dim=-1)
        v_337 = self.transformer_layers_3_cross_attn_ffn_mlp_0(v_336)
        v_338 = self.transformer_layers_3_cross_attn_ffn_mlp_1(v_337)
        v_339 = self.transformer_layers_3_cross_attn_ffn_mlp_2(v_338)
        v_340 = self.transformer_layers_3_cross_attn_ffn_norm2(v_339)
        v_341 = (v_303 + v_340)
        v_342, v_343 = torch.chunk(input=v_341, chunks=2, dim=0)
        v_344 = self.transformer_layers_4_self_attn_q_proj(v_341)
        v_345 = self.transformer_layers_4_self_attn_k_proj(v_341)
        v_346 = self.transformer_layers_4_self_attn_v_proj(v_341)
        v_347 = v_344.reshape(2, 2, 18, 2, 30, 128)
        v_348 = v_345.reshape(2, 2, 18, 2, 30, 128)
        v_349 = v_346.reshape(2, 2, 18, 2, 30, 128)
        v_350 = torch.permute(input=v_347, dims=(0, 1, 3, 2, 4, 5))
        v_351 = torch.permute(input=v_348, dims=(0, 1, 3, 2, 4, 5))
        v_352 = v_351.reshape(8, 540, 128)
        v_353 = v_350.reshape(8, 540, 128)
        v_354 = torch.permute(input=v_352, dims=(0, 2, 1))
        v_355 = torch.matmul(input=v_353, other=v_354)
        v_356 = (v_355 / 11.313708)
        v_357 = torch.permute(input=v_349, dims=(0, 1, 3, 2, 4, 5))
        v_358 = F.softmax(input=v_356, dim=-1)
        v_359 = v_357.reshape(8, 540, 128)
        v_360 = torch.matmul(input=v_358, other=v_359)
        v_361 = v_360.reshape(2, 2, 2, 18, 30, 128)
        v_362 = torch.permute(input=v_361, dims=(0, 1, 3, 2, 4, 5))
        v_363 = v_362.reshape(2, 2160, 128)
        v_364 = self.transformer_layers_4_self_attn_merge(v_363)
        v_365 = self.transformer_layers_4_self_attn_norm1(v_364)
        v_366 = (v_341 + v_365)
        v_367 = self.transformer_layers_4_cross_attn_ffn_q_proj(v_366)
        v_368 = torch.cat((v_343, v_342), dim=0)
        v_369 = self.transformer_layers_4_cross_attn_ffn_k_proj(v_368)
        v_370 = self.transformer_layers_4_cross_attn_ffn_v_proj(v_368)
        v_371 = v_367.reshape(2, 2, 18, 2, 30, 128)
        v_372 = v_369.reshape(2, 2, 18, 2, 30, 128)
        v_373 = v_370.reshape(2, 2, 18, 2, 30, 128)
        v_374 = torch.permute(input=v_371, dims=(0, 1, 3, 2, 4, 5))
        v_375 = torch.permute(input=v_372, dims=(0, 1, 3, 2, 4, 5))
        v_376 = v_375.reshape(8, 540, 128)
        v_377 = v_374.reshape(8, 540, 128)
        v_378 = torch.permute(input=v_376, dims=(0, 2, 1))
        v_379 = torch.matmul(input=v_377, other=v_378)
        v_380 = (v_379 / 11.313708)
        v_381 = torch.permute(input=v_373, dims=(0, 1, 3, 2, 4, 5))
        v_382 = F.softmax(input=v_380, dim=-1)
        v_383 = v_381.reshape(8, 540, 128)
        v_384 = torch.matmul(input=v_382, other=v_383)
        v_385 = v_384.reshape(2, 2, 2, 18, 30, 128)
        v_386 = torch.permute(input=v_385, dims=(0, 1, 3, 2, 4, 5))
        v_387 = v_386.reshape(2, 2160, 128)
        v_388 = self.transformer_layers_4_cross_attn_ffn_merge(v_387)
        v_389 = self.transformer_layers_4_cross_attn_ffn_norm1(v_388)
        v_390 = torch.cat((v_366, v_389), dim=-1)
        v_391 = self.transformer_layers_4_cross_attn_ffn_mlp_0(v_390)
        v_392 = self.transformer_layers_4_cross_attn_ffn_mlp_1(v_391)
        v_393 = self.transformer_layers_4_cross_attn_ffn_mlp_2(v_392)
        v_394 = self.transformer_layers_4_cross_attn_ffn_norm2(v_393)
        v_395 = (v_366 + v_394)
        v_396, v_397 = torch.chunk(input=v_395, chunks=2, dim=0)
        v_398 = self.transformer_layers_5_self_attn_q_proj(v_395)
        v_399 = self.transformer_layers_5_self_attn_k_proj(v_395)
        v_400 = self.transformer_layers_5_self_attn_v_proj(v_395)
        v_401 = v_398.view(2, 36, 60, 128)
        v_402 = torch.roll(input=v_401, dims=(1, 2), shifts=(-9, -15))
        v_403 = v_402.view(2, 2, 18, 2, 30, 128)
        v_404 = v_399.view(2, 36, 60, 128)
        v_405 = torch.roll(input=v_404, dims=(1, 2), shifts=(-9, -15))
        v_406 = v_405.view(2, 2, 18, 2, 30, 128)
        v_407 = v_400.view(2, 36, 60, 128)
        v_408 = torch.roll(input=v_407, dims=(1, 2), shifts=(-9, -15))
        v_409 = v_408.view(2, 2, 18, 2, 30, 128)
        v_410 = torch.permute(input=v_403, dims=(0, 1, 3, 2, 4, 5))
        v_411 = torch.permute(input=v_406, dims=(0, 1, 3, 2, 4, 5))
        v_412 = v_411.reshape(8, 540, 128)
        v_413 = v_410.reshape(8, 540, 128)
        v_414 = torch.permute(input=v_412, dims=(0, 2, 1))
        v_415 = torch.matmul(input=v_413, other=v_414)
        v_416 = self.pnnx_fold_3981_pnnx_fold_3981
        v_417 = ((v_415 / 11.313708) + v_416)
        v_418 = torch.permute(input=v_409, dims=(0, 1, 3, 2, 4, 5))
        v_419 = F.softmax(input=v_417, dim=-1)
        v_420 = v_418.reshape(8, 540, 128)
        v_421 = torch.matmul(input=v_419, other=v_420)
        v_422 = v_421.reshape(2, 2, 2, 18, 30, 128)
        v_423 = torch.permute(input=v_422, dims=(0, 1, 3, 2, 4, 5))
        v_424 = v_423.reshape(2, 36, 60, 128)
        v_425 = torch.roll(input=v_424, dims=(1, 2), shifts=(9, 15))
        v_426 = v_425.view(2, -1, 128)
        v_427 = self.transformer_layers_5_self_attn_merge(v_426)
        v_428 = self.transformer_layers_5_self_attn_norm1(v_427)
        v_429 = (v_395 + v_428)
        v_430 = self.transformer_layers_5_cross_attn_ffn_q_proj(v_429)
        v_431 = torch.cat((v_397, v_396), dim=0)
        v_432 = self.transformer_layers_5_cross_attn_ffn_k_proj(v_431)
        v_433 = self.transformer_layers_5_cross_attn_ffn_v_proj(v_431)
        v_434 = v_430.view(2, 36, 60, 128)
        v_435 = torch.roll(input=v_434, dims=(1, 2), shifts=(-9, -15))
        v_436 = v_435.view(2, 2, 18, 2, 30, 128)
        v_437 = v_432.view(2, 36, 60, 128)
        v_438 = torch.roll(input=v_437, dims=(1, 2), shifts=(-9, -15))
        v_439 = v_438.view(2, 2, 18, 2, 30, 128)
        v_440 = v_433.view(2, 36, 60, 128)
        v_441 = torch.roll(input=v_440, dims=(1, 2), shifts=(-9, -15))
        v_442 = v_441.view(2, 2, 18, 2, 30, 128)
        v_443 = torch.permute(input=v_436, dims=(0, 1, 3, 2, 4, 5))
        v_444 = torch.permute(input=v_439, dims=(0, 1, 3, 2, 4, 5))
        v_445 = v_444.reshape(8, 540, 128)
        v_446 = v_443.reshape(8, 540, 128)
        v_447 = torch.permute(input=v_445, dims=(0, 2, 1))
        v_448 = torch.matmul(input=v_446, other=v_447)
        v_449 = self.pnnx_fold_4198_pnnx_fold_4198
        v_450 = ((v_448 / 11.313708) + v_449)
        v_451 = torch.permute(input=v_442, dims=(0, 1, 3, 2, 4, 5))
        v_452 = F.softmax(input=v_450, dim=-1)
        v_453 = v_451.reshape(8, 540, 128)
        v_454 = torch.matmul(input=v_452, other=v_453)
        v_455 = v_454.reshape(2, 2, 2, 18, 30, 128)
        v_456 = torch.permute(input=v_455, dims=(0, 1, 3, 2, 4, 5))
        v_457 = v_456.reshape(2, 36, 60, 128)
        v_458 = torch.roll(input=v_457, dims=(1, 2), shifts=(9, 15))
        v_459 = v_458.view(2, -1, 128)
        v_460 = self.transformer_layers_5_cross_attn_ffn_merge(v_459)
        v_461 = self.transformer_layers_5_cross_attn_ffn_norm1(v_460)
        v_462 = torch.cat((v_429, v_461), dim=-1)
        v_463 = self.transformer_layers_5_cross_attn_ffn_mlp_0(v_462)
        v_464 = self.transformer_layers_5_cross_attn_ffn_mlp_1(v_463)
        v_465 = self.transformer_layers_5_cross_attn_ffn_mlp_2(v_464)
        v_466 = self.transformer_layers_5_cross_attn_ffn_norm2(v_465)
        v_467 = (v_429 + v_466)
        v_468, v_469 = torch.chunk(input=v_467, chunks=2, dim=0)
        v_470 = v_468.view(1, 36, 60, 128)
        v_471 = v_469.view(1, 36, 60, 128)
        v_472 = torch.permute(input=v_471, dims=(0, 3, 1, 2))
        v_473 = torch.permute(input=v_470, dims=(0, 3, 1, 2))
        v_474 = v_473.contiguous(memory_format=torch.contiguous_format)
        v_475 = v_474.view(1, 128, -1)
        v_476 = torch.permute(input=v_475, dims=(0, 2, 1))
        v_477 = v_472.reshape(1, 128, -1)
        v_478 = torch.matmul(input=v_476, other=v_477)
        v_479 = v_478.view(1, 36, 60, 36, 60)
        v_480 = (v_479 / 11.313708)
        v_481 = self.pnnx_fold_init_grid_1_pnnx_fold_init_grid_1
        v_482 = v_480.view(1, 2160, 2160)
        v_483 = F.softmax(input=v_482, dim=-1)
        v_484 = self.pnnx_fold_grid_5_pnnx_fold_grid_5
        v_485 = torch.matmul(input=v_483, other=v_484)
        v_486 = v_485.view(1, 36, 60, 2)
        v_487 = torch.permute(input=v_486, dims=(0, 3, 1, 2))
        v_488 = (v_487 - v_481)
        v_489 = v_474.view(1, 128, 2160)
        v_490 = torch.permute(input=v_489, dims=(0, 2, 1))
        v_491 = self.feature_flow_attn_q_proj(v_490)
        v_492 = self.feature_flow_attn_k_proj(v_491)
        v_493 = v_488.view(1, 2, 2160)
        v_494 = torch.permute(input=v_492, dims=(0, 2, 1))
        v_495 = torch.matmul(input=v_491, other=v_494)
        v_496 = (v_495 / 11.313708)
        v_497 = F.softmax(input=v_496, dim=-1)
        v_498 = torch.permute(input=v_493, dims=(0, 2, 1))
        v_499 = torch.matmul(input=v_497, other=v_498)
        v_500 = v_499.view(1, 36, 60, 2)
        v_501 = torch.permute(input=v_500, dims=(0, 3, 1, 2))
        v_502 = F.upsample(input=v_501, align_corners=True, mode='bilinear', scale_factor=(2.000000, 2.000000))
        v_503 = (v_502 * 2)
        v_504 = self.pnnx_fold_732_pnnx_fold_732
        v_505 = (v_504 + v_503)
        v_506, v_507 = torch.unbind(v_505, dim=1)
        v_508 = (((v_506 * 2) / 119.000000) - 1)
        v_509 = (((v_507 * 2) / 71.000000) - 1)
        v_510 = v_69.view(1, 128, 8, 9, 8, 15)
        v_511 = torch.stack((v_508, v_509), dim=-1)
        v_512 = F.grid_sample(input=v_70, grid=v_511, align_corners=True, mode='bilinear', padding_mode='zeros')
        v_513 = v_512.view(1, 128, 8, 9, 8, 15)
        v_514 = torch.permute(input=v_510, dims=(0, 2, 4, 1, 3, 5))
        v_515 = v_514.reshape(64, 128, 9, 15)
        v_516 = self.pnnx_fold_position0_1_pnnx_fold_position0_1
        v_517 = self.pnnx_fold_position0_1_1_pnnx_fold_position0_1
        v_518 = (v_515 + v_516)
        v_519 = torch.permute(input=v_513, dims=(0, 2, 4, 1, 3, 5))
        v_520 = v_519.reshape(64, 128, 9, 15)
        v_521 = (v_520 + v_517)
        v_522 = v_518.view(1, 8, 8, 128, 9, 15)
        v_523 = torch.permute(input=v_522, dims=(0, 3, 1, 4, 2, 5))
        v_524 = v_521.view(1, 8, 8, 128, 9, 15)
        v_525 = torch.permute(input=v_524, dims=(0, 3, 1, 4, 2, 5))
        v_526 = v_523.reshape(1, 128, 72, 120)
        v_527 = torch.flatten(input=v_526, end_dim=-1, start_dim=-2)
        v_528 = v_525.reshape(1, 128, 72, 120)
        v_529 = torch.flatten(input=v_528, end_dim=-1, start_dim=-2)
        v_530 = torch.permute(input=v_529, dims=(0, 2, 1))
        v_531 = torch.permute(input=v_527, dims=(0, 2, 1))
        v_532 = torch.cat((v_531, v_530), dim=0)
        v_533 = self.pnnx_unique_12(v_532)
        v_534 = self.pnnx_unique_13(v_532)
        v_535 = self.pnnx_unique_14(v_532)
        v_536 = v_533.reshape(2, 8, 9, 8, 15, 128)
        v_537 = v_534.reshape(2, 8, 9, 8, 15, 128)
        v_538 = v_535.reshape(2, 8, 9, 8, 15, 128)
        v_539 = torch.permute(input=v_536, dims=(0, 1, 3, 2, 4, 5))
        v_540 = torch.permute(input=v_537, dims=(0, 1, 3, 2, 4, 5))
        v_541 = v_540.reshape(128, 135, 128)
        v_542 = v_539.reshape(128, 135, 128)
        v_543 = torch.permute(input=v_541, dims=(0, 2, 1))
        v_544 = torch.matmul(input=v_542, other=v_543)
        v_545 = (v_544 / 11.313708)
        v_546 = torch.permute(input=v_538, dims=(0, 1, 3, 2, 4, 5))
        v_547 = F.softmax(input=v_545, dim=-1)
        v_548 = v_546.reshape(128, 135, 128)
        v_549 = torch.matmul(input=v_547, other=v_548)
        v_550 = v_549.reshape(2, 8, 8, 9, 15, 128)
        v_551 = torch.permute(input=v_550, dims=(0, 1, 3, 2, 4, 5))
        v_552 = v_551.reshape(2, 8640, 128)
        v_553 = self.pnnx_unique_15(v_552)
        v_554 = self.pnnx_unique_16(v_553)
        v_555 = (v_532 + v_554)
        v_556 = self.pnnx_unique_17(v_555)
        v_557 = torch.cat((v_530, v_531), dim=0)
        v_558 = self.pnnx_unique_18(v_557)
        v_559 = self.pnnx_unique_19(v_557)
        v_560 = v_556.reshape(2, 8, 9, 8, 15, 128)
        v_561 = v_558.reshape(2, 8, 9, 8, 15, 128)
        v_562 = v_559.reshape(2, 8, 9, 8, 15, 128)
        v_563 = torch.permute(input=v_560, dims=(0, 1, 3, 2, 4, 5))
        v_564 = torch.permute(input=v_561, dims=(0, 1, 3, 2, 4, 5))
        v_565 = v_564.reshape(128, 135, 128)
        v_566 = v_563.reshape(128, 135, 128)
        v_567 = torch.permute(input=v_565, dims=(0, 2, 1))
        v_568 = torch.matmul(input=v_566, other=v_567)
        v_569 = (v_568 / 11.313708)
        v_570 = torch.permute(input=v_562, dims=(0, 1, 3, 2, 4, 5))
        v_571 = F.softmax(input=v_569, dim=-1)
        v_572 = v_570.reshape(128, 135, 128)
        v_573 = torch.matmul(input=v_571, other=v_572)
        v_574 = v_573.reshape(2, 8, 8, 9, 15, 128)
        v_575 = torch.permute(input=v_574, dims=(0, 1, 3, 2, 4, 5))
        v_576 = v_575.reshape(2, 8640, 128)
        v_577 = self.pnnx_unique_20(v_576)
        v_578 = self.pnnx_unique_21(v_577)
        v_579 = torch.cat((v_555, v_578), dim=-1)
        v_580 = self.pnnx_unique_22(v_579)
        v_581 = self.pnnx_unique_23(v_580)
        v_582 = self.pnnx_unique_24(v_581)
        v_583 = self.pnnx_unique_25(v_582)
        v_584 = (v_555 + v_583)
        v_585, v_586 = torch.chunk(input=v_584, chunks=2, dim=0)
        v_587 = self.pnnx_unique_26(v_584)
        v_588 = self.pnnx_unique_27(v_584)
        v_589 = self.pnnx_unique_28(v_584)
        v_590 = v_587.view(2, 72, 120, 128)
        v_591 = torch.roll(input=v_590, dims=(1, 2), shifts=(-4, -7))
        v_592 = v_591.view(2, 8, 9, 8, 15, 128)
        v_593 = v_588.view(2, 72, 120, 128)
        v_594 = torch.roll(input=v_593, dims=(1, 2), shifts=(-4, -7))
        v_595 = v_594.view(2, 8, 9, 8, 15, 128)
        v_596 = v_589.view(2, 72, 120, 128)
        v_597 = torch.roll(input=v_596, dims=(1, 2), shifts=(-4, -7))
        v_598 = v_597.view(2, 8, 9, 8, 15, 128)
        v_599 = torch.permute(input=v_592, dims=(0, 1, 3, 2, 4, 5))
        v_600 = torch.permute(input=v_595, dims=(0, 1, 3, 2, 4, 5))
        v_601 = v_600.reshape(128, 135, 128)
        v_602 = v_599.reshape(128, 135, 128)
        v_603 = torch.permute(input=v_601, dims=(0, 2, 1))
        v_604 = torch.matmul(input=v_602, other=v_603)
        v_605 = self.pnnx_fold_5210_pnnx_fold_5210
        v_606 = ((v_604 / 11.313708) + v_605)
        v_607 = torch.permute(input=v_598, dims=(0, 1, 3, 2, 4, 5))
        v_608 = F.softmax(input=v_606, dim=-1)
        v_609 = v_607.reshape(128, 135, 128)
        v_610 = torch.matmul(input=v_608, other=v_609)
        v_611 = v_610.reshape(2, 8, 8, 9, 15, 128)
        v_612 = torch.permute(input=v_611, dims=(0, 1, 3, 2, 4, 5))
        v_613 = v_612.reshape(2, 72, 120, 128)
        v_614 = torch.roll(input=v_613, dims=(1, 2), shifts=(4, 7))
        v_615 = v_614.view(2, -1, 128)
        v_616 = self.pnnx_unique_29(v_615)
        v_617 = self.pnnx_unique_30(v_616)
        v_618 = (v_584 + v_617)
        v_619 = self.pnnx_unique_31(v_618)
        v_620 = torch.cat((v_586, v_585), dim=0)
        v_621 = self.pnnx_unique_32(v_620)
        v_622 = self.pnnx_unique_33(v_620)
        v_623 = v_619.view(2, 72, 120, 128)
        v_624 = torch.roll(input=v_623, dims=(1, 2), shifts=(-4, -7))
        v_625 = v_624.view(2, 8, 9, 8, 15, 128)
        v_626 = v_621.view(2, 72, 120, 128)
        v_627 = torch.roll(input=v_626, dims=(1, 2), shifts=(-4, -7))
        v_628 = v_627.view(2, 8, 9, 8, 15, 128)
        v_629 = v_622.view(2, 72, 120, 128)
        v_630 = torch.roll(input=v_629, dims=(1, 2), shifts=(-4, -7))
        v_631 = v_630.view(2, 8, 9, 8, 15, 128)
        v_632 = torch.permute(input=v_625, dims=(0, 1, 3, 2, 4, 5))
        v_633 = torch.permute(input=v_628, dims=(0, 1, 3, 2, 4, 5))
        v_634 = v_633.reshape(128, 135, 128)
        v_635 = v_632.reshape(128, 135, 128)
        v_636 = torch.permute(input=v_634, dims=(0, 2, 1))
        v_637 = torch.matmul(input=v_635, other=v_636)
        v_638 = self.pnnx_fold_5429_pnnx_fold_5429
        v_639 = ((v_637 / 11.313708) + v_638)
        v_640 = torch.permute(input=v_631, dims=(0, 1, 3, 2, 4, 5))
        v_641 = F.softmax(input=v_639, dim=-1)
        v_642 = v_640.reshape(128, 135, 128)
        v_643 = torch.matmul(input=v_641, other=v_642)
        v_644 = v_643.reshape(2, 8, 8, 9, 15, 128)
        v_645 = torch.permute(input=v_644, dims=(0, 1, 3, 2, 4, 5))
        v_646 = v_645.reshape(2, 72, 120, 128)
        v_647 = torch.roll(input=v_646, dims=(1, 2), shifts=(4, 7))
        v_648 = v_647.view(2, -1, 128)
        v_649 = self.pnnx_unique_34(v_648)
        v_650 = self.pnnx_unique_35(v_649)
        v_651 = torch.cat((v_618, v_650), dim=-1)
        v_652 = self.pnnx_unique_36(v_651)
        v_653 = self.pnnx_unique_37(v_652)
        v_654 = self.pnnx_unique_38(v_653)
        v_655 = self.pnnx_unique_39(v_654)
        v_656 = (v_618 + v_655)
        v_657, v_658 = torch.chunk(input=v_656, chunks=2, dim=0)
        v_659 = self.pnnx_unique_40(v_656)
        v_660 = self.pnnx_unique_41(v_656)
        v_661 = self.pnnx_unique_42(v_656)
        v_662 = v_659.reshape(2, 8, 9, 8, 15, 128)
        v_663 = v_660.reshape(2, 8, 9, 8, 15, 128)
        v_664 = v_661.reshape(2, 8, 9, 8, 15, 128)
        v_665 = torch.permute(input=v_662, dims=(0, 1, 3, 2, 4, 5))
        v_666 = torch.permute(input=v_663, dims=(0, 1, 3, 2, 4, 5))
        v_667 = v_666.reshape(128, 135, 128)
        v_668 = v_665.reshape(128, 135, 128)
        v_669 = torch.permute(input=v_667, dims=(0, 2, 1))
        v_670 = torch.matmul(input=v_668, other=v_669)
        v_671 = (v_670 / 11.313708)
        v_672 = torch.permute(input=v_664, dims=(0, 1, 3, 2, 4, 5))
        v_673 = F.softmax(input=v_671, dim=-1)
        v_674 = v_672.reshape(128, 135, 128)
        v_675 = torch.matmul(input=v_673, other=v_674)
        v_676 = v_675.reshape(2, 8, 8, 9, 15, 128)
        v_677 = torch.permute(input=v_676, dims=(0, 1, 3, 2, 4, 5))
        v_678 = v_677.reshape(2, 8640, 128)
        v_679 = self.pnnx_unique_43(v_678)
        v_680 = self.pnnx_unique_44(v_679)
        v_681 = (v_656 + v_680)
        v_682 = self.pnnx_unique_45(v_681)
        v_683 = torch.cat((v_658, v_657), dim=0)
        v_684 = self.pnnx_unique_46(v_683)
        v_685 = self.pnnx_unique_47(v_683)
        v_686 = v_682.reshape(2, 8, 9, 8, 15, 128)
        v_687 = v_684.reshape(2, 8, 9, 8, 15, 128)
        v_688 = v_685.reshape(2, 8, 9, 8, 15, 128)
        v_689 = torch.permute(input=v_686, dims=(0, 1, 3, 2, 4, 5))
        v_690 = torch.permute(input=v_687, dims=(0, 1, 3, 2, 4, 5))
        v_691 = v_690.reshape(128, 135, 128)
        v_692 = v_689.reshape(128, 135, 128)
        v_693 = torch.permute(input=v_691, dims=(0, 2, 1))
        v_694 = torch.matmul(input=v_692, other=v_693)
        v_695 = (v_694 / 11.313708)
        v_696 = torch.permute(input=v_688, dims=(0, 1, 3, 2, 4, 5))
        v_697 = F.softmax(input=v_695, dim=-1)
        v_698 = v_696.reshape(128, 135, 128)
        v_699 = torch.matmul(input=v_697, other=v_698)
        v_700 = v_699.reshape(2, 8, 8, 9, 15, 128)
        v_701 = torch.permute(input=v_700, dims=(0, 1, 3, 2, 4, 5))
        v_702 = v_701.reshape(2, 8640, 128)
        v_703 = self.pnnx_unique_48(v_702)
        v_704 = self.pnnx_unique_49(v_703)
        v_705 = torch.cat((v_681, v_704), dim=-1)
        v_706 = self.pnnx_unique_50(v_705)
        v_707 = self.pnnx_unique_51(v_706)
        v_708 = self.pnnx_unique_52(v_707)
        v_709 = self.pnnx_unique_53(v_708)
        v_710 = (v_681 + v_709)
        v_711, v_712 = torch.chunk(input=v_710, chunks=2, dim=0)
        v_713 = self.pnnx_unique_54(v_710)
        v_714 = self.pnnx_unique_55(v_710)
        v_715 = self.pnnx_unique_56(v_710)
        v_716 = v_713.view(2, 72, 120, 128)
        v_717 = torch.roll(input=v_716, dims=(1, 2), shifts=(-4, -7))
        v_718 = v_717.view(2, 8, 9, 8, 15, 128)
        v_719 = v_714.view(2, 72, 120, 128)
        v_720 = torch.roll(input=v_719, dims=(1, 2), shifts=(-4, -7))
        v_721 = v_720.view(2, 8, 9, 8, 15, 128)
        v_722 = v_715.view(2, 72, 120, 128)
        v_723 = torch.roll(input=v_722, dims=(1, 2), shifts=(-4, -7))
        v_724 = v_723.view(2, 8, 9, 8, 15, 128)
        v_725 = torch.permute(input=v_718, dims=(0, 1, 3, 2, 4, 5))
        v_726 = torch.permute(input=v_721, dims=(0, 1, 3, 2, 4, 5))
        v_727 = v_726.reshape(128, 135, 128)
        v_728 = v_725.reshape(128, 135, 128)
        v_729 = torch.permute(input=v_727, dims=(0, 2, 1))
        v_730 = torch.matmul(input=v_728, other=v_729)
        v_731 = self.pnnx_fold_6044_pnnx_fold_6044
        v_732 = ((v_730 / 11.313708) + v_731)
        v_733 = torch.permute(input=v_724, dims=(0, 1, 3, 2, 4, 5))
        v_734 = F.softmax(input=v_732, dim=-1)
        v_735 = v_733.reshape(128, 135, 128)
        v_736 = torch.matmul(input=v_734, other=v_735)
        v_737 = v_736.reshape(2, 8, 8, 9, 15, 128)
        v_738 = torch.permute(input=v_737, dims=(0, 1, 3, 2, 4, 5))
        v_739 = v_738.reshape(2, 72, 120, 128)
        v_740 = torch.roll(input=v_739, dims=(1, 2), shifts=(4, 7))
        v_741 = v_740.view(2, -1, 128)
        v_742 = self.pnnx_unique_57(v_741)
        v_743 = self.pnnx_unique_58(v_742)
        v_744 = (v_710 + v_743)
        v_745 = self.pnnx_unique_59(v_744)
        v_746 = torch.cat((v_712, v_711), dim=0)
        v_747 = self.pnnx_unique_60(v_746)
        v_748 = self.pnnx_unique_61(v_746)
        v_749 = v_745.view(2, 72, 120, 128)
        v_750 = torch.roll(input=v_749, dims=(1, 2), shifts=(-4, -7))
        v_751 = v_750.view(2, 8, 9, 8, 15, 128)
        v_752 = v_747.view(2, 72, 120, 128)
        v_753 = torch.roll(input=v_752, dims=(1, 2), shifts=(-4, -7))
        v_754 = v_753.view(2, 8, 9, 8, 15, 128)
        v_755 = v_748.view(2, 72, 120, 128)
        v_756 = torch.roll(input=v_755, dims=(1, 2), shifts=(-4, -7))
        v_757 = v_756.view(2, 8, 9, 8, 15, 128)
        v_758 = torch.permute(input=v_751, dims=(0, 1, 3, 2, 4, 5))
        v_759 = torch.permute(input=v_754, dims=(0, 1, 3, 2, 4, 5))
        v_760 = v_759.reshape(128, 135, 128)
        v_761 = v_758.reshape(128, 135, 128)
        v_762 = torch.permute(input=v_760, dims=(0, 2, 1))
        v_763 = torch.matmul(input=v_761, other=v_762)
        v_764 = self.pnnx_fold_6263_pnnx_fold_6263
        v_765 = ((v_763 / 11.313708) + v_764)
        v_766 = torch.permute(input=v_757, dims=(0, 1, 3, 2, 4, 5))
        v_767 = F.softmax(input=v_765, dim=-1)
        v_768 = v_766.reshape(128, 135, 128)
        v_769 = torch.matmul(input=v_767, other=v_768)
        v_770 = v_769.reshape(2, 8, 8, 9, 15, 128)
        v_771 = torch.permute(input=v_770, dims=(0, 1, 3, 2, 4, 5))
        v_772 = v_771.reshape(2, 72, 120, 128)
        v_773 = torch.roll(input=v_772, dims=(1, 2), shifts=(4, 7))
        v_774 = v_773.view(2, -1, 128)
        v_775 = self.pnnx_unique_62(v_774)
        v_776 = self.pnnx_unique_63(v_775)
        v_777 = torch.cat((v_744, v_776), dim=-1)
        v_778 = self.pnnx_unique_64(v_777)
        v_779 = self.pnnx_unique_65(v_778)
        v_780 = self.pnnx_unique_66(v_779)
        v_781 = self.pnnx_unique_67(v_780)
        v_782 = (v_744 + v_781)
        v_783, v_784 = torch.chunk(input=v_782, chunks=2, dim=0)
        v_785 = self.pnnx_unique_68(v_782)
        v_786 = self.pnnx_unique_69(v_782)
        v_787 = self.pnnx_unique_70(v_782)
        v_788 = v_785.reshape(2, 8, 9, 8, 15, 128)
        v_789 = v_786.reshape(2, 8, 9, 8, 15, 128)
        v_790 = v_787.reshape(2, 8, 9, 8, 15, 128)
        v_791 = torch.permute(input=v_788, dims=(0, 1, 3, 2, 4, 5))
        v_792 = torch.permute(input=v_789, dims=(0, 1, 3, 2, 4, 5))
        v_793 = v_792.reshape(128, 135, 128)
        v_794 = v_791.reshape(128, 135, 128)
        v_795 = torch.permute(input=v_793, dims=(0, 2, 1))
        v_796 = torch.matmul(input=v_794, other=v_795)
        v_797 = (v_796 / 11.313708)
        v_798 = torch.permute(input=v_790, dims=(0, 1, 3, 2, 4, 5))
        v_799 = F.softmax(input=v_797, dim=-1)
        v_800 = v_798.reshape(128, 135, 128)
        v_801 = torch.matmul(input=v_799, other=v_800)
        v_802 = v_801.reshape(2, 8, 8, 9, 15, 128)
        v_803 = torch.permute(input=v_802, dims=(0, 1, 3, 2, 4, 5))
        v_804 = v_803.reshape(2, 8640, 128)
        v_805 = self.pnnx_unique_71(v_804)
        v_806 = self.pnnx_unique_72(v_805)
        v_807 = (v_782 + v_806)
        v_808 = self.pnnx_unique_73(v_807)
        v_809 = torch.cat((v_784, v_783), dim=0)
        v_810 = self.pnnx_unique_74(v_809)
        v_811 = self.pnnx_unique_75(v_809)
        v_812 = v_808.reshape(2, 8, 9, 8, 15, 128)
        v_813 = v_810.reshape(2, 8, 9, 8, 15, 128)
        v_814 = v_811.reshape(2, 8, 9, 8, 15, 128)
        v_815 = torch.permute(input=v_812, dims=(0, 1, 3, 2, 4, 5))
        v_816 = torch.permute(input=v_813, dims=(0, 1, 3, 2, 4, 5))
        v_817 = v_816.reshape(128, 135, 128)
        v_818 = v_815.reshape(128, 135, 128)
        v_819 = torch.permute(input=v_817, dims=(0, 2, 1))
        v_820 = torch.matmul(input=v_818, other=v_819)
        v_821 = (v_820 / 11.313708)
        v_822 = torch.permute(input=v_814, dims=(0, 1, 3, 2, 4, 5))
        v_823 = F.softmax(input=v_821, dim=-1)
        v_824 = v_822.reshape(128, 135, 128)
        v_825 = torch.matmul(input=v_823, other=v_824)
        v_826 = v_825.reshape(2, 8, 8, 9, 15, 128)
        v_827 = torch.permute(input=v_826, dims=(0, 1, 3, 2, 4, 5))
        v_828 = v_827.reshape(2, 8640, 128)
        v_829 = self.pnnx_unique_76(v_828)
        v_830 = self.pnnx_unique_77(v_829)
        v_831 = torch.cat((v_807, v_830), dim=-1)
        v_832 = self.pnnx_unique_78(v_831)
        v_833 = self.pnnx_unique_79(v_832)
        v_834 = self.pnnx_unique_80(v_833)
        v_835 = self.pnnx_unique_81(v_834)
        v_836 = (v_807 + v_835)
        v_837, v_838 = torch.chunk(input=v_836, chunks=2, dim=0)
        v_839 = self.pnnx_unique_82(v_836)
        v_840 = self.pnnx_unique_83(v_836)
        v_841 = self.pnnx_unique_84(v_836)
        v_842 = v_839.view(2, 72, 120, 128)
        v_843 = torch.roll(input=v_842, dims=(1, 2), shifts=(-4, -7))
        v_844 = v_843.view(2, 8, 9, 8, 15, 128)
        v_845 = v_840.view(2, 72, 120, 128)
        v_846 = torch.roll(input=v_845, dims=(1, 2), shifts=(-4, -7))
        v_847 = v_846.view(2, 8, 9, 8, 15, 128)
        v_848 = v_841.view(2, 72, 120, 128)
        v_849 = torch.roll(input=v_848, dims=(1, 2), shifts=(-4, -7))
        v_850 = v_849.view(2, 8, 9, 8, 15, 128)
        v_851 = torch.permute(input=v_844, dims=(0, 1, 3, 2, 4, 5))
        v_852 = torch.permute(input=v_847, dims=(0, 1, 3, 2, 4, 5))
        v_853 = v_852.reshape(128, 135, 128)
        v_854 = v_851.reshape(128, 135, 128)
        v_855 = torch.permute(input=v_853, dims=(0, 2, 1))
        v_856 = torch.matmul(input=v_854, other=v_855)
        v_857 = self.pnnx_fold_6878_pnnx_fold_6878
        v_858 = ((v_856 / 11.313708) + v_857)
        v_859 = torch.permute(input=v_850, dims=(0, 1, 3, 2, 4, 5))
        v_860 = F.softmax(input=v_858, dim=-1)
        v_861 = v_859.reshape(128, 135, 128)
        v_862 = torch.matmul(input=v_860, other=v_861)
        v_863 = v_862.reshape(2, 8, 8, 9, 15, 128)
        v_864 = torch.permute(input=v_863, dims=(0, 1, 3, 2, 4, 5))
        v_865 = v_864.reshape(2, 72, 120, 128)
        v_866 = torch.roll(input=v_865, dims=(1, 2), shifts=(4, 7))
        v_867 = v_866.view(2, -1, 128)
        v_868 = self.pnnx_unique_85(v_867)
        v_869 = self.pnnx_unique_86(v_868)
        v_870 = (v_836 + v_869)
        v_871 = self.pnnx_unique_87(v_870)
        v_872 = torch.cat((v_838, v_837), dim=0)
        v_873 = self.pnnx_unique_88(v_872)
        v_874 = self.pnnx_unique_89(v_872)
        v_875 = v_871.view(2, 72, 120, 128)
        v_876 = torch.roll(input=v_875, dims=(1, 2), shifts=(-4, -7))
        v_877 = v_876.view(2, 8, 9, 8, 15, 128)
        v_878 = v_873.view(2, 72, 120, 128)
        v_879 = torch.roll(input=v_878, dims=(1, 2), shifts=(-4, -7))
        v_880 = v_879.view(2, 8, 9, 8, 15, 128)
        v_881 = v_874.view(2, 72, 120, 128)
        v_882 = torch.roll(input=v_881, dims=(1, 2), shifts=(-4, -7))
        v_883 = v_882.view(2, 8, 9, 8, 15, 128)
        v_884 = torch.permute(input=v_877, dims=(0, 1, 3, 2, 4, 5))
        v_885 = torch.permute(input=v_880, dims=(0, 1, 3, 2, 4, 5))
        v_886 = v_885.reshape(128, 135, 128)
        v_887 = v_884.reshape(128, 135, 128)
        v_888 = torch.permute(input=v_886, dims=(0, 2, 1))
        v_889 = torch.matmul(input=v_887, other=v_888)
        v_890 = self.pnnx_fold_7097_pnnx_fold_7097
        v_891 = ((v_889 / 11.313708) + v_890)
        v_892 = torch.permute(input=v_883, dims=(0, 1, 3, 2, 4, 5))
        v_893 = F.softmax(input=v_891, dim=-1)
        v_894 = v_892.reshape(128, 135, 128)
        v_895 = torch.matmul(input=v_893, other=v_894)
        v_896 = v_895.reshape(2, 8, 8, 9, 15, 128)
        v_897 = torch.permute(input=v_896, dims=(0, 1, 3, 2, 4, 5))
        v_898 = v_897.reshape(2, 72, 120, 128)
        v_899 = torch.roll(input=v_898, dims=(1, 2), shifts=(4, 7))
        v_900 = v_899.view(2, -1, 128)
        v_901 = self.pnnx_unique_90(v_900)
        v_902 = self.pnnx_unique_91(v_901)
        v_903 = torch.cat((v_870, v_902), dim=-1)
        v_904 = self.pnnx_unique_92(v_903)
        v_905 = self.pnnx_unique_93(v_904)
        v_906 = self.pnnx_unique_94(v_905)
        v_907 = self.pnnx_unique_95(v_906)
        v_908 = (v_870 + v_907)
        v_909, v_910 = torch.chunk(input=v_908, chunks=2, dim=0)
        v_911 = v_909.view(1, 72, 120, 128)
        v_912 = v_910.view(1, 72, 120, 128)
        v_913 = torch.permute(input=v_912, dims=(0, 3, 1, 2))
        v_914 = v_913.contiguous(memory_format=torch.contiguous_format)
        v_915 = torch.permute(input=v_911, dims=(0, 3, 1, 2))
        v_916 = v_915.contiguous(memory_format=torch.contiguous_format)
        v_917 = self.pnnx_fold_coords_init_1_pnnx_fold_coords_init_1
        v_918 = self.pnnx_fold_coords0_1_pnnx_fold_coords0_1
        v_919 = self.pnnx_fold_grid_1_pnnx_fold_grid_1
        v_920 = F.grid_sample(input=v_914, grid=v_919, align_corners=True, mode='bilinear', padding_mode='zeros')
        v_921 = torch.permute(input=v_916, dims=(0, 2, 3, 1))
        v_922 = v_921.view(1, 8640, 1, 128)
        v_923 = torch.permute(input=v_920, dims=(0, 2, 1, 3))
        v_924 = torch.matmul(input=v_922, other=v_923)
        v_925 = v_924.view(1, 8640, -1)
        v_926 = (v_925 / 11.313708)
        v_927 = F.softmax(input=v_926, dim=-1)
        v_928 = torch.unsqueeze(input=v_927, dim=-2)
        v_929 = torch.matmul(input=v_928, other=v_918)
        v_930 = v_929.reshape(1, 72, 120, 2)
        v_931 = torch.permute(input=v_930, dims=(0, 3, 1, 2))
        v_932 = (v_503 + (v_931 - v_917))
        v_933 = v_916.view(1, 128, -1)
        v_934 = torch.permute(input=v_933, dims=(0, 2, 1))
        v_935 = self.pnnx_unique_96(v_934)
        v_936 = self.pnnx_unique_97(v_934)
        v_937 = torch.permute(input=v_936, dims=(0, 2, 1))
        v_938 = v_937.reshape(1, 128, 72, 120)
        v_939 = F.unfold(input=v_938, dilation=(1, 1), kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        v_940 = v_939.view(1, 128, 9, 72, 120)
        v_941 = F.unfold(input=v_932, dilation=(1, 1), kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        v_942 = v_941.view(1, 2, 9, 72, 120)
        v_943 = torch.permute(input=v_940, dims=(0, 3, 4, 1, 2))
        v_944 = v_943.reshape(8640, 128, 9)
        v_945 = v_935.reshape(8640, 1, 128)
        v_946 = torch.matmul(input=v_945, other=v_944)
        v_947 = (v_946 / 11.313708)
        v_948 = F.softmax(input=v_947, dim=-1)
        v_949 = torch.permute(input=v_942, dims=(0, 3, 4, 2, 1))
        v_950 = v_949.reshape(8640, 9, 2)
        v_951 = torch.matmul(input=v_948, other=v_950)
        v_952 = v_951.view(1, 72, 120, 2)
        v_953 = torch.permute(input=v_952, dims=(0, 3, 1, 2))
        v_954 = v_953.contiguous(memory_format=torch.contiguous_format)
        v_955 = torch.cat((v_954, v_916), dim=1)
        v_956 = self.upsampler_0(v_955)
        v_957 = self.upsampler_1(v_956)
        v_958 = self.upsampler_2(v_957)
        v_959 = (v_954 * 4)
        v_960 = F.unfold(input=v_959, dilation=(1, 1), kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        v_961 = v_958.view(1, 1, 9, 4, 4, 72, 120)
        v_962 = F.softmax(input=v_961, dim=2)
        v_963 = v_960.view(1, 2, 9, 1, 1, 72, 120)
        v_964 = (v_962 * v_963)
        v_965 = torch.sum(input=v_964, dim=(2,), keepdim=False)
        v_966 = torch.permute(input=v_965, dims=(0, 1, 4, 2, 5, 3))
        v_967 = v_966.reshape(1, 2, 288, 480)
        return v_967


def export_torchscript():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 288, 480, dtype=torch.float)
    v_1 = torch.rand(1, 3, 288, 480, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1))
    mod.save("D:/60-fps-Project/VFI/GMFSS2NCNN/flownet_288_pnnx.py.pt")


def export_onnx():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 288, 480, dtype=torch.float)
    v_1 = torch.rand(1, 3, 288, 480, dtype=torch.float)

    torch.onnx._export(net, (v_0, v_1), "D:/60-fps-Project/VFI/GMFSS2NCNN/flownet_288_pnnx.py.onnx", export_params=True,
                       operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13,
                       input_names=['in0', 'in1'], output_names=['out0'])


def test_inference():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    device = torch.device("cpu")

    import cv2
    _image_path = r'input'
    # shape = (960, 544)  # override the shape variable above, indent if necessary
    shape = (480, 288)  # override the shape variable above, indent if necessary
    shape_t = (shape[1], shape[0])
    _i0 = cv2.resize(cv2.imread(os.path.join(_image_path, r'0022.jpg')), shape)
    _i1 = cv2.resize(cv2.imread(os.path.join(_image_path, r'0023.jpg')), shape)
    v_0 = torch.from_numpy(_i0).to(device).unsqueeze(0).permute(0, 3, 1, 2) / 255.
    v_1 = torch.from_numpy(_i1).to(device).unsqueeze(0).permute(0, 3, 1, 2) / 255.
    bs = 1

    return net(v_0, v_1)


if __name__ == '__main__':
    # export_torchscript()
    # export_onnx()
    import numpy as np

    output = test_inference()
    np.save("output.npy", output.detach().numpy())
    print(output.shape)
