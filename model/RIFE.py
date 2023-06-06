import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.gmflow.gmflow import GMFlow
from model.IFNet_HDv3 import IFNet
from model.MetricNet import MetricNet
from model.FeatureNet import FeatureNet
from model.FusionNet import GridNet
from model.log import print_mat
from model.softsplat_v1 import Softsplat

# device = torch.device("cuda")
torch.ops.load_library("softsplat_cuda.pyd")

class ReuseNet(nn.Module):
    def __init__(self):
        super(ReuseNet, self).__init__()
        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.featnet = FeatureNet()

    def normalize_img(self, img):
        img0 = img[:, 0, :, :]
        img1 = img[:, 1, :, :]
        img2 = img[:, 2, :, :]
        img0 = (img0 - 0.485) / 0.229
        img1 = (img1 - 0.456) / 0.224
        img2 = (img2 - 0.406) / 0.225
        return torch.cat((img0.unsqueeze(1), img1.unsqueeze(1), img2.unsqueeze(1)), dim=1)


    def forward(self, img0, img1):
        # lock scale == 1.0
        feat11, feat12, feat13 = self.featnet(img0)
        feat21, feat22, feat23 = self.featnet(img1)

        img0 = F.interpolate(img0, scale_factor = 0.5, mode="bilinear", align_corners=False)
        img1 = F.interpolate(img1, scale_factor = 0.5, mode="bilinear", align_corners=False)

        imgf0 = self.normalize_img(img0)
        imgf1 = self.normalize_img(img1)

        flow01 = self.flownet(imgf0, imgf1)
        flow10 = self.flownet(imgf1, imgf0)

        metric = self.metricnet(img0, img1, flow01, flow10)

        # print_mat(flow01, "flow01")
        # print_mat(flow10, "flow10")
        # print_mat(metric, "metric")
        # print_mat(feat11, "feat11")
        # print_mat(feat12, "feat12")
        # print_mat(feat13, "feat13")
        # print_mat(feat21, "feat21")
        # print_mat(feat22, "feat22")
        # print_mat(feat23, "feat23")

        return flow01, flow10, metric, feat11, feat12, feat13, feat21, feat22, feat23


class InferNet(nn.Module):
    def __init__(self):
        super(InferNet, self).__init__()
        self.ifnet = IFNet()
        self.fusionnet = GridNet(9, 64*2, 128*2, 192*2, 3)
        self.softsplat = Softsplat()
    
    def warp(self, tenIn, tenFlow, tenMetric):
        exp = torch.exp(tenMetric)
        tenIn = torch.cat([torch.mul(tenIn, exp), exp], 1)
        # print_mat(tenIn, "ss_in")
        # print_mat(tenFlow, "ss_flow")
        tenOut = torch.ops.softsplat.forward(tenIn, tenFlow)
        # print_mat(tenOut, "ss_out")
        tenNormalize = tenOut[:, -1, :, :]
        tenNormalize = tenNormalize + 0.00001
        tenOut = tenOut[:, :-1, :, :] / tenNormalize
        return tenOut

    def forward(self, img0, img1, timestep, flow01, flow10, metric, feat11, feat12, feat13, feat21, feat22, feat23):
        # t_f = timestep.repeat(1, flow01.shape[1], flow01.shape[2], flow01.shape[3])
        F1t = torch.mul(flow01, timestep)  # B, 2, H, W
        F2t = torch.mul(flow10, (1-timestep))

        Z1t = torch.mul(metric, timestep)[:, 0:1, :, :]  # B, 1, H, W
        Z2t = torch.mul(metric, (1-timestep))[:, 1:2, :, :]

        img0 = F.interpolate(img0, scale_factor = 0.5, mode="bilinear", align_corners=False)
        img1 = F.interpolate(img1, scale_factor = 0.5, mode="bilinear", align_corners=False)

        F1td = F.interpolate(F1t, scale_factor = 0.5, mode="bilinear", align_corners=False) * 0.5
        Z1d = F.interpolate(Z1t, scale_factor = 0.5, mode="bilinear", align_corners=False)        
        
        F2td = F.interpolate(F2t, scale_factor = 0.5, mode="bilinear", align_corners=False) * 0.5
        Z2d = F.interpolate(Z2t, scale_factor = 0.5, mode="bilinear", align_corners=False)
        
        F1tdd = F.interpolate(F1t, scale_factor = 0.25, mode="bilinear", align_corners=False) * 0.25
        Z1dd = F.interpolate(Z1t, scale_factor = 0.25, mode="bilinear", align_corners=False)
        
        F2tdd = F.interpolate(F2t, scale_factor = 0.25, mode="bilinear", align_corners=False) * 0.25
        Z2dd = F.interpolate(Z2t, scale_factor = 0.25, mode="bilinear", align_corners=False)
        
        I1t = self.warp(img0, F1t, Z1t)
        I2t = self.warp(img1, F2t, Z2t)
        feat1t1 = self.warp(feat11, F1t, Z1t)
        feat2t1 = self.warp(feat21, F2t, Z2t)

        feat1t2 = self.warp(feat12, F1td, Z1d)
        feat2t2 = self.warp(feat22, F2td, Z2d)

        feat1t3 = self.warp(feat13, F1tdd, Z1dd)
        feat2t3 = self.warp(feat23, F2tdd, Z2dd)

        imgs = torch.cat((img0, img1), 1)
        rife = self.ifnet(imgs, timestep, scale_list=[8, 4, 2, 1])
        # print_mat(I2t, "I2t")
        # print_mat(I2t, "I2t")
        # print_mat(rife, "rife")

        out = self.fusionnet(torch.cat([I1t, rife, I2t], dim=1), torch.cat([feat1t1, feat2t1], dim=1), torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))
        out = torch.clamp(out, 0, 1)
        # print_mat(out, "out")
        return out


class Model:
    def __init__(self):
        self.reusenet = ReuseNet()
        self.infernet = InferNet()
        self.version = 3.9

    def eval(self):
        self.reusenet.eval()
        self.infernet.eval()

    def device(self, device='cuda'):
        self.reusenet.to(device)
        self.infernet.to(device)

    def load_model(self, path, rank):
        def convert(param, title):
            return {
                f"{title}.{k}" : v
                for k, v in param.items()
            }
        reuse_param: dict = convert(torch.load('{}/flownet.pkl'.format(path)), 'flownet')
        reuse_param.update(convert(torch.load('{}/metric.pkl'.format(path)), 'metricnet'))
        reuse_param.update(convert(torch.load('{}/feat.pkl'.format(path)), 'featnet'))
        self.reusenet.load_state_dict(reuse_param)

        infer_param: dict = convert(torch.load('{}/rife.pkl'.format(path)), 'ifnet')
        infer_param.update(convert(torch.load('{}/fusionnet.pkl'.format(path)), 'fusionnet'))
        self.infernet.load_state_dict(infer_param)

    def reuse(self, img0, img1, scale):
        flow01, flow10, metric, feat11, feat12, feat13, feat21, feat22, feat23 = self.reusenet(img0, img1)
        # feat_ext0 = [feat11, feat12, feat13]
        # feat_ext1 = [feat21, feat22, feat23]

        return flow01, flow10, metric, feat11, feat12, feat13, feat21, feat22, feat23

    def inference(self, img0, img1, timestep, flow01, flow10, metric, feat11, feat12, feat13, feat21, feat22, feat23):
        # flow01, metric0, feat11, feat12, feat13 = reuse_things[0], reuse_things[2], reuse_things[4][0], reuse_things[4][1], reuse_things[4][2]
        # flow10, metric1, feat21, feat22, feat23 = reuse_things[1], reuse_things[3], reuse_things[5][0], reuse_things[5][1], reuse_things[5][2]
        return self.infernet(img0, img1, timestep, flow01, flow10, metric, feat11, feat12, feat13, feat21, feat22, feat23)
