import os
import cv2
import torch
import argparse
import numpy as np
from torch.nn import functional as F
import warnings
from queue import Queue, Empty

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default="input")
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--multi', dest='multi', type=int, default=2)

args = parser.parse_args()
if args.exp != 1:
    args.multi = (2 ** args.exp)
assert (not args.video is None or not args.img is None)
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if(args.fp16):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

try:
    from model.RIFE import Model
except:
    print("Please download our model from model list")
model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model(args.modelDir, -1)
print("Loaded model")
model.eval()
model.device()

videogen = []
for f in os.listdir(args.img):
    if 'png' in f or 'jpg' in f:
        videogen.append(f)
tot_frame = len(videogen)
videogen.sort(key= lambda x:int(x[:-4]))
lastframe = cv2.resize(cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy(), (480, 288))
nextframe = cv2.resize(cv2.imread(os.path.join(args.img, videogen[1]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy(), (480, 288))

videogen = videogen[1:]

h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
if not os.path.exists('vid_out'):
    os.mkdir('vid_out')


def make_inference(I0, I1, reuse_things, n):    
    global model
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, reuse_things, (i+1) * 1. / (n+1)))
        return res
    else:
        middle = model.inference(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

tmp = max(64, int(64 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)

I0 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I0 = F.interpolate(I0, (ph, pw), mode='bilinear', align_corners=False)
I1 = torch.from_numpy(np.transpose(nextframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = F.interpolate(I1, (ph, pw), mode='bilinear', align_corners=False)
    
reuse_things = model.reuse(I0, I1, args.scale)
output = make_inference(I0, I1, reuse_things, args.multi-1)
cv2.imwrite('check.png', (output[0] * 255.).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)[:, :, ::-1])