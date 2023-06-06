import numpy as np
import torch
from model.RIFE import InferNet, ReuseNet
from model.gmflow.gmflow import GMFlow
import os
import cv2
from model.log import LOG_STATE
LOG_STATE.is_log = False

from model.gmflow.utils import normalize_img
import pickle

device = torch.device("cuda")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

_image_path = r'input2'
shape = (960, 576)  # override the shape variable above, indent if necessary
# shape = (480, 288)  # override the shape variable above, indent if necessary
shape_t = (shape[1], shape[0])
# _i0 = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(_image_path, r'0022.jpg')), cv2.COLOR_BGR2RGB), shape)
# _i1 = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(_image_path, r'0023.jpg')), cv2.COLOR_BGR2RGB), shape)
_i0 = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(_image_path, r'0022.png')), cv2.COLOR_BGR2RGB), shape)
_i1 = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(_image_path, r'0023.png')), cv2.COLOR_BGR2RGB), shape)

i0 = torch.from_numpy(_i0).to(device).unsqueeze(0).permute(0,3,1,2) / 255.
i1 = torch.from_numpy(_i1).to(device).unsqueeze(0).permute(0,3,1,2) / 255.
bs = 1

# GMF
# net = GMFlow()
# net.load_state_dict(torch.load('train_log/flownet.pkl'))
# net_input = normalize_img(i0, i1)
# pt_path = "flownet_288.pt"

def convert(param, title):
    return {
        f"{title}.{k}" : v
        for k, v in param.items()
    }
# ReuseNet
# net = ReuseNet()
# reuse_param: dict = convert(torch.load('train_log/flownet.pkl'), 'flownet')
# reuse_param.update(convert(torch.load('train_log/metric.pkl'), 'metricnet'))
# reuse_param.update(convert(torch.load('train_log/feat.pkl'), 'featnet'))
# net.load_state_dict(reuse_param)
# net_input = i0, i1
# pt_path = "reuse_576.pt"
# net = net.to(device)

# with torch.no_grad():
#     LOG_STATE.is_log = True
#     net_output = net(*net_input)
# with open('reuse_output.pkl', 'wb') as f:
#     pickle.dump([i.cpu() for i in net_output], f)

# InferNet
net = InferNet()
infer_param: dict = convert(torch.load('train_log/rife.pkl'), 'ifnet')
infer_param.update(convert(torch.load('train_log/fusionnet.pkl'), 'fusionnet'))
net.load_state_dict(infer_param)
with open('reuse_output.pkl', 'rb') as f:
    net_input_ = pickle.load(f)
net_input = (i0, i1, torch.full((1, 1, 1, 1), 0.5).to(device), *[i.to(device) for i in net_input_])
pt_path = "infer_576.pt"

net = net.eval()
net = net.to(device)


# debug below, for comparison between ncnn and gt
# with torch.no_grad():
#     LOG_STATE.is_log = True
#     net_output = net(*net_input)
    

# gen pt
with torch.no_grad():
    LOG_STATE.is_log = False
    # model = torch.jit.optimize_for_inference(torch.jit.trace(flownet, model_input, ))
    model = torch.jit.trace(net, [i.to(device) for i in net_input])
    torch.jit.save(model, pt_path)

# ref_output = flownet(*model_input)
# print(ref_output.shape)
# np.save('ref_output.npy', to_numpy(ref_output))
# ref_pt_output = model(*model_input)
# print(ref_pt_output.shape)
# np.save('ref_pt_output.npy', to_numpy(ref_pt_output))

# onnx_path = f"flownet_op16_{shape_t[0]}x{shape_t[1]}.onnx"
# with torch.no_grad():
#     torch.onnx.export(
#                     model, 
#                     model_input,  # model input (or a tuple for multiple inputs)
#                     onnx_path,  # where to save the model (can be a file or file-like object)
#                     export_params=True,  # store the trained parameter weights inside the model file
#                     opset_version=16,  # the ONNX version to export the model to
#                     do_constant_folding=True,  # whether to execute constant folding for optimization
#                     input_names=['i0', 'i1'],  # the model's input names
#                     output_names=['output'],  # the model's output names
#                     # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                     #               'output' : {0 : 'batch_size'}}
#                     # dynamic_axes={'i0' : {2 : 'height', 3: 'width'},    # variable length axes
#                     #               'i1' : {2 : 'height', 3: 'width'},
#                     #               't' : {2 : 'height', 3: 'width'},
#                     #               'output' : {2 : 'height', 3: 'width'},}
#                     )
