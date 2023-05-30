import numpy as np
import torch
from model.gmflow.gmflow import GMFlow
import os
import cv2

device = torch.device("cpu")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

_image_path = r'input'
# shape = (960, 544)  # override the shape variable above, indent if necessary
shape = (480, 288)  # override the shape variable above, indent if necessary
shape_t = (shape[1], shape[0])
_i0 = cv2.resize(cv2.imread(os.path.join(_image_path, r'0022.jpg')), shape)
_i1 = cv2.resize(cv2.imread(os.path.join(_image_path, r'0023.jpg')), shape)
i0 = torch.from_numpy(_i0).to(device).unsqueeze(0).permute(0,3,1,2) / 255.
i1 = torch.from_numpy(_i1).to(device).unsqueeze(0).permute(0,3,1,2) / 255.
bs = 1

flownet = GMFlow()
flownet.load_state_dict(torch.load('train_log/flownet.pkl'))
flownet = flownet.to(device)
flownet = flownet.eval()

# model_input = torch.cat((i0, i1), dim=1)
# print(model_input.shape)
model_input = (i0, i1)
# model = torch.jit.optimize_for_inference(torch.jit.trace(flownet, (model_input, )))
with torch.no_grad():
    # model = torch.jit.optimize_for_inference(torch.jit.trace(flownet, model_input, ))
    model = torch.jit.trace(flownet, model_input)
    torch.jit.save(model, "flownet_288.pt")

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
