import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 576, 960, dtype=torch.float)
    in1 = torch.rand(1, 3, 576, 960, dtype=torch.float)
    in2 = torch.rand(1, 1, 1, 1, dtype=torch.float)
    in3 = torch.rand(1, 2, 288, 480, dtype=torch.float)
    in4 = torch.rand(1, 2, 288, 480, dtype=torch.float)
    in5 = torch.rand(1, 2, 288, 480, dtype=torch.float)
    in6 = torch.rand(1, 64, 288, 480, dtype=torch.float)
    in7 = torch.rand(1, 128, 144, 240, dtype=torch.float)
    in8 = torch.rand(1, 192, 72, 120, dtype=torch.float)
    in9 = torch.rand(1, 64, 288, 480, dtype=torch.float)
    in10 = torch.rand(1, 128, 144, 240, dtype=torch.float)
    in11 = torch.rand(1, 192, 72, 120, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
         net.load_param("D:/60-fps-Project/VFI/GMFSS2NCNN/infer_576.ncnn.param")
         net.load_model("D:/60-fps-Project/VFI/GMFSS2NCNN/infer_576.ncnn.bin")

         with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())
            ex.input("in1", ncnn.Mat(in1.squeeze(0).numpy()).clone())
            ex.input("in2", ncnn.Mat(in2.squeeze(0).numpy()).clone())
            ex.input("in3", ncnn.Mat(in3.squeeze(0).numpy()).clone())
            ex.input("in4", ncnn.Mat(in4.squeeze(0).numpy()).clone())
            ex.input("in5", ncnn.Mat(in5.squeeze(0).numpy()).clone())
            ex.input("in6", ncnn.Mat(in6.squeeze(0).numpy()).clone())
            ex.input("in7", ncnn.Mat(in7.squeeze(0).numpy()).clone())
            ex.input("in8", ncnn.Mat(in8.squeeze(0).numpy()).clone())
            ex.input("in9", ncnn.Mat(in9.squeeze(0).numpy()).clone())
            ex.input("in10", ncnn.Mat(in10.squeeze(0).numpy()).clone())
            ex.input("in11", ncnn.Mat(in11.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)
