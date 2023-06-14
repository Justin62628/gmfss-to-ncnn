# Make GMFSS-NCNN!

![download](https://img.shields.io/github/downloads/pnnx/pnnx/total.svg)

About GMFSS: , https://github.com/98mxr/GMFSS_Fortuna

Steps below.

## Get PNNX

1. Download and setup the libtorch from https://pytorch.org/

2. Clone pnnx (inside Tencent/ncnn tools/pnnx folder)

```shell
git clone https://github.com/Tencent/ncnn.git
```

3. Build with CMake

```shell
mkdir ncnn/tools/pnnx/build
cd ncnn/tools/pnnx/build
cmake -DCMAKE_INSTALL_PREFIX=install -DTorch_INSTALL_DIR=<your libtorch dir> ..
cmake --build . --config Release -j 2
cmake --build . --config Release --target install
```

## Read GMF2NCNN.ipynb and Execute Code Blocks

- Only Python 3.7 + pytorch11.7+cu117 is tested. Due to limited support of Softsplat op, other Python version may not work. 

- There are certain parameters you may need to modify:
  - Input Resolution

- Currently, only static input shape is tested, welcome to contribute dynamic input shape support!
- If lucky, you'll get reuse_xxx.ncnn.param and infer_xxx.ncnn.bin for inference, the demo locates at https://github.com/Justin62628/gmfss-ncnn-vulkan

