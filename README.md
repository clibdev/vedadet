# Fork of [Media-Smart/vedadet](https://github.com/Media-Smart/vedadet)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.5. (🔥)
* Original pretrained models and converted ONNX models from GitHub [releases page](https://github.com/clibdev/vedadet/releases). (🔥)
* Installation with [requirements.txt](requirements.txt) file.
* The following deprecations and errors has been fixed:
  * Fatal error: THC/THC.h: No such file or directory.
  * FutureWarning: You are using 'torch.load' with 'weights_only=False'.

# Installation

```shell
pip install -r requirements.txt
```
```shell
# First terminal
docker run -it --rm --entrypoint=bash --name=cuda-container nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Second terminal
docker cp cuda-container:/usr/local/cuda/ ./cuda-runtime
```
```shell
CUDA_HOME=$(realpath cuda-runtime) python setup.py build_ext --inplace
```

# Pretrained models

* Download links:

| Name                      | Model Size (MB) | Link                                                                                                                                                                                          | SHA-256                                                                                                                              |
|---------------------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| TinaFace (R50-FPN-BN)     | 143.4<br>143.4  | [PyTorch](https://github.com/clibdev/vedadet/releases/latest/download/tinaface-r50-fpn-bn.pt)<br>[ONNX](https://github.com/clibdev/vedadet/releases/latest/download/tinaface-r50-fpn-bn.onnx) | b59b4c57d6042676b0aa95861da432c1eed4cd77a91e0bd7c597b8387b400fb9<br>df84456cffa7257b2733368109a9f81e1bb2945bbf33fc3e4b99035730296442 |
| TinaFace (R50-FPN-GN-DCN) | 145.0           | [PyTorch](https://github.com/clibdev/vedadet/releases/latest/download/tinaface-r50-fpn-gn-dcn.pt)                                                                                             | d804dc59639109ea301756e116baf7da45a380dced26126b45e9410da6b8c1c9                                                                     |

# Inference

```shell
python tools/infer.py configs/infer/tinaface/tinaface_r50_fpn_bn.py img/bus.jpg
python tools/infer.py configs/infer/tinaface/tinaface_r50_fpn_gn_dcn.py img/bus.jpg
```

# Export to ONNX format

```shell
pip install onnx
```
```shell
python tools/torch2onnx.py configs/trainval/tinaface/tinaface_r50_fpn_bn.py tinaface-r50-fpn-bn.pt tinaface-r50-fpn-bn.onnx --dummy_input_shape 3,1650,1100 --dynamic_shape
```
