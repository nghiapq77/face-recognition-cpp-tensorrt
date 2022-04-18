# Face Recognition using C++ TensorRT
Face Recognition with [RetinaFace](https://github.com/biubug6/Face-Detector-1MB-with-landmark) and [ArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch).

This project uses [TensorRT](https://developer.nvidia.com/tensorrt), [OpenCV](https://opencv.org/) and [cuBLAS](https://developer.nvidia.com/cublas) 
to implement a face recognition system with simple [SQLite](https://www.sqlite.org/) database and
Web APIs via [Crow](https://github.com/CrowCpp/Crow).

## Requirements
- CUDA 11.3
- TensorRT 8.2.2.1
- OpenCV 4.5.5
- SQLite 3.31.1
- Crow 1.0
- Boost 1.79.0
- CURL 7.68

## Installation
```bash
git clone https://github.com/nghiapq77/face-recognition-cpp-tensorrt.git
cd face-recognition-cpp-tensorrt
cd app
mkdir build && cd build
cmake ..
make -j$(nproc)
cd main
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Pretrained
### Requirements
- Python 3.8
- torch 1.11.0+cu113
- torchvision 0.12.0+cu113

### Pretrained models
- [RetinaFace](https://github.com/biubug6/Face-Detector-1MB-with-landmark/tree/master/weights)
- [ArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#model-zoo)

### Convert torch model weights to serialized TensorRT engines
Using [torch2trt\_dynamic](https://github.com/grimoire/torch2trt_dynamic).

```bash
cd conversion/[retina/arcface]
python torch2trt.py
```

## Usage
```bash
cd app/build
./app -c <config-file>
```

Example:
```bash
curl localhost:18080/insert/user -d '{"userId": "morty", "userName": "Morty Smith"}'
curl localhost:18080/insert/face -d '{"data": [{"userId": "morty", "imgPath": "<absolute-path-to-this-repo>/imgs/1.jpg"}]}'
curl localhost:18080/reload
cd main/build
./main
```
<p align="center">
    <img src="imgs/vis.jpg">
</p>
<p align="center">
Visualized result from example.
</p>

## References
- [Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark)
- [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)
- [mtcnn\_facenet\_cpp\_tensorRT](https://github.com/nwesem/mtcnn_facenet_cpp_tensorRT)
- [tensorrtx](https://github.com/wang-xinyu/tensorrtx)
