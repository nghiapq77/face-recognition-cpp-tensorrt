# Face Recognition using C++ TensorRT
Face Recognition with [RetinaFace](https://github.com/biubug6/Face-Detector-1MB-with-landmark) and [ArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch).

## Requirements
- CUDA 10.2 + cuDNN 8.0
- TensorRT 7.1.3
- OpenCV 4.1.1
- CURL 7.75.0

## Installation
```bash
git clone https://github.com/nghiapq77/face-recognition-cpp-tensorrt.git
cd face-recognition-cpp-tensorrt
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Pretrained
### Requirements
- Python 3.6
- torch 1.7.0
- torchvision 0.8.1

### Pretrained models
- [RetinaFace](https://github.com/biubug6/Face-Detector-1MB-with-landmark/tree/master/weights)
- [ArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#model-zoo)

### Convert torch model weights to serialized TensorRT engines
```bash
cd convert/[retina/arc]
python torch_to_onnx.py
cd ..
python onnx_to_tensorrt.py -t [retina/arc]
```

## Usage
```bash
./face -c <config-file>
```
Modify `input_embeddingsFile`, `input_numImagesFile`, `gen_imgSource`, `gen_imgIsCropped` in `config.json` to generate embeddings for known faces.  
See `main.cpp` for more detail.

## References
- [Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark)
- [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)
- [mtcnn\_facenet\_cpp\_tensorRT](https://github.com/nwesem/mtcnn_facenet_cpp_tensorRT)
- [tensorrtx](https://github.com/wang-xinyu/tensorrtx)
