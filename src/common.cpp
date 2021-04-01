//
// Created by zhou on 18-4-30.
//

#include "common.h"

void *safeCudaMalloc(size_t memSize) {
    void *deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr) {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

std::vector<std::pair<int64_t, nvinfer1::DataType>> calculateBindingBufferSizes(const nvinfer1::ICudaEngine &engine,
                                                                                int nbBindings, int batchSize) {
    std::vector<std::pair<int64_t, nvinfer1::DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine.getBindingDimensions(i);
        nvinfer1::DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

inline int64_t volume(const nvinfer1::Dims &d) {
    int64_t v = 1;
    for (int64_t i = 0; i < d.nbDims; i++)
        v *= d.d[i];
    return v;
}

void loadInputImage(std::string inputFilePath, cv::Mat &image, int videoFrameWidth, int videoFrameHeight) {
    image = cv::imread(inputFilePath.c_str());
    cv::resize(image, image, cv::Size(videoFrameWidth, videoFrameHeight));
}
