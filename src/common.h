//
// Created by zhou on 18-4-30.
//

#ifndef _TRT_COMMON_H_
#define _TRT_COMMON_H_
#include "NvInfer.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#define CHECK(status)                                                                                                  \
    {                                                                                                                  \
        if (status != 0) {                                                                                             \
            std::cout << "Cuda failure: " << status;                                                                   \
            abort();                                                                                                   \
        }                                                                                                              \
    }

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger {
  public:
    void log(nvinfer1::ILogger::Severity severity, const char *msg) override {
        // suppress info-level messages
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

struct Paths {
    std::string absPath;
    std::string fileName;
};

inline bool fileExists(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void *safeCudaMalloc(size_t memSize);
inline int64_t volume(const nvinfer1::Dims &d);
std::vector<std::pair<int64_t, nvinfer1::DataType>> calculateBindingBufferSizes(const nvinfer1::ICudaEngine &engine,
                                                                                int nbBindings, int batchSize);
void getFilePaths(std::string imagesPath, std::vector<struct Paths> &paths);
void loadInputImage(std::string inputFilePath, cv::Mat &image, int videoFrameWidth, int videoFrameHeight);

#endif // _TRT_COMMON_H_
