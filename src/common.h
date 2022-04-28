#ifndef COMMON_H
#define COMMON_H

#include "NvInfer.h"
// #include "cuda_runtime_api.h"
#include <cublasLt.h>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Bbox {
    int x1, y1, x2, y2;
    float score;
};

struct Paths {
    std::string absPath;
    std::string className;
};

bool fileExists(const std::string &name);
void getFilePaths(std::string rootPath, std::vector<struct Paths> &paths);
void checkCudaStatus(cudaError_t status);
void checkCublasStatus(cublasStatus_t status);

class TRTLogger : public nvinfer1::ILogger {
  public:
    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override {
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
        case Severity::kVERBOSE:
            std::cerr << "VERBOSE: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

#endif // COMMON_H
