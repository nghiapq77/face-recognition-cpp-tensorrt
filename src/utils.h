#ifndef UTILS_H
#define UTILS_H

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <chrono>
#include <cublasLt.h>
#include <curl/curl.h>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "base64.h"
#include "json.hpp"

using json = nlohmann::json;

struct Bbox {
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
};

struct CroppedFace {
    cv::Mat face;
    cv::Mat faceMat;
    int x1, y1, x2, y2;
};

struct Paths {
    std::string absPath;
    std::string className;
};

void getFilePaths(std::string rootPath, std::vector<struct Paths> &paths);
bool fileExists(const std::string &name);
void checkCudaStatus(cudaError_t status);
void checkCublasStatus(cublasStatus_t status);
void getCroppedFaces(cv::Mat frame, std::vector<struct Bbox> &outputBbox, int resize_w, int resize_h, std::vector<struct CroppedFace> &croppedFaces);

class CosineSimilarityCalculator {
    /*
    Matrix multiplication C = A x B
    Input:
        A: m x k, row-major matrix
        B: n x k, row-major matrix
        both are l2-normed
    Output:
        C: m x n, row-major matrix

    NOTE: Since cuBLAS use column-major matrix as input, we need to transpose A (transA=CUBLAS_OP_T).
    */
  public:
    CosineSimilarityCalculator();
    ~CosineSimilarityCalculator();
    void init(float *knownEmbeds, int numRow, int numCol);
    void calculate(float *embeds, int embedCount, float *outputs);

  private:
    cudaDataType_t cudaDataType = CUDA_R_32F;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cublasLtHandle_t ltHandle;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    void *workspace;
    const size_t workspaceSize = 1024 * 1024 * 4;
    cudaStream_t stream;
    float *dA, *dB, *dC;
    const float alpha = 1, beta = 0;
    int m, n, k, lda, ldb, ldc;
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;
};

class Requests {
  public:
    Requests(std::string server);
    ~Requests();
    void init_get();
    void init_send();
    void send(json j);
    json get(std::string encodedImage);

    CURLcode res;

  private:
    std::string m_server;
    CURL *m_curl;
    struct curl_slist *m_headers = NULL; // init to NULL is important
    std::string m_readBuffer;

    static size_t writeCallback(void *contents, size_t size, size_t nmemb, void *userp);
};

class Logger : public nvinfer1::ILogger {
  public:
    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override {
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

#endif // UTILS_H
