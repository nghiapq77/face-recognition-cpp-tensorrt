#ifndef UTILS_H
#define UTILS_H

#include "NvInfer.h"
//#include "cblas.h"
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

struct KnownID {
    std::string className;
    std::vector<float> embeddedFace;
};

struct Paths {
    std::string absPath;
    std::string className;
};

void getFilePaths(std::string rootPath, std::vector<struct Paths> &paths);
bool fileExists(const std::string &name);
void checkCudaStatus(cudaError_t status);
void checkCublasStatus(cublasStatus_t status);
/*
void l2_norm(float *p, int size = 512);
float cosine_similarity(std::vector<float> &A, std::vector<float> &B);
std::vector<std::vector<float>> batch_cosine_similarity(std::vector<std::vector<float>> &A,
                                                        std::vector<struct KnownID> &B, const int size,
                                                        bool normalize = false);
void cublas_batch_cosine_similarity(float *A, float *B, int embedCount, int classCount, int size, float *outputs);
void batch_cosine_similarity(std::vector<std::vector<float>> A, std::vector<struct KnownID> B, int size,
                             float *outputs);
void batch_cosine_similarity(float *A, float *B, int embedCount, int classCount, int size, float *outputs);
*/
void getCroppedFaces(cv::Mat frame, std::vector<struct Bbox> &outputBbox, int resize_w, int resize_h,
                     std::vector<struct CroppedFace> &croppedFaces);

class CosineSimilarityCalculator {
  public:
    CosineSimilarityCalculator();
    ~CosineSimilarityCalculator();
    void init(float *knownEmbeds, int numRow, int numCol);
    void calculate(float *embeds, int embedCount, float *outputs);

  private:
    cudaDataType_t dataType = CUDA_R_32F;
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
    //void send(std::vector<std::string> names, std::vector<float> sims, std::vector<struct CroppedFace> &croppedFaces,
              //int classCount, float threshold, std::string check_type);
    void send(json j);
    json get(std::string encodedImage);

    CURLcode res;

  private:
    std::string m_server;
    CURL *m_curl;
    struct curl_slist *m_headers = NULL; // init to NULL is important
    std::string m_readBuffer;
};

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
