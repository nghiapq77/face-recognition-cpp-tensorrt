#ifndef FACE_RECOGNITION_ARCFACE_IR50_H
#define FACE_RECOGNITION_ARCFACE_IR50_H

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include <NvInferPlugin.h>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "utils.h"

using namespace std;
using namespace nvinfer1;

class ArcFaceIR50 {
  public:
    ArcFaceIR50(Logger gLogger, const string engineFile, float knownPersonThreshold, int maxFacesPerScene,
                int frameWidth, int frameHeight);
    ~ArcFaceIR50();

    void preprocessFace(cv::Mat &face);
    void preprocessFaces();
    void doInference(float *input, float *output, bool normalize = false);
    void forwardAddFace(cv::Mat image, std::vector<struct Bbox> outputBbox, const string className);
    void addEmbedding(const string className, std::vector<float> embedding);
    void forward(cv::Mat image, std::vector<struct Bbox> outputBbox);
    std::vector<std::vector<float>> featureMatching();
    float *featureMatching(float *outputs);
    void visualize(cv::Mat &image, std::vector<std::vector<float>> &outputs);
    void visualize(cv::Mat &image, float *outputs);
    void addNewFace(cv::Mat &image, std::vector<struct Bbox> outputBbox);
    void resetVariables();
    void init_knownEmbeds(int num);

    std::vector<struct KnownID> m_knownFaces;

  private:
    static int m_classCount;
    const char *m_INPUT_BLOB_NAME;
    const char *m_OUTPUT_BLOB_NAME;
    static const int m_INPUT_C = 3;
    static const int m_INPUT_H = 112;
    static const int m_INPUT_W = 112;
    static const int m_OUTPUT_SIZE = 512;
    static const int m_batchSize = 1;
    cv::Mat m_input;
    int m_frameWidth, m_frameHeight;
    Logger m_gLogger;
    string m_engineFile;
    int m_maxFacesPerScene;
    float m_knownPersonThresh;
    ICudaEngine *m_engine;
    IExecutionContext *m_context;
    float m_output[m_OUTPUT_SIZE];
    float *m_outputs;
    std::vector<std::vector<float>> m_embeddings;
    float *m_embeds;
    float *m_knownEmbeds;
    std::vector<struct CroppedFace> m_croppedFaces;

    void createOrLoadEngine(Logger gLogger, const string engineFile);
};

#endif // FACE_RECOGNITION_ARCFACE_IR50_H
