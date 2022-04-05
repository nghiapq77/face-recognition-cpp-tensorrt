#ifndef ARCFACE_IR50_H
#define ARCFACE_IR50_H

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include <NvInferPlugin.h>
#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "utils.h"

using namespace nvinfer1;

class ArcFaceIR50 {
  public:
    ArcFaceIR50(Logger gLogger, const std::string engineFile, int frameWidth, int frameHeight,
                std::vector<int> inputShape, int outputDim, int maxBatchSize, int maxFacesPerScene,
                float knownPersonThreshold);
    ~ArcFaceIR50();

    void preprocessFace(cv::Mat &face, cv::Mat &output);
    void preprocessFaces();
    void preprocessFaces_();
    void doInference(float *input, float *output);
    void doInference(float *input, float *output, int batchSize);
    void forwardAddFace(cv::Mat image, std::vector<struct Bbox> outputBbox, const std::string className);
    void addEmbedding(const std::string className, float embedding[]);
    void addEmbedding(const std::string className, std::vector<float> embedding);
    void forward(cv::Mat image, std::vector<struct Bbox> outputBbox);
    float *featureMatching();
    std::tuple<std::vector<std::string>, std::vector<float>> getOutputs(float *output_sims);
    void visualize(cv::Mat &image, std::vector<std::string> names, std::vector<float> sims);
    void addNewFace(cv::Mat &image, std::vector<struct Bbox> outputBbox);
    void resetVariables();
    void resetEmbeddings();
    void initKnownEmbeds(int num);
    void initCosSim();

    std::vector<struct CroppedFace> croppedFaces;
    std::vector<struct KnownID> knownFaces;
    static int classCount;

  private:
    const char *m_INPUT_BLOB_NAME = "input";
    const char *m_OUTPUT_BLOB_NAME = "output";
    int m_frameWidth, m_frameHeight, m_INPUT_C, m_INPUT_H, m_INPUT_W, m_OUTPUT_D, m_OUTPUT_SIZE_BASE, m_INPUT_SIZE,
        m_OUTPUT_SIZE, m_maxBatchSize, m_maxFacesPerScene;
    cv::Mat m_input;
    float *m_embed, *m_embeds, *m_knownEmbeds, *m_outputs;
    //std::vector<std::vector<float>> m_embeddings;

    Logger m_gLogger;
    std::string m_engineFile;
    float m_knownPersonThresh;
    ICudaEngine *m_engine;
    IExecutionContext *m_context;
    cudaStream_t stream;
    void *buffers[2];
    int inputIndex, outputIndex;

    CosineSimilarityCalculator cossim;

    void loadEngine(Logger gLogger, const std::string engineFile);
    void preInference();
};

#endif // ARCFACE_IR50_H
