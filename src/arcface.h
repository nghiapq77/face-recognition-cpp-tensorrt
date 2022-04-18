#ifndef ARCFACE_H
#define ARCFACE_H

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
    ArcFaceIR50(Logger gLogger, const std::string engineFile, int frameWidth, int frameHeight, std::string inputName, std::string outputName,
                std::vector<int> inputShape, int outputDim, int maxBatchSize, int maxFacesPerScene, float knownPersonThreshold);
    ~ArcFaceIR50();

    void preprocessFace(cv::Mat &face, cv::Mat &output);
    void doInference(float *input, float *output);
    void doInference(float *input, float *output, int batchSize);
    void addEmbedding(const std::string className, float embedding[]);
    void addEmbedding(const std::string className, std::vector<float> embedding);
    void forward(cv::Mat image, std::vector<struct Bbox> outputBbox);
    float *featureMatching();
    std::tuple<std::vector<std::string>, std::vector<float>> getOutputs(float *output_sims);
    void resetEmbeddings();
    void initKnownEmbeds(int num);
    void initCosSim();
    void visualize(cv::Mat &image, std::vector<std::string> names, std::vector<float> sims);

    std::vector<struct CroppedFace> croppedFaces;
    static int classCount;

  private:
    void loadEngine(Logger gLogger, const std::string engineFile);
    void preInference();
    void preInference(std::string inputName, std::string outputName);
    void preprocessFaces();

    int m_frameWidth, m_frameHeight, m_INPUT_C, m_INPUT_H, m_INPUT_W, m_OUTPUT_D, m_INPUT_SIZE, m_OUTPUT_SIZE, m_maxBatchSize, m_maxFacesPerScene;
    float m_knownPersonThresh;
    cv::Mat m_input;
    float *m_embed, *m_embeds, *m_knownEmbeds, *m_outputs;
    std::vector<std::string> classNames;

    ICudaEngine *m_engine;
    IExecutionContext *m_context;
    cudaStream_t stream;
    void *buffers[2];
    int inputIndex, outputIndex;

    CosineSimilarityCalculator cossim;
};

#endif // ARCFACE_H
