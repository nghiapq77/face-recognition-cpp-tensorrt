#ifndef ARCFACE_H
#define ARCFACE_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <tuple>

#include "common.h"
#include "matmul.h"

struct CroppedFace {
    cv::Mat face;
    cv::Mat faceMat;
    int x1, y1, x2, y2;
};

void getCroppedFaces(cv::Mat frame, std::vector<struct Bbox> &outputBbox, int resize_w, int resize_h, std::vector<struct CroppedFace> &croppedFaces);

class ArcFaceIR50 {
  public:
    ArcFaceIR50(TRTLogger gLogger, const std::string engineFile, int frameWidth, int frameHeight, std::string inputName, std::string outputName,
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
    void initMatMul();
    void visualize(cv::Mat &image, std::vector<std::string> names, std::vector<float> sims);

    std::vector<struct CroppedFace> croppedFaces;
    static int classCount;

  private:
    void loadEngine(TRTLogger gLogger, const std::string engineFile);
    void preInference();
    void preInference(std::string inputName, std::string outputName);
    void preprocessFaces();

    int m_frameWidth, m_frameHeight, m_INPUT_C, m_INPUT_H, m_INPUT_W, m_OUTPUT_D, m_INPUT_SIZE, m_OUTPUT_SIZE, m_maxBatchSize, m_maxFacesPerScene;
    float m_knownPersonThresh;
    cv::Mat m_input;
    float *m_embed, *m_embeds, *m_knownEmbeds, *m_outputs;
    std::vector<std::string> classNames;

    nvinfer1::ICudaEngine *m_engine;
    nvinfer1::IExecutionContext *m_context;
    cudaStream_t stream;
    void *buffers[2];
    int inputIndex, outputIndex;

    MatMul matmul;
};

#endif // ARCFACE_H
