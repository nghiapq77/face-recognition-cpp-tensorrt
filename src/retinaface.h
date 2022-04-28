#ifndef RETINAFACE_H
#define RETINAFACE_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "common.h"

#define CLIP(a, min, max) (MAX(MIN(a, max), min)) // MIN, MAX defined in opencv

struct anchorBox {
    float cx;
    float cy;
    float sx;
    float sy;
};

class RetinaFace {
  public:
    RetinaFace(TRTLogger gLogger, const std::string engineFile, int frameWidth, int frameHeight, std::string inputName, std::vector<std::string> outputNames,
               std::vector<int> inputShape, int maxBatchSize, int maxFacesPerScene, float nms_threshold, float bbox_threshold);
    ~RetinaFace();
    std::vector<struct Bbox> findFace(cv::Mat &img);

  private:
    void loadEngine(TRTLogger gLogger, const std::string engineFile);
    void preInference();
    void preInference(std::string inputNames, std::vector<std::string> outputNames);
    void doInference(float *input, float *output0, float *output1);
    void preprocess(cv::Mat &img);
    void postprocessing(float *bbox, float *conf);
    void create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h);
    static inline bool m_cmp(Bbox a, Bbox b);
    void nms(std::vector<Bbox> &input_boxes, float NMS_THRESH);

    int m_frameWidth, m_frameHeight, m_INPUT_C, m_INPUT_H, m_INPUT_W, m_INPUT_SIZE, m_OUTPUT_SIZE_BASE, m_maxBatchSize, m_maxFacesPerScene;
    float m_nms_threshold, m_bbox_threshold, m_scale_h, m_scale_w;
    cv::Mat m_input;
    float *m_output0, *m_output1;
    std::vector<struct Bbox> m_outputBbox;

    nvinfer1::ICudaEngine *m_engine;
    nvinfer1::IExecutionContext *m_context;
    cudaStream_t stream;
    void *buffers[3];
    int inputIndex, outputIndex0, outputIndex1;
};

#endif // RETINAFACE_H
