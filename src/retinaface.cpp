#include "retinaface.h"

RetinaFace::RetinaFace(TRTLogger gLogger, const std::string engineFile, int frameWidth, int frameHeight, std::string inputName,
                       std::vector<std::string> outputNames, std::vector<int> inputShape, int maxBatchSize, int maxFacesPerScene, float nms_threshold,
                       float bbox_threshold) {
    m_frameWidth = static_cast<const int>(frameWidth);
    m_frameHeight = static_cast<const int>(frameHeight);
    assert(inputShape.size() == 3);
    m_INPUT_C = static_cast<const int>(inputShape[0]);
    m_INPUT_H = static_cast<const int>(inputShape[1]);
    m_INPUT_W = static_cast<const int>(inputShape[2]);
    m_INPUT_SIZE = static_cast<const int>(m_INPUT_C * m_INPUT_H * m_INPUT_W * sizeof(float));
    m_OUTPUT_SIZE_BASE = static_cast<const int>((m_INPUT_H / 8 * m_INPUT_W / 8 + m_INPUT_H / 16 * m_INPUT_W / 16 + m_INPUT_H / 32 * m_INPUT_W / 32) * 2);
    m_output0 = new float[m_OUTPUT_SIZE_BASE * 4];
    m_output1 = new float[m_OUTPUT_SIZE_BASE * 2];
    m_maxBatchSize = static_cast<const int>(maxBatchSize);
    m_maxFacesPerScene = static_cast<const int>(maxFacesPerScene);
    m_nms_threshold = static_cast<const float>(nms_threshold);
    m_bbox_threshold = static_cast<const float>(bbox_threshold);

    m_scale_h = (float)m_INPUT_H / m_frameHeight;
    m_scale_w = (float)m_INPUT_W / m_frameWidth;

    // load engine from .engine file
    loadEngine(gLogger, engineFile);

    // create stream and pre-allocate GPU buffers memory
    preInference(inputName, outputNames);
}

void RetinaFace::loadEngine(TRTLogger gLogger, const std::string engineFile) {
    if (fileExists(engineFile)) {
        std::cout << "[INFO] Loading RetinaFace Engine...\n";
        std::vector<char> trtModelStream_;
        size_t size{0};

        std::ifstream file(engineFile, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream_.resize(size);
            file.read(trtModelStream_.data(), size);
            file.close();
        }
        nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
        assert(runtime != nullptr);
        m_engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size);
        assert(m_engine != nullptr);
        m_context = m_engine->createExecutionContext();
        assert(m_context != nullptr);
    } else {
        throw std::logic_error("Cant find engine file");
    }
}

void RetinaFace::preInference() {
    /*
    Does not make landmark head as we do not use face alignment.
    */
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(m_engine->getNbBindings() == 3);

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = m_engine->getBindingIndex("input_det");
    outputIndex0 = m_engine->getBindingIndex("output_det0");
    outputIndex1 = m_engine->getBindingIndex("output_det1");
    // outputIndex2 = m_engine->getBindingIndex("output_det2");

    // Create GPU buffers on device
    checkCudaStatus(cudaMalloc(&buffers[inputIndex], m_maxBatchSize * m_INPUT_SIZE));
    checkCudaStatus(cudaMalloc(&buffers[outputIndex0], m_maxBatchSize * m_OUTPUT_SIZE_BASE * 4 * sizeof(float)));
    checkCudaStatus(cudaMalloc(&buffers[outputIndex1], m_maxBatchSize * m_OUTPUT_SIZE_BASE * 2 * sizeof(float)));
    // checkCudaStatus(cudaMalloc(&buffers[outputIndex2], m_maxBatchSize * m_OUTPUT_SIZE_BASE * 10 * sizeof(float)));

    // Create stream
    checkCudaStatus(cudaStreamCreate(&stream));
}

void RetinaFace::preInference(std::string inputName, std::vector<std::string> outputNames) {
    /*
    Does not make landmark head as we do not use face alignment.
    */
    // Assert
    assert(outputNames.size() == 2);

    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(m_engine->getNbBindings() == 3);

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = m_engine->getBindingIndex(inputName.c_str());
    outputIndex0 = m_engine->getBindingIndex(outputNames[0].c_str());
    outputIndex1 = m_engine->getBindingIndex(outputNames[1].c_str());

    // Create GPU buffers on device
    checkCudaStatus(cudaMalloc(&buffers[inputIndex], m_maxBatchSize * m_INPUT_SIZE));
    checkCudaStatus(cudaMalloc(&buffers[outputIndex0], m_maxBatchSize * m_OUTPUT_SIZE_BASE * 4 * sizeof(float)));
    checkCudaStatus(cudaMalloc(&buffers[outputIndex1], m_maxBatchSize * m_OUTPUT_SIZE_BASE * 2 * sizeof(float)));

    // Create stream
    checkCudaStatus(cudaStreamCreate(&stream));
}

void RetinaFace::preprocess(cv::Mat &img) {
    // Release input vector
    m_input.release();

    // Resize
    int w, h, x, y;
    if (m_scale_h > m_scale_w) {
        w = m_INPUT_W;
        h = m_scale_w * img.rows;
        x = 0;
        y = (m_INPUT_H - h) / 2;
    } else {
        w = m_scale_h * img.cols;
        h = m_INPUT_H;
        x = (m_INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(m_INPUT_H, m_INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    // Normalize
    out.convertTo(out, CV_32F);
    out = out - cv::Scalar(104, 117, 123);
    std::vector<cv::Mat> temp;
    cv::split(out, temp);
    for (int i = 0; i < temp.size(); i++) {
        m_input.push_back(temp[i]);
    }
}

void RetinaFace::doInference(float *input, float *output0, float *output1) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    checkCudaStatus(cudaMemcpyAsync(buffers[inputIndex], input, m_maxBatchSize * m_INPUT_SIZE, cudaMemcpyHostToDevice, stream));
    m_context->enqueueV2(buffers, stream, nullptr);
    checkCudaStatus(cudaMemcpyAsync(output0, buffers[outputIndex0], m_maxBatchSize * m_OUTPUT_SIZE_BASE * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkCudaStatus(cudaMemcpyAsync(output1, buffers[outputIndex1], m_maxBatchSize * m_OUTPUT_SIZE_BASE * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

std::vector<struct Bbox> RetinaFace::findFace(cv::Mat &img) {
    preprocess(img);
    doInference((float *)m_input.ptr<float>(0), m_output0, m_output1);
    postprocessing(m_output0, m_output1);
    return m_outputBbox;
}

void RetinaFace::postprocessing(float *bbox, float *conf) {
    m_outputBbox.clear();
    std::vector<anchorBox> anchor;
    create_anchor_retinaface(anchor, m_INPUT_W, m_INPUT_H);

    for (int i = 0; i < anchor.size(); ++i) {
        if (*(conf + 1) > m_bbox_threshold) {
            anchorBox tmp = anchor[i];
            anchorBox tmp1;
            Bbox result;

            // decode bbox (y - W; x - H)
            tmp1.cx = tmp.cx + *bbox * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(bbox + 1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(bbox + 2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(bbox + 3) * 0.2);

            result.y1 = (tmp1.cx - tmp1.sx / 2) * m_INPUT_W;
            result.x1 = (tmp1.cy - tmp1.sy / 2) * m_INPUT_H;
            result.y2 = (tmp1.cx + tmp1.sx / 2) * m_INPUT_W;
            result.x2 = (tmp1.cy + tmp1.sy / 2) * m_INPUT_H;

            // rescale to original size
            if (m_scale_h > m_scale_w) {
                result.y1 = result.y1 / m_scale_w;
                result.y2 = result.y2 / m_scale_w;
                result.x1 = (result.x1 - (m_INPUT_H - m_scale_w * m_frameHeight) / 2) / m_scale_w;
                result.x2 = (result.x2 - (m_INPUT_H - m_scale_w * m_frameHeight) / 2) / m_scale_w;
            } else {
                result.y1 = (result.y1 - (m_INPUT_W - m_scale_h * m_frameWidth) / 2) / m_scale_h;
                result.y2 = (result.y2 - (m_INPUT_W - m_scale_h * m_frameWidth) / 2) / m_scale_h;
                result.x1 = result.x1 / m_scale_h;
                result.x2 = result.x2 / m_scale_h;
            }

            // Clip object box coordinates to network resolution
            result.y1 = CLIP(result.y1, 0, m_frameWidth - 1);
            result.x1 = CLIP(result.x1, 0, m_frameHeight - 1);
            result.y2 = CLIP(result.y2, 0, m_frameWidth - 1);
            result.x2 = CLIP(result.x2, 0, m_frameHeight - 1);

            // Get confidence
            result.score = *(conf + 1);

            // Push to result vector
            m_outputBbox.push_back(result);
        }
        bbox += 4;
        conf += 2;
    }
    std::sort(m_outputBbox.begin(), m_outputBbox.end(), m_cmp);
    nms(m_outputBbox, m_nms_threshold);
    if (m_outputBbox.size() > m_maxFacesPerScene)
        m_outputBbox.resize(m_maxFacesPerScene);
}

void RetinaFace::create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h) {
    anchor.clear();
    std::vector<std::vector<int>> feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h / steps[i]));
        feature_map[i].push_back(ceil(w / steps[i]));
    }
    std::vector<int> minsize1 = {10, 20};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 64};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {128, 256};
    min_sizes[2] = minsize3;

    for (int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (int l = 0; l < min_size.size(); ++l) {
                    float s_kx = min_size[l] * 1.0 / w;
                    float s_ky = min_size[l] * 1.0 / h;
                    float cx = (j + 0.5) * steps[k] / w;
                    float cy = (i + 0.5) * steps[k] / h;
                    anchorBox axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }
    }
}

inline bool RetinaFace::m_cmp(Bbox a, Bbox b) {
    if (a.score > b.score)
        return true;
    return false;
}

void RetinaFace::nms(std::vector<Bbox> &input_boxes, float NMS_THRESH) {
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

RetinaFace::~RetinaFace() {
    // Release stream and buffers
    checkCudaStatus(cudaStreamDestroy(stream));
    checkCudaStatus(cudaFree(buffers[inputIndex]));
    checkCudaStatus(cudaFree(buffers[outputIndex0]));
    checkCudaStatus(cudaFree(buffers[outputIndex1]));
    // checkCudaStatus(cudaFree(buffers[outputIndex2]));
}
