#include "arcface.h"

void getCroppedFaces(cv::Mat frame, std::vector<struct Bbox> &outputBbox, int resize_w, int resize_h, std::vector<struct CroppedFace> &croppedFaces) {
    croppedFaces.clear();
    for (std::vector<struct Bbox>::iterator it = outputBbox.begin(); it != outputBbox.end(); it++) {
        cv::Rect facePos(cv::Point((*it).y1, (*it).x1), cv::Point((*it).y2, (*it).x2));
        cv::Mat tempCrop = frame(facePos);
        struct CroppedFace currFace;
        cv::resize(tempCrop, currFace.faceMat, cv::Size(resize_h, resize_w), 0, 0, cv::INTER_CUBIC);
        currFace.face = currFace.faceMat.clone();
        currFace.x1 = it->x1;
        currFace.y1 = it->y1;
        currFace.x2 = it->x2;
        currFace.y2 = it->y2;
        croppedFaces.push_back(currFace);
    }
}

int ArcFaceIR50::classCount = 0;

ArcFaceIR50::ArcFaceIR50(TRTLogger gLogger, const std::string engineFile, int frameWidth, int frameHeight, std::string inputName, std::string outputName,
                         std::vector<int> inputShape, int outputDim, int maxBatchSize, int maxFacesPerScene, float knownPersonThreshold) {
    m_frameWidth = static_cast<const int>(frameWidth);
    m_frameHeight = static_cast<const int>(frameHeight);
    assert(inputShape.size() == 3);
    m_INPUT_C = static_cast<const int>(inputShape[0]);
    m_INPUT_H = static_cast<const int>(inputShape[1]);
    m_INPUT_W = static_cast<const int>(inputShape[2]);
    m_OUTPUT_D = static_cast<const int>(outputDim);
    m_INPUT_SIZE = static_cast<const int>(m_INPUT_C * m_INPUT_H * m_INPUT_W * sizeof(float));
    m_OUTPUT_SIZE = static_cast<const int>(m_OUTPUT_D * sizeof(float));
    m_maxBatchSize = static_cast<const int>(maxBatchSize);
    m_embed = new float[m_OUTPUT_D];
    croppedFaces.reserve(maxFacesPerScene);
    m_embeds = new float[maxFacesPerScene * m_OUTPUT_D];
    m_knownPersonThresh = knownPersonThreshold;

    // load engine from .engine file or create new engine
    loadEngine(gLogger, engineFile);

    // create stream and pre-allocate GPU buffers memory
    preInference(inputName, outputName);
}

void ArcFaceIR50::loadEngine(TRTLogger gLogger, const std::string engineFile) {
    if (fileExists(engineFile)) {
        std::cout << "[INFO] Loading ArcFace Engine...\n";
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

void ArcFaceIR50::preInference() {
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(m_engine->getNbBindings() == 2);

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = m_engine->getBindingIndex("input");
    outputIndex = m_engine->getBindingIndex("output");

    // Create GPU buffers on device
    checkCudaStatus(cudaMalloc(&buffers[inputIndex], m_maxBatchSize * m_INPUT_SIZE));
    checkCudaStatus(cudaMalloc(&buffers[outputIndex], m_maxBatchSize * m_OUTPUT_SIZE));

    // Create stream
    checkCudaStatus(cudaStreamCreate(&stream));
}

void ArcFaceIR50::preInference(std::string inputName, std::string outputName) {
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(m_engine->getNbBindings() == 2);

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = m_engine->getBindingIndex(inputName.c_str());
    outputIndex = m_engine->getBindingIndex(outputName.c_str());

    // Create GPU buffers on device
    checkCudaStatus(cudaMalloc(&buffers[inputIndex], m_maxBatchSize * m_INPUT_SIZE));
    checkCudaStatus(cudaMalloc(&buffers[outputIndex], m_maxBatchSize * m_OUTPUT_SIZE));

    // Create stream
    checkCudaStatus(cudaStreamCreate(&stream));
}

void ArcFaceIR50::preprocessFace(cv::Mat &face, cv::Mat &output) {
    cv::cvtColor(face, face, cv::COLOR_BGR2RGB);
    face.convertTo(face, CV_32F);
    face = (face - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
    std::vector<cv::Mat> temp;
    cv::split(face, temp);
    for (int i = 0; i < temp.size(); i++) {
        output.push_back(temp[i]);
    }
}

void ArcFaceIR50::preprocessFaces() {
    for (int i = 0; i < croppedFaces.size(); i++) {
        cv::cvtColor(croppedFaces[i].faceMat, croppedFaces[i].faceMat, cv::COLOR_BGR2RGB);
        croppedFaces[i].faceMat.convertTo(croppedFaces[i].faceMat, CV_32F);
        croppedFaces[i].faceMat = (croppedFaces[i].faceMat - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
        std::vector<cv::Mat> temp;
        cv::split(croppedFaces[i].faceMat, temp);
        for (int i = 0; i < temp.size(); i++) {
            m_input.push_back(temp[i]);
        }
        croppedFaces[i].faceMat = m_input.clone();
        m_input.release();
    }
}

void ArcFaceIR50::doInference(float *input, float *output) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    checkCudaStatus(cudaMemcpyAsync(buffers[inputIndex], input, m_INPUT_SIZE, cudaMemcpyHostToDevice, stream));
    m_context->enqueueV2(buffers, stream, nullptr);
    checkCudaStatus(cudaMemcpyAsync(output, buffers[outputIndex], m_OUTPUT_SIZE, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void ArcFaceIR50::doInference(float *input, float *output, int batchSize) {
    // Set input dimensions
    m_context->setBindingDimensions(inputIndex, nvinfer1::Dims4(batchSize, m_INPUT_C, m_INPUT_H, m_INPUT_W));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    checkCudaStatus(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * m_INPUT_SIZE, cudaMemcpyHostToDevice, stream));
    m_context->enqueueV2(buffers, stream, nullptr);
    checkCudaStatus(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * m_OUTPUT_SIZE, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void ArcFaceIR50::addEmbedding(const std::string className, float embedding[]) {
    classNames.push_back(className);
    std::copy(embedding, embedding + m_OUTPUT_D, m_knownEmbeds + classCount * m_OUTPUT_D);
    classCount++;
}

void ArcFaceIR50::addEmbedding(const std::string className, std::vector<float> embedding) {
    classNames.push_back(className);
    std::copy(embedding.begin(), embedding.end(), m_knownEmbeds + classCount * m_OUTPUT_D);
    classCount++;
}

void ArcFaceIR50::initKnownEmbeds(int num) { m_knownEmbeds = new float[num * m_OUTPUT_D]; }

void ArcFaceIR50::initMatMul() { matmul.init(m_knownEmbeds, classCount, m_OUTPUT_D); }

void ArcFaceIR50::forward(cv::Mat frame, std::vector<struct Bbox> outputBbox) {
    getCroppedFaces(frame, outputBbox, m_INPUT_W, m_INPUT_H, croppedFaces);
    preprocessFaces();
    if (m_maxBatchSize < 2) {
        for (int i = 0; i < croppedFaces.size(); i++) {
            doInference((float *)croppedFaces[i].faceMat.ptr<float>(0), m_embed);
            std::copy(m_embed, m_embed + m_OUTPUT_D, m_embeds + i * m_OUTPUT_D);
        }
    } else {
        int num = croppedFaces.size();
        int end = 0;
        for (int beg = 0; beg < croppedFaces.size(); beg = beg + m_maxBatchSize) {
            end = std::min(num, beg + m_maxBatchSize);
            cv::Mat input;
            for (int i = beg; i < end; ++i) {
                input.push_back(croppedFaces[i].faceMat);
            }
            doInference((float *)input.ptr<float>(0), m_embed, end - beg);
            std::copy(m_embed, m_embed + (end - beg) * m_OUTPUT_D, m_embeds + (end - beg) * beg * m_OUTPUT_D);
        }
    }
}

float *ArcFaceIR50::featureMatching() {
    /*
        Get cosine similarity matrix of known embeddings and new embeddings.
        Since output is l2-normed already, only need to perform matrix multiplication.
    */
    m_outputs = new float[croppedFaces.size() * classCount];
    if (classNames.size() > 0 && croppedFaces.size() > 0) {
        matmul.calculate(m_embeds, croppedFaces.size(), m_outputs);
    } else {
        throw "Feature matching: No faces in database or no faces found";
    }
    return m_outputs;
}

std::tuple<std::vector<std::string>, std::vector<float>> ArcFaceIR50::getOutputs(float *output_sims) {
    /*
        Get person corresponding to maximum similarity score based on cosine similarity matrix.
    */
    std::vector<std::string> names;
    std::vector<float> sims;
    for (int i = 0; i < croppedFaces.size(); ++i) {
        int argmax = std::distance(output_sims + i * classCount, std::max_element(output_sims + i * classCount, output_sims + (i + 1) * classCount));
        float sim = *(output_sims + i * classCount + argmax);
        std::string name = classNames[argmax];
        names.push_back(name);
        sims.push_back(sim);
    }
    return std::make_tuple(names, sims);
}

void ArcFaceIR50::visualize(cv::Mat &image, std::vector<std::string> names, std::vector<float> sims) {
    for (int i = 0; i < croppedFaces.size(); ++i) {
        float fontScaler = static_cast<float>(croppedFaces[i].x2 - croppedFaces[i].x1) / static_cast<float>(m_frameWidth);
        cv::Scalar color;
        if (sims[i] >= m_knownPersonThresh)
            color = cv::Scalar(0, 255, 0);
        else
            color = cv::Scalar(0, 0, 255);
        cv::rectangle(image, cv::Point(croppedFaces[i].y1, croppedFaces[i].x1), cv::Point(croppedFaces[i].y2, croppedFaces[i].x2), color, 2, 8, 0);
        cv::putText(image, names[i] + " " + std::to_string(sims[i]), cv::Point(croppedFaces[i].y1 + 2, croppedFaces[i].x2 - 3), cv::FONT_HERSHEY_DUPLEX,
                    0.1 + 2 * fontScaler, color, 1);
    }
}

void ArcFaceIR50::resetEmbeddings() {
    classCount = 0;
    classNames.clear();
}

ArcFaceIR50::~ArcFaceIR50() {
    // Release stream and buffers
    checkCudaStatus(cudaStreamDestroy(stream));
    checkCudaStatus(cudaFree(buffers[inputIndex]));
    checkCudaStatus(cudaFree(buffers[outputIndex]));
}
