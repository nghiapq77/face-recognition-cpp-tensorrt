#include "arcface-ir50.h"

int ArcFaceIR50::m_classCount = 0;

ArcFaceIR50::ArcFaceIR50(Logger gLogger, const string engineFile, float knownPersonThreshold, int maxFacesPerScene,
                         int frameWidth, int frameHeight) {
    m_frameWidth = static_cast<const int>(frameWidth);
    m_frameHeight = static_cast<const int>(frameHeight);
    m_knownPersonThresh = knownPersonThreshold;
    m_croppedFaces.reserve(maxFacesPerScene);
    m_embeds = new float[maxFacesPerScene * m_OUTPUT_D];

    // load engine from .engine file or create new engine
    this->createOrLoadEngine(gLogger, engineFile);
}

void ArcFaceIR50::createOrLoadEngine(Logger gLogger, const string engineFile) {
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
            // std::cout << "size: " << trtModelStream_.size() << std::endl;
            file.read(trtModelStream_.data(), size);
            file.close();
        }
        IRuntime *runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        m_engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
        assert(m_engine != nullptr);
        m_context = m_engine->createExecutionContext();
        assert(m_context != nullptr);
        std::cout << std::endl;
    } else {
        // TODO: implement in C++
        throw std::logic_error("Cant find engine file");
    }
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
    for (int i = 0; i < m_croppedFaces.size(); i++) {
        cv::cvtColor(m_croppedFaces[i].faceMat, m_croppedFaces[i].faceMat, cv::COLOR_BGR2RGB);
        m_croppedFaces[i].faceMat.convertTo(m_croppedFaces[i].faceMat, CV_32F);
        m_croppedFaces[i].faceMat = (m_croppedFaces[i].faceMat - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
        std::vector<cv::Mat> temp;
        cv::split(m_croppedFaces[i].faceMat, temp);
        for (int i = 0; i < temp.size(); i++) {
            m_input.push_back(temp[i]);
        }
        m_croppedFaces[i].faceMat = m_input.clone();
        m_input.release();
    }
}

void ArcFaceIR50::preprocessFaces_() {
    float *m_input_;
    m_input_ = new float[m_croppedFaces.size() * m_INPUT_C * m_INPUT_H * m_INPUT_W];
    for (int i = 0; i < m_croppedFaces.size(); i++) {
        cv::cvtColor(m_croppedFaces[i].faceMat, m_croppedFaces[i].faceMat, cv::COLOR_BGR2RGB);
        m_croppedFaces[i].faceMat.convertTo(m_croppedFaces[i].faceMat, CV_32F);
        m_croppedFaces[i].faceMat = (m_croppedFaces[i].faceMat - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
        std::vector<cv::Mat> temp;
        cv::split(m_croppedFaces[i].faceMat, temp);
        for (int j = 0; j < temp.size(); j++) {
            m_input.push_back(temp[j]);
            std::copy(temp[j].ptr<float>(0), temp[j].ptr<float>(0) + m_INPUT_H * m_INPUT_W,
                      m_input_ + i * m_INPUT_C * m_INPUT_H * m_INPUT_W + j * m_INPUT_H * m_INPUT_W);
        }
        m_croppedFaces[i].faceMat = m_input.clone();
        m_input.release();
    }
}

void ArcFaceIR50::doInference(float *input, float *output) {
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(m_engine->getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = m_engine->getBindingIndex(m_INPUT_BLOB_NAME);
    const int outputIndex = m_engine->getBindingIndex(m_OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], m_INPUT_SIZE));
    CHECK(cudaMalloc(&buffers[outputIndex], m_OUTPUT_SIZE));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, m_INPUT_SIZE, cudaMemcpyHostToDevice, stream));
    m_context->enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], m_OUTPUT_SIZE, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // L2-norm
    l2_norm(output);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void ArcFaceIR50::doInference(float *input, float *output, int batchSize) {
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(m_engine->getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = m_engine->getBindingIndex(m_INPUT_BLOB_NAME);
    const int outputIndex = m_engine->getBindingIndex(m_OUTPUT_BLOB_NAME);

    // Set input dimensions
    std::cout << "batchSize: " << batchSize << "\n";
    // m_context->setOptimizationProfile(batchSize - 1);
    m_context->setBindingDimensions(inputIndex, Dims4(batchSize, m_INPUT_C, m_INPUT_H, m_INPUT_W));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * m_INPUT_SIZE));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * m_OUTPUT_SIZE));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * m_INPUT_SIZE, cudaMemcpyHostToDevice, stream));
    // m_context->enqueue(batchSize, buffers, stream, nullptr);
    m_context->enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * m_OUTPUT_SIZE, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // L2-norm
    l2_norm(output);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void ArcFaceIR50::forwardAddFace(cv::Mat image, std::vector<struct Bbox> outputBbox, const string className) {
    getCroppedFaces(image, outputBbox, m_INPUT_W, m_INPUT_H, m_croppedFaces);
    if (!m_croppedFaces.empty()) {
        preprocessFaces();
        doInference((float *)m_croppedFaces[0].faceMat.ptr<float>(0), m_embed);
        struct KnownID person;
        person.className = className;
        person.classNumber = m_classCount;
        person.embeddedFace.insert(person.embeddedFace.begin(), m_embed, m_embed + m_OUTPUT_D);
        m_knownFaces.push_back(person);
        m_classCount++;
    }
    m_croppedFaces.clear();
}

void ArcFaceIR50::addEmbedding(const string className, std::vector<float> embedding) {
    struct KnownID person;
    person.className = className;
    person.classNumber = m_classCount;
    //person.embeddedFace = embedding;
    m_knownFaces.push_back(person);
    std::copy(embedding.begin(), embedding.end(), m_knownEmbeds + m_classCount * m_OUTPUT_D);
    m_classCount++;
}

void ArcFaceIR50::initKnownEmbeds(int num) { m_knownEmbeds = new float[num * m_OUTPUT_D]; }

void ArcFaceIR50::initCosSim() { cossim.init(m_knownEmbeds, m_classCount, m_OUTPUT_D); }

void ArcFaceIR50::forward(cv::Mat frame, std::vector<struct Bbox> outputBbox) {
    getCroppedFaces(frame, outputBbox, m_INPUT_W, m_INPUT_H, m_croppedFaces);
    preprocessFaces();
    for (int i = 0; i < m_croppedFaces.size(); i++) {
        doInference((float *)m_croppedFaces[i].faceMat.ptr<float>(0), m_embed);
        std::copy(m_embed, m_embed + m_OUTPUT_D, m_embeds + i * m_OUTPUT_D);
    }
}

float *ArcFaceIR50::featureMatching() {
    m_outputs = new float[m_croppedFaces.size() * m_classCount];
    //float *m_outputs_ = new float[m_croppedFaces.size() * m_classCount];
    if (m_knownFaces.size() > 0 && m_croppedFaces.size() > 0) {
        //batch_cosine_similarity(m_embeds, m_knownEmbeds, m_croppedFaces.size(), m_classCount, m_OUTPUT_D, m_outputs_);
        cossim.calculate(m_embeds, m_croppedFaces.size(), m_outputs);
        // assertion
        //for (int i = 0; i < m_croppedFaces.size(); ++i) {
            //for (int j = 0; j < m_classCount; ++j) {
                ////std::cout << *(m_outputs + i * m_classCount + j) << " " << *(m_outputs_ + j * m_croppedFaces.size() + i) << "\n";
                ////std::cout << *(m_outputs + i * m_classCount + j) << " " << *(m_outputs_ + i * m_classCount + j) << "\n";
                ////std::cout << "=================\n";
                //assert(fabs(*(m_outputs + i * m_classCount + j) - *(m_outputs_ + i * m_classCount + j)) <= 0.000001);
            //}
        //}
        ////
    } else {
        std::cout << "No faces in database or no faces found\n";
    }
    return m_outputs;
}

std::tuple<std::vector<std::string>, std::vector<float>> ArcFaceIR50::getOutputs(float *output_sims) {
    std::vector<std::string> names;
    std::vector<float> sims;
    for (int i = 0; i < m_croppedFaces.size(); ++i) {
        int argmax =
            std::distance(output_sims + i * m_classCount,
                          std::max_element(output_sims + i * m_classCount, output_sims + (i + 1) * m_classCount));
        float sim = *(output_sims + i * m_classCount + argmax);
        std::string name = m_knownFaces[argmax].className;
        names.push_back(name);
        sims.push_back(sim);
    }
    return std::make_tuple(names, sims);
}

void ArcFaceIR50::visualize(cv::Mat &image, std::vector<std::string> names, std::vector<float> sims) {
    for (int i = 0; i < m_croppedFaces.size(); ++i) {
        float fontScaler =
            static_cast<float>(m_croppedFaces[i].x2 - m_croppedFaces[i].x1) / static_cast<float>(m_frameWidth);
        cv::Scalar color;
        if (sims[i] >= m_knownPersonThresh)
            color = cv::Scalar(0, 255, 0);
        else
            color = cv::Scalar(0, 0, 255);
        cv::rectangle(image, cv::Point(m_croppedFaces[i].y1, m_croppedFaces[i].x1),
                      cv::Point(m_croppedFaces[i].y2, m_croppedFaces[i].x2), color, 2, 8, 0);
        cv::putText(image, names[i] + " " + std::to_string(sims[i]),
                    cv::Point(m_croppedFaces[i].y1 + 2, m_croppedFaces[i].x2 - 3), cv::FONT_HERSHEY_DUPLEX,
                    0.1 + 2 * fontScaler, color, 1);
    }
}

void ArcFaceIR50::addNewFace(cv::Mat &image, std::vector<struct Bbox> outputBbox) {
    std::cout << "Adding new person...\nPlease make sure there is only one "
                 "face in the current frame.\n"
              << "What's your name? ";
    string newName;
    std::cin >> newName;
    std::cout << "Hi " << newName << ", you will be added to the database.\n";
    forwardAddFace(image, outputBbox, newName);
    string filePath = "../imgs/";
    filePath.append(newName);
    filePath.append(".jpg");
    cv::imwrite(filePath, image);
}

void ArcFaceIR50::resetVariables() {
    // m_embeddings.clear();
    m_croppedFaces.clear();
}

ArcFaceIR50::~ArcFaceIR50() {}
