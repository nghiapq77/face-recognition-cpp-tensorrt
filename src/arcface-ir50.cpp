#include "arcface-ir50.h"

int ArcFaceIR50::m_classCount = 0;

ArcFaceIR50::ArcFaceIR50(Logger gLogger, const string engineFile, const string onnxFile, float knownPersonThreshold,
                         int maxFacesPerScene, int frameWidth, int frameHeight) {
    m_INPUT_BLOB_NAME = "input";
    m_OUTPUT_BLOB_NAME = "output";
    m_frameWidth = static_cast<const int>(frameWidth);
    m_frameHeight = static_cast<const int>(frameHeight);
    m_gLogger = gLogger;
    m_engineFile = static_cast<const string>(engineFile);
    m_onnxFile = static_cast<const string>(onnxFile);
    m_croppedFaces.reserve(maxFacesPerScene);
    m_knownPersonThresh = knownPersonThreshold;

    // load engine from .engine file or create new engine
    this->createOrLoadEngine();
}

void ArcFaceIR50::createOrLoadEngine() {
    if (fileExists(m_engineFile)) {
        std::cout << "[INFO] Loading ArcFace Engine...\n";
        std::vector<char> trtModelStream_;
        size_t size{0};

        std::ifstream file(m_engineFile, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream_.resize(size);
            // std::cout << "size: " << trtModelStream_.size() << std::endl;
            file.read(trtModelStream_.data(), size);
            file.close();
        }
        IRuntime *runtime = createInferRuntime(m_gLogger);
        assert(runtime != nullptr);
        m_engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
        assert(m_engine != nullptr);
        m_context = m_engine->createExecutionContext();
        assert(m_context != nullptr);
        std::cout << std::endl;
    } else {
        // TODO: implement in C++
        throw std::logic_error("NOT IMPLEMENTED");
    }
}

void ArcFaceIR50::preprocessFace(cv::Mat &face) {
    // Release input mat
    m_input.release();

    // Preprocess
    cv::cvtColor(face, face, cv::COLOR_BGR2RGB);
    face.convertTo(face, CV_32F);
    face = (face - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
    std::vector<cv::Mat> temp;
    cv::split(face, temp);
    for (int i = 0; i < temp.size(); i++) {
        m_input.push_back(temp[i]);
    }
    ////
    // cv::Mat face0 = cv::imread("/home/jetson/b.jpg");
    // cv::cvtColor(face0, face0, cv::COLOR_BGR2RGB);
    // face0.convertTo(face0, CV_32F);
    // face0 = (face0 - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
    // cv::Mat input0;
    // std::vector<cv::Mat> temp;
    // cv::split(face0, temp);
    // for (int i = 0; i < temp.size(); i++) {
    // input0.push_back(temp[i]);
    //}
}

void ArcFaceIR50::preprocessFaces() {
    for (int i = 0; i < m_croppedFaces.size(); i++) {
        cv::cvtColor(m_croppedFaces[i].faceMat, m_croppedFaces[i].faceMat, cv::COLOR_BGR2RGB);
        m_croppedFaces[i].faceMat.convertTo(m_croppedFaces[i].faceMat, CV_32F);
        m_croppedFaces[i].faceMat = (m_croppedFaces[i].faceMat - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
        std::vector<cv::Mat> temp;
        cv::split(m_croppedFaces[i].faceMat, temp);
        m_input.release();
        for (int i = 0; i < temp.size(); i++) {
            m_input.push_back(temp[i]);
        }
        m_croppedFaces[i].faceMat = m_input.clone();
    }
}

void ArcFaceIR50::doInference(float *input, float *output) {
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(m_engine->getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than
    // IEngine::getNbBindings()
    const int inputIndex = m_engine->getBindingIndex(m_INPUT_BLOB_NAME);
    const int outputIndex = m_engine->getBindingIndex(m_OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], m_batchSize * m_INPUT_C * m_INPUT_H * m_INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], m_batchSize * m_OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and
    // DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, m_batchSize * m_INPUT_C * m_INPUT_H * m_INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    m_context->enqueue(m_batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], m_batchSize * m_OUTPUT_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void ArcFaceIR50::forwardAddFace(cv::Mat image, std::vector<struct Bbox> outputBbox, const string className) {
    get_croppedFaces(image, outputBbox, m_INPUT_W, m_INPUT_H, m_croppedFaces);
    if (!m_croppedFaces.empty()) {
        preprocessFaces();
        doInference((float *)m_croppedFaces[0].faceMat.ptr<float>(0), m_output);
        // preprocessFace(image);
        // doInference((float*)m_input.ptr<float>(0), m_output);
        struct KnownID person;
        person.className = className;
        person.classNumber = m_classCount;
        person.embeddedFace.insert(person.embeddedFace.begin(), m_output, m_output + m_OUTPUT_SIZE);
        m_knownFaces.push_back(person);
        m_classCount++;
    }
    m_croppedFaces.clear();
}

void ArcFaceIR50::addEmbedding(const string className, std::vector<float> embedding) {
    struct KnownID person;
    person.className = className;
    person.classNumber = m_classCount;
    person.embeddedFace = embedding;
    m_knownFaces.push_back(person);
    m_classCount++;
}

void ArcFaceIR50::forward(cv::Mat frame, std::vector<struct Bbox> outputBbox) {
    std::cout << "ArcFace: " << std::endl;
    std::clock_t start = std::clock();
    get_croppedFaces(frame, outputBbox, m_INPUT_W, m_INPUT_H, m_croppedFaces);
    std::clock_t end = std::clock();
    std::cout << "\tGetCropped: " << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << "ms" << std::endl;
    start = std::clock();
    preprocessFaces();
    end = std::clock();
    std::cout << "\tPreprocess: " << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << "ms" << std::endl;
    start = std::clock();
    for (int i = 0; i < m_croppedFaces.size(); i++) {
        // preprocessFace(m_croppedFaces[i].faceMat);
        // doInference((float*)m_input.ptr<float>(0), m_output);
        doInference((float *)m_croppedFaces[i].faceMat.ptr<float>(0), m_output);
        std::vector<float> e{m_output, m_output + m_OUTPUT_SIZE};
        m_embeddings.push_back(e);
        // m_embeddings.insert(m_embeddings.end(), m_output,
        // m_output+m_OUTPUT_SIZE);
    }
    end = std::clock();
    std::cout << "\tInference: " << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << "ms" << std::endl;
}

std::vector<std::vector<float>> ArcFaceIR50::featureMatching() {
    std::clock_t start = std::clock();
    std::vector<std::vector<float>> outputs;
    if (m_knownFaces.size() > 0 && m_embeddings.size() > 0) {
        outputs = batch_cosine_similarity(m_embeddings, m_knownFaces, m_OUTPUT_SIZE);
        // std::cout << m_embeddings.size() << " " << m_knownFaces.size() << " "
        // << outputs.size() << std::endl; std::cout << outputs[0][0] <<
        // std::endl; std::vector<std::vector<float>> outputs_ =
        // batch_cosine_similarity_(m_embeddings, m_knownFaces, m_OUTPUT_SIZE);
        // std::cout << outputs_[0][0] << std::endl;
    } else {
        std::cout << "No faces in database" << std::endl;
    }
    std::clock_t end = std::clock();
    std::cout << "\tFeature matching: " << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << "ms" << std::endl;
    return outputs;
}

void ArcFaceIR50::visualize(cv::Mat &image, std::vector<std::vector<float>> &outputs) {
    for (int i = 0; i < outputs.size(); ++i) {
        int argmax = std::distance(outputs[i].begin(), std::max_element(outputs[i].begin(), outputs[i].end()));
        float fontScaler =
            static_cast<float>(m_croppedFaces[i].x2 - m_croppedFaces[i].x1) / static_cast<float>(m_frameWidth);
        cv::rectangle(image, cv::Point(m_croppedFaces[i].y1, m_croppedFaces[i].x1),
                      cv::Point(m_croppedFaces[i].y2, m_croppedFaces[i].x2), cv::Scalar(0, 255, 0), 2, 8, 0);
        if (outputs[i][argmax] >= m_knownPersonThresh) {
            cv::putText(image, m_knownFaces[argmax].className + " " + std::to_string(outputs[i][argmax]),
                        cv::Point(m_croppedFaces[i].y1 + 2, m_croppedFaces[i].x2 - 3), cv::FONT_HERSHEY_DUPLEX,
                        0.1 + 2 * fontScaler, cv::Scalar(0, 255, 0, 255), 1);
        } else {
            string un = "unknown";
            cv::putText(image, un + " " + std::to_string(outputs[i][argmax]),
                        cv::Point(m_croppedFaces[i].y1 + 2, m_croppedFaces[i].x2 - 3), cv::FONT_HERSHEY_DUPLEX,
                        0.1 + 2 * fontScaler, cv::Scalar(0, 0, 255, 255), 1);
        }
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
    m_embeddings.clear();
    m_croppedFaces.clear();
}

ArcFaceIR50::~ArcFaceIR50() {}
