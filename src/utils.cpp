#include "utils.h"

void getFilePaths(std::string rootPath, std::vector<struct Paths> &paths) {
    /*
    imagesPath--|
                |--class0--|
                |          |--f0.jpg
                |          |--f1.jpg
                |
                |--class1--|
                           |--f0.jpg
                           |--f1.jpg
    ...
    */
    DIR *dir;
    struct dirent *entry;
    std::string postfix = ".jpg";
    if ((dir = opendir(rootPath.c_str())) != NULL) {
        while ((entry = readdir(dir)) != NULL) {
            std::string class_path = rootPath + "/" + entry->d_name;
            DIR *class_dir = opendir(class_path.c_str());
            struct dirent *file_entry;
            while ((file_entry = readdir(class_dir)) != NULL) {
                std::string name(file_entry->d_name);
                if (name.length() >= postfix.length() && 0 == name.compare(name.length() - postfix.length(), postfix.length(), postfix))
                    if (file_entry->d_type != DT_DIR) {
                        struct Paths tempPaths;
                        tempPaths.className = std::string(entry->d_name);
                        tempPaths.absPath = class_path + "/" + name;
                        paths.push_back(tempPaths);
                    }
            }
        }
        closedir(dir);
    }
}

bool fileExists(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("CUDA API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("CUDA API failed");
    }
}

void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS API failed with status " << status << "\n";
        throw std::logic_error("cuBLAS API failed");
    }
}

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

CosineSimilarityCalculator::CosineSimilarityCalculator() {
    checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
    checkCublasStatus(cublasLtCreate(&ltHandle));
    checkCudaStatus(cudaStreamCreate(&stream));
}

void CosineSimilarityCalculator::init(float *knownEmbeds, int numRow, int numCol) {
    m = static_cast<const int>(numRow);
    k = static_cast<const int>(numCol);
    lda = static_cast<const int>(numCol);
    ldb = static_cast<const int>(numCol);
    ldc = static_cast<const int>(numRow);

    // alloc and copy known embeddings to GPU
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dA), m * k * sizeof(float)));
    checkCudaStatus(cudaMemcpyAsync(dA, knownEmbeds, m * k * sizeof(float), cudaMemcpyHostToDevice, stream));

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults;
    // here we just need to set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, computeType, cudaDataType));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, cudaDataType, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
}

void CosineSimilarityCalculator::calculate(float *embeds, int embedCount, float *outputs) {
    n = embedCount;

    // Allocate arrays on GPU
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dB), k * n * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dC), m * n * sizeof(float)));
    checkCudaStatus(cudaMemcpyAsync(dB, embeds, k * n * sizeof(float), cudaMemcpyHostToDevice, stream));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    cublasLtMatrixLayout_t Bdesc = NULL, Cdesc = NULL;
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, cudaDataType, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, cudaDataType, m, n, ldc));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // Do the actual multiplication
    checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc, &heuristicResult.algo, workspace,
                                     workspaceSize, stream));

    // Cleanup: descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));

    // Copy the result on host memory
    checkCudaStatus(cudaMemcpyAsync(outputs, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // CUDA stream sync
    checkCudaStatus(cudaStreamSynchronize(stream));

    // Free GPU memory
    checkCudaStatus(cudaFree(dB));
    checkCudaStatus(cudaFree(dC));
}

CosineSimilarityCalculator::~CosineSimilarityCalculator() {
    if (preference)
        checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Adesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc)
        checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));

    checkCublasStatus(cublasLtDestroy(ltHandle));
    checkCudaStatus(cudaFree(dA));
    checkCudaStatus(cudaFree(workspace));
    checkCudaStatus(cudaStreamDestroy(stream));
}

Requests::Requests(std::string server) {
    m_server = server;
    m_curl = curl_easy_init();
}

size_t Requests::writeCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

void Requests::init_send() {
    m_headers = curl_slist_append(m_headers, "Content-Type: application/json");
    m_headers = curl_slist_append(m_headers, "Authorization: Bearer <token>");
    curl_easy_setopt(m_curl, CURLOPT_HTTPHEADER, m_headers);
    curl_easy_setopt(m_curl, CURLOPT_CUSTOMREQUEST, "POST");
    curl_easy_setopt(m_curl, CURLOPT_URL, m_server.c_str());
}

void Requests::send(json j) {
    std::string payload = j.dump();

    // Send
    curl_easy_setopt(m_curl, CURLOPT_POSTFIELDS, payload.c_str());
    res = curl_easy_perform(m_curl);
    std::cout << "Send response: " << res << std::endl;
}

void Requests::init_get() {
    curl_easy_setopt(m_curl, CURLOPT_URL, m_server.c_str());
    curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(m_curl, CURLOPT_WRITEDATA, &m_readBuffer);
}

json Requests::get(std::string encodedImage) {
    json j;

    // create sending data
    json d = {
        {"image", encodedImage},
    };
    std::string payload = d.dump();

    // Send
    curl_easy_setopt(m_curl, CURLOPT_POSTFIELDS, payload.c_str());
    res = curl_easy_perform(m_curl);
    if (m_readBuffer.empty()) {
        std::cout << "No response from server\n";
    } else {
        j = json::parse(m_readBuffer);
        m_readBuffer.clear();
    }
    return j;
}

Requests::~Requests() {
    // Clean up
    curl_easy_cleanup(m_curl);
    m_curl = NULL;
    curl_slist_free_all(m_headers);
    m_headers = NULL;
}
