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

WebSocketClient::WebSocketClient(std::string host, std::string port, std::string url) {
    // Look up the domain name
    tcp::resolver resolver{m_ioc};
    auto const results = resolver.resolve(host, port);

    // Make the connection on the IP address we get from a lookup
    auto ep = net::connect(m_ws.next_layer(), results);

    // Update the host_ string. This will provide the value of the
    // Host HTTP header during the WebSocket handshake.
    // See https://tools.ietf.org/html/rfc7230#section-5.4
    host += ':' + std::to_string(ep.port());

    // Perform the websocket handshake
    m_ws.handshake(host, url);
}

std::string WebSocketClient::send(std::string s) {
    // Clear read buffer
    m_buffer.clear();

    // Send the message
    m_ws.write(net::buffer(s));

    // Read a message into our buffer
    m_ws.read(m_buffer);
    return beast::buffers_to_string(m_buffer.data());
}

WebSocketClient::~WebSocketClient() {
    // Close the WebSocket connection
    m_ws.close(websocket::close_code::normal);
}

HttpClient::HttpClient(std::string host, std::string port, std::string url) {
    // Look up the domain name
    tcp::resolver resolver{m_ioc};
    m_results = resolver.resolve(host, port);

    // Set up an HTTP POST request message
    m_req.method(beast::http::verb::post);
    m_req.target(url);
    m_req.set(http::field::host, host);
    m_req.set(http::field::content_type, "application/json");
}

std::string HttpClient::send(std::string s) {
    // Make the connection on the IP address we get from a lookup
    m_stream.connect(m_results);

    // Clear read buffer
    // m_buffer.clear();

    // Clear response
    m_res = {};

    // Prepare response
    m_req.body() = s;
    m_req.prepare_payload();

    // Send the HTTP request to the remote host
    http::write(m_stream, m_req);

    // Receive the HTTP response
    http::read(m_stream, m_buffer, m_res);

    // Gracefully close the socket
    beast::error_code ec;
    m_stream.socket().shutdown(tcp::socket::shutdown_both, ec);

    // not_connected happens sometimes, so don't bother reporting it.
    if (ec && ec != beast::errc::not_connected)
        throw beast::system_error{ec};

    // If we get here then the connection is closed gracefully
    return beast::buffers_to_string(m_res.body().data());
}

HttpClient::~HttpClient() {}
