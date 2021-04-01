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
                if (name.length() >= postfix.length() &&
                    0 == name.compare(name.length() - postfix.length(), postfix.length(), postfix))
                    if (file_entry->d_type != DT_DIR) {
                        struct Paths tempPaths;
                        // tempPaths.fileName = std::string(file_entry->d_name);
                        tempPaths.className = std::string(entry->d_name);
                        tempPaths.absPath = class_path + "/" + name;
                        paths.push_back(tempPaths);
                    }
            }
            // closedir(class_dir);
        }
        closedir(dir);
    }
}

bool fileExists(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void l2_norm(float *p, int size) {
    float norm = cblas_snrm2((blasint)size, p, 1);
    cblas_sscal((blasint)size, 1 / norm, p, 1);
}

void batch_cosine_similarity(float *A, float *B, int embedCount, int classCount, int size, float *outputs) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (blasint)embedCount, (blasint)classCount, size, 1, A, size, B,
                size, 0, outputs, classCount);
}

std::vector<std::vector<float>> batch_cosine_similarity(std::vector<std::vector<float>> &A,
                                                        std::vector<struct KnownID> &B, const int size,
                                                        bool normalize) {
    std::vector<std::vector<float>> outputs;
    if (normalize) {
        // Calculate cosine similarity
        for (int A_index = 0; A_index < A.size(); ++A_index) {
            std::vector<float> output;
            for (int B_index = 0; B_index < B.size(); ++B_index) {
                float *p_A = &A[A_index][0];
                float *p_B = &B[B_index].embeddedFace[0];
                float sim = cblas_sdot((blasint)size, p_A, 1, p_B, 1);
                output.push_back(sim);
            }
            outputs.push_back(output);
        }
    } else {
        // Pre-calculate norm for all elements
        std::vector<float> A_norms, B_norms;
        for (int i = 0; i < A.size(); ++i) {
            float *p = &A[i][0];
            float norm = cblas_snrm2((blasint)size, p, 1);
            A_norms.push_back(norm);
        }
        for (int i = 0; i < B.size(); ++i) {
            float *p = &B[i].embeddedFace[0];
            float norm = cblas_snrm2((blasint)size, p, 1);
            B_norms.push_back(norm);
        }
        // Calculate cosine similarity
        for (int A_index = 0; A_index < A.size(); ++A_index) {
            std::vector<float> output;
            for (int B_index = 0; B_index < B.size(); ++B_index) {
                float *p_A = &A[A_index][0];
                float *p_B = &B[B_index].embeddedFace[0];
                float sim = cblas_sdot((blasint)size, p_A, 1, p_B, 1) / (A_norms[A_index] * B_norms[B_index]);
                output.push_back(sim);
            }
            outputs.push_back(output);
        }
    }
    return outputs;
}

float cosine_similarity(std::vector<float> &A, std::vector<float> &B) {
    if (A.size() != B.size()) {
        std::cout << A.size() << " " << B.size() << std::endl;
        throw std::logic_error("Vector A and Vector B are not the same size");
    }

    // Prevent Division by zero
    if (A.size() < 1) {
        throw std::logic_error("Vector A and Vector B are empty");
    }

    float *p_A = &A[0];
    float *p_B = &B[0];
    float mul = cblas_sdot((blasint)(A.size()), p_A, 1, p_B, 1);
    float d_a = cblas_sdot((blasint)(A.size()), p_A, 1, p_A, 1);
    float d_b = cblas_sdot((blasint)(A.size()), p_B, 1, p_B, 1);

    if (d_a == 0.0f || d_b == 0.0f) {
        throw std::logic_error("cosine similarity is not defined whenever one or both "
                               "input vectors are zero-vectors.");
    }

    return mul / (sqrt(d_a) * sqrt(d_b));
}

void get_croppedFaces(cv::Mat frame, std::vector<struct Bbox> &outputBbox, int resize_w, int resize_h,
                      std::vector<struct CroppedFace> &croppedFaces) {
    for (std::vector<struct Bbox>::iterator it = outputBbox.begin(); it != outputBbox.end(); it++) {
        // if ((*it).exist) {
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
        //}
    }
}

Requests::Requests(std::string server, short int location) {
    m_headers = curl_slist_append(m_headers, "Content-Type: application/json");
    m_curl = curl_easy_init();
    curl_easy_setopt(m_curl, CURLOPT_URL, server.c_str());
    curl_easy_setopt(m_curl, CURLOPT_HTTPHEADER, m_headers);
    curl_easy_setopt(m_curl, CURLOPT_CUSTOMREQUEST, "POST");

    m_location = std::to_string(location);
}

void Requests::send(std::vector<std::string> names, std::vector<float> sims,
                    std::vector<struct CroppedFace> &croppedFaces, int classCount, float threshold, std::string check_type) {
    std::vector<json> data;
    for (int i = 0; i < croppedFaces.size(); ++i) {
        std::cout << names[i] << " " << sims[i] << "\n";
        if (sims[i] < threshold)
            continue;

        // cv::Mat to base64
        std::vector<uchar> buf;
        cv::imencode(".jpg", croppedFaces[i].face, buf);
        auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
        std::string encoded = base64_encode(enc_msg, buf.size());

        // create json element
        json d = {
            {"image", encoded},
            {"userId", names[i]},
            {"conf", sims[i]},
            {"type", check_type},
        };
        data.push_back(d);
    }
    if (data.size() < 1)
        return;

    // payload prepare
    json info_detection = {
        {"location", m_location},
        {"array", data},
    };
    std::string payload = info_detection.dump();
    // std::cout << payload.c_str() << "\n";

    // Send
    curl_easy_setopt(m_curl, CURLOPT_POSTFIELDS, payload.c_str());
    res = curl_easy_perform(m_curl);
}

Requests::~Requests() {
    // Clean up
    curl_easy_cleanup(m_curl);
    m_curl = NULL;
    curl_slist_free_all(m_headers);
    m_headers = NULL;
}
