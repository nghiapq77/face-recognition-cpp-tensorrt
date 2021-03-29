#include "utils.h"

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
        currFace.x1 = it->x1;
        currFace.y1 = it->y1;
        currFace.x2 = it->x2;
        currFace.y2 = it->y2;
        croppedFaces.push_back(currFace);
        //}
    }
}
