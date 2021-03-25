#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <stdlib.h>
#include <vector>

#include "cblas.h"

struct pBox {
    float *pdata;
    int width;
    int height;
    int channel;
};

struct Bbox {
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    float ppoint[10];
    float regreCoord[4];
};

//struct orderScore {
    //float score;
    //int oriOrder;
//};

struct CroppedFace {
    cv::Mat faceMat;
    int x1, y1, x2, y2;
};

struct KnownID {
    std::string className;
    int classNumber;
    std::vector<float> embeddedFace;
};

float cosine_similarity(std::vector<float> &A, std::vector<float> &B);
std::vector<std::vector<float>> batch_cosine_similarity(std::vector<std::vector<float>> &A,
                                                        std::vector<struct KnownID> &B, const int size);
std::vector<std::vector<float>> batch_cosine_similarity_(std::vector<std::vector<float>> A,
                                                         std::vector<struct KnownID> B, int size);
void get_croppedFaces(cv::Mat frame, std::vector<struct Bbox> &outputBbox, int resize_w, int resize_h,
                      std::vector<struct CroppedFace> &croppedFaces);

#endif // UTILS_H
