#ifndef UTILS_H
#define UTILS_H

#include "base64.h"
#include "cblas.h"
#include "json.hpp"
#include <cstring>
#include <curl/curl.h>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <stdlib.h>
#include <vector>

using json = nlohmann::json;

struct Bbox {
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    // bool exist;
    // float area;
    // float ppoint[10];
    // float regreCoord[4];
};

struct CroppedFace {
    cv::Mat face;
    cv::Mat faceMat;
    int x1, y1, x2, y2;
};

struct KnownID {
    std::string className;
    int classNumber;
    std::vector<float> embeddedFace;
};

struct Paths {
    std::string absPath;
    // std::string fileName;
    std::string className;
};

void getFilePaths(std::string rootPath, std::vector<struct Paths> &paths);
bool fileExists(const std::string &name);
void l2_norm(float *p, int size = 512);
float cosine_similarity(std::vector<float> &A, std::vector<float> &B);
std::vector<std::vector<float>> batch_cosine_similarity(std::vector<std::vector<float>> &A,
                                                        std::vector<struct KnownID> &B, const int size,
                                                        bool normalize = false);
void batch_cosine_similarity(std::vector<std::vector<float>> A, std::vector<struct KnownID> B, int size,
                             float *outputs);
void batch_cosine_similarity(float *A, float *B, int embedCount, int classCount, int size, float *outputs);
void get_croppedFaces(cv::Mat frame, std::vector<struct Bbox> &outputBbox, int resize_w, int resize_h,
                      std::vector<struct CroppedFace> &croppedFaces);

class Requests {
  public:
    Requests(std::string server, short int location);
    ~Requests();
    void send(std::vector<std::string> names, std::vector<float> sims, std::vector<struct CroppedFace> &croppedFaces,
              int classCount, float threshold, std::string check_type);

    CURLcode res;

  private:
    CURL *m_curl;
    struct curl_slist *m_headers = NULL; // init to NULL is important
    std::string m_location;
};

#endif // UTILS_H
