#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/highgui.hpp>
#include <string>

#include "arcface-ir50.h"
#include "json.hpp"
#include "retinaface.h"
#include "videoStreamer.h"

using json = nlohmann::json;
//#define LOG_TIMES

int main(int argc, const char **argv) {
    std::cout << "[INFO] Loading config if exists" << std::endl;
    std::ifstream configStream("../config.json");
    json config;
    configStream >> config;
    configStream.close();

    // TRT Logger
    Logger gLogger = Logger();

    // DataType dtype = DataType::kHALF;
    int nbFrames = 0;
    int videoFrameWidth = 640;
    int videoFrameHeight = 480;
    // int videoFrameWidth = 1920;
    // int videoFrameHeight = 1080;
    // int maxFacesPerScene = 5;
    // float knownPersonThreshold = 0.75;
    // string embeddingsFile = "embeddings.json";
    int maxFacesPerScene = config["maxFacesPerScene"];
    float knownPersonThreshold = config["knownPersonThreshold"];
    string embeddingsFile = config["embeddingsFile"];
    bool isCSICam = false;

    string engineFile = "../weights/ir50_asia.engine";
    string modelFile = "../weights/ir50_asia.onnx";
    ArcFaceIR50 recognizer = ArcFaceIR50(gLogger, engineFile, modelFile, knownPersonThreshold, maxFacesPerScene,
                                         videoFrameWidth, videoFrameHeight);

    // init opencv stuff
    VideoStreamer videoStreamer = VideoStreamer(0, videoFrameWidth, videoFrameHeight, 60, isCSICam);
    cv::Mat frame;

    // init retina
    engineFile = "../weights/retina-mobile025-320x320.engine";
    modelFile = "../weights/retina-mobile025.onnx";
    RetinaFace detector(gLogger, engineFile, modelFile, videoFrameWidth, videoFrameHeight);

    // init Bbox and allocate memory for "maxFacesPerScene" faces per scene
    std::vector<struct Bbox> outputBbox;
    outputBbox.reserve(maxFacesPerScene);

    // get embeddings of known faces
    if (fileExists(embeddingsFile)) {
        std::cout << "[INFO] Reading embeddings from file..." << std::endl;
        std::ifstream i(embeddingsFile);
        json j;
        i >> j;
        i.close();
        for (json::iterator it = j.begin(); it != j.end(); ++it) {
            recognizer.addEmbedding(it.key(), it.value());
        }
    } else {
        std::cout << "[INFO] Reading embeddings from img folder..." << std::endl;
        json j;
        ofstream o("embeddings.json");
        std::vector<struct Paths> paths;
        cv::Mat image;
        getFilePaths("../imgs", paths);
        for (int i = 0; i < paths.size(); i++) {
            loadInputImage(paths[i].absPath, image, videoFrameWidth, videoFrameHeight);
            outputBbox = detector.findFace(image);
            std::size_t index = paths[i].fileName.find_last_of(".");
            std::string rawName = paths[i].fileName.substr(0, index);
            // std::cout << rawName << std::endl;
            recognizer.forwardAddFace(image, outputBbox, rawName);
            recognizer.resetVariables();

            // write to file
            for (int i = 0; i < recognizer.m_knownFaces.size(); ++i) {
                j[recognizer.m_knownFaces[i].className] = recognizer.m_knownFaces[i].embeddedFace;
            }
        }
        o << std::setw(4) << j << std::endl;
        o.close();
        outputBbox.clear();
    }

    // loop over frames with inference
    auto globalTimeStart = chrono::steady_clock::now();
    while (true) {
        videoStreamer.getFrame(frame);
        if (frame.empty()) {
            std::cout << "Empty frame! Exiting...\n Try restarting nvargus-daemon by "
                         "doing: sudo systemctl restart nvargus-daemon"
                      << std::endl;
            break;
        }
        std::cout << "Input: " << frame.size() << std::endl;

        auto startMTCNN = chrono::steady_clock::now();
        outputBbox = detector.findFace(frame);
        auto endMTCNN = chrono::steady_clock::now();
        auto startForward = chrono::steady_clock::now();
        recognizer.forward(frame, outputBbox);
        auto endForward = chrono::steady_clock::now();
        auto startFeatM = chrono::steady_clock::now();
        std::vector<std::vector<float>> outputs = recognizer.featureMatching();
        auto endFeatM = chrono::steady_clock::now();
        recognizer.visualize(frame, outputs);
        recognizer.resetVariables();

        cv::imshow("VideoSource", frame);
        nbFrames++;
        outputBbox.clear();
        frame.release();

        char keyboard = cv::waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
            break;
        else if (keyboard == 'n') {
            auto dTimeStart = chrono::steady_clock::now();
            videoStreamer.getFrame(frame);
            outputBbox = detector.findFace(frame);
            cv::imshow("VideoSource", frame);
            recognizer.addNewFace(frame, outputBbox);
            auto dTimeEnd = chrono::steady_clock::now();
            globalTimeStart += (dTimeEnd - dTimeStart);
        }

#ifdef LOG_TIMES
        std::cout << "Detector took " << std::chrono::duration_cast<chrono::milliseconds>(endMTCNN - startMTCNN).count()
                  << "ms\n";
        std::cout << "Forward took "
                  << std::chrono::duration_cast<chrono::milliseconds>(endForward - startForward).count() << "ms\n";
        std::cout << "Feature matching took "
                  << std::chrono::duration_cast<chrono::milliseconds>(endFeatM - startFeatM).count() << "ms\n\n";
        std::cout << "-------------------------" << std::endl;
#endif // LOG_TIMES
    }
    auto globalTimeEnd = chrono::steady_clock::now();
    cv::destroyAllWindows();
    videoStreamer.release();
    auto milliseconds = chrono::duration_cast<chrono::milliseconds>(globalTimeEnd - globalTimeStart).count();
    double seconds = double(milliseconds) / 1000.;
    double fps = nbFrames / seconds;

    std::cout << "Counted " << nbFrames << " frames in " << double(milliseconds) / 1000. << " seconds!"
              << " This equals " << fps << "fps.\n";

    return 0;
}
