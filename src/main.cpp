//#include <NvInfer.h>
//#include <NvInferPlugin.h>
//#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#include "arcface-ir50.h"
#include "json.hpp"
#include "retinaface.h"
#include "utils.h"

using json = nlohmann::json;
#define LOG_TIMES
#define NUM_REPEAT_EMBED 1

int main(int argc, const char **argv) {
    std::cout << "[INFO] Loading config..." << std::endl;
    std::ifstream configStream("../config.json");
    json config;
    configStream >> config;
    configStream.close();

    // TRT Logger
    Logger gLogger = Logger();

    // curl request
    int location = 7;
    Requests r(config["server"], location);

    int nbFrames = 0;
    int videoFrameWidth = 640;
    int videoFrameHeight = 480;
    int maxFacesPerScene = config["maxFacesPerScene"];
    float knownPersonThreshold = config["knownPersonThreshold"];
    std::string embeddingsFile = config["embeddingsFile"];
    bool isCSICam = false;

    std::string engineFile = "../weights/ir50_asia-fp16-b1.engine";
    ArcFaceIR50 recognizer =
        ArcFaceIR50(gLogger, engineFile, knownPersonThreshold, maxFacesPerScene, videoFrameWidth, videoFrameHeight);

    // init retina
    engineFile = "../weights/retina-mobile025-320x320-fp16.engine";
    RetinaFace detector(gLogger, engineFile, videoFrameWidth, videoFrameHeight, maxFacesPerScene);

    // init Bbox and allocate memory for "maxFacesPerScene" faces per scene
    std::vector<struct Bbox> outputBbox;
    outputBbox.reserve(maxFacesPerScene);

    // get embeddings of known faces
    if (fileExists(embeddingsFile)) {
        std::cout << "[INFO] Reading embeddings from file...\n";
        ifstream i(config["numImagesFile"]);
        std::string numImages_str;
        std::getline(i, numImages_str);
        unsigned int numImages = stoi(numImages_str);
        i.close();
        i.clear();
        i.open(embeddingsFile);
        json j;
        i >> j;
        i.close();
        recognizer.initKnownEmbeds(numImages);
        for (json::iterator it = j.begin(); it != j.end(); ++it)
            for (int i = 0; i < it.value().size(); ++i)
                recognizer.addEmbedding(it.key(), it.value()[i]);
        std::cout << "[INFO] Init cuBLASLt cosine similarity calculator...\n";
        recognizer.initCosSim();
    } else {
        std::cout << "[INFO] Parsing images from " << config["imgSource"] << "\n";
        std::vector<struct Paths> paths;
        getFilePaths(config["imgSource"], paths);
        unsigned int img_count = paths.size();
        ofstream o(config["numImagesFile"]);
        o << img_count << std::endl;
        o.close();
        o.clear();
        o.open(embeddingsFile);
        json j;
        cv::Mat image;
        if (config["imgIsCropped"]) {
            cv::Mat input;
            float output[512];
            std::vector<float> embeddedFace;
            for (int i = 0; i < paths.size(); i++) {
                image = cv::imread(paths[i].absPath.c_str());
                std::string className = paths[i].className;
                // std::cout << paths[i].absPath << "\n";
                recognizer.preprocessFace(image, input);
                recognizer.doInference((float *)input.ptr<float>(0), output);
                embeddedFace.insert(embeddedFace.begin(), output, output + 512);
                if (j.contains(className)) {
                    j[className].push_back(embeddedFace);
                } else {
                    std::vector<std::vector<float>> temp;
                    temp.push_back(embeddedFace);
                    j[className] = temp;
                }
                input.release();
                embeddedFace.clear();
            }
        } else {
            for (int i = 0; i < paths.size(); i++) {
                image = cv::imread(paths[i].absPath.c_str());
                cv::resize(image, image, cv::Size(videoFrameWidth, videoFrameHeight));
                outputBbox = detector.findFace(image);
                std::string rawName = paths[i].className;
                recognizer.forwardAddFace(image, outputBbox, rawName);
                recognizer.resetVariables();
            }
            // to json
            for (int i = 0; i < NUM_REPEAT_EMBED; ++i) {
                for (int k = 0; k < recognizer.m_knownFaces.size(); ++k) {
                    std::string className = recognizer.m_knownFaces[k].className + std::to_string(i);
                    std::vector<std::vector<float>> temp;
                    temp.push_back(recognizer.m_knownFaces[k].embeddedFace);
                    j[className] = temp;
                }
            }
        }
        // write result to json file
        o << std::setw(4) << j << std::endl;
        std::cout << "[INFO] Embeddings saved to json. Exitting..." << std::endl;
        exit(0);
    }

    // init opencv stuff
    std::string camera_input = config["camera_input"];
    cv::VideoCapture vc(camera_input);
    if (!vc.isOpened()) {
        // error in opening the video input
        std::cerr << "Failed to open camera.\n";
        return -1;
    }
    cv::Mat rawInput;
    cv::Mat frame;
    float *output_sims;
    std::vector<std::string> names;
    std::vector<float> sims;
    std::cout << "[INFO] Start video stream\n";

    // loop over frames with inference
    auto globalTimeStart = std::chrono::high_resolution_clock::now();
    while (true) {
        bool ret = vc.read(rawInput); // read a new frame from video
        if (!ret) {
            std::cerr << "ERROR: Cannot read frame from stream\n";
            continue;
        }
        // std::cout << "Input: " << rawInput.size() << "\n";
        if (config["crop_input"]) {
            cv::Rect cropPos(cv::Point(470, 400), cv::Point(1150, 900));
            rawInput = rawInput(cropPos);
        }
        cv::resize(rawInput, frame, cv::Size(videoFrameWidth, videoFrameHeight));

        auto startDetect = std::chrono::high_resolution_clock::now();
        outputBbox = detector.findFace(frame);
        auto endDetect = std::chrono::high_resolution_clock::now();
        auto startRecognize = std::chrono::high_resolution_clock::now();
        recognizer.forward(frame, outputBbox);
        auto endRecognize = std::chrono::high_resolution_clock::now();
        auto startFeatM = std::chrono::high_resolution_clock::now();
        //float *output_sims = recognizer.featureMatching();
        output_sims = recognizer.featureMatching();
        auto endFeatM = std::chrono::high_resolution_clock::now();
        std::tie(names, sims) = recognizer.getOutputs(output_sims);

        // curl request
        //std::string check_type = "in";
        //r.send(names, sims, recognizer.m_croppedFaces, recognizer.m_classCount, knownPersonThreshold, check_type);

        // visualize and clean
        recognizer.visualize(frame, names, sims);
        cv::imshow("VideoSource", frame);
        recognizer.resetVariables();
        outputBbox.clear();
        names.clear();
        sims.clear();
        rawInput.release();
        frame.release();
        nbFrames++;

        char keyboard = cv::waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
            break;
        else if (keyboard == 'n') {
            auto dTimeStart = std::chrono::high_resolution_clock::now();
            vc.read(frame);
            outputBbox = detector.findFace(frame);
            cv::imshow("VideoSource", frame);
            recognizer.addNewFace(frame, outputBbox);
            auto dTimeEnd = std::chrono::high_resolution_clock::now();
            globalTimeStart += (dTimeEnd - dTimeStart);
        }

#ifdef LOG_TIMES
        std::cout << "Detector took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(endDetect - startDetect).count() << "ms\n";
        std::cout << "Recognizer took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(endRecognize - startRecognize).count()
                  << "ms\n";
        std::cout << "Feature matching took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(endFeatM - startFeatM).count() << "ms\n";
        std::cout << "-------------------------" << std::endl;
#endif // LOG_TIMES
    }
    auto globalTimeEnd = std::chrono::high_resolution_clock::now();
    cv::destroyAllWindows();
    vc.release();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(globalTimeEnd - globalTimeStart).count();
    double seconds = double(milliseconds) / 1000.;
    double fps = nbFrames / seconds;

    std::cout << "Counted " << nbFrames << " frames in " << double(milliseconds) / 1000. << " seconds!"
              << " This equals " << fps << "fps.\n";

    return 0;
}
