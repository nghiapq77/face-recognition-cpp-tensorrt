#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <opencv2/highgui.hpp>
#include <string>

#include "arcface-ir50.h"
#include "json.hpp"
#include "retinaface.h"
#include "utils.h"

using json = nlohmann::json;
#define LOG_TIMES

int main(int argc, const char **argv) {
    // Config
    std::cout << "[INFO] Loading config..." << std::endl;
    std::string configPath = "../config.json";
    if (argc < 2 || (strcmp(argv[1], "-c") != 0)) {
        std::cout << "\tPlease specify config file path with -c option. Use default path: \"" << configPath << "\"\n";
    } else {
        configPath = argv[2];
        std::cout << "\tConfig path: \"" << configPath << "\"\n";
    }
    std::ifstream configStream(configPath);
    json config;
    configStream >> config;
    configStream.close();

    // TRT Logger
    Logger gLogger = Logger();

    // curl request
    Requests r(config["send_server"], config["send_location"]);

    // params
    int numFrames = 0;
    std::string detEngineFile = config["det_engine"];
    std::vector<int> detInputShape = config["det_inputShape"];
    float det_threshold_nms = config["det_threshold_nms"];
    float det_threshold_bbox = config["det_threshold_bbox"];
    std::vector<int> recInputShape = config["rec_inputShape"];
    int recOutputDim = config["rec_outputDim"];
    std::string recEngineFile = config["rec_engine"];
    int videoFrameWidth = config["input_frameWidth"];
    int videoFrameHeight = config["input_frameHeight"];
    int maxFacesPerScene = config["det_maxFacesPerScene"];
    float knownPersonThreshold = config["rec_knownPersonThreshold"];
    std::string embeddingsFile = config["input_embeddingsFile"];

    // init arcface
    ArcFaceIR50 recognizer(gLogger, recEngineFile, videoFrameWidth, videoFrameHeight, recInputShape, recOutputDim,
                           maxFacesPerScene, knownPersonThreshold);

    // init retinaface
    RetinaFace detector(gLogger, detEngineFile, videoFrameWidth, videoFrameHeight, detInputShape, maxFacesPerScene,
                        det_threshold_nms, det_threshold_bbox);

    // init bbox and allocate memory according to maxFacesPerScene
    std::vector<struct Bbox> outputBbox;
    outputBbox.reserve(maxFacesPerScene);

    // create or get embeddings of known faces
    if (fileExists(embeddingsFile)) {
        std::cout << "[INFO] Reading embeddings from file...\n";
        std::ifstream i(config["input_numImagesFile"]);
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
        std::cout << "[INFO] Parsing images from " << config["gen_imgSource"] << "\n";
        std::vector<struct Paths> paths;
        getFilePaths(config["gen_imgSource"], paths);
        unsigned int img_count = paths.size();
        std::ofstream o(config["input_numImagesFile"]);
        o << img_count << std::endl;
        o.close();
        o.clear();
        o.open(embeddingsFile);
        json j;
        cv::Mat image;
        if (config["gen_imgIsCropped"]) {
            cv::Mat input;
            float output[recOutputDim];
            std::vector<float> embeddedFace;
            for (int i = 0; i < paths.size(); i++) {
                image = cv::imread(paths[i].absPath.c_str());
                std::string className = paths[i].className;
                recognizer.preprocessFace(image, input);
                recognizer.doInference((float *)input.ptr<float>(0), output);
                embeddedFace.insert(embeddedFace.begin(), output, output + recOutputDim);
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
            for (int k = 0; k < recognizer.knownFaces.size(); ++k) {
                std::string className = recognizer.knownFaces[k].className;
                std::vector<std::vector<float>> temp;
                temp.push_back(recognizer.knownFaces[k].embeddedFace);
                j[className] = temp;
            }
        }
        // write result to json file
        o << std::setw(4) << j << std::endl;
        std::cout << "[INFO] Embeddings saved to json. Exitting..." << std::endl;
        exit(0);
    }

    // init opencv and output vectors
    std::string camera_input = config["input_camera"];
    cv::VideoCapture vc(camera_input);
    if (!vc.isOpened()) {
        // error in opening the video input
        std::cerr << "Failed to open camera.\n";
        return -1;
    }
    cv::Mat rawInput;
    std::vector<int> coord = config["input_cropPos"]; // x1 y1 x2 y2
    cv::Rect cropPos(cv::Point(coord[0], coord[1]), cv::Point(coord[2], coord[3]));
    cv::Mat frame;
    float *output_sims;
    std::vector<std::string> names;
    std::vector<float> sims;

    std::cout << "[INFO] Start video stream\n";
    auto globalTimeStart = std::chrono::high_resolution_clock::now();
    // loop over frames with inference
    while (true) {
        bool ret = vc.read(rawInput);
        if (!ret) {
            std::cerr << "ERROR: Cannot read frame from stream\n";
            continue;
        }
        //std::cout << "Input: " << rawInput.size() << "\n";
        if (config["input_takeCrop"])
            rawInput = rawInput(cropPos);
        cv::resize(rawInput, frame, cv::Size(videoFrameWidth, videoFrameHeight));

        auto startDetect = std::chrono::high_resolution_clock::now();
        outputBbox = detector.findFace(frame);
        auto endDetect = std::chrono::high_resolution_clock::now();
        auto startRecognize = std::chrono::high_resolution_clock::now();
        recognizer.forward(frame, outputBbox);
        auto endRecognize = std::chrono::high_resolution_clock::now();
        auto startFeatM = std::chrono::high_resolution_clock::now();
        output_sims = recognizer.featureMatching();
        auto endFeatM = std::chrono::high_resolution_clock::now();
        std::tie(names, sims) = recognizer.getOutputs(output_sims);

        // curl request
        if (config["send_request"]) {
            std::string check_type = "in";
            r.send(names, sims, recognizer.croppedFaces, recognizer.classCount, knownPersonThreshold, check_type);
        }

        // visualize
        if (config["out_visualize"]) {
            recognizer.visualize(frame, names, sims);
            cv::imshow("frame", frame);

            char keyboard = cv::waitKey(1);
            if (keyboard == 'q' || keyboard == 27)
                break;
            //else if (keyboard == 'n') {
                //auto dTimeStart = std::chrono::high_resolution_clock::now();
                //recognizer.addNewFace(frame, outputBbox);
                //auto dTimeEnd = std::chrono::high_resolution_clock::now();
                //globalTimeStart += (dTimeEnd - dTimeStart);
            //}
        }

        // clean
        recognizer.resetVariables();
        outputBbox.clear();
        names.clear();
        sims.clear();
        rawInput.release();
        frame.release();
        numFrames++;

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
    double fps = numFrames / seconds;

    std::cout << "Counted " << numFrames << " frames in " << double(milliseconds) / 1000. << " seconds!"
              << " This equals " << fps << "fps.\n";

    return 0;
}
