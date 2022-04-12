#include "json.hpp"
#include "utils.h"
#include <opencv2/highgui.hpp>

using json = nlohmann::json;

int main(int argc, const char **argv) {
    // Config
    std::cout << "[INFO] Loading config..." << std::endl;
    std::string configPath = "../config.json";
    if (argc < 2 || (strcmp(argv[1], "-c") != 0)) {
        std::cout << "\tPlease specify config file path with -c option. Use "
                     "default path: \""
                  << configPath << "\"\n";
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
    Requests req_send(config["send_server"]);
    req_send.init_send();
    Requests req_get(config["get_server"]);
    req_get.init_get();
    json retval;

    // params
    int videoFrameWidth = config["input_frameWidth"];
    int videoFrameHeight = config["input_frameHeight"];

    // init opencv and output vectors
    //std::string camera_input = config["input_camera"];
    //cv::VideoCapture vc(camera_input);
    //if (!vc.isOpened()) {
        //// error in opening the video input
        //std::cerr << "Failed to open camera.\n";
        //return -1;
    //}

    cv::Mat rawInput;
    std::vector<int> coord = config["input_cropPos"]; // x1 y1 x2 y2
    cv::Rect cropPos(cv::Point(coord[0], coord[1]),
                     cv::Point(coord[2], coord[3]));
    cv::Mat frame;

    std::cout << "[INFO] Start video stream\n";
    // auto globalTimeStart = std::chrono::high_resolution_clock::now();
    // loop over frames with inference
    while (true) {
        //bool ret = vc.read(rawInput);
        //if (!ret) {
            //std::cerr << "ERROR: Cannot read frame from stream\n";
            //continue;
        //}
        //std::cout << "Input: " << rawInput.size() << "\n";
        //if (config["input_takeCrop"])
            //rawInput = rawInput(cropPos);
        //cv::resize(rawInput, frame,
                   //cv::Size(videoFrameWidth, videoFrameHeight));
         std::string path = "/home/jetson/face/data/1.jpg";
         frame = cv::imread(path.c_str());

        // cv::Mat to base64
        std::vector<uchar> buf;
        cv::imencode(".jpg", frame, buf);
        auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
        std::string encoded = base64_encode(enc_msg, buf.size());

        // inference api
        retval = req_get.get(encoded);
        std::cout << retval.empty() << " " << retval["userId"] << "\n";
        std::cout << "====================\n";
        if (!retval.empty() == 0){
            req_send.send(retval);
        }
        break;
    }
    return 0;
}
