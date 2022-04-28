#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "json.hpp"
#include "webclient.h"

using json = nlohmann::json;

int main(int argc, const char **argv) {
    // HTTP request client
    std::string host = "localhost";
    std::string port = "18080";
    std::string url = "/recognize";
    HttpClient http(host, port, url);

    // Variables for response
    json j;
    std::string res;

    // Read image
    cv::Mat frame;
    std::string path = "../../imgs/2.jpg";
    frame = cv::imread(path.c_str());

    // Encode
    std::vector<uchar> buf;
    cv::imencode(".jpg", frame, buf);
    std::string s = std::string(buf.begin(), buf.end());

    // Send
    res = http.send(s);

    // Buffer to json
    j = json::parse(res);
    if (j.size() > 0) {
        std::cout << "Prediction: " << j["userId"] << " " << j["similarity"] << "\n";

        // visualization
        cv::Mat frame1;
        path = "../../imgs/1.jpg";
        frame1 = cv::imread(path.c_str());
        cv::hconcat(frame1, frame, frame);
        cv::resize(frame, frame, cv::Size(448, 224));
        path = "../../imgs/vis.jpg";
        cv::Scalar color = cv::Scalar(0, 255, 0);
        float sim = j["similarity"];
        cv::putText(frame, "Similarity: " + std::to_string(sim), cv::Point(140, 220), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        std::cout << "Visualization saved at `" << path << "`\n";
        cv::imwrite(path, frame);
    } else {
        std::cout << "No prediction\n";
    }
    return 0;
}
