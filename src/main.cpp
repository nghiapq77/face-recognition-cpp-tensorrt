#include <opencv2/highgui.hpp>

#include "json.hpp"
#include "base64.h"
#include "utils.h"

using json = nlohmann::json;

int main(int argc, const char **argv) {
    // curl request
    std::string server = "localhost:18080/recognize";
    Requests req(server);
    req.init_get();
    json retval;

    // read frame
    cv::Mat frame;
    std::string path = "../../imgs/2.jpg";
    frame = cv::imread(path.c_str());

    // cv::Mat to base64
    std::vector<uchar> buf;
    cv::imencode(".jpg", frame, buf);
    auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
    std::string encoded = base64_encode(enc_msg, buf.size());

    // inference api
    retval = req.get(encoded);
    if (retval.size() > 0) {
        std::cout << "Prediction: " << retval["userId"] << " " << retval["similarity"] << "\n";

        // visualization
        cv::Mat frame1;
        path = "../../imgs/1.jpg";
        frame1 = cv::imread(path.c_str());
        cv::hconcat(frame1, frame, frame);
        cv::resize(frame, frame, cv::Size(448, 224));
        path = "../../imgs/vis.jpg";
        cv::Scalar color = cv::Scalar(0, 255, 0);
        float sim = retval["similarity"];
        cv::putText(frame, "Similarity: " + std::to_string(sim), cv::Point(140, 220), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        std::cout << "Visualization saved at `" << path << "`\n";
        cv::imwrite(path, frame);
    }
    return 0;
}
