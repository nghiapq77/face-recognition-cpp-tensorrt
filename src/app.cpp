//#include <fstream>
//#include <iomanip>
//#include <iostream>
//#include <iterator>
//#include <opencv2/highgui.hpp>
//#include <string>

#include "arcface-ir50.h"
#include "crow.h"
#include "db.h"
#include "json.hpp"
#include "retinaface.h"
#include "utils.h"

using json = nlohmann::json;

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

    // params
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
    int recMaxBatchSize = config["rec_maxBatchSize"];

    // init arcface
    ArcFaceIR50 recognizer(gLogger, recEngineFile, videoFrameWidth, videoFrameHeight, recInputShape, recOutputDim,
                           recMaxBatchSize, maxFacesPerScene, knownPersonThreshold);

    // init retinaface
    RetinaFace detector(gLogger, detEngineFile, videoFrameWidth, videoFrameHeight, detInputShape, maxFacesPerScene,
                        det_threshold_nms, det_threshold_bbox);

    // init bbox and allocate memory according to maxFacesPerScene
    std::vector<struct Bbox> outputBbox;
    outputBbox.reserve(maxFacesPerScene);

    // init db
    Database db = Database(config["database_path"], recOutputDim);

    // dict that map USR_ID to USR_NM
    //std::map<std::string, std::string> userDict = db.getUserDict();

    if (config["gen"]) {
        // generate db
        std::cout << "[INFO] Parsing images from " << config["gen_imgSource"] << "\n";
        std::vector<struct Paths> paths;
        getFilePaths(config["gen_imgSource"], paths);
        cv::Mat image;
        cv::Mat input;
        float output[recOutputDim];
        // create database from folder
        std::cout << "Creating database...\n";
        for (int i = 0; i < paths.size(); i++) {
            image = cv::imread(paths[i].absPath.c_str());
            int height = image.size[0];
            int width = image.size[1];
            // Only accept 112x112 images
            //assert(height == recInputShape[2]);
            //assert(width == recInputShape[1]);
            // resize if diff size
            if ((height != recInputShape[2]) || (width != recInputShape[1])){
                std::cout << "Resizing input to " << recInputShape[1] << "x" << recInputShape[2] << "\n";
                cv::resize(image, image, cv::Size(recInputShape[1], recInputShape[2]));
            }
            std::string className = paths[i].className;
            recognizer.preprocessFace(image, input);
            recognizer.doInference((float *)input.ptr<float>(0), output);
            //std::string id = db.insertPersonIfNotExist(className);
            db.insertUser(className, className, className, "", 1, 1, 1, 1, 1);
            db.insertFace(className, paths[i].absPath, output, 1, 1, 1, 1, 1);
            //std::cout << paths[i].absPath << " " << id << "\n";
            input.release();
        }
        std::cout << "[INFO] Database generated. Exitting..." << std::endl;
        exit(0);
    } else {
        // load from database
        std::cout << "[INFO] Reading embeddings from database...\n";
        db.getEmbeddings(recognizer);
        std::cout << "[INFO] Init cuBLASLt cosine similarity calculator...\n";
        recognizer.initCosSim();
    }

    // init opencv and output vectors
    cv::Mat rawInput;
    cv::Mat frame;
    float *output_sims;
    std::vector<std::string> names;
    std::vector<float> sims;

    /*
    std::cout << "[INFO] Start video stream\n";
    std::string path = "/home/jetson/face/retina_arcface/data/1.jpg";
    std::string outpath = "/home/jetson/face/retina_arcface/data/1_output.jpg";
    //std::string foutpath = "/home/jetson/face/retina_arcface/data/1_output.txt";
    std::string foutpath = "./1_base64.txt";
        //visualize and save
        recognizer.visualize(frame, names, sims);
        //cv::imwrite(outpath, frame);
        std::vector<struct CroppedFace> croppedFaces = recognizer.croppedFaces;
        std::ofstream f;
        f.open (foutpath);
        //for (int i = 0; i < croppedFaces.size(); ++i) {
            //cv::imwrite(outpath, croppedFaces[i].face);
            //// cv::Mat to base64
            //std::vector<uchar> buf;
            //cv::imencode(".jpg", croppedFaces[i].face, buf);
            //auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
            //std::string encoded = base64_encode(enc_msg, buf.size());
            //f << encoded;
        //}

        //std::vector<uchar> buf;
        //cv::imencode(".jpg", rawInput, buf);
        //auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
        //std::string encoded = base64_encode(enc_msg, buf.size());
        //f << encoded;
        //f.close();
        //break;
        //
    */

    crow::SimpleApp app;

    CROW_ROUTE(app, "/insert/user")
        .methods("POST"_method)([&db](const crow::request &req) {
            auto x = crow::json::load(req.body);
            if (!x)
                return crow::response(crow::status::BAD_REQUEST);
            std::string userId = x["userId"].s();
            std::string userName = x["userName"].s();
            std::string userFullName = x["userFullName"].s();
            std::string ofcCd = x["ofcCd"].s();
            int active = x["active"].i();
            int creDt = x["creDt"].i();
            int creUser = x["creUser"].i();
            int updDt = x["updDt"].i();
            int updUser = x["updUser"].i();
            int ret = db.insertUser(userId, userName, userFullName, ofcCd, active, creDt, creUser, updDt, updUser);
            std::string response = "Fail! User `" + userId + "` already in database.\n";
            if (ret == 1) response = "Success! User `" + userId + "` inserted.\n";
            return crow::response(response);
    });

    CROW_ROUTE(app, "/insert/face")
        .methods("POST"_method)([&db, &recognizer, &recInputShape, &recOutputDim](const crow::request &req) {
            auto x = crow::json::load(req.body);
            if (!x)
                return crow::response(crow::status::BAD_REQUEST);
            std::string userId = x["userId"].s();
            std::string imgName = x["imgName"].s();
            int trainingFlag = x["trainingFlag"].i();
            int creDt = x["creDt"].i();
            int creUser = x["creUser"].i();
            int updDt = x["updDt"].i();
            int updUser = x["updUser"].i();

            /*
            std::string embedding_str = x["embedding"].s();
            std::string decoded = base64_decode(embedding_str);
            std::vector<uchar> data(decoded.begin(), decoded.end());
            cv::Mat image = cv::imdecode(data, cv::IMREAD_UNCHANGED);
            */
            cv::Mat image = cv::imread(imgName.c_str());
            int height = image.size[0];
            int width = image.size[1];
            // resize if diff size
            if ((height != recInputShape[2]) || (width != recInputShape[1])){
                CROW_LOG_INFO << "Resizing input to " << recInputShape[1] << "x" << recInputShape[2] << "\n";
                cv::resize(image, image, cv::Size(recInputShape[1], recInputShape[2]));
            }
            cv::Mat input;
            float output[recOutputDim];
            recognizer.preprocessFace(image, input);
            recognizer.doInference((float *)input.ptr<float>(0), output);
            //std::cout << person_id << " " << image.size() << "\n";
            int ret = db.insertFace(userId, imgName, output, trainingFlag, creDt, creUser, updDt, updUser);
            std::string response = "Fail! Embedding for `" + userId + "` cannot be inserted.\n";
            if (ret == 1) response = "Success! Embedding for `" + userId + "` inserted successfully.\n";
            return crow::response(response);
        });

    CROW_ROUTE(app, "/delete/user")
    ([&db](const crow::request &req) {
        if (req.url_params.get("id") == nullptr)
            return crow::response("Failed\n");
        else {
            std::string userId = req.url_params.get("id");
            db.deleteUser(userId);
        }

        return crow::response("Success\n");
    });

    CROW_ROUTE(app, "/delete/face")
    ([&db](const crow::request &req) {
        if (req.url_params.get("id") == nullptr)
            return crow::response("Failed\n");
        else {
            int id = boost::lexical_cast<int>(req.url_params.get("id"));
            db.deleteFace(id);
        }

        return crow::response("Success\n");
    });

    CROW_ROUTE(app, "/inference")
        .methods("POST"_method)([&db, &rawInput, &frame, &detector, &recognizer, &videoFrameWidth, &videoFrameHeight,
                                 &outputBbox, &output_sims, &names, &sims, &knownPersonThreshold](const crow::request &req) {
            auto x = crow::json::load(req.body);
            if (!x)
                return crow::response(crow::status::BAD_REQUEST);
            auto start = std::chrono::high_resolution_clock::now();
            std::string base64_image = x["image"].s();
            std::string decoded = base64_decode(base64_image);
            std::vector<uchar> data(decoded.begin(), decoded.end());
            rawInput = cv::imdecode(data, cv::IMREAD_UNCHANGED);
            auto endDecode = std::chrono::high_resolution_clock::now();
            CROW_LOG_INFO << "Image: " << rawInput.size() << "\n";
            CROW_LOG_INFO << "Resizing input to " << videoFrameWidth << "x" << videoFrameHeight << "\n";
            cv::resize(rawInput, frame, cv::Size(videoFrameWidth, videoFrameHeight));

            auto startInfer = std::chrono::high_resolution_clock::now();
            outputBbox = detector.findFace(frame);
            auto endDetect = std::chrono::high_resolution_clock::now();
            recognizer.forward(frame, outputBbox);
            auto endRecog = std::chrono::high_resolution_clock::now();
            output_sims = recognizer.featureMatching();
            std::tie(names, sims) = recognizer.getOutputs(output_sims);
            auto endInfer = std::chrono::high_resolution_clock::now();

            // get json result
            //std::vector<struct CroppedFace> croppedFaces = recognizer.croppedFaces;
            std::vector<crow::json::wvalue> data_list;
            bool isUnknown;
            for (int i = 0; i < recognizer.croppedFaces.size(); ++i) {
                // debug
                //std::string outpath = "./output.jpg";
                //cv::imwrite(outpath, croppedFaces[i].face);
                CROW_LOG_INFO << "Prediction: " << names[i] << " " << sims[i] << "\n";
                isUnknown = false;
                if (sims[i] < knownPersonThreshold) isUnknown = true;

                // cv::Mat to base64
                std::vector<uchar> buf;
                cv::imencode(".jpg", recognizer.croppedFaces[i].face, buf);
                auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
                std::string encoded = base64_encode(enc_msg, buf.size());

                // create json element
                crow::json::wvalue d = {
                    {"image", encoded},
                    {"userId", names[i]},
                    //{"userName", userDict[names[i]]},
                    {"similarity", sims[i]},
                    {"isUnknown", isUnknown},
                };
                data_list.push_back(d);
            }
            crow::json::wvalue retval({{"result", data_list}, {"count", recognizer.croppedFaces.size()}});

            // clean
            recognizer.resetVariables();
            outputBbox.clear();
            names.clear();
            sims.clear();
            rawInput.release();
            frame.release();

            //timing
            CROW_LOG_DEBUG << "Decode took "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(endDecode - start).count() << "ms\n";
            CROW_LOG_DEBUG << "Detection took "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(endDetect - startInfer).count() << "ms\n";
            CROW_LOG_DEBUG << "Recognition took "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(endRecog - endDetect).count() << "ms\n";
            CROW_LOG_DEBUG << "Matching took "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(endInfer - endRecog).count() << "ms\n";
            CROW_LOG_DEBUG << "Total inference took "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(endInfer - startInfer).count() << "ms\n";
            return crow::response(retval);
        });

    CROW_ROUTE(app, "/reload")([&db, &recognizer]() {
        CROW_LOG_INFO << "[INFO] Reset embeddings from recognizer...\n";
        recognizer.resetEmbeddings();
        CROW_LOG_INFO << "[INFO] Reading embeddings from database...\n";
        db.getEmbeddings(recognizer);
        CROW_LOG_INFO << "[INFO] Init cuBLASLt cosine similarity calculator...\n";
        recognizer.initCosSim();
        return crow::response("Success\n");
    });

    // enables all log
    //app.loglevel(crow::LogLevel::Debug);

    app.port(18080).multithreaded().run();
}
