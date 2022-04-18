#include "arcface.h"
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
    std::string databasePath = config["database_path"];
    int videoFrameWidth = config["input_frameWidth"];
    int videoFrameHeight = config["input_frameHeight"];
    std::string detEngineFile = config["det_engine"];
    std::vector<int> detInputShape = config["det_inputShape"];
    std::string detInputName = config["det_inputName"];
    std::vector<std::string> detOutputNames = config["det_outputNames"];
    int detMaxBatchSize = config["det_maxBatchSize"];
    float det_threshold_nms = config["det_threshold_nms"];
    float det_threshold_bbox = config["det_threshold_bbox"];
    std::vector<int> recInputShape = config["rec_inputShape"];
    int recOutputDim = config["rec_outputDim"];
    std::string recEngineFile = config["rec_engine"];
    int maxFacesPerScene = config["det_maxFacesPerScene"];
    float knownPersonThreshold = config["rec_knownPersonThreshold"];
    int recMaxBatchSize = config["rec_maxBatchSize"];
    std::string recInputName = config["rec_inputName"];
    std::string recOutputName = config["rec_outputName"];
    bool apiImageIsCropped = config["api_imgIsCropped"];

    // init arcface
    ArcFaceIR50 recognizer(gLogger, recEngineFile, videoFrameWidth, videoFrameHeight, recInputName, recOutputName, recInputShape, recOutputDim, recMaxBatchSize,
                           maxFacesPerScene, knownPersonThreshold);

    // init retinaface
    RetinaFace detector(gLogger, detEngineFile, videoFrameWidth, videoFrameHeight, detInputName, detOutputNames, detInputShape, detMaxBatchSize,
                        maxFacesPerScene, det_threshold_nms, det_threshold_bbox);

    // init bbox and allocate memory according to maxFacesPerScene
    std::vector<struct Bbox> outputBbox;
    outputBbox.reserve(maxFacesPerScene);

    // init db
    Database db = Database(databasePath, recOutputDim);

    // dict that map USR_ID to USR_NM
    std::map<std::string, std::string> userDict = db.getUserDict();

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
            // resize if diff size
            if ((height != recInputShape[2]) || (width != recInputShape[1])) {
                std::cout << "Resizing input to " << recInputShape[1] << "x" << recInputShape[2] << "\n";
                cv::resize(image, image, cv::Size(recInputShape[1], recInputShape[2]));
            }
            std::string className = paths[i].className;
            recognizer.preprocessFace(image, input);
            if (recMaxBatchSize < 2)
                recognizer.doInference((float *)input.ptr<float>(0), output);
            else
                recognizer.doInference((float *)input.ptr<float>(0), output, 1);
            db.insertUser(className, className);
            db.insertFace(className, paths[i].absPath, output);
            // std::cout << paths[i].absPath << " " << id << "\n";
            input.release();
        }
        std::cout << "[INFO] Database generated. Exitting..." << std::endl;
        exit(0);
    } else {
        // load from database
        CROW_LOG_INFO << "Reading embeddings from database...";
        db.getEmbeddings(recognizer);
        CROW_LOG_INFO << "Init cuBLASLt cosine similarity calculator...";
        recognizer.initCosSim();
    }

    // init opencv and output vectors
    cv::Mat rawInput;
    cv::Mat frame;
    float *output_sims;
    std::vector<std::string> names;
    std::vector<float> sims;

    // crow app
    crow::SimpleApp app;

    CROW_ROUTE(app, "/insert/user").methods("POST"_method)([&db](const crow::request &req) {
        auto x = crow::json::load(req.body);
        if (!x)
            return crow::response(crow::status::BAD_REQUEST);
        std::string userId = x["userId"].s();
        std::string userName = x["userName"].s();
        int ret = db.insertUser(userId, userName);
        std::string response = "Fail! User `" + userId + "` already in database.\n";
        if (ret == 1)
            response = "Success! User `" + userId + "` inserted.\n";
        return crow::response(response);
    });

    CROW_ROUTE(app, "/insert/face")
        .methods("POST"_method)([&db, &recognizer, &recInputShape, &recOutputDim, &recMaxBatchSize, &apiImageIsCropped, &rawInput, &frame, &videoFrameWidth,
                                 &videoFrameHeight, &detector, &outputBbox](const crow::request &req) {
            json j;
            std::string response = "";
            std::string info;
            int ret = 0;
            try {
                j = json::parse(req.body);
            } catch (json::parse_error &e) {
                // output exception information
                std::cout << "message: " << e.what() << '\n' << "exception id: " << e.id << '\n' << "byte position of error: " << e.byte << std::endl;
                return crow::response(crow::status::BAD_REQUEST);
            }

            if (j.contains("data")) {
                for (auto &el : j["data"].items()) {
                    std::string userId = el.value()["userId"];
                    std::string imgPath = el.value()["imgPath"];

                    cv::Mat image = cv::imread(imgPath.c_str());
                    cv::Mat input;
                    float output[recOutputDim];
                    if (apiImageIsCropped) {
                        int height = image.size[0];
                        int width = image.size[1];
                        // resize if diff size
                        if ((height != recInputShape[1]) || (width != recInputShape[2])) {
                            CROW_LOG_INFO << "Resizing input to " << recInputShape[1] << "x" << recInputShape[2] << "\n";
                            cv::resize(image, image, cv::Size(recInputShape[1], recInputShape[2]));
                        }
                        CROW_LOG_INFO << "Getting embedding...\n";
                        recognizer.preprocessFace(image, input);
                        if (recMaxBatchSize < 2)
                            recognizer.doInference((float *)input.ptr<float>(0), output);
                        else
                            recognizer.doInference((float *)input.ptr<float>(0), output, 1);
                        ret = 1;
                    } else {
                        CROW_LOG_INFO << "Image: " << image.size() << "\n";
                        CROW_LOG_INFO << "Resizing input to " << videoFrameWidth << "x" << videoFrameHeight << "\n";
                        cv::resize(image, frame, cv::Size(videoFrameWidth, videoFrameHeight));
                        CROW_LOG_INFO << "Finding faces in image...\n";
                        outputBbox = detector.findFace(frame);
                        std::vector<struct CroppedFace> croppedFaces;
                        getCroppedFaces(frame, outputBbox, recInputShape[2], recInputShape[1], croppedFaces);
                        CROW_LOG_INFO << "There are " << croppedFaces.size() << " face(s) in image.";
                        if (croppedFaces.size() > 1) {
                            response += "There are more than 1 faces in input image from `" + imgPath + "`\n";
                            ret = 2;
                        } else if (croppedFaces.size() == 0) {
                            response += "Cant find any faces in input image from `" + imgPath + "`\n";
                            ret = 3;
                        } else {
                            CROW_LOG_INFO << "Getting embedding...\n";
                            response += "1 face found in input image from `" + imgPath + "`, processing...\n";
                            recognizer.preprocessFace(croppedFaces[0].faceMat, input);
                            if (recMaxBatchSize < 2)
                                recognizer.doInference((float *)input.ptr<float>(0), output);
                            else
                                recognizer.doInference((float *)input.ptr<float>(0), output, 1);
                            ret = 1;
                        }
                        // clean
                        outputBbox.clear();
                        rawInput.release();
                        frame.release();
                    }

                    if (ret != 1) {
                        response += "Fail! Embedding for `" + userId + "` cannot be inserted.\n";
                    } else {
                        ret = db.insertFace(userId, imgPath, output);
                        if (ret == 1) {
                            response += "Success! Embedding for `" + userId + "` inserted successfully.\n";
                        } else {
                            response += "Fail! Embedding for `" + userId + "` cannot be inserted.\n";
                        }
                    }
                }
            } else {
                response = "Cant find field `data` in input!\n";
            }
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

    CROW_ROUTE(app, "/recognize")
        .methods("POST"_method)(
            [&frame, &recInputShape, &recognizer, &recOutputDim, &recMaxBatchSize, &outputBbox, &output_sims, &names, &sims](const crow::request &req) {
                auto x = crow::json::load(req.body);
                crow::json::wvalue retval;
                if (!x)
                    return crow::response(crow::status::BAD_REQUEST);
                std::string base64_image = x["image"].s();
                std::string decoded = base64_decode(base64_image);
                std::vector<uchar> data(decoded.begin(), decoded.end());
                frame = cv::imdecode(data, cv::IMREAD_UNCHANGED);
                int height = frame.size[0];
                int width = frame.size[1];
                CROW_LOG_INFO << "Image: " << frame.size();
                // resize if diff size
                if ((height != recInputShape[1]) || (width != recInputShape[2])) {
                    CROW_LOG_INFO << "Resizing input to " << recInputShape[1] << "x" << recInputShape[2] << "\n";
                    cv::resize(frame, frame, cv::Size(recInputShape[1], recInputShape[2]));
                }
                CROW_LOG_INFO << "Getting embedding...";
                Bbox bbox;
                bbox.x1 = 0;
                bbox.y1 = 0;
                bbox.x2 = recInputShape[1];
                bbox.y2 = recInputShape[2];
                bbox.score = 1;
                outputBbox.push_back(bbox);
                recognizer.forward(frame, outputBbox);
                CROW_LOG_INFO << "Feature matching";
                try {
                    output_sims = recognizer.featureMatching();
                } catch (std::logic_error &e) {
                    // output exception information
                    std::cout << "message: " << e.what() << std::endl;
                    return crow::response(retval);
                }
                std::tie(names, sims) = recognizer.getOutputs(output_sims);
                retval = {
                    {"userId", names[0]},
                    {"similarity", sims[0]},
                };
                CROW_LOG_INFO << "Prediction: " << names[0] << " " << sims[0];

                // clean
                outputBbox.clear();
                names.clear();
                sims.clear();
                frame.release();

                return crow::response(retval);
            });

    CROW_ROUTE(app, "/inference")
        .methods("POST"_method)([&db, &rawInput, &frame, &detector, &recognizer, &videoFrameWidth, &videoFrameHeight, &outputBbox, &output_sims, &names, &sims,
                                 &knownPersonThreshold, &userDict](const crow::request &req) {
            auto x = crow::json::load(req.body);
            crow::json::wvalue retval;
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

            CROW_LOG_INFO << "Inferencing...\n";
            auto startInfer = std::chrono::high_resolution_clock::now();
            outputBbox = detector.findFace(frame);
            if (outputBbox.size() < 1) {
                CROW_LOG_INFO << "No faces found, returning...\n";
                return crow::response(retval);
            }
            auto endDetect = std::chrono::high_resolution_clock::now();
            recognizer.forward(frame, outputBbox);
            auto endRecog = std::chrono::high_resolution_clock::now();
            try {
                output_sims = recognizer.featureMatching();
            } catch (std::logic_error &e) {
                // output exception information
                std::cout << "message: " << e.what() << std::endl;
                return crow::response(retval);
            }
            std::tie(names, sims) = recognizer.getOutputs(output_sims);
            auto endInfer = std::chrono::high_resolution_clock::now();

            int maxSimIdx = 0;
            float maxSim = -1;
            bool isUnknown;
            for (int i = 0; i < recognizer.croppedFaces.size(); ++i) {
                if (sims[i] > maxSim) {
                    maxSim = sims[i];
                    maxSimIdx = i;
                }
            }
            CROW_LOG_INFO << "Prediction: " << names[maxSimIdx] << " " << sims[maxSimIdx] << "\n";
            isUnknown = false;
            if (sims[maxSimIdx] < knownPersonThreshold)
                isUnknown = true;

            // cv::Mat to base64
            std::vector<uchar> buf;
            cv::imencode(".jpg", recognizer.croppedFaces[maxSimIdx].face, buf);
            auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
            std::string encoded = base64_encode(enc_msg, buf.size());

            // create json element
            retval = {
                {"image", encoded},       {"userId", names[maxSimIdx]}, {"userName", userDict[names[maxSimIdx]]}, {"similarity", sims[maxSimIdx]},
                {"isUnknown", isUnknown},
            };

            // clean
            outputBbox.clear();
            names.clear();
            sims.clear();
            rawInput.release();
            frame.release();

            // timing
            CROW_LOG_DEBUG << "Decode took " << std::chrono::duration_cast<std::chrono::milliseconds>(endDecode - start).count() << "ms\n";
            CROW_LOG_DEBUG << "Detection took " << std::chrono::duration_cast<std::chrono::milliseconds>(endDetect - startInfer).count() << "ms\n";
            CROW_LOG_DEBUG << "Recognition took " << std::chrono::duration_cast<std::chrono::milliseconds>(endRecog - endDetect).count() << "ms\n";
            CROW_LOG_DEBUG << "Matching took " << std::chrono::duration_cast<std::chrono::milliseconds>(endInfer - endRecog).count() << "ms\n";
            CROW_LOG_DEBUG << "Total inference took " << std::chrono::duration_cast<std::chrono::milliseconds>(endInfer - startInfer).count() << "ms\n";
            return crow::response(retval);
        });

    CROW_ROUTE(app, "/reload")
    ([&db, &recognizer, &userDict]() {
        CROW_LOG_INFO << "[INFO] Reset embeddings from recognizer...\n";
        recognizer.resetEmbeddings();
        CROW_LOG_INFO << "[INFO] Reading embeddings from database...\n";
        db.getEmbeddings(recognizer);
        CROW_LOG_INFO << "[INFO] Init cuBLASLt cosine similarity calculator...\n";
        recognizer.initCosSim();
        CROW_LOG_INFO << "[INFO] Create user dictionary...\n";
        userDict = db.getUserDict();
        return crow::response("Success\n");
    });

    // enables all log
    // app.loglevel(crow::LogLevel::Debug);

    app.port(18080).multithreaded().run();
}