# API usage documentation
- localhost:18080/insert/user  
    Input: json {"userId": string, "userName": string}  
    Example: `curl localhost:18080/insert/user -d '{"userId": "morty", "userName": "Morty Smith"}'`  
- localhost:18080/insert/face  
    Input: json {"data": [{"userId": string, "imgName": string, "trainingFlag": int, "creDt": int, "creUser": int, "updDt": int, "updUser": int}]}  
    Example: `curl localhost:18080/insert/face -d '{"data": [{"userId": "morty", "imgPath": "/home/nghiapham/face/face-recognition-cpp-tensorrt/imgs/1.jpg"}]}'`  
- localhost:18080/delete/user  
    Input: use param input  
    Example: `curl localhost:18080/delete/user?id='ABC'`  
- localhost:18080/delete/face  
    Input: use param input  
    Example: `curl localhost:18080/delete/face?id=100`  
- localhost:18080/recognize  
    Input: encoded cv::Mat  
    Example: See src/main.cpp
- localhost:18080/inference  
    Input: encoded cv::Mat 
- localhost:18080/reload  
    Use to reload database when adding/deleting  
