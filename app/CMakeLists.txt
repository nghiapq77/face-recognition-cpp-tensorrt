cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(PROJECT_NAME app)
project(${PROJECT_NAME})
set (CMAKE_CXX_STANDARD 11)

# OpenCV
message(STATUS "Finding OpenCV...")
find_package(OpenCV REQUIRED)
list(APPEND CMAKE_MESSAGE_INDENT "    ")
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "Using OpenCV include dir(s): ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "Using OpenCV lib(s): ${OpenCV_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not find OpenCV")
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)

# CUDA
message(STATUS "Finding CUDA...")
find_package(CUDA REQUIRED)
list(APPEND CMAKE_MESSAGE_INDENT "    ")
if(CUDA_FOUND)
    message(STATUS "CUDA version: ${CUDA_VERSION}")
    message(STATUS "CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_75")
    set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}; 
        -gencode arch=compute_75,code=sm_75
    )

    # cublasLt
    message(STATUS "Setting cublasLt library variable...")
    set(CUBLASLT_LIBRARY cublasLt)
else()
    message(FATAL_ERROR "Could not find CUDA")
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)

# TensorRT
message(STATUS "Finding TensorRT...")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/include
    PATH_SUFFIXES include)
find_path(TENSORRT_INCLUDE_DIR NvInferPlugin.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
list(APPEND CMAKE_MESSAGE_INDENT "    ")
if(TENSORRT_FOUND)
    message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "Found TensorRT libs at ${TENSORRT_LIBRARY}")
else()
    message(FATAL_ERROR "Cannot find TensorRT library.")
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)

# boost
message(STATUS "Finding Boost...")
find_package(Boost REQUIRED)
list(APPEND CMAKE_MESSAGE_INDENT "    ")
if(Boost_FOUND)
    message(STATUS "Boost version: ${Boost_VERSION_STRING}")
    message(STATUS "Using Boost include dir(s): ${Boost_INCLUDE_DIRS}")
    message(STATUS "Using Boost lib(s): ${Boost_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not find Boost")
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)

# sqlite3
message(STATUS "Finding SQLite3...")
find_package(SQLite3 REQUIRED)
list(APPEND CMAKE_MESSAGE_INDENT "    ")
if(SQLite3_FOUND)
    message(STATUS "SQLite3 version: ${SQLite3_VERSION}")
    message(STATUS "Using SQLite3 include dir(s): ${SQLite3_INCLUDE_DIRS}")
    message(STATUS "Using SQLite3 lib(s): ${SQLite3_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not find SQLite3")
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)

# add the executable
AUX_SOURCE_DIRECTORY(../src SRC)
list(REMOVE_ITEM SRC "../src/main.cpp" "../src/webclient.h" "../src/webclient.cpp")
message(STATUS "SOURCE FILES: ${SRC}")
cuda_add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${CUBLASLT_LIBRARY}
    ${Boost_LIBRARIES}
    ${SQLite3_LIBRARIES}
)
