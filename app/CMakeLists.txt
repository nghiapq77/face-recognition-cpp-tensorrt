cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(PROJECT_NAME face)
project(${PROJECT_NAME})
set (CMAKE_CXX_STANDARD 11)

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "Using OpenCV include dir(s): ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "Using OpenCV lib(s): ${OpenCV_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not find OpenCV")
endif()

# CUDA
find_package(CUDA)
message(STATUS "CUDA version: ${CUDA_VERSION}")

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
        -gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

if(CUDA_VERSION_MAJOR GREATER 9)
    message(STATUS "CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS}; 
            -gencode arch=compute_72,code=sm_72
    )

endif()

# TensorRT
#message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/include
  PATH_SUFFIXES include)
find_path(TENSORRT_INCLUDE_DIR NvInferPlugin.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
  find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
message(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()
message("TENSORRT_LIBRARY = ${TENSORRT_LIBRARY}")

## cublas
set(CUBLASLT_LIBRARY cublasLt)
find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
message(STATUS "cuBLASLt library found")

# curl
find_package(CURL REQUIRED)
if(CURL_FOUND)
    message(STATUS "Found CURL version: ${CURL_VERSION_STRING}")
    message(STATUS "Using CURL include dir(s): ${CURL_INCLUDE_DIRS}")
    message(STATUS "Using CURL lib(s): ${CURL_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not find CURL")
endif()

# sqlite3
set(SQLITE_LIBRARY sqlite3)

AUX_SOURCE_DIRECTORY(../src DIR_SRCS)
list(REMOVE_ITEM DIR_SRCS "../src/main.cpp")
message("DIR_SRCS = ${DIR_SRCS}")
cuda_add_executable(${PROJECT_NAME} ${DIR_SRCS})
target_link_libraries(${PROJECT_NAME}
    ${TENSORRT_LIBRARY}
    ${OpenCV_LIBS}
    ${CUBLASLT_LIBRARY}
    ${CURL_LIBRARIES}
    ${SQLITE_LIBRARY}
)
