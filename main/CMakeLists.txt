cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(PROJECT_NAME main)
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

# boost
message(STATUS "Finding Boost...")
find_package(Boost REQUIRED COMPONENTS thread)
list(APPEND CMAKE_MESSAGE_INDENT "    ")
if(Boost_FOUND)
    message(STATUS "Boost version: ${Boost_VERSION_STRING}")
    message(STATUS "Using Boost include dir(s): ${Boost_INCLUDE_DIRS}")
    message(STATUS "Using Boost lib(s): ${Boost_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not find Boost")
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)

# add the executable
add_executable(${PROJECT_NAME} 
    ../src/main.cpp
    ../src/json.hpp
    ../src/webclient.cpp
    ../src/webclient.h
)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)
