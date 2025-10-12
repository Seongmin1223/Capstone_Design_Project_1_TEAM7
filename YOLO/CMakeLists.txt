# CMake 최소 요구 버전 설정
cmake_minimum_required(VERSION 3.20)
project(YOLO_Inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(YOLO_Inference
    main.cpp
    inference.cpp
)

target_link_libraries(YOLO_Inference ${OpenCV_LIBS})