#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct Detection
{
    cv::Rect box;
    float confidence{ 0.0 };
    int class_id{ 0 };
    std::string className{};
    cv::Scalar color{}; // [수정] color 변수 추가
};

class Inference
{
public:
    Inference(const std::string& onnxModelPath, const cv::Size& modelInputShape);
    std::vector<Detection> runInference(const cv::Mat& input);

private:
    void loadOnnxNetwork();

    std::string modelPath;
    cv::Size modelShape;
    float modelScoreThreshold{ 0.45f };
    float modelNMSThreshold{ 0.50f };

    std::vector<std::string> classes{
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    // [개선] 클래스별 색상을 저장할 벡터 추가
    std::vector<cv::Scalar> class_colors;

    cv::dnn::Net net;
};