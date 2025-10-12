#include "inference.h"
#include <iostream>
#include <random>

Inference::Inference(const std::string& onnxModelPath, const cv::Size& modelInputShape)
{
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    loadOnnxNetwork();

    // [개선] 생성자에서 클래스별 색상을 미리 한 번만 생성합니다.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);
    for (size_t i = 0; i < classes.size(); ++i) {
        class_colors.push_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
    }
}

void Inference::loadOnnxNetwork()
{
    net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "ONNX model loaded successfully." << std::endl;
}

std::vector<Detection> Inference::runInference(const cv::Mat& input)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    if (outputs.empty())
        return {};

    cv::Mat output_buffer = outputs[0];
    int rows = output_buffer.size[2];
    int dimensions = output_buffer.size[1];

    output_buffer = output_buffer.reshape(1, dimensions);
    cv::transpose(output_buffer, output_buffer);
    float* data = (float*)output_buffer.data;

    float x_factor = (float)input.cols / modelShape.width;
    float y_factor = (float)input.rows / modelShape.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        float* classes_scores = data + 4;
        cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
        cv::Point class_id_point;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

        if (max_class_score > modelScoreThreshold)
        {
            confidences.push_back(max_class_score);
            class_ids.push_back(class_id_point.x);

            float x = data[0], y = data[1], w = data[2], h = data[3];
            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    std::vector<Detection> detections{};
    for (int idx : nms_result)
    {
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.className = classes[result.class_id];
        result.box = boxes[idx];
        // [개선] 매번 랜덤 색상을 생성하는 대신, 미리 만들어둔 색상을 사용합니다.
        result.color = class_colors[result.class_id];
        detections.push_back(result);
    }
    return detections;
}