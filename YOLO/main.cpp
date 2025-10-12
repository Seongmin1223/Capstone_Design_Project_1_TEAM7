#include "inference.h"
#include <iostream>

int main()
{
    
    std::string model_path = "model/yolov8n.onnx";

    try
    {
        Inference inf(model_path, cv::Size(640, 640));
        cv::VideoCapture cap(0, cv::CAP_DSHOW);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open camera!" << std::endl;
            return -1;
        }

        cv::Mat frame;
        while (true)
        {
            cap.read(frame);
            if (frame.empty())
                break;

            std::vector<Detection> output = inf.runInference(frame);

            for (const auto& detection : output)
            {
                cv::rectangle(frame, detection.box, detection.color, 2);

                std::string label = detection.className + " " + std::to_string(detection.confidence).substr(0, 4);

                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
                cv::rectangle(frame, cv::Point(detection.box.x, detection.box.y - labelSize.height - 10),
                    cv::Point(detection.box.x + labelSize.width, detection.box.y),
                    detection.color, cv::FILLED);
                cv::putText(frame, label, cv::Point(detection.box.x, detection.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            }

            cv::imshow("YOLOv8 C++ Inference", frame);

            if (cv::waitKey(1) == 27) // ESC
                break;
        }
        cap.release();
        cv::destroyAllWindows();
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}