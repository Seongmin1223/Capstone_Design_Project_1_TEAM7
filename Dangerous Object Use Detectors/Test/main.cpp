#include <dlib/image_processing/frontal_face_detector.h>
#include "LandmarkCoreIncludes.h"
#include <FaceAnalyser.h>
#include <GazeEstimation.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <filesystem>
namespace fs = std::filesystem;
#pragma comment(lib, "opencv_world4120.lib")
#pragma comment(lib, "Utilities.lib")
#pragma comment(lib, "LandmarkDetector.lib")
#pragma comment(lib, "FaceAnalyser.lib")
#pragma comment(lib, "GazeAnalyser.lib")
#pragma comment(lib, "dlib.lib")
#pragma comment(lib, "openblas.lib")

// YOLO 탐지 결과 구조체
struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

// YOLO 후처리 함수 (YOLOv8 형식)
std::vector<Detection> postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs,
    float confThreshold, float nmsThreshold) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // YOLOv8 출력 형식: [1, 84, 8400] (전치된 형태)
    for (size_t i = 0; i < outs.size(); ++i) {
        cv::Mat output = outs[i];

        // [1, 84, 8400] -> [8400, 84]로 변환
        if (output.dims == 3) {
            output = output.reshape(1, output.size[1]);  // [84, 8400]
            cv::transpose(output, output);  // [8400, 84]
        }

        int rows = output.rows;
        int dimensions = output.cols;

        // 각 검출 결과 처리
        for (int j = 0; j < rows; ++j) {
            float* data = output.ptr<float>(j);

            // x, y, w, h는 처음 4개 값
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            // 클래스 점수는 4번 인덱스부터
            float maxScore = 0;
            int maxClassId = 0;
            for (int k = 4; k < dimensions; ++k) {
                if (data[k] > maxScore) {
                    maxScore = data[k];
                    maxClassId = k - 4;
                }
            }

            // 신뢰도 체크
            if (maxScore > confThreshold) {
                // 좌표 변환 (정규화된 중심점 -> 픽셀 좌표)
                int centerX = (int)(x * frame.cols);
                int centerY = (int)(y * frame.rows);
                int width = (int)(w * frame.cols);
                int height = (int)(h * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(maxClassId);
                confidences.push_back(maxScore);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    std::vector<Detection> detections;
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.classId = classIds[idx];
        detections.push_back(det);
    }

    return detections;
}

int main(int argc, char** argv)
{
    std::vector<std::string> arguments = {
        "FaceTest.exe",
        "-wild",
        "-mloc", "model/main_ceclm_general.txt"
    };

    // OpenFace 초기화
    LandmarkDetector::FaceModelParameters det_parameters(arguments);
    LandmarkDetector::CLNF face_model(det_parameters.model_location);
    if (!face_model.loaded_successfully) {
        std::cerr << "모델오류" << std::endl;
        return 1;
    }

    FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
    face_analysis_params.OptimizeForImages();
    FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

    dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();
    Utilities::Visualizer visualizer(arguments);

    // YOLO 모델 로드 - 여러 경로 시도
    std::vector<std::string> model_paths = {
        "yolov8n.onnx",              // 실행 파일과 같은 위치
        "../yolov8n.onnx",           // 한 단계 상위
        "../../yolov8n.onnx",        // 두 단계 상위
        "model/yolov8n.onnx",       // models 폴더 내
        "../model/yolov8n.onnx"     // 상위의 models 폴더
    };

    cv::dnn::Net yolo_net;
    bool model_loaded = false;
    std::string loaded_path;

    for (const auto& path : model_paths) {
        if (fs::exists(path)) {
            try {
                yolo_net = cv::dnn::readNetFromONNX(path);
                yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                loaded_path = path;
                model_loaded = true;
                std::cout << "YOLO 모델 로드 성공: " << path << std::endl;
                break;
            }
            catch (const cv::Exception& e) {
                std::cerr << "경로에서 로드 실패 (" << path << "): " << e.what() << std::endl;
            }
        }
    }

    if (!model_loaded) {
        std::cerr << "YOLO 모델을 찾을 수 없습니다. 다음 위치에 yolov8n.onnx를 배치하세요:" << std::endl;
        for (const auto& path : model_paths) {
            std::cerr << "  - " << fs::absolute(path) << std::endl;
        }
        return 1;
    }

    // COCO 클래스 이름 (휴대폰은 인덱스 67)
    std::vector<std::string> classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    const int CELL_PHONE_CLASS_ID = 67;
    float confThreshold = 0.15f;  // 임계값 더 낮춤 (휴대폰 탐지 민감도 증가)
    float nmsThreshold = 0.4f;

    // 웹캠 열기
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "웹캠을 열 수 없습니다." << std::endl;
        return 1;
    }

    float fx = 500, fy = 500;
    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        cv::Mat display_frame = frame.clone();
        cv::Mat_<uchar> grayscale_image;
        cv::cvtColor(frame, grayscale_image, cv::COLOR_BGR2GRAY);

        // === YOLO로 휴대폰 탐지 ===
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
        yolo_net.setInput(blob);

        std::vector<cv::Mat> yolo_outs;
        yolo_net.forward(yolo_outs, yolo_net.getUnconnectedOutLayersNames());

        std::vector<Detection> detections = postprocess(frame, yolo_outs, confThreshold, nmsThreshold);

        // 모든 탐지 결과 출력 (디버깅용)
        if (!detections.empty()) {
            std::cout << "=== Frame Detection Results ===" << std::endl;
            std::cout << "Total detections: " << detections.size() << std::endl;
        }

        // 휴대폰 탐지 결과 표시
        bool phone_detected = false;
        int phone_count = 0;

        for (const auto& det : detections) {
            // 모든 탐지 객체 출력
            if (!detections.empty()) {
                std::cout << "  - " << classNames[det.classId]
                    << " (confidence: " << (det.confidence * 100) << "%)" << std::endl;
            }

            // 휴대폰 탐지 시
            if (det.classId == CELL_PHONE_CLASS_ID) {
                phone_detected = true;
                phone_count++;

                // 빨간색 박스로 휴대폰 표시
                cv::rectangle(display_frame, det.box, cv::Scalar(0, 0, 255), 3);

                std::string label = "PHONE DETECTED! " + std::to_string((int)(det.confidence * 100)) + "%";
                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseLine);

                // 레이블 배경
                cv::rectangle(display_frame,
                    cv::Point(det.box.x, det.box.y - labelSize.height - 10),
                    cv::Point(det.box.x + labelSize.width, det.box.y),
                    cv::Scalar(0, 0, 255), cv::FILLED);

                // 레이블 텍스트
                cv::putText(display_frame, label,
                    cv::Point(det.box.x, det.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            }
        }

        // 화면 상단에 큰 경고 메시지
        if (phone_detected) {
            std::string warning = "!!! WARNING: PHONE IN VIEW !!!";
            cv::Size textSize = cv::getTextSize(warning, cv::FONT_HERSHEY_SIMPLEX, 1.5, 3, nullptr);

            // 경고 배경 (빨간색 반투명)
            cv::Mat overlay = display_frame.clone();
            cv::rectangle(overlay,
                cv::Point(0, 0),
                cv::Point(display_frame.cols, 80),
                cv::Scalar(0, 0, 255), cv::FILLED);
            cv::addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame);

            // 경고 텍스트
            cv::putText(display_frame, warning,
                cv::Point((display_frame.cols - textSize.width) / 2, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 3);

            std::cout << ">>> PHONE DETECTED! Count: " << phone_count << " <<<" << std::endl;
        }

        // === OpenFace로 얼굴 및 시선 탐지 ===
        dlib::cv_image<unsigned char> dlib_img(grayscale_image);
        std::vector<dlib::rectangle> hog_face_detections = face_detector_hog(dlib_img);
        std::vector<cv::Rect_<float>> face_detections;
        for (const auto& rect : hog_face_detections) {
            face_detections.push_back(cv::Rect_<float>(rect.left(), rect.top(), rect.width(), rect.height()));
        }

        if (!face_detections.empty()) {
            float cx = frame.cols / 2.0f, cy = frame.rows / 2.0f;
            visualizer.SetImage(display_frame, fx, fy, cx, cy);

            for (const auto& face_rect : face_detections) {
                bool success = LandmarkDetector::DetectLandmarksInImage(frame, face_rect, face_model, det_parameters, grayscale_image);

                if (success) {
                    cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, fx, fy, cx, cy);
                    cv::Point3f gaze_direction0(0, 0, -1), gaze_direction1(0, 0, -1);

                    if (face_model.eye_model) {
                        GazeAnalysis::EstimateGaze(face_model, gaze_direction0, fx, fy, cx, cy, true);
                        GazeAnalysis::EstimateGaze(face_model, gaze_direction1, fx, fy, cx, cy, false);

                        std::cout << "Left Eye Gaze (X, Y, Z): " << gaze_direction0 << std::endl;
                        std::cout << "Right Eye Gaze (X, Y, Z): " << gaze_direction1 << std::endl;
                    }

                    face_analyser.PredictStaticAUsAndComputeFeatures(frame, face_model.detected_landmarks);

                    visualizer.SetObservationLandmarks(face_model.detected_landmarks, 1.0, face_model.GetVisibilities());
                    visualizer.SetObservationPose(pose_estimate, 1.0);
                    visualizer.SetObservationGaze(gaze_direction0, gaze_direction1,
                        LandmarkDetector::CalculateAllEyeLandmarks(face_model),
                        LandmarkDetector::Calculate3DEyeLandmarks(face_model, fx, fy, cx, cy),
                        face_model.detection_certainty);
                    visualizer.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
                }
            }

            cv::imshow("결과", visualizer.GetVisImage());
            visualizer.ShowObservation();
        }
        else {
            cv::imshow("결과", display_frame);
        }

        if (cv::waitKey(1) == 27) break; // ESC 누르면 종료
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}