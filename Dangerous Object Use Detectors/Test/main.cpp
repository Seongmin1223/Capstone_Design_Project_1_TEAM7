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

struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

//욜로 처리 함수
std::vector<Detection> postprocess(cv::Mat& frame, const cv::Size& blobSize, const std::vector<cv::Mat>& outs,
    float confThreshold, float nmsThreshold)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    float scale_x = (float)frame.cols / blobSize.width;
    float scale_y = (float)frame.rows / blobSize.height;
    for (size_t i = 0; i < outs.size(); ++i) {
        cv::Mat output = outs[i];
        if (output.dims == 3) {
            output = output.reshape(1, output.size[1]);
            cv::transpose(output, output);
        }
        int rows = output.rows;
        int dims = output.cols;
        for (int j = 0; j < rows; ++j) {
            float* data = output.ptr<float>(j);
            float x = data[0], y = data[1], w = data[2], h = data[3];
            float maxScore = 0;
            int maxClassId = -1;
            for (int k = 4; k < dims; ++k) {
                if (data[k] > maxScore) {
                    maxScore = data[k];
                    maxClassId = k - 4;
                }
            }
            if (maxScore > confThreshold && maxClassId >= 0) {
                int centerX = static_cast<int>(x * scale_x);
                int centerY = static_cast<int>(y * scale_y);
                int width = static_cast<int>(w * scale_x);
                int height = static_cast<int>(h * scale_y);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                left = std::max(0, std::min(left, frame.cols - 1));
                top = std::max(0, std::min(top, frame.rows - 1));
                width = std::max(1, std::min(width, frame.cols - left));
                height = std::max(1, std::min(height, frame.rows - top));
                classIds.push_back(maxClassId);
                confidences.push_back(maxScore);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    std::vector<Detection> detections;
    for (int idx : indices)
        detections.push_back({ boxes[idx], confidences[idx], classIds[idx] });
    return detections;
}


int main(int argc, char** argv)
{
    std::vector<std::string> arguments = { "FaceTest.exe", "-wild", "-mloc", "model/main_ceclm_general.txt" };
    LandmarkDetector::FaceModelParameters det_parameters(arguments);
    LandmarkDetector::CLNF face_model(det_parameters.model_location);
    if (!face_model.loaded_successfully) {
        std::cerr << "OpenFace 모델 로드 실패" << std::endl;
        return 1;
    }
    FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
    face_analysis_params.OptimizeForImages();
    FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);
    Utilities::Visualizer visualizer(arguments);
    std::vector<std::string> model_paths = { "v1_opencv.onnx", "model/v1_opencv.onnx" }; //모델 설정하는 파트, 모델의 추가 강화 학습이 이루어지면 model/~~ 로 저장바람(onnx 파일)
    cv::dnn::Net yolo_net;
    bool model_loaded = false;
    for (const auto& path : model_paths) {
        if (fs::exists(path)) {
            try {
                yolo_net = cv::dnn::readNetFromONNX(path);
                yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                model_loaded = true;
                std::cout << "YOLO 모델 로드 성공: " << path << std::endl;
                break;
            }
            catch (const cv::Exception& e) { std::cerr << "YOLO 로드 실패 (" << path << "): " << e.what() << std::endl; }
        }
    }
    if (!model_loaded) { std::cerr << "YOLOv8 모델 파일을 찾을 수 없습니다." << std::endl; return 1; }

    std::vector<std::string> classNames = { "cell phone" };
    const int CELL_PHONE_CLASS_ID = 0;
    float confThreshold = 0.25f;
    float nmsThreshold = 0.4f;
    const cv::Size yoloInputSize(640, 640);
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "웹캠 열기 실패" << std::endl; return 1; }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat display_frame = frame.clone();
        cv::Mat_<uchar> grayscale_image;
        cv::cvtColor(frame, grayscale_image, cv::COLOR_BGR2GRAY);

        // 이 '추측'이 시선 교차의 정확도를 결정합니다.
        double fx = 1000.0, fy = 1000.0; // 800~1200으로 임의 추정바람, 실제 캘리브레이션을 위해서는 체스판?.. 필요
        double cx = frame.cols / 2.0, cy = frame.rows / 2.0;
        cv::Matx33d K(fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, yoloInputSize, cv::Scalar(), true, false);
        yolo_net.setInput(blob);
        std::vector<cv::Mat> yolo_outs;
        yolo_net.forward(yolo_outs, yolo_net.getUnconnectedOutLayersNames());
        std::vector<Detection> detections = postprocess(frame, yoloInputSize, yolo_outs, confThreshold, nmsThreshold);

        LandmarkDetector::DetectLandmarksInVideo(frame, face_model, det_parameters, grayscale_image);
        bool face_detected = face_model.detection_success;
        std::vector<cv::Point2f> gaze_line_2d; // 2D 시선 벡터 (시작점, 끝점)

        if (face_detected) {
            // '추측'한 fx, fy, cx, cy로 3D 눈/시선 계산
            auto eye_landmarks_3D = LandmarkDetector::Calculate3DEyeLandmarks(face_model, fx, fy, cx, cy);
            cv::Point3f gaze0(0, 0, -1), gaze1(0, 0, -1);
            GazeAnalysis::EstimateGaze(face_model, gaze0, fx, fy, cx, cy, true);
            GazeAnalysis::EstimateGaze(face_model, gaze1, fx, fy, cx, cy, false);
            cv::Point3f gaze_vector_head = (gaze0 + gaze1) / 2.0f;

            cv::Point3f eye_center_head(0, 0, 0);
            if (!eye_landmarks_3D.empty()) {
                for (const auto& p : eye_landmarks_3D) { eye_center_head += p; }
                eye_center_head /= (float)eye_landmarks_3D.size();
            }

            float gaze_line_length = 10000.0; // 10미터 (충분히 길게)
            cv::Point3f gaze_endpoint_head = eye_center_head + (gaze_vector_head * gaze_line_length);
            std::vector<cv::Point3f> gaze_line_3d = { eye_center_head, gaze_endpoint_head };

            cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, fx, fy, cx, cy);
            cv::Vec3d tvec(pose_estimate[0], pose_estimate[1], pose_estimate[2]);
            cv::Vec3d rvec(pose_estimate[3], pose_estimate[4], pose_estimate[5]);
            cv::projectPoints(gaze_line_3d, rvec, tvec, K, dist, gaze_line_2d);
        }

        visualizer.SetImage(display_frame, fx, fy, cx, cy);
        if (face_detected) {
            cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, fx, fy, cx, cy);
            cv::Point3f gaze0(0, 0, -1), gaze1(0, 0, -1);
            if (face_model.eye_model) {
                GazeAnalysis::EstimateGaze(face_model, gaze0, fx, fy, cx, cy, true);
                GazeAnalysis::EstimateGaze(face_model, gaze1, fx, fy, cx, cy, false);
            }
            face_analyser.PredictStaticAUsAndComputeFeatures(frame, face_model.detected_landmarks);
            visualizer.SetObservationLandmarks(face_model.detected_landmarks, 1.0, face_model.GetVisibilities());
            visualizer.SetObservationPose(pose_estimate, 1.0);
            visualizer.SetObservationGaze(gaze0, gaze1,
                LandmarkDetector::CalculateAllEyeLandmarks(face_model),
                LandmarkDetector::Calculate3DEyeLandmarks(face_model, fx, fy, cx, cy),
                face_model.detection_certainty);
            visualizer.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
        }

        cv::Mat visImg = visualizer.GetVisImage().clone();

        for (const auto& det : detections) {
            std::string className = classNames[det.classId];
            cv::Scalar color = cv::Scalar(0, 0, 255); // 기본색 빨간색
            bool is_looking = false;

            if (det.classId == CELL_PHONE_CLASS_ID && face_detected && gaze_line_2d.size() == 2) {
                cv::Rect phone_box = det.box;
                cv::Point2f p1_f = gaze_line_2d[0]; // 눈 위치
                cv::Point2f p2_f = gaze_line_2d[1]; // 10미터 앞 투영점
                cv::Point2f gaze_vec_2d = p2_f - p1_f;
                cv::Point p1 = p1_f;
                cv::Point p_far = p1_f + (gaze_vec_2d * 5000.0);
                if (cv::clipLine(phone_box, p1, p_far)) {
                    is_looking = true;
                    color = cv::Scalar(255, 0, 0);
                }
            }

            cv::rectangle(visImg, det.box, color, 2);
            std::string label = className + " " + std::to_string((int)(det.confidence * 100)) + "%";
            if (is_looking) label += " (WATCHING)";

            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseLine);
            cv::rectangle(visImg,
                cv::Point(det.box.x, det.box.y - labelSize.height - 5),
                cv::Point(det.box.x + labelSize.width, det.box.y),
                color, cv::FILLED);
            cv::putText(visImg, label, cv::Point(det.box.x, det.box.y - 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1);
        }

        if (face_detected && gaze_line_2d.size() == 2)
        {
            cv::Point2f p1_f = gaze_line_2d[0];
            cv::Point2f p2_f = gaze_line_2d[1];
            cv::Point2f gaze_vec_2d = p2_f - p1_f;
            cv::Point p1 = p1_f;
            cv::Point p_far = p1_f + (gaze_vec_2d * 5000.0);
            cv::line(visImg, p1, p_far, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            cv::circle(visImg, p1, 5, cv::Scalar(0, 255, 255), -1);
        }

        cv::imshow("결과", visImg);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}