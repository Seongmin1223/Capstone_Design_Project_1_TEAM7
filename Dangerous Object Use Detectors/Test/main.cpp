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

void Project(cv::Mat_<float>& dest, const cv::Mat_<float>& mesh, float fx, float fy, float cx, float cy)
{
    dest = cv::Mat_<float>(mesh.rows, 2, 0.0);

    for (int i = 0; i < mesh.rows; i++)
    {
        float X = mesh.at<float>(i, 0);
        float Y = mesh.at<float>(i, 1);
        float Z = mesh.at<float>(i, 2);

        float x = ((X * fx / Z) + cx);
        float y = ((Y * fy / Z) + cy);

        dest.at<float>(i, 0) = x;
        dest.at<float>(i, 1) = y;
    }
}

//openface와 동일한 기능
struct GazeLine {
    cv::Point2f start;
    cv::Point2f end;
    bool valid;
};

GazeLine calculateAccurateGazeLine(
    const LandmarkDetector::CLNF& face_model,
    double fx, double fy, double cx, double cy,
    int imageWidth, int imageHeight)
{
    GazeLine result = { cv::Point2f(0,0), cv::Point2f(0,0), false };

    if (!face_model.detection_success) {
        return result;
    }

    // 1. 3D 눈 랜드마크 가져오기
    auto eye_landmarks3d = LandmarkDetector::Calculate3DEyeLandmarks(face_model, fx, fy, cx, cy);

    if (eye_landmarks3d.empty()) {
        return result;
    }

    // 2. 왼쪽/오른쪽 눈의 시선 방향 계산
    cv::Point3f gaze_direction0(0, 0, -1);
    cv::Point3f gaze_direction1(0, 0, -1);
    GazeAnalysis::EstimateGaze(face_model, gaze_direction0, fx, fy, cx, cy, true);  // left eye
    GazeAnalysis::EstimateGaze(face_model, gaze_direction1, fx, fy, cx, cy, false); // right eye

    // 3. Pupil 위치 계산 (왼쪽 눈 iris 8개 점의 평균)
    cv::Point3f pupil_left(0, 0, 0);
    cv::Point3f pupil_right(0, 0, 0);
    for (size_t i = 0; i < 8; ++i)
    {
        pupil_left = pupil_left + eye_landmarks3d[i];
        pupil_right = pupil_right + eye_landmarks3d[i + eye_landmarks3d.size() / 2];
    }
    pupil_left = pupil_left / 8.0f;
    pupil_right = pupil_right / 8.0f;

    // 4. 평균 pupil 위치와 시선 방향
    cv::Point3f pupil_avg = (pupil_left + pupil_right) / 2.0f;
    cv::Point3f gaze_avg = (gaze_direction0 + gaze_direction1) / 2.0f;

    // 5. 시선의 시작점과 끝점 (3D)
    std::vector<cv::Point3f> points;
    points.push_back(pupil_avg);
    points.push_back(pupil_avg + gaze_avg * 500.0f);  // 길이 조절 (50.0 -> 150.0)

    // 6. 3D -> 2D 투영 (OpenFace의 Project 함수 사용)
    cv::Mat_<float> mesh = (cv::Mat_<float>(2, 3) <<
        points[0].x, points[0].y, points[0].z,
        points[1].x, points[1].y, points[1].z);

    cv::Mat_<float> proj_points;
    Project(proj_points, mesh, fx, fy, cx, cy);

    if (proj_points.rows == 2) {
        result.start = cv::Point2f(proj_points.at<float>(0, 0), proj_points.at<float>(0, 1));
        result.end = cv::Point2f(proj_points.at<float>(1, 0), proj_points.at<float>(1, 1));
        result.valid = true;
    }

    return result;
}

//정확도 관련 함수
struct GazeAccuracy {
    float angle_diff_deg;
    float endpoint_distance_px;

    std::string toString() const {
        char buf[256];
        snprintf(buf, sizeof(buf), "Angle: %.2f deg | Distance: %.0fpx",
            angle_diff_deg, endpoint_distance_px);
        return std::string(buf);
    }
};

GazeAccuracy compareGazeLines(
    const GazeLine& openface_line,
    const GazeLine& custom_line)
{
    GazeAccuracy accuracy = { 0.0f, 0.0f };

    if (!openface_line.valid || !custom_line.valid) {
        return accuracy;
    }

    // 1. 방향 벡터 계산
    cv::Point2f v1 = openface_line.end - openface_line.start;
    cv::Point2f v2 = custom_line.end - custom_line.start;

    // 2. 벡터 정규화
    float len1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
    float len2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);

    if (len1 < 0.001f || len2 < 0.001f) {
        return accuracy;
    }

    v1 *= (1.0f / len1);
    v2 *= (1.0f / len2);

    // 3. 각도 차이 계산
    float dot_product = v1.x * v2.x + v1.y * v2.y;
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
    float angle_rad = std::acos(dot_product);
    accuracy.angle_diff_deg = angle_rad * 180.0f / CV_PI;

    // 4. 끝점 거리 계산
    cv::Point2f diff = openface_line.end - custom_line.end;
    accuracy.endpoint_distance_px = std::sqrt(diff.x * diff.x + diff.y * diff.y);

    return accuracy;
}

// YOLO Postprocessing
std::vector<Detection> postprocess(
    cv::Mat& frame,
    const cv::Size& blobSize,
    const std::vector<cv::Mat>& outs,
    float confThreshold,
    float nmsThreshold)
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
                int centerX = (int)(x * scale_x);
                int centerY = (int)(y * scale_y);
                int width = (int)(w * scale_x);
                int height = (int)(h * scale_y);

                int left = centerX - width / 2;
                int top = centerY - height / 2;

                left = std::max(0, std::min(left, frame.cols - 1));
                top = std::max(0, std::min(top, frame.rows - 1));
                width = std::max(1, std::min(width, frame.cols - left));
                height = std::max(1, std::min(height, frame.rows - top));

                boxes.push_back(cv::Rect(left, top, width, height));
                classIds.push_back(maxClassId);
                confidences.push_back(maxScore);
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
    std::vector<std::string> arguments = {
        "FaceTest.exe", "-wild",
        "-mloc", "model/main_ceclm_general.txt"
    };

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

    // ---------------------------
    // YOLO ONNX Load
    // ---------------------------
    std::vector<std::string> model_paths = {
        "v1_opencv.onnx",
        "model/v1_opencv.onnx"
    };

    cv::dnn::Net yolo_net;
    bool model_loaded = false;

    for (const auto& path : model_paths) {
        if (!fs::exists(path)) {
            std::cerr << "[WARN] 모델 파일 없음: " << path << "\n";
            continue;
        }

        try {
            yolo_net = cv::dnn::readNetFromONNX(path);
            yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

            model_loaded = true;
            std::cout << "YOLO 모델 로드 성공: " << path << std::endl;
            break;
        }
        catch (const cv::Exception& e) {
            std::cerr << "[ERROR] YOLO 로드 실패 (" << path << ")\n";
            std::cerr << "OpenCV 예외 메시지: " << e.what() << "\n";
        }
    }

    if (!model_loaded) {
        std::cerr << "ERROR: YOLOv8 모델 로드 실패" << std::endl;
        return 1;
    }

    // YOLO Settings
    std::vector<std::string> classNames = { "cell phone" };
    const int CELL_PHONE_CLASS_ID = 0;
    float confThreshold = 0.25f;
    float nmsThreshold = 0.4f;
    const cv::Size yoloInputSize(640, 640);

    // Webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "웹캠 열기 실패" << std::endl;
        return 1;
    }

    cv::Mat frame;

    std::cout << "\n=== Gaze Detection Accuracy Comparison System ===" << std::endl;
    std::cout << "Green Line: OpenFace Original | Red Line: Accurate Implementation" << std::endl;
    std::cout << "Press ESC to exit" << std::endl;

    // 정확도 통계
    std::vector<float> angle_diffs;
    std::vector<float> distance_diffs;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat display_frame = frame.clone();
        cv::Mat_<uchar> grayscale_image;
        cv::cvtColor(frame, grayscale_image, cv::COLOR_BGR2GRAY);

        double fx = 1000.0, fy = 1000.0;
        double cx = frame.cols / 2.0, cy = frame.rows / 2.0;

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, yoloInputSize, cv::Scalar(), true, false);
        yolo_net.setInput(blob);

        std::vector<cv::Mat> yolo_outs;
        yolo_net.forward(yolo_outs, yolo_net.getUnconnectedOutLayersNames());
        auto detections = postprocess(frame, yoloInputSize, yolo_outs, confThreshold, nmsThreshold);

        LandmarkDetector::DetectLandmarksInVideo(frame, face_model, det_parameters, grayscale_image);
        bool face_detected = face_model.detection_success;
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
            visualizer.SetObservationGaze(gaze0, gaze1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, fx, fy, cx, cy), face_model.detection_certainty);

            visualizer.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
        }

        cv::Mat visImg = visualizer.GetVisImage().clone();
        GazeLine accurate_gaze = calculateAccurateGazeLine(face_model, fx, fy, cx, cy, frame.cols, frame.rows);

        if (accurate_gaze.valid) {
            // 빨간색으로 정확한 시선 라인 그리기
            cv::Point p1(accurate_gaze.start.x, accurate_gaze.start.y);
            cv::Point p2(accurate_gaze.end.x, accurate_gaze.end.y);

            cv::line(visImg, p1, p2, cv::Scalar(0, 0, 255), 3);
            cv::circle(visImg, p1, 6, cv::Scalar(0, 0, 255), -1);
            cv::circle(visImg, p2, 6, cv::Scalar(0, 0, 255), -1);
        }

        for (const auto& det : detections) {
            cv::rectangle(visImg, det.box, cv::Scalar(255, 255, 0), 2);

            std::string label = classNames[det.classId] + " " + std::to_string((int)(det.confidence * 100)) + "%";

            // 교차 판정 (정확한 라인 사용)
            if (det.classId == CELL_PHONE_CLASS_ID && accurate_gaze.valid) {
                cv::Point p1(accurate_gaze.start.x, accurate_gaze.start.y);
                cv::Point p2(accurate_gaze.end.x, accurate_gaze.end.y);

                if (cv::clipLine(det.box, p1, p2)) {
                    label += " [시선 감지]";
                    cv::rectangle(visImg, det.box, cv::Scalar(0, 0, 255), 3);
                }
            }

            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);

            cv::rectangle(visImg, cv::Point(det.box.x, det.box.y - labelSize.height - 5), cv::Point(det.box.x + labelSize.width, det.box.y), cv::Scalar(255, 255, 0), cv::FILLED);

            cv::putText(visImg, label, cv::Point(det.box.x, det.box.y - 3), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        }

        int y = 30;
        cv::putText(visImg, "Green: OpenFace Original", cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        y += 30;
        cv::putText(visImg, "Red: Accurate Implementation", cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

        // 평균 정확도 표시
        if (!angle_diffs.empty()) {
            float avg_angle = 0.0f, avg_dist = 0.0f;
            for (float a : angle_diffs) avg_angle += a;
            for (float d : distance_diffs) avg_dist += d;
            avg_angle /= angle_diffs.size();
            avg_dist /= distance_diffs.size();

            y += 40;
            char buf[256];
            snprintf(buf, sizeof(buf), "Avg Angle Diff: %.2f deg", avg_angle);
            cv::putText(visImg, buf, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

            y += 25;
            snprintf(buf, sizeof(buf), "Avg Distance: %.0fpx", avg_dist);
            cv::putText(visImg, buf, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }

        cv::imshow("Gaze Detection Comparison", visImg);
        if (cv::waitKey(1) == 27) break;
    }

    // 최종 통계 출력
    if (!angle_diffs.empty()) {
        float avg_angle = 0.0f, avg_dist = 0.0f;
        for (float a : angle_diffs) avg_angle += a;
        for (float d : distance_diffs) avg_dist += d;
        avg_angle /= angle_diffs.size();
        avg_dist /= distance_diffs.size();

        std::cout << "\n=== Final Accuracy Statistics ===" << std::endl;
        std::cout << "Average Angle Difference: " << avg_angle << " degrees" << std::endl;
        std::cout << "Average Endpoint Distance: " << avg_dist << " pixels" << std::endl;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}