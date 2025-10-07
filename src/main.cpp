#include <dlib/image_processing/frontal_face_detector.h>
#include "LandmarkCoreIncludes.h"
#include <FaceAnalyser.h>
#include <GazeEstimation.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>

#include <filesystem>
namespace fs = std::filesystem;
#pragma comment(lib, "opencv_world410.lib")
#pragma comment(lib, "Utilities.lib")
#pragma comment(lib, "LandmarkDetector.lib")
#pragma comment(lib, "FaceAnalyser.lib")
#pragma comment(lib, "GazeAnalyser.lib")
#pragma comment(lib, "dlib.lib")
#pragma comment(lib, "openblas.lib")

int main(int argc, char** argv)
{
    std::vector<std::string> arguments = {
        "FaceTest.exe",
        "-wild",
        "-mloc", "model/main_ceclm_general.txt"
    };

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

    // --- 웹캠 열기 ---
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

        cv::Mat_<uchar> grayscale_image;
        cv::cvtColor(frame, grayscale_image, cv::COLOR_BGR2GRAY);

        dlib::cv_image<unsigned char> dlib_img(grayscale_image);
        std::vector<dlib::rectangle> hog_face_detections = face_detector_hog(dlib_img);
        std::vector<cv::Rect_<float>> face_detections;
        for (const auto& rect : hog_face_detections) {
            face_detections.push_back(cv::Rect_<float>(rect.left(), rect.top(), rect.width(), rect.height()));
        }

        if (face_detections.empty()) {
            cv::imshow("결과", frame);
            if (cv::waitKey(1) == 27) break; // ESC 누르면 종료
            continue;
        }

        float cx = frame.cols / 2.0f, cy = frame.rows / 2.0f;
        visualizer.SetImage(frame, fx, fy, cx, cy);

        for (const auto& face_rect : face_detections)
        {
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
                visualizer.SetObservationGaze(gaze_direction0, gaze_direction1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, fx, fy, cx, cy), face_model.detection_certainty);
                visualizer.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
            }
        }

        cv::imshow("결과", visualizer.GetVisImage());
        visualizer.ShowObservation();

        if (cv::waitKey(1) == 27) break; // ESC 누르면 종료
    }

    return 0;
}
