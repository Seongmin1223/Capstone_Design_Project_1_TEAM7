#include <dlib/image_processing/frontal_face_detector.h>
#include "LandmarkCoreIncludes.h"
#include <FaceAnalyser.h>
#include <GazeEstimation.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>

namespace fs = std::filesystem;

#ifdef _WIN32
#pragma comment(lib, "opencv_world4120.lib")
#pragma comment(lib, "Utilities.lib")
#pragma comment(lib, "LandmarkDetector.lib")
#pragma comment(lib, "FaceAnalyser.lib")
#pragma comment(lib, "GazeAnalyser.lib")
#pragma comment(lib, "dlib.lib")
#pragma comment(lib, "openblas.lib")
#endif

struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

struct AnalysisResult {
    bool face_detected;
    bool phone_detected;
    bool looking_at_phone;
    float phone_confidence;
    std::string message;
    std::string filename;
};

struct Statistics {
    int total_images;
    int TP;
    int FP;
    int FN;
    int TN;

    std::vector<std::string> TP_files;
    std::vector<std::string> FP_files;
    std::vector<std::string> FN_files;
    std::vector<std::string> TN_files;
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

    auto eye_landmarks3d = LandmarkDetector::Calculate3DEyeLandmarks(face_model, fx, fy, cx, cy);
    if (eye_landmarks3d.empty()) {
        return result;
    }

    cv::Point3f gaze_direction0(0, 0, -1);
    cv::Point3f gaze_direction1(0, 0, -1);
    GazeAnalysis::EstimateGaze(face_model, gaze_direction0, fx, fy, cx, cy, true);
    GazeAnalysis::EstimateGaze(face_model, gaze_direction1, fx, fy, cx, cy, false);

    cv::Point3f pupil_left(0, 0, 0);
    cv::Point3f pupil_right(0, 0, 0);
    for (size_t i = 0; i < 8; ++i)
    {
        pupil_left = pupil_left + eye_landmarks3d[i];
        pupil_right = pupil_right + eye_landmarks3d[i + eye_landmarks3d.size() / 2];
    }
    pupil_left = pupil_left / 8.0f;
    pupil_right = pupil_right / 8.0f;

    cv::Point3f pupil_avg = (pupil_left + pupil_right) / 2.0f;
    cv::Point3f gaze_avg = (gaze_direction0 + gaze_direction1) / 2.0f;

    std::vector<cv::Point3f> points;
    points.push_back(pupil_avg);
    points.push_back(pupil_avg + gaze_avg * 1500.0f);

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

AnalysisResult analyzeImage(const std::string& imagePath,
    LandmarkDetector::CLNF& face_model,
    LandmarkDetector::FaceModelParameters& det_parameters,
    cv::dnn::Net& yolo_net)
{
    AnalysisResult result = { false, false, false, 0.0f, "", "" };
    result.filename = fs::path(imagePath).filename().string();

    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty()) {
        result.message = "ERROR: Cannot open image file";
        return result;
    }

    cv::Mat_<uchar> grayscale_image;
    cv::cvtColor(frame, grayscale_image, cv::COLOR_BGR2GRAY);

    double fx = 1000.0, fy = 1000.0;
    double cx = frame.cols / 2.0, cy = frame.rows / 2.0;

    LandmarkDetector::DetectLandmarksInVideo(frame, face_model, det_parameters, grayscale_image);
    result.face_detected = face_model.detection_success;

    if (!result.face_detected) {
        result.message = "No face detected";
        return result;
    }

    GazeLine gaze_line = calculateAccurateGazeLine(face_model, fx, fy, cx, cy, frame.cols, frame.rows);

    if (!gaze_line.valid) {
        result.message = "Cannot analyze gaze";
        return result;
    }

    const int CELL_PHONE_CLASS_ID = 0;
    float confThreshold = 0.25f;
    float nmsThreshold = 0.4f;
    const cv::Size yoloInputSize(640, 640);

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, yoloInputSize, cv::Scalar(), true, false);
    yolo_net.setInput(blob);

    std::vector<cv::Mat> yolo_outs;
    yolo_net.forward(yolo_outs, yolo_net.getUnconnectedOutLayersNames());
    auto detections = postprocess(frame, yoloInputSize, yolo_outs, confThreshold, nmsThreshold);

    for (const auto& det : detections) {
        if (det.classId == CELL_PHONE_CLASS_ID) {
            result.phone_detected = true;
            result.phone_confidence = det.confidence;

            cv::Point p1(gaze_line.start.x, gaze_line.start.y);
            cv::Point p2(gaze_line.end.x, gaze_line.end.y);
            bool intersects_strict = cv::clipLine(det.box, p1, p2);

            int margin = 100;
            cv::Rect expanded_box(
                det.box.x - margin,
                det.box.y - margin,
                det.box.width + 2 * margin,
                det.box.height + 2 * margin
            );

            cv::Point p1_expanded = cv::Point(gaze_line.start.x, gaze_line.start.y);
            cv::Point p2_expanded = cv::Point(gaze_line.end.x, gaze_line.end.y);
            bool intersects_expanded = cv::clipLine(expanded_box, p1_expanded, p2_expanded);

            cv::Point2f phone_center(det.box.x + det.box.width / 2.0f,
                det.box.y + det.box.height / 2.0f);
            cv::Point2f gaze_direction = gaze_line.end - gaze_line.start;
            cv::Point2f to_phone = phone_center - gaze_line.start;

            float gaze_len = std::sqrt(gaze_direction.x * gaze_direction.x +
                gaze_direction.y * gaze_direction.y);
            float phone_len = std::sqrt(to_phone.x * to_phone.x + to_phone.y * to_phone.y);

            float angle_to_phone = 180.0f;
            if (gaze_len > 0.001f && phone_len > 0.001f) {
                gaze_direction /= gaze_len;
                to_phone /= phone_len;

                float dot = gaze_direction.x * to_phone.x + gaze_direction.y * to_phone.y;
                dot = std::max(-1.0f, std::min(1.0f, dot));
                angle_to_phone = std::acos(dot) * 180.0f / CV_PI;
            }

            bool looking_by_strict = intersects_strict;
            bool looking_by_expanded = intersects_expanded;
            bool looking_by_angle = (angle_to_phone < 30.0f);

            result.looking_at_phone = looking_by_strict || looking_by_expanded || looking_by_angle;

            if (result.looking_at_phone) {
                result.message = "Looking at phone";
            }
            else {
                result.message = "Phone present but not looking";
            }

            break;
        }
    }

    if (!result.phone_detected) {
        result.message = "No phone detected";
    }

    return result;
}

Statistics analyzeDataset(const std::string& phoneFolder, const std::string& noGazeFolder, const std::string& noPhoneFolder)
{
    Statistics stats = { 0, 0, 0, 0, 0 };

    std::vector<std::string> arguments = {
        "FaceTest.exe", "-wild",
        "-mloc", "model/main_ceclm_general.txt"
    };

    LandmarkDetector::FaceModelParameters det_parameters(arguments);
    LandmarkDetector::CLNF face_model(det_parameters.model_location);

    if (!face_model.loaded_successfully) {
        std::cerr << "ERROR: OpenFace model loading failed" << std::endl;
        return stats;
    }

    std::vector<std::string> model_paths = {
        "v4_opencv.onnx",
        "model/v4_opencv.onnx"
    };

    cv::dnn::Net yolo_net;
    bool model_loaded = false;

    for (const auto& path : model_paths) {
        if (!fs::exists(path)) continue;

        try {
            yolo_net = cv::dnn::readNetFromONNX(path);
            yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            model_loaded = true;
            std::cout << "YOLO model loaded: " << path << std::endl;
            break;
        }
        catch (const cv::Exception& e) {
            std::cerr << "YOLO loading failed: " << path << std::endl;
        }
    }

    if (!model_loaded) {
        std::cerr << "ERROR: YOLO model loading failed" << std::endl;
        return stats;
    }

    std::vector<std::string> image_extensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };

    struct ImageInfo {
        std::string path;
        int ground_truth;
    };

    std::vector<ImageInfo> all_images;

    if (fs::exists(phoneFolder) && fs::is_directory(phoneFolder)) {
        for (const auto& entry : fs::directory_iterator(phoneFolder)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (std::find(image_extensions.begin(), image_extensions.end(), ext) != image_extensions.end()) {
                    all_images.push_back({ entry.path().string(), 1 });
                }
            }
        }
    }

    if (fs::exists(noGazeFolder) && fs::is_directory(noGazeFolder)) {
        for (const auto& entry : fs::directory_iterator(noGazeFolder)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (std::find(image_extensions.begin(), image_extensions.end(), ext) != image_extensions.end()) {
                    all_images.push_back({ entry.path().string(), 0 });
                }
            }
        }
    }

    if (fs::exists(noPhoneFolder) && fs::is_directory(noPhoneFolder)) {
        for (const auto& entry : fs::directory_iterator(noPhoneFolder)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (std::find(image_extensions.begin(), image_extensions.end(), ext) != image_extensions.end()) {
                    all_images.push_back({ entry.path().string(), 0 });
                }
            }
        }
    }

    stats.total_images = all_images.size();

    int phone_count = 0, no_gaze_count = 0, no_phone_count = 0;

    for (const auto& img : all_images) {
        std::string parent = fs::path(img.path).parent_path().filename().string();
        std::transform(parent.begin(), parent.end(), parent.begin(), ::tolower);

        if (parent.find("phone") != std::string::npos && parent.find("no") == std::string::npos) {
            phone_count++;
        }
        else if (parent.find("gaze") != std::string::npos || parent.find("no_gaze") != std::string::npos) {
            no_gaze_count++;
        }
        else if (parent.find("no_phone") != std::string::npos || parent.find("nophone") != std::string::npos) {
            no_phone_count++;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Found " << stats.total_images << " images total" << std::endl;
    std::cout << "  'phone' folder (looking at phone):        " << phone_count << " images [GT = 1]" << std::endl;
    std::cout << "  'no_gaze' folder (phone but not looking): " << no_gaze_count << " images [GT = 0]" << std::endl;
    std::cout << "  'no_phone' folder (no phone):             " << no_phone_count << " images [GT = 0]" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "Starting analysis...\n" << std::endl;

    int processed = 0;
    for (const auto& img_info : all_images) {
        processed++;
        std::string filename = fs::path(img_info.path).filename().string();
        std::string folder_name = fs::path(img_info.path).parent_path().filename().string();

        std::cout << "[" << processed << "/" << stats.total_images << "] "
            << folder_name << "/" << filename << " ";

        AnalysisResult result = analyzeImage(img_info.path, face_model, det_parameters, yolo_net);

        int actual = img_info.ground_truth;
        int predicted = result.looking_at_phone ? 1 : 0;

        std::string status;
        if (actual == 1 && predicted == 1) {
            stats.TP++;
            stats.TP_files.push_back(folder_name + "/" + filename);
            status = "[TP] CORRECT";
        }
        else if (actual == 0 && predicted == 1) {
            stats.FP++;
            stats.FP_files.push_back(folder_name + "/" + filename);
            status = "[FP] FALSE ALARM";
        }
        else if (actual == 1 && predicted == 0) {
            stats.FN++;
            stats.FN_files.push_back(folder_name + "/" + filename);
            status = "[FN] MISSED";
        }
        else if (actual == 0 && predicted == 0) {
            stats.TN++;
            stats.TN_files.push_back(folder_name + "/" + filename);
            status = "[TN] CORRECT";
        }

        std::cout << status << std::endl;
    }

    return stats;
}

void printStatistics(const Statistics& stats)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "         PERFORMANCE EVALUATION" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total images: " << stats.total_images << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n[CONFUSION MATRIX]" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "TP (True Positive):  " << stats.TP << " images" << std::endl;
    std::cout << "   -> Actually looking at phone & Detected as looking" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "FP (False Positive): " << stats.FP << " images" << std::endl;
    std::cout << "   -> Actually NOT looking & Detected as looking (FALSE ALARM)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "FN (False Negative): " << stats.FN << " images" << std::endl;
    std::cout << "   -> Actually looking & NOT detected (MISSED)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "TN (True Negative):  " << stats.TN << " images" << std::endl;
    std::cout << "   -> Actually NOT looking & NOT detected" << std::endl;
    std::cout << "========================================" << std::endl;

    float accuracy = 0.0f, precision = 0.0f, recall = 0.0f, f1_score = 0.0f;

    if (stats.total_images > 0) {
        accuracy = (float)(stats.TP + stats.TN) / stats.total_images * 100.0f;
    }

    if ((stats.TP + stats.FP) > 0) {
        precision = (float)stats.TP / (stats.TP + stats.FP) * 100.0f;
    }

    if ((stats.TP + stats.FN) > 0) {
        recall = (float)stats.TP / (stats.TP + stats.FN) * 100.0f;
    }

    if ((precision + recall) > 0) {
        f1_score = 2.0f * precision * recall / (precision + recall);
    }

    std::cout << "\n[PERFORMANCE METRICS]" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Accuracy:  " << accuracy << "%" << std::endl;
    std::cout << "           = (TP + TN) / Total" << std::endl;
    std::cout << "           = (" << stats.TP << " + " << stats.TN << ") / " << stats.total_images << std::endl;
    std::cout << "           [How often is the system correct overall?]" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Precision: " << precision << "%" << std::endl;
    std::cout << "           = TP / (TP + FP)" << std::endl;
    std::cout << "           = " << stats.TP << " / (" << stats.TP << " + " << stats.FP << ")" << std::endl;
    std::cout << "           [When it says 'looking', how often is it right?]" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Recall:    " << recall << "%" << std::endl;
    std::cout << "           = TP / (TP + FN)" << std::endl;
    std::cout << "           = " << stats.TP << " / (" << stats.TP << " + " << stats.FN << ")" << std::endl;
    std::cout << "           [Of all actual phone-looking cases, how many did we catch?]" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "F1-Score:  " << f1_score << "%" << std::endl;
    std::cout << "           = 2 * Precision * Recall / (Precision + Recall)" << std::endl;
    std::cout << "           [Balanced measure of Precision and Recall]" << std::endl;
    std::cout << "========================================" << std::endl;

    if (!stats.TP_files.empty()) {
        std::cout << "\n[TRUE POSITIVES - " << stats.TP << " files]" << std::endl;
        std::cout << "Correctly detected as looking at phone:" << std::endl;
        for (const auto& file : stats.TP_files) {
            std::cout << "  ✓ " << file << std::endl;
        }
    }

    if (!stats.FP_files.empty()) {
        std::cout << "\n[FALSE POSITIVES - " << stats.FP << " files]" << std::endl;
        std::cout << "Incorrectly detected as looking (False Alarms):" << std::endl;
        for (const auto& file : stats.FP_files) {
            std::cout << "  ✗ " << file << std::endl;
        }
    }

    if (!stats.FN_files.empty()) {
        std::cout << "\n[FALSE NEGATIVES - " << stats.FN << " files]" << std::endl;
        std::cout << "Missed detections (Should have detected):" << std::endl;
        for (const auto& file : stats.FN_files) {
            std::cout << "  ✗ " << file << std::endl;
        }
    }

    if (!stats.TN_files.empty()) {
        std::cout << "\n[TRUE NEGATIVES - " << stats.TN << " files]" << std::endl;
        std::cout << "Correctly identified as NOT looking:" << std::endl;
        for (const auto& file : stats.TN_files) {
            std::cout << "  ✓ " << file << std::endl;
        }
    }
}

int main(int argc, char** argv)
{
    std::cout << "=== Phone Gaze Detection - Performance Evaluation ===" << std::endl;
    std::cout << "\nThis program evaluates detection performance using three folders:" << std::endl;
    std::cout << "1. 'phone' folder:    Person IS looking at phone (Ground Truth = 1)" << std::endl;
    std::cout << "2. 'no_gaze' folder:  Phone present but NOT looking (Ground Truth = 0)" << std::endl;
    std::cout << "3. 'no_phone' folder: No phone in image (Ground Truth = 0)" << std::endl;
    std::cout << "\n========================================\n" << std::endl;

    std::string phoneFolder, noGazeFolder, noPhoneFolder;

    std::cout << "Enter path to 'phone' folder (looking at phone): " << std::endl;
    std::cout << "> ";
    std::getline(std::cin, phoneFolder);

    std::cout << "\nEnter path to 'no_gaze' folder (phone but not looking): " << std::endl;
    std::cout << "> ";
    std::getline(std::cin, noGazeFolder);

    std::cout << "\nEnter path to 'no_phone' folder (no phone): " << std::endl;
    std::cout << "> ";
    std::getline(std::cin, noPhoneFolder);

    if (phoneFolder.empty() || noGazeFolder.empty() || noPhoneFolder.empty()) {
        std::cerr << "ERROR: All three folder paths are required" << std::endl;
        return 1;
    }

    if (!fs::exists(phoneFolder)) {
        std::cerr << "ERROR: Phone folder does not exist: " << phoneFolder << std::endl;
        return 1;
    }

    if (!fs::exists(noGazeFolder)) {
        std::cerr << "ERROR: No-gaze folder does not exist: " << noGazeFolder << std::endl;
        return 1;
    }

    if (!fs::exists(noPhoneFolder)) {
        std::cerr << "ERROR: No-phone folder does not exist: " << noPhoneFolder << std::endl;
        return 1;
    }

    Statistics stats = analyzeDataset(phoneFolder, noGazeFolder, noPhoneFolder);
    printStatistics(stats);

    return 0;
}