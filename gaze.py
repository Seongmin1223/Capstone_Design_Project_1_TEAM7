
# 2025/10/06 시선처리 + 객체 탐지

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import winsound
import math

# ----------------------------------------------------------------------------
# 1. 초기 설정
# ----------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

try:
    yolo_model = YOLO('yolov8s.pt')
except Exception as e:
    print(f"YOLO 모델 로드 실패: {e}\nyolov8s.pt 모델을 다운로드하거나 경로를 확인하세요.")
    exit()

CELLPHONE_CLASS_ID = 67
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

GAZE_DURATION_THRESHOLD = 2.0
gaze_start_time = None
is_looking_continuously = False

# ----------------------------------------------------------------------------
# 2. 유틸리티 함수
# ----------------------------------------------------------------------------
def liang_barsky_clip(p1, p2, rect):
    x_min, y_min, x_max, y_max = rect; x1, y1 = p1; x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    p = [-dx, dx, -dy, dy]; q = [x1 - x_min, x_max - x1, y1 - y_min, y_max - y1]
    t0, t1 = 0.0, 1.0
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0: return False
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                if t > t1: return False
                t0 = max(t0, t)
            else:
                if t < t0: return False
                t1 = min(t1, t)
    return t0 < t1

def is_point_in_rect(point, rect):
    x, y = point; x_min, y_min, x_max, y_max = rect
    return x_min <= x <= x_max and y_min <= y <= y_max

# ----------------------------------------------------------------------------
# 3. 메인 루프
# ----------------------------------------------------------------------------
main_window_name = 'Final Gaze Tracking System'

# 눈의 경계를 정의하는 랜드마크 인덱스 (가장 바깥쪽 점들)
L_EYE_BOUNDS = [33, 160, 158, 133, 153, 144]
R_EYE_BOUNDS = [362, 385, 387, 263, 373, 380]
R_IRIS_INDEXES = [474, 475, 476, 477]
L_IRIS_INDEXES = [469, 470, 471, 472]

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(rgb_frame)
    
    phone_coords = None
    is_looking_at_phone_now = False

    results_yolo = yolo_model(frame, classes=[CELLPHONE_CLASS_ID], verbose=False)
    if len(results_yolo) > 0 and len(results_yolo[0].boxes) > 0:
        phone_coords = results_yolo[0].boxes[0].xyxy[0].cpu().numpy().astype(int)

    if results_face.multi_face_landmarks:
        landmarks = results_face.multi_face_landmarks[0].landmark
        
        # 가려짐 판정
        if phone_coords is not None:
            all_eye_indices = L_EYE_BOUNDS + R_EYE_BOUNDS
            for i in all_eye_indices:
                lx, ly = int(landmarks[i].x * w), int(landmarks[i].y * h)
                if is_point_in_rect((lx, ly), phone_coords):
                    is_looking_at_phone_now = True
                    break
        
        if not is_looking_at_phone_now:
            # --- Y축 문제 해결 및 시선 시각화 로직 ---
            
            # 1. 왼쪽 눈 처리
            l_eye_region = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in L_EYE_BOUNDS], np.int32)
            lx, ly, lw, lh = cv2.boundingRect(l_eye_region)
            l_iris_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in L_IRIS_INDEXES])
            l_pupil = np.mean(l_iris_points, axis=0).astype(int)
            l_ratio_x = (l_pupil[0] - lx) / (lw + 1e-6)
            l_ratio_y = (l_pupil[1] - ly) / (lh + 1e-6)

            # 2. 오른쪽 눈 처리
            r_eye_region = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in R_EYE_BOUNDS], np.int32)
            rx, ry, rw, rh = cv2.boundingRect(r_eye_region)
            r_iris_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in R_IRIS_INDEXES])
            r_pupil = np.mean(r_iris_points, axis=0).astype(int)
            r_ratio_x = (r_pupil[0] - rx) / (rw + 1e-6)
            r_ratio_y = (r_pupil[1] - ry) / (rh + 1e-6)

            # 3. 시선 비율을 이용해 시선 끝점 계산 및 시각화
            # 눈동자 중심에서 화면 경계까지 선을 그림
            l_gaze_direction = (l_ratio_x - 0.5, l_ratio_y - 0.5)
            r_gaze_direction = (r_ratio_x - 0.5, r_ratio_y - 0.5)
            
            # 시선 벡터를 길게 늘여서 끝점 계산
            l_end_point = (int(l_pupil[0] + l_gaze_direction[0] * w * 2), int(l_pupil[1] + l_gaze_direction[1] * h * 2))
            r_end_point = (int(r_pupil[0] + r_gaze_direction[0] * w * 2), int(r_pupil[1] + r_gaze_direction[1] * h * 2))

            cv2.line(frame, tuple(l_pupil), l_end_point, (0, 255, 255), 2)  # 왼쪽 눈: 노란색
            cv2.line(frame, tuple(r_pupil), r_end_point, (255, 255, 0), 2)  # 오른쪽 눈: 청록색
            
            # '선' 교차 판정 로직
            if phone_coords is not None:
                if liang_barsky_clip(tuple(l_pupil), l_end_point, phone_coords) or \
                   liang_barsky_clip(tuple(r_pupil), r_end_point, phone_coords):
                    is_looking_at_phone_now = True

    # 2초 주시 감지 및 알림
    if is_looking_at_phone_now:
        if gaze_start_time is None: gaze_start_time = time.time()
        elif time.time() - gaze_start_time >= GAZE_DURATION_THRESHOLD:
            is_looking_continuously = True
    else:
        gaze_start_time = None
        is_looking_continuously = False
        
    # 최종 시각화
    if phone_coords is not None:
        box_color = (0, 0, 255) if is_looking_continuously or (is_looking_at_phone_now and gaze_start_time is not None) else (255, 0, 255)
        cv2.rectangle(frame, tuple(phone_coords[0:2]), tuple(phone_coords[2:4]), box_color, 2)

    if is_looking_continuously:
        cv2.putText(frame, "WARNING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        winsound.Beep(1200, 200)

    cv2.imshow(main_window_name, frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()