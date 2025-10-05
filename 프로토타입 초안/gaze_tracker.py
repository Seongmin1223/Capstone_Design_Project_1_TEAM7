# -*- coding: utf-8 -*-
#2025/10/05 시선처리 + 객체 탐지 프로토타입

import cv2 #open_cv용
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import winsound
import math
import collections #시선 좌표 저장용

# ----------------------------------------------------------------------------
# 1. 초기 설정 (Initialization)
# ----------------------------------------------------------------------------

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# YOLOv8 모델 로드
yolo_model = YOLO('yolov8n.pt')
CELLPHONE_CLASS_ID = 67

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")
    
# [추가] 시선 안정화를 위한 deque 생성
# 최근 5개 프레임의 시선 끝점을 저장하여 평균내기용
GAZE_HISTORY_LENGTH = 5
gaze_history = collections.deque(maxlen=GAZE_HISTORY_LENGTH)

# ----------------------------------------------------------------------------
# 2. 교차 판정 함수 (Intersection Logic - 이전과 동일)
# ----------------------------------------------------------------------------
def line_intersects_rect(p1, p2, rect):
    try:
        x_min, y_min, x_max, y_max = rect
        if (p1[0] > x_max and p2[0] > x_max) or \
           (p1[0] < x_min and p2[0] < x_min) or \
           (p1[1] > y_max and p2[1] > y_max) or \
           (p1[1] < y_min and p2[1] < y_min):
            return False
        m = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else float('inf')
        for y_boundary in [y_min, y_max]:
            if m != 0:
                x_intersect = (y_boundary - p1[1]) / m + p1[0]
                if x_min <= x_intersect <= x_max:
                    if min(p1[1], p2[1]) <= y_boundary <= max(p1[1], p2[1]):
                        return True
        for x_boundary in [x_min, x_max]:
            if m != float('inf'):
                y_intersect = m * (x_boundary - p1[0]) + p1[1]
                if y_min <= y_intersect <= y_max:
                     if min(p1[0], p2[0]) <= x_boundary <= max(p1[0], p2[0]):
                        return True
    except ZeroDivisionError: return False
    return False

# ----------------------------------------------------------------------------
# 3. 메인 루프 (Main Loop)
# ----------------------------------------------------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("카메라 프레임을 무시합니다.")
        continue

    frame.flags.writeable = False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(rgb_frame)
    frame.flags.writeable = True
    
    frame_height, frame_width, _ = frame.shape
    phone_coords = None
    is_looking_at_phone = False
    
    # --- YOLO 휴대폰 탐지 ---
    results_yolo = yolo_model(frame, classes=[CELLPHONE_CLASS_ID], verbose=False)
    if len(results_yolo) > 0 and len(results_yolo[0].boxes) > 0:
        phone_coords = results_yolo[0].boxes[0].xyxy[0].cpu().numpy().astype(int)

    # --- MediaPipe 시선 계산 및 시각화 ---
    if results_face.multi_face_landmarks:
        landmarks = results_face.multi_face_landmarks[0].landmark
        
        PUPIL_INDEX = 473
        EYE_LEFT_CORNER_INDEX = 362
        EYE_RIGHT_CORNER_INDEX = 263

        pupil_pos = (int(landmarks[PUPIL_INDEX].x * frame_width), int(landmarks[PUPIL_INDEX].y * frame_height))
        left_corner_pos = (int(landmarks[EYE_LEFT_CORNER_INDEX].x * frame_width), int(landmarks[EYE_LEFT_CORNER_INDEX].y * frame_height))
        right_corner_pos = (int(landmarks[EYE_RIGHT_CORNER_INDEX].x * frame_width), int(landmarks[EYE_RIGHT_CORNER_INDEX].y * frame_height))

        eye_center_pos = (int((left_corner_pos[0] + right_corner_pos[0]) / 2), int((left_corner_pos[1] + right_corner_pos[1]) / 2))
        
        gaze_vec_x = pupil_pos[0] - eye_center_pos[0]
        gaze_vec_y = pupil_pos[1] - eye_center_pos[1]
        
        # 민감도 조절 파라미터 이걸로 선 길이 조절하면 됩니다!!!
        SENSITIVITY = 80 #30했는데 짧아서 바꿨는데 크게 다른지 모르겠음
        gaze_end_point = (int(eye_center_pos[0] + gaze_vec_x * SENSITIVITY), int(eye_center_pos[1] + gaze_vec_y * SENSITIVITY))

        # 계산된 끝점을 히스토리에 추가
        gaze_history.append(gaze_end_point)
        
        # 히스토리 평균값으로 부드러운 끝점 생성
        if len(gaze_history) > 0:
            avg_x = sum(p[0] for p in gaze_history) / len(gaze_history)
            avg_y = sum(p[1] for p in gaze_history) / len(gaze_history)
            smooth_gaze_end_point = (int(avg_x), int(avg_y))

            # 시각화 및 교차 판정에 부드러운 끝점 사용
            cv2.line(frame, eye_center_pos, smooth_gaze_end_point, (0, 255, 0), 2)
            cv2.circle(frame, eye_center_pos, 3, (0, 0, 255), -1)

            if phone_coords is not None:
                px, py = smooth_gaze_end_point # raw 값 대신 부드러운 값으로 판정
                x_min, y_min, x_max, y_max = phone_coords
                if x_min < px < x_max and y_min < py < y_max:
                    is_looking_at_phone = True

    # --- 최종 시각화 및 경고 ---
    if phone_coords is not None:
        box_color = (0, 0, 255) if is_looking_at_phone else (255, 0, 255)
        cv2.rectangle(frame, (phone_coords[0], phone_coords[1]), (phone_coords[2], phone_coords[3]), box_color, 2)

    if is_looking_at_phone:
        cv2.putText(frame, "LOOKING AT PHONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        winsound.Beep(1200, 100)

    cv2.imshow('MediaPipe Gaze and YOLO Phone Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- 종료 처리 ---
cap.release()
cv2.destroyAllWindows()
face_mesh.close()