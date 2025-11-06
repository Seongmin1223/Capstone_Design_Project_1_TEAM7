import os
from ultralytics import YOLO

# ----------------------------
# 환경 설정
# ----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 사용
data_yaml = "/home/driver/workspace/venvs/gaze_track/YOLO_F/data.yaml"  # 데이터셋 yaml
pretrained_weights = "yolov8n.pt"  # YOLOv8n pretrained

# ----------------------------
# 학습 파라미터
# ----------------------------
batch_size = 16       # GPU 메모리 상황에 따라 줄일 수 있음
epochs = 150          # 필요하면 늘리기
img_size = 640
lr = 0.001

# ----------------------------
# YOLO 모델 불러오기
# ----------------------------
model = YOLO(pretrained_weights)  # pretrained weights 로드

# ----------------------------
# 학습
# ----------------------------
model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    lr0=lr,
    workers=0,         # DataLoader shared memory 문제 방지
    device=0,          # GPU 0
    pretrained=True,   # pretrained weights 활용
    name="phone_only_training"
)

print("학습 시작 완료. nohup 등으로 실행 시 SSH 연결 끊겨도 학습 계속됨.")
print("학습 로그와 가중치는 'runs/train/phone_only_training' 폴더에 저장됨.")
