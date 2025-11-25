import os
from ultralytics import YOLO

# ----------------------------
# 환경 설정
# ----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 사용

data_yaml = "/home/driver/workspace/venvs/gaze_track/YOLO_F8/data.yaml"
pretrained_weights = "yolov8n.pt"  # 기본 pretrained

# ----------------------------
# 학습 파라미터
# ----------------------------
batch_size = 16
epochs = 150
img_size = 640
lr = 0.001

# ----------------------------
# YOLO 모델 불러오기
# ----------------------------
model = YOLO(pretrained_weights)

# ----------------------------
# 학습 실행 (폰 최적화 버전)
# ----------------------------
model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    lr0=lr,
    workers=0,
    device=0,
    pretrained=True,
    name="phone_only_training",

    augment=True,
    hsv_h=0.015,       # 색조 변화 → 빛/반사 대응
    hsv_s=0.7,         # 채도 변화 → 다양한 조명 대응
    hsv_v=0.4,         # 밝기 변화 → 밤/실내 환경
    translate=0.1,     # 위치 이동 → 손에 의해 가림 보정
    scale=0.5,         # 크기 다양화 → 작은 폰 강화
    fliplr=0.5,        # 좌우 반전 → 운전자/조수석 대응
    erasing=0.4,       # 부분 가림 → 손이 폰 가리는 상황 대비
    auto_augment="randaugment",  # 강력한 랜덤 강화

    mosaic=1.0,        # mosaic 사용
    close_mosaic=10,   # epoch 10 이후 mosaic OFF (안정 수렴)
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,

    deterministic=True
)

print("✅ 학습 시작됨. SSH 끊겨도 계속 진행.")
print("📁 결과 저장 폴더: runs/train/phone_only_training")
