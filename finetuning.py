# finetune_driver_phone.py
from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 사전 학습된 YOLOv8n 모델 불러오기
    # 새로운 데이터셋으로 학습할 때는 항상 공식 사전학습 모델로 시작하는 것이 좋습니다.
    model = YOLO("yolov8n.pt")

    # 2. 모델 학습(Fine-tuning) 시작
    # 이전 학습에서 mAP 성능이 잘 나왔던 핵심 파라미터들을 적용했습니다.
    model.train(
        # --- 기본 설정 ---
        data="data.yaml",           # 데이터셋 정보가 담긴 yaml 파일 경로
        imgsz=640,                  # 학습 이미지 사이즈
        batch=6,                    # 배치 크기 (GPU 메모리에 맞게 유지)
        epochs=100,                 # 💡 학습 횟수 (Epochs): 충분히 학습되도록 100으로 설정
        device=0,                   # 사용할 GPU 번호 (0번 GPU)
        workers=4,                  # 데이터 로딩에 사용할 CPU 스레드 수

        # --- 성능 향상을 위한 핵심 파라미터 ---
        lr0=0.0001,                 # 💡 학습률 (Learning Rate): Fine-tuning에 적합하게 낮은 값으로 설정
        augment=True,               # 💡 데이터 증강 (Augmentation): 과적합을 방지하고 일반화 성능 향상
        freeze=10,                  # 💡 백본 동결 (Freeze): 모델의 0~9번 레이어를 얼려서 안정적인 특징 추출을 보장
        
        # --- 기타 설정 ---
        optimizer="AdamW",          # 최적화 함수
        project="runs/train",       # 학습 결과가 저장될 기본 폴더
        name="driver_phone_train_v1", # 이번 학습 세션의 이름 (결과 폴더명)
        exist_ok=True,              # 같은 이름의 폴더가 있어도 덮어쓰기 허용
        val=True                    # 매 epoch마다 검증 데이터셋으로 성능을 평가
    )
    
    print("✅ 모델 학습 완료!")