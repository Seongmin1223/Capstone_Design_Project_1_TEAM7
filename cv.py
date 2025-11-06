from ultralytics import YOLO

# ----------------------------
# 경로 설정 (Windows 환경)
# ----------------------------
pt_model_path = r"C:\Users\nhj43\OneDrive\바탕 화면\convert\new3\bestf.pt"  # 원본 학습 모델
onnx_output_path = r"C:\Users\nhj43\OneDrive\바탕 화면\convert\new3\best_op12.onnx"  # 변환 후 저장될 파일

# ----------------------------
# 모델 로드 및 변환
# ----------------------------
model = YOLO(pt_model_path)

model.export(
    format="onnx",     # ONNX 형식
    opset=12,          # opset version 12
    simplify=True,     # 그래프 단순화
    dynamic=False      # 고정 입력 크기 (True면 가변 입력)
)

print(f"\n✅ ONNX 변환 완료!\n출력 경로: {onnx_output_path}")
