import onnxruntime as ort
import numpy as np
import cv2

# 1. 모델 파일 경로
MODEL_PATH = 'last.onnx'

# 2. InferenceSession 생성
session = ort.InferenceSession(MODEL_PATH)

# 3. 입력 이름 확인
input_name = session.get_inputs()[0].name
print(f"Input tensor name: {input_name}")

# 4. 테스트용 입력 데이터 (이미지 예시)
# 예: YOLO 모델이라면 640x640으로 리사이즈 후 float32로 변환
image_path = 'test.jpg'  # 테스트 이미지 파일 경로
image = cv2.imread(image_path)
image = cv2.resize(image, (640, 640))
input_data = image.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) / 255.0

# 5. 추론 실행
outputs = session.run(None, {input_name: input_data})

# 6. 출력 결과 확인
print("\n===== MODEL OUTPUT =====")
for i, output in enumerate(outputs):
    print(f"Output {i} shape: {output.shape}")
    print(output[:5])  # 일부 결과 출력

# 7. 후처리 (YOLO 형식일 경우 박스 시각화 등 추가 가능)
# 예시: outputs[0] 형태에 따라 시각화 가능
# cv2.imshow('Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
