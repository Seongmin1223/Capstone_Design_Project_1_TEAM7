import onnxruntime as ort
import numpy as np
import cv2, glob, os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# ONNX 모델 로드
session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

image_dir = "images/"
label_dir = "labels/"

y_true_all, y_score_all = [], []

for img_path in glob.glob(os.path.join(image_dir, "*.jpg")):
    img = cv2.imread(img_path)
    img_input = cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    outputs = session.run(None, {input_name: img_input})[0]
    boxes, scores, classes = outputs[..., :4], outputs[..., 4], outputs[..., 5]

    label_path = os.path.join(label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        continue
    gt = np.loadtxt(label_path).reshape(-1, 5)

    y_true = np.zeros_like(scores)
    y_true[:len(gt)] = 1
    y_true_all.extend(y_true)
    y_score_all.extend(scores)

precision, recall, _ = precision_recall_curve(y_true_all, y_score_all)
ap = average_precision_score(y_true_all, y_score_all)
print(f"Average Precision (AP): {ap:.4f}")

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='b', lw=2, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (ONNX model)')
plt.legend()
plt.grid(True)
plt.show()
