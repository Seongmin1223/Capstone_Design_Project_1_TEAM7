import os
# YOLO_F 데이터셋 경로
DATASET_DIR = "/home/driver/workspace/venvs/gaze_track/YOLO_F"

# 남길 클래스 이름
KEEP_CLASS = "phone"
CLASS_NAMES = ['6', 'phone', 'undefined']

KEEP_CLASS_NUM = CLASS_NAMES.index(KEEP_CLASS)

SPLITS = ['train', 'valid', 'test']

for split in SPLITS:
    images_dir = os.path.join(DATASET_DIR, split, 'images')
    labels_dir = os.path.join(DATASET_DIR, split, 'labels')

    print(f"=== {split} 처리 시작 ===")

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(labels_dir, label_file)
        # phone 클래스만 남기기
        with open(label_path, 'r') as f:
            lines = f.readlines()
        new_lines = [line for line in lines if line.split()[0] == str(KEEP_CLASS_NUM)]
        with open(label_path, 'w') as f:
            f.writelines(new_lines)

    # 라벨 없는 이미지 삭제
    for image_file in os.listdir(images_dir):
        if not image_file.endswith(('.jpg', '.png', '.jpeg')):
            continue
        label_file = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')
        if not os.path.exists(label_file) or os.path.getsize(label_file) == 0:
            os.remove(os.path.join(images_dir, image_file))
            print(f"삭제됨: {image_file}")

print("=== 모든 SPLIT 정리 완료 ===")
