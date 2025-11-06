import os

DATASET_DIR = "/home/driver/workspace/venvs/gaze_track/YOLO_F"
SPLITS = ['train', 'valid', 'test']

for split in SPLITS:
    labels_dir = os.path.join(DATASET_DIR, split, 'labels')
    print(f"=== {split} 라벨 클래스 번호 변경 시작 ===")

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 모든 라인에서 클래스 번호 1 -> 0
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts[0] == '1':
                parts[0] = '0'
            new_lines.append(' '.join(parts) + '\n')

        with open(label_path, 'w') as f:
            f.writelines(new_lines)

print("=== 모든 SPLIT 라벨 클래스 번호 변경 완료 ===")
