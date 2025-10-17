# update_labels.py
import os

# 라벨 파일이 있는 폴더 경로 (C:\YOLOT2 기준)
label_folders = [
    'train/labels',
    'valid/labels',
    'test/labels'
]

# 통계를 위한 카운터
removed_count = 0
processed_files = 0

print("YOLO 라벨 파일 필터링을 시작합니다...")
print("-" * 30)

# 각 폴더를 순회
for folder in label_folders:
    # 폴더가 존재하는지 확인
    if not os.path.exists(folder):
        print(f"⚠️  경고: '{folder}' 폴더를 찾을 수 없어 건너뜁니다.")
        continue

    print(f"'{folder}' 폴더를 처리 중입니다...")
    
    # 폴더 안의 모든 .txt 파일을 대상으로 작업
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            lines_to_keep = []
            file_modified = False
            
            try:
                # 파일을 열어 유효한 라인만 골라내기
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if not parts: continue # 빈 줄은 건너뛰기
                        
                        # 클래스 ID가 0 또는 1인 줄만 lines_to_keep 리스트에 추가
                        class_id = int(parts[0])
                        if class_id in [0, 1]:
                            lines_to_keep.append(line)
                        else:
                            # 0 또는 1이 아닌 라인은 삭제 대상
                            removed_count += 1
                            file_modified = True
                
                # 파일 내용이 변경된 경우에만 덮어쓰기
                if file_modified:
                    with open(filepath, 'w') as f:
                        f.writelines(lines_to_keep)
                
                processed_files += 1

            except Exception as e:
                print(f"❌ 오류: '{filepath}' 파일 처리 중 문제가 발생했습니다 - {e}")

print("-" * 30)
print(f"✅ 작업 완료!")
print(f"총 {processed_files}개의 파일을 검사하여 {removed_count}개의 불필요한 라벨을 삭제했습니다.")