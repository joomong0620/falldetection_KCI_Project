import os
import csv

# YOLO predict 결과 라벨 폴더 경로
label_dir = r'D:\falldetection\runs\detect\predict\labels'
output_csv = 'yolo_as_gt.csv'

# 결과 저장 리스트
gt_labels = []

for file in os.listdir(label_dir):
    if not file.endswith('.txt'):
        continue

    path = os.path.join(label_dir, file)
    with open(path, 'r') as f:
        lines = f.readlines()

    # 객체가 하나라도 있으면 "fallen"으로 간주
    label = 'fallen' if len(lines) > 0 else 'normal'

    # 프레임 이름 = 파일명에서 .txt 제거
    frame_name = file.replace('.txt', '')
    gt_labels.append((frame_name, label))

# CSV 저장
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'label'])
    writer.writerows(gt_labels)

print(f"✅ yolo_as_gt.csv 생성 완료! 총 {len(gt_labels)}개 프레임 라벨링됨.")
