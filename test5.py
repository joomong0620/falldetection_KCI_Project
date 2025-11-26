import csv
import os
from ultralytics import YOLO

# ✅ 모델 로드
model = YOLO('yolov8s-pose.pt')  # 또는 너가 훈련한 best.pt

# ✅ 데이터 경로
val_dir = 'D:/falldetection/dataset_split/images/val'
output_csv = 'pose_as_gt.csv'

# ✅ 결과 저장 리스트 초기화
gt_labels = []

# ✅ 모든 이미지에 대해 추론
for img_file in sorted(os.listdir(val_dir)):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join(val_dir, img_file)
        results = model(img_path)

        # ✅ Keypoints 수집
        keypoints = results[0].keypoints

        # 간단한 낙상 판별 로직 예시 (관절이 존재하는지 여부만 판단)
        label = 'fallen' if keypoints.shape[1] > 0 else 'normal'

        # ✅ 프레임명과 라벨 기록
        frame_name = os.path.splitext(img_file)[0]
        gt_labels.append([frame_name, label])

# ✅ CSV 저장
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'label'])
    writer.writerows(gt_labels)

print(f"✅ {output_csv} 생성 완료! 총 {len(gt_labels)}개 프레임 라벨링됨.")
