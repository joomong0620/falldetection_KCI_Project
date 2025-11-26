# import os
# from ultralytics import YOLO
#
# # ✅ 모델 로드
# model = YOLO('yolov8s-pose.pt')  # 또는 best.pt
#
# # ✅ 데이터 경로
# val_dir = 'D:/falldetection/dataset_split/images/val'
# output_dir = 'pose_vis_results'
# os.makedirs(output_dir, exist_ok=True)
#
# # ✅ 이미지별 시각화 예측 및 저장
# for img_file in sorted(os.listdir(val_dir)):
#     if img_file.endswith(('.jpg', '.png')):
#         img_path = os.path.join(val_dir, img_file)
#         # 시각화 예측 수행 (save=True가 저장해줌)
#         model.predict(img_path, save=True, project=output_dir, name='vis')
#
# print(f"✅ 시각화된 결과가 {output_dir}/vis 폴더에 저장되었습니다.")

from ultralytics import YOLO
import os

import os
import csv
from ultralytics import YOLO

# ✅ 모델 로드
model = YOLO('yolov8s-pose.pt')

# ✅ 이미지 폴더 경로
val_dir = 'D:/falldetection/dataset_split/images/val'

# ✅ 임계값 (필요에 따라 조절)
CONFIDENCE_THRESHOLD = 0.5

# ✅ 결과 수집
total = 0
correct = 0

for img_file in sorted(os.listdir(val_dir)):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join(val_dir, img_file)
        results = model(img_path)

        keypoints = results[0].keypoints

        # ✅ 관절 개수로 낙상 여부 단순 판별 (네가 설계한 논리로 바꿔도 됨)
        label = 'fallen' if keypoints.shape[1] > 0 else 'normal'

        # ✅ Ground Truth는 너가 만든 yolo_as_gt.csv 또는 기타 소스에서 불러와야 함
        # 여기선 예시로 'fallen'만 정답이라고 가정
        true_label = 'fallen'

        # ✅ 평가
        total += 1
        if label == true_label:
            correct += 1

# ✅ 최종 결과 출력
accuracy = correct / total * 100
print(f"✅ 총 {total}개 이미지 중 {correct}개 정답 맞춤 (정확도: {accuracy:.2f}%)")
