import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# 📄 YOLO+Pose 예측 결과
pose_df = pd.read_csv('fall_pose_results.csv')

# 📄 YOLO 예측 결과를 GT로 사용한다고 가정
# 👉 같은 형식의 'frame,label' 형태로 yolo_results.csv 파일 있어야 함
# 예시로 만든다고 치면:
#   frame,label
#   frame00123,fallen
#   frame00124,normal

yolo_df = pd.read_csv('yolo_as_gt.csv')  # ← 이 파일만 네가 만들어줘야 함 (YOLO 결과 기반 GT)

# ✅ 프레임 이름 기준 정렬
pose_df = pose_df.sort_values('frame').reset_index(drop=True)
yolo_df = yolo_df.sort_values('frame').reset_index(drop=True)

# 🔁 교집합만 비교 (혹시 빠진 프레임 있을까봐)
merged = pd.merge(pose_df, yolo_df, on='frame', suffixes=('_pose', '_gt'))

# 🎯 비교
y_true = merged['label_gt']
y_pred = merged['label_pose']

print("=== YOLO vs YOLO+Pose 성능 비교 ===")
print(classification_report(y_true, y_pred, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("YOLO Pose 결과 프레임 수:", len(pose_df))
print("YOLO 결과 프레임 수:", len(yolo_df))
print("Merge 후 프레임 수:", len(merged))
