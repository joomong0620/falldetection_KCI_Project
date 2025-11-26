from sklearn.metrics import classification_report, confusion_matrix
import os

# 📂 Pose 결과 경로
pose_label_path = 'fall_pose_results.csv'
yolo_label_path = 'yolo_as_gt.csv'

# ✅ CSV를 dict로 읽는 함수
def load_labels_as_dict(csv_path):
    data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # 첫 줄(header) 스킵
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            frame, label = parts
            data[frame.strip()] = label.strip()
    return data

# ✅ 두 결과를 dict로 불러옴
pose_dict = load_labels_as_dict(pose_label_path)
yolo_dict = load_labels_as_dict(yolo_label_path)

# ✅ 공통 frame만 추출
common_keys = set(pose_dict.keys()) & set(yolo_dict.keys())
print(f"🔍 공통 비교 대상 frame 수: {len(common_keys)}")

# ✅ 정답(y_true) / 예측값(y_pred) 리스트 생성
y_true = [yolo_dict[k] for k in sorted(common_keys)]
y_pred = [pose_dict[k] for k in sorted(common_keys)]

# ✅ 성능 비교 출력
print("\n=== YOLO vs YOLO+Pose 성능 비교 결과 ===")
print(classification_report(y_true, y_pred, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
