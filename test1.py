import os
import csv

# 📂 YOLOv8 Pose 결과 .txt 경로
label_dir = r'D:\falldetection\runs\pose\predict2\labels'
output_csv = 'fall_pose_results.csv'

results = []


# ✅ 완화된 낙상 판단 기준 함수
def is_fallen(keypoints):
    if len(keypoints) < 13:
        return False  # 판단할 관절 부족 시 normal 처리

    try:
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        vertical_diff = abs(shoulder_y - hip_y)

        shoulder_x_dist = abs(left_shoulder[0] - right_shoulder[0])
        hip_x_dist = abs(left_hip[0] - right_hip[0])

        # 💡 기준 완화: 40 → 60, 60 → 30
        if vertical_diff < 60 and (shoulder_x_dist > 30 or hip_x_dist > 30):
            return True
    except Exception as e:
        print(f"⚠️ 낙상 판단 오류: {e}")

    return False


# 📄 Pose .txt 파일 순회
for filename in os.listdir(label_dir):
    if not filename.endswith('.txt'):
        continue

    file_path = os.path.join(label_dir, filename)

    with open(file_path, 'r') as f:
        line = f.readline().strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 4:
            print(f"⚠️ {filename} - 데이터 부족")
            continue

        try:
            parts = list(map(float, parts))
            keypoints = []
            for i in range(1, len(parts), 3):  # parts[0]은 class
                if i + 2 < len(parts):
                    x = parts[i]
                    y = parts[i + 1]
                    conf = parts[i + 2]
                    keypoints.append([x, y, conf])

            if len(keypoints) >= 12:
                fallen = is_fallen(keypoints)
                results.append((filename.replace('.txt', ''), 'fallen' if fallen else 'normal'))
            else:
                print(f"⚠️ {filename} - 관절 부족 ({len(keypoints)}개), 스킵")
        except Exception as e:
            print(f"⚠️ {filename} - 파싱 오류: {e}")

# ✅ 결과 CSV로 저장
try:
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'label'])
        writer.writerows(results)
    print(f"\n✅ fall_pose_results.csv 생성 완료! 총 {len(results)}개 프레임 분류됨.")
except PermissionError:
    print("❌ fall_pose_results.csv 파일이 열려 있어서 저장 실패! 닫고 다시 시도해줘.")
