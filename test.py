import os
import random
import shutil

# 📂 기본 경로 (images, labels 위치)
base_img_dir = r'D:\falldetection\processed\images'
base_lbl_dir = r'D:\falldetection\processed\labels'

# 📂 train/val 나눠서 저장할 경로
output_img_train = r'D:\falldetection\dataset_split\images\train'
output_img_val = r'D:\falldetection\dataset_split\images\val'
output_lbl_train = r'D:\falldetection\dataset_split\labels\train'
output_lbl_val = r'D:\falldetection\dataset_split\labels\val'

# 📂 폴더 생성
os.makedirs(output_img_train, exist_ok=True)
os.makedirs(output_img_val, exist_ok=True)
os.makedirs(output_lbl_train, exist_ok=True)
os.makedirs(output_lbl_val, exist_ok=True)

# 📄 전체 이미지 파일 리스트
image_files = [f for f in os.listdir(base_img_dir) if f.endswith('.jpg')]

# 🔀 랜덤 셔플
random.seed(42)  # 결과 재현성을 위해
random.shuffle(image_files)

# 📈 train/val 분할 비율
train_ratio = 0.8
train_size = int(len(image_files) * train_ratio)

train_files = image_files[:train_size]
val_files = image_files[train_size:]

# 📦 파일 복사 함수 (라벨 없으면 빈 파일 생성)
def copy_files(file_list, img_dest, lbl_dest):
    for img_file in file_list:
        img_src_path = os.path.join(base_img_dir, img_file)
        lbl_src_path = os.path.join(base_lbl_dir, img_file.replace('.jpg', '.txt'))

        img_dst_path = os.path.join(img_dest, img_file)
        lbl_dst_path = os.path.join(lbl_dest, img_file.replace('.jpg', '.txt'))

        # 이미지 복사
        shutil.copy(img_src_path, img_dst_path)

        # 라벨 복사 (없으면 빈 파일 생성)
        if os.path.exists(lbl_src_path):
            shutil.copy(lbl_src_path, lbl_dst_path)
        else:
            open(lbl_dst_path, 'w').close()

# 📦 복사 실행
copy_files(train_files, output_img_train, output_lbl_train)
copy_files(val_files, output_img_val, output_lbl_val)

print(f"✅ 전처리 완료! Train: {len(train_files)}개, Val: {len(val_files)}개")
