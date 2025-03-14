import os
import zipfile
import argparse
from tqdm import tqdm
import cv2
import numpy as np

def make_zip_dataset(source_folder, output_zip, image_size=256):
    """
    지정된 폴더의 이미지들을 StyleGAN2-ADA 학습을 위해 ZIP 파일로 변환.
    비율을 유지하면서 정사각형(256x256)으로 변환 후 압축.
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_STORED) as zipf:
        for root, _, files in tqdm(os.walk(source_folder), desc="Processing Images"):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    
                    # 이미지 로드
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"⚠ {img_path} 로드 실패, 스킵합니다.")
                        continue

                    h, w, _ = img.shape
                    max_dim = max(w, h)

                    # 정사각형 크기로 패딩 추가
                    pad_top = (max_dim - h) // 2
                    pad_bottom = max_dim - h - pad_top
                    pad_left = (max_dim - w) // 2
                    pad_right = max_dim - w - pad_left

                    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))

                    # 256x256으로 리사이즈
                    img_resized = cv2.resize(img_padded, (image_size, image_size))

                    # 임시 파일로 저장 후 ZIP 압축
                    temp_img_path = os.path.join(source_folder, f"temp_{file}")
                    cv2.imwrite(temp_img_path, img_resized)

                    # ZIP 파일에 추가
                    zipf.write(temp_img_path, arcname=f"dataset/{file}")

                    # 임시 파일 삭제
                    os.remove(temp_img_path)

    print(f"✅ 정사각형 변환 + 압축 완료: {output_zip}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2-ADA 학습용 ZIP 데이터셋 변환")
    parser.add_argument('--source', type=str, required=True, help="원본 이미지 폴더 경로")
    parser.add_argument('--dest', type=str, required=True, help="출력 ZIP 파일 경로")
    parser.add_argument('--size', type=int, default=256, help="변환할 이미지 크기 (256 또는 512)")

    args = parser.parse_args()
    make_zip_dataset(args.source, args.dest, args.size)
