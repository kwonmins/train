import requests
import tarfile
import os

# 저장할 폴더 생성
os.makedirs("ffhq-dataset", exist_ok=True)

# FFHQ 데이터셋 URL
url = "https://github.com/NVlabs/ffhq-dataset/releases/download/v1.0/ffhq-images.tar"
file_path = "ffhq-dataset/ffhq-images.tar"

# 파일 다운로드
print("Downloading FFHQ dataset...")
response = requests.get(url, stream=True)
with open(file_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB씩 저장
        file.write(chunk)

# 압축 해제
print("Extracting FFHQ dataset...")
with tarfile.open(file_path, "r") as tar:
    tar.extractall(path="ffhq-dataset")

print("FFHQ dataset is ready!")
