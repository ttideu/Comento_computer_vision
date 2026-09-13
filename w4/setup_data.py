# setup_data.py
import os
import cv2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def setup_dataset():
    # 폴더 구조 생성
    paths = [
        'datasets/train/images', 'datasets/train/labels',
        'datasets/valid/images', 'datasets/valid/labels'
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)

    # Hugging Face Food-101 데이터 로드
    # streaming=True로 설정하여 메모리 효율적으로 필요한 만큼만 다운로드
    print("데이터셋 다운로드 중 (Target: 200 images)...")
    ds = load_dataset("ethz/food101", split='train', streaming=True)
    
    # 200장 추출 (기존 100장 -> 200장으로 증량)
    samples = list(ds.take(200)) 

    print("이미지 처리 및 라벨링 생성 중...")
    for i, item in enumerate(tqdm(samples)):
        # 1. 이미지 변환 (PIL -> OpenCV BGR)
        image = np.array(item['image'])
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 2. 크기 조정 (YOLO 입력 표준 640x640)
        image = cv2.resize(image, (640, 640))
        
        # 3. 데이터 분할 (8:2)
        if i < 160:
            base_path = 'datasets/train'
        else:
            base_path = 'datasets/valid'
        
        img_save_path = f'{base_path}/images/food_{i}.jpg'
        lbl_save_path = f'{base_path}/labels/food_{i}.txt'
        
        # 4. 이미지 저장
        cv2.imwrite(img_save_path, image)
        
        # 5. 라벨 파일 생성 (Dummy Label)
        # 이미지 중앙에 객체가 있다고 가정 (class 0)
        with open(lbl_save_path, 'w') as f:
            f.write("0 0.5 0.5 0.8 0.8") # class_id center_x center_y width height

    print("\n[완료] 데이터셋 구축이 끝났습니다.")
    print(f" - Train: 160장")
    print(f" - Valid: 40장")

if __name__ == "__main__":
    setup_dataset()
