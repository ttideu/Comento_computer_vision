# train.py
from ultralytics import YOLO

def train_model():
    # 1. 모델 로드
    model = YOLO("yolov8n.pt") 

    # 2. 학습 시작
    print("=== 모델 학습 시작 Epochs: 50 ===")
    model.train(
        data="data.yaml",      
        epochs=50,             # 학습 횟수
        imgsz=640,             # 이미지 크기
        batch=16,
        
        # 데이터 증강 (Augmentation)
        augment=True,          
        degrees=15.0,          # 회전
        fliplr=0.5,            # 좌우 반전
        hsv_v=0.3,             # 밝기 변화
        mosaic=0.0,            # 모자이크
    )
    print("=== 학습 완료 ===")

if __name__ == "__main__":
    train_model()
