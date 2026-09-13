# AI Computer Vision 기반 식단 관리 및 칼로리 분석 시스템

## 개요
YOLOv8 객체 탐지 모델과 OpenCV 이미지 처리 기술을 결합하여, 음식 이미지를 인식하고 영양 정보를 분석하는 AI 식단 관리 솔루션입니다.<br>
객체를 탐지하여 칼로리를 산출하며, Unit Test를 통해 비즈니스 로직의 안정성을 검증했습니다.

## 실행 순서
본 프로젝트는 아래 순서대로 실행해야 정상적으로 작동합니다.

1. **데이터셋 구축**: `setup_data.py`
   - Food-101 데이터 다운로드 및 640x640 전처리
2. **모델 학습**: `train.py`
   - YOLOv8 모델 학습
3. **로직 검증**: `test_calorie.py`
   - Pytest를 활용한 영양소 계산 로직 무결성 검증
4. **서비스 실행**: `main_image.py`
   - 최종 결과물 시연

## 구조
```text
📦 Project Root
 ┣ 📂 datasets            # 학습용 데이터셋 (setup_data.py 실행 시 생성)
 ┣ 📂 runs                # 학습 결과, 모델 가중치 및 그래프 (train.py 실행 시 생성)
 ┣ 📜 setup_data.py       # Food-101 데이터 다운로드, 리사이즈 및 라벨링
 ┣ 📜 train.py            # YOLOv8 모델 학습 설정 및 실행
 ┣ 📜 calorie_calc.py     # [Core] 영양소 계산기 및 브랜드 평균 DB 클래스
 ┣ 📜 test_calorie.py     # [Test] 계산 로직 검증을 위한 Unit Test (Pytest)
 ┣ 📜 main_image.py       # [Main] AI 객체 탐지 및 영양 정보 시각화
 ┣ 📜 data.yaml           # 데이터셋 경로 및 클래스 설정
 ┗ 📜 README.md           # 프로젝트 설명 문서
```

## 주요 기능
* **AI 객체 탐지**: YOLOv8n 모델을 활용한 음식 위치 및 종류 인식
* **영양 분석**: 메이저 브랜드 평균 데이터를 기반으로 칼로리/탄단지 계산
* **AR 시각화**: OpenCV를 활용하여 음식 위에 영양 정보를 오버레이
* **검증 자동화**: Pytest를 통한 로직 단위 테스트 수행

## 환경 및 설치
* Python 3.8+
* Ultralytics YOLOv8
* OpenCV, NumPy, Pytest, Hugging Face Datasets

### 필수 라이브러리 설치
```bash
pip install ultralytics opencv-python numpy pandas matplotlib datasets tqdm pytest

```



