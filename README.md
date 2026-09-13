# Smart Calorie

**MobileNetV3-Small 기반 음식 이미지 분류 및 섭취량 보정형 영양정보 추정 웹앱**

Food-101에서 10개 음식 클래스를 구성해 전이학습하고, 사용자가 업로드한 음식 이미지를 분류합니다. 이후 사용자가 입력한 예상 섭취량(g)을 기준으로 100g 기준 영양정보를 환산해 칼로리와 탄수화물·단백질·지방을 제공합니다.

> Food-101은 이미지 분류 데이터셋이므로 객체 탐지용 임의 Bounding Box를 만들지 않고, 데이터 구조에 맞춰 Classification 문제로 구성했습니다.

## 주요 결과

| 항목 | 결과 |
|---|---:|
| Test Accuracy | **87.6%** |
| Macro Precision | **87.60%** |
| Macro Recall | **87.60%** |
| Macro F1 | **87.48%** |
| Best Validation Accuracy | **90.4%** |
| Test Images | 500 |
| Classes | 10 |

ONNX FP32 변환 후에도 Test Accuracy 87.6%를 유지했습니다. 동일 환경의 CPU 추론 벤치마크에서는 PyTorch FP32 중앙값 6.7009 ms, ONNX Runtime FP32 1.3187 ms로 측정되어 약 5.08× 빠른 추론 시간을 확인했습니다.

INT8 정적 양자화도 실험했지만 Test Accuracy가 11.2%까지 하락해 최종 적용에서 제외했습니다. 정확도 손실을 확인한 뒤 FP32 모델을 유지하는 방향으로 결정했습니다.

## 프로젝트 구조

```text
smart_calorie_ai/
├─ app.py                    # Streamlit 웹앱
├─ README.md
├─ requirements.txt
├─ data/
│  └─ nutrition_db.json      # 100g 기준 영양정보
├─ src/
│  ├─ config.py              # 경로/클래스 설정
│  ├─ data.py                # 데이터 로더/전처리
│  ├─ model.py               # MobileNetV3-Small 모델
│  ├─ inference.py           # PyTorch/ONNX 추론
│  └─ nutrition.py           # 섭취량 기반 영양정보 계산
├─ scripts/                  # 학습·평가·배포 실험 재현용 코드
│  ├─ prepare_food101.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ export_onnx.py
│  ├─ evaluate_onnx.py
│  ├─ quantize_onnx.py
│  └─ benchmark.py
├─ tests/
│  └─ test_nutrition.py
└─ artifacts/
   ├─ models/
   │  ├─ best_model.pth
   │  └─ classes.json
   └─ reports/
      ├─ test_metrics.json
      ├─ confusion_matrix.png
      ├─ onnx_fp32_metrics.json
      ├─ onnx_int8_metrics.json
      └─ final_fp32_benchmark.json
```

루트에는 실제 실행 진입점인 `app.py`만 두고, 학습·평가·변환·벤치마크 코드는 `scripts/`에 분리했습니다. 데이터셋과 중간 실험 모델은 저장소에서 제외했습니다.

## 대상 음식 10종

`cheesecake`, `chicken_wings`, `fried_rice`, `hamburger`, `hot_dog`, `ice_cream`, `pizza`, `spaghetti_bolognese`, `steak`, `sushi`

## 실행

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

저장소에는 학습된 PyTorch FP32 모델이 포함되어 있어 데이터셋을 다시 학습하지 않아도 웹앱을 실행할 수 있습니다.

## 학습 및 평가 재현

Food-101 10-class subset 구성:

```powershell
python -m scripts.prepare_food101
```

학습 및 Test 평가:

```powershell
python -m scripts.train
python -m scripts.evaluate
```

ONNX 변환 및 평가:

```powershell
python -m scripts.export_onnx
python -m scripts.evaluate_onnx --model artifacts/models/food_classifier.onnx --name onnx_fp32
```

INT8 정적 양자화 실험:

```powershell
python -m scripts.quantize_onnx
python -m scripts.evaluate_onnx --model artifacts/models/food_classifier_int8.onnx --name onnx_int8
```

CPU 벤치마크:

```powershell
python -m scripts.benchmark
```

## 데이터 및 학습 방식

- Food-101 중 10개 클래스 사용
- 클래스당 Train 200 / Validation 50 / Test 50
- 총 3,000장 구성
- ImageNet 사전학습 MobileNetV3-Small 사용
- 초기 Classifier Head 학습 후 전체 네트워크 Fine-tuning
- Validation Accuracy 기준 최적 가중치 저장
- 독립 Test split으로 Accuracy / Precision / Recall / F1 평가

원본 Food-101 이미지는 저장소 용량과 중복 배포를 줄이기 위해 포함하지 않았으며 `scripts/prepare_food101.py`로 다시 구성할 수 있습니다.

## 영양정보 계산 방식

단일 RGB 이미지에서 실제 음식 중량을 신뢰성 있게 추정하기 어렵기 때문에 이미지에서는 음식 종류만 분류합니다. 이후 사용자가 섭취량(g)을 보정하면 100g 기준 영양정보를 비례 환산합니다.

따라서 이 프로젝트의 출력은 의료·영양 진단값이 아니라 **사용자 입력 섭취량에 기반한 예상 영양정보**입니다.
