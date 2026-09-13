from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
NUTRITION_DB_PATH = ROOT / "data" / "nutrition_db.json"
MODEL_DIR = ROOT / "artifacts" / "models"
REPORT_DIR = ROOT / "artifacts" / "reports"

IMAGE_SIZE = 224

# Food-101에서 사용하는 10개 클래스.
# 서로 시각적으로 어느 정도 구분되면서 영양정보 연결이 쉬운 메뉴 위주로 선정.
TARGET_CLASSES = [
    "cheesecake",
    "fried_rice",
    "hamburger",
    "hot_dog",
    "ice_cream",
    "pizza",
    "spaghetti_bolognese",
    "steak",
    "sushi",
    "chicken_wings",
]
