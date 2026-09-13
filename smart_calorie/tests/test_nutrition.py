from pathlib import Path

import pytest

from src.nutrition import NutritionDB


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "nutrition_db.json"


def test_100g_is_identity():
    db = NutritionDB(DB_PATH)
    raw = db.get("pizza")["per_100g"]
    estimate = db.estimate("pizza", 100)
    assert estimate.calories == raw["calories"]
    assert estimate.carbs == raw["carbs_g"]


def test_double_grams_double_calories():
    db = NutritionDB(DB_PATH)
    a = db.estimate("hamburger", 100)
    b = db.estimate("hamburger", 200)
    assert b.calories == pytest.approx(a.calories * 2)


def test_unknown_food_raises():
    db = NutritionDB(DB_PATH)
    with pytest.raises(KeyError):
        db.get("unknown_food")


def test_non_positive_grams_raises():
    db = NutritionDB(DB_PATH)
    with pytest.raises(ValueError):
        db.estimate("sushi", 0)


def test_all_target_classes_have_nutrition_entry():
    from src.config import TARGET_CLASSES

    db = NutritionDB(DB_PATH)
    missing = [name for name in TARGET_CLASSES if not db.contains(name)]
    assert missing == []
