from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class NutritionEstimate:
    food_key: str
    display_name: str
    grams: float
    calories: float
    carbs: float
    protein: float
    fat: float


class NutritionDB:
    """100g 기준 영양정보를 실제 섭취량(g)에 맞춰 환산한다."""

    def __init__(self, json_path: str | Path):
        self.path = Path(json_path)
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.meta = payload.get("meta", {})
        self.foods: Dict[str, dict] = payload["foods"]

    def contains(self, food_key: str) -> bool:
        return food_key in self.foods

    def get(self, food_key: str) -> dict:
        if food_key not in self.foods:
            raise KeyError(f"영양 DB에 없는 음식입니다: {food_key}")
        return self.foods[food_key]

    def default_serving_grams(self, food_key: str) -> int:
        return int(self.get(food_key)["default_serving_g"])

    def estimate(self, food_key: str, grams: float) -> NutritionEstimate:
        if grams <= 0:
            raise ValueError("grams는 0보다 커야 합니다.")

        item = self.get(food_key)
        factor = grams / 100.0
        per100 = item["per_100g"]

        return NutritionEstimate(
            food_key=food_key,
            display_name=item["display_name"],
            grams=float(grams),
            calories=round(per100["calories"] * factor, 1),
            carbs=round(per100["carbs_g"] * factor, 1),
            protein=round(per100["protein_g"] * factor, 1),
            fat=round(per100["fat_g"] * factor, 1),
        )
