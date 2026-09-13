from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image

from src.config import MODEL_DIR, NUTRITION_DB_PATH
from src.inference import ONNXClassifier, TorchClassifier
from src.nutrition import NutritionDB


st.set_page_config(page_title="Smart Calorie AI", page_icon="🍽️", layout="centered")

st.markdown(
    """
    <style>
    .block-container {max-width: 900px; padding-top: 2rem;}
    .hero {padding: 1.2rem 1.4rem; border: 1px solid rgba(128,128,128,.25); border-radius: 16px; margin-bottom: 1rem;}
    .muted {opacity: .72; font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0">🍽️ Smart Calorie AI</h1>
      <p style="margin:.5rem 0 0 0">경량 이미지 분류 기반 음식 인식 · 섭취량 기반 영양정보 추정</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_classifier():
    int8_path = MODEL_DIR / "food_classifier_int8.onnx"
    onnx_path = MODEL_DIR / "food_classifier.onnx"
    classes_path = MODEL_DIR / "classes.json"
    torch_path = MODEL_DIR / "best_model.pth"

    if classes_path.exists():
        class_names = json.loads(classes_path.read_text(encoding="utf-8"))
        if int8_path.exists():
            return ONNXClassifier(int8_path, class_names), "ONNX Runtime INT8 (CPU)"
        if onnx_path.exists():
            return ONNXClassifier(onnx_path, class_names), "ONNX Runtime FP32 (CPU)"
    if torch_path.exists():
        return TorchClassifier(torch_path), "PyTorch"
    return None, None


nutrition_db = NutritionDB(NUTRITION_DB_PATH)
classifier, backend_name = load_classifier()

if classifier is None:
    st.error("학습된 모델이 없습니다.")
    st.code("python -m scripts.prepare_food101\npython -m scripts.train\npython -m scripts.evaluate\npython -m scripts.export_onnx")
    st.stop()

st.caption(f"Inference backend: {backend_name}")

uploaded = st.file_uploader("음식 사진을 업로드하세요", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="입력 이미지", use_container_width=True)

    with st.spinner("이미지를 분석하고 있습니다..."):
        preds = classifier.predict(image, top_k=3)

    top1 = preds[0]
    food = nutrition_db.get(top1.label)

    st.subheader("AI 인식 결과")
    col1, col2 = st.columns(2)
    col1.metric("예측 음식", food["display_name"])
    col2.metric("Confidence", f"{top1.confidence:.1%}")

    if top1.confidence < 0.55:
        st.warning("예측 확신도가 낮습니다. 음식이 화면에 크게 나오도록 다른 사진을 사용하는 것이 좋습니다.")

    with st.expander("Top-3 예측 보기"):
        for pred in preds:
            display = nutrition_db.get(pred.label)["display_name"]
            st.write(f"**{display}** — {pred.confidence:.1%}")

    st.subheader("섭취량 및 영양 추정")
    default_g = nutrition_db.default_serving_grams(top1.label)
    grams = st.slider(
        "예상 섭취량(g)",
        min_value=50,
        max_value=600,
        value=max(50, min(default_g, 600)),
        step=10,
        help="사진 한 장만으로 실제 중량을 정확히 알 수 없으므로 사용자가 섭취량을 보정합니다.",
    )

    estimate = nutrition_db.estimate(top1.label, grams)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("칼로리", f"{estimate.calories:.0f} kcal")
    c2.metric("탄수화물", f"{estimate.carbs:.1f} g")
    c3.metric("단백질", f"{estimate.protein:.1f} g")
    c4.metric("지방", f"{estimate.fat:.1f} g")

    st.info(nutrition_db.meta["disclaimer"])
else:
    st.write("사진을 한 장 업로드하면 음식 종류를 분류하고, 섭취량에 맞춰 예상 영양정보를 계산합니다.")
