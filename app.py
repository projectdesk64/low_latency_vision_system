"""
Low-Latency Vision System - Streamlit Application
Run: streamlit run app.py
"""
import time
import streamlit as st
from PIL import Image

from src.config import (
    CLASSIFICATION_MODELS,
    RESOLUTION_OPTIONS,
    RESOLUTION_MAP,
    PAGE_TITLE,
    PAGE_LAYOUT,
    APP_TITLE
)
from src.models import load_classifier, run_classification, load_detector, run_detection

# Page configuration
st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)
st.title(APP_TITLE)

# UI Controls
model_choice = st.selectbox("Select Model", list(CLASSIFICATION_MODELS.keys()))
resolution = st.selectbox("Input Resolution", RESOLUTION_OPTIONS)
enable_detection = st.checkbox(
    "Enable Object Detection (for explanation, not classification)"
)
st.caption(
    "Detected objects are shown to explain classification results. "
    "They are not used to modify or override the final prediction."
)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


@st.cache_resource
def cached_load_classifier(model_id):
    return load_classifier(model_id)


@st.cache_resource
def cached_load_detector():
    return load_detector()


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", width="stretch")

    processor, model = cached_load_classifier(CLASSIFICATION_MODELS[model_choice])

    if st.button("Run Inference"):
        total_start = time.time()

        # ---------- Object Detection ----------
        if enable_detection:
            det_processor, det_model = cached_load_detector()
            det_result = run_detection(image, det_processor, det_model)
            detection_time = det_result["inference_time_ms"]

            st.subheader("Detected Objects")
            st.image(det_result["annotated_image"], width="stretch")
        else:
            detection_time = 0.0

        # ---------- Classification ----------
        size = RESOLUTION_MAP[resolution]
        cls_result = run_classification(image, processor, model, size=size)
        classification_time = cls_result["inference_time_ms"]

        total_end = time.time()
        total_time = (total_end - total_start) * 1000

        # ---------- Results ----------
        st.success("Top-3 Predictions:")
        for i, (label, score) in enumerate(cls_result["top3_predictions"], 1):
            st.write(f"{i}. {label} — {score:.4f}")

        st.subheader("Latency Breakdown")
        st.write(f"Detection Time: {detection_time:.2f} ms")
        st.write(f"Classification Time: {classification_time:.2f} ms")
        st.write(f"Total Pipeline Time: {total_time:.2f} ms")
