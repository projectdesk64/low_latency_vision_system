"""
Streamlit UI components for the vision system.
"""
import streamlit as st


def render_sidebar(models: dict, resolutions: list) -> dict:
    """
    Render the sidebar with model and resolution selection.
    
    Returns:
        Dictionary with selected options
    """
    model_choice = st.selectbox("Select Model", list(models.keys()))
    resolution = st.selectbox("Input Resolution", resolutions)
    enable_detection = st.checkbox("Enable Object Detection (YOLO-style)")
    
    return {
        "model_choice": model_choice,
        "resolution": resolution,
        "enable_detection": enable_detection
    }


def render_results(prediction: str, confidence: float):
    """
    Render classification results.
    """
    st.success(f"Prediction: {prediction}")
    st.write(f"Confidence: {confidence:.4f}")


def render_latency_breakdown(
    detection_time: float,
    classification_time: float,
    total_time: float
):
    """
    Render latency breakdown section.
    """
    st.subheader("Latency Breakdown")
    st.write(f"Detection Time: {detection_time:.2f} ms")
    st.write(f"Classification Time: {classification_time:.2f} ms")
    st.write(f"Total Pipeline Time: {total_time:.2f} ms")
