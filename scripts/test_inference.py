"""
Test script for single model inference.
Run: python -m scripts.test_inference
"""
import sys
sys.path.insert(0, ".")

from PIL import Image
from src.models.classifier import load_classifier, run_classification
from src.config import CLASSIFICATION_MODELS

MODEL_NAME = "google/mobilenet_v2_1.0_224"

if __name__ == "__main__":
    print("Loading model...")
    processor, model = load_classifier(MODEL_NAME)
    
    # Load test image
    image = Image.open("assets/test.png").convert("RGB")
    
    print("Running inference...")
    result = run_classification(image, processor, model, size=224)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Inference time (ms): {result['inference_time_ms']:.2f}")
