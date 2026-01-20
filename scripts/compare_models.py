"""
Model comparison script - compares MobileNetV2 vs ResNet-50.
Run: python -m scripts.compare_models
"""
import sys
sys.path.insert(0, ".")

from PIL import Image
from src.models.classifier import load_classifier, run_classification
from src.config import CLASSIFICATION_MODELS

if __name__ == "__main__":
    # Load test image
    image = Image.open("assets/test.png").convert("RGB")
    
    for name, model_id in CLASSIFICATION_MODELS.items():
        print(f"\nRunning: {name}")
        
        processor, model = load_classifier(model_id)
        result = run_classification(image, processor, model, size=224)
        
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference time (ms): {result['inference_time_ms']:.2f}")
