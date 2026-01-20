"""
Image Classification module using HuggingFace Transformers.
"""
import time
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def load_classifier(model_id: str):
    """
    Load a classification model and processor from HuggingFace.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Tuple of (processor, model)
    """
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    return processor, model


def run_classification(
    image: Image.Image,
    processor,
    model,
    size: int = 224
) -> dict:
    """
    Run image classification on an input image.
    
    Args:
        image: PIL Image to classify
        processor: HuggingFace image processor
        model: Classification model
        size: Input resolution (default 224)
        
    Returns:
        Dictionary with prediction, confidence, and inference_time_ms
    """
    start = time.time()
    
    # Handle different processor size formats
    try:
        inputs = processor(
            images=image,
            size={"height": size, "width": size},
            return_tensors="pt"
        )
    except ValueError:
        # Some models (like ResNet) expect 'shortest_edge' format
        inputs = processor(
            images=image,
            size={"shortest_edge": size},
            return_tensors="pt"
        )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    end = time.time()
    
    logits = outputs.logits
    pred_id = logits.argmax(-1).item()
    confidence = torch.softmax(logits, dim=1)[0][pred_id].item()
    
    return {
        "prediction": model.config.id2label[pred_id],
        "confidence": confidence,
        "inference_time_ms": (end - start) * 1000
    }
