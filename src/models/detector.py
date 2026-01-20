"""
Object Detection module using DETR (DEtection TRansformer).
"""
import time
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from ..config import DETECTION_MODEL, DETECTION_THRESHOLD


def load_detector():
    """
    Load the DETR object detection model.
    
    Returns:
        Tuple of (processor, model)
    """
    processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL)
    model = AutoModelForObjectDetection.from_pretrained(DETECTION_MODEL)
    model.eval()
    return processor, model


def run_detection(
    image: Image.Image,
    processor,
    model,
    threshold: float = DETECTION_THRESHOLD
) -> dict:
    """
    Run object detection on an input image.
    
    Args:
        image: PIL Image for detection
        processor: HuggingFace image processor
        model: Detection model
        threshold: Confidence threshold for detections
        
    Returns:
        Dictionary with annotated_image (numpy array), detections list, and inference_time_ms
    """
    start = time.time()
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]
    
    # Draw bounding boxes
    img_np = np.array(image)
    detections = []
    
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box_coords = [int(i) for i in box.tolist()]
        label_text = model.config.id2label[label.item()]
        
        cv2.rectangle(
            img_np,
            (box_coords[0], box_coords[1]),
            (box_coords[2], box_coords[3]),
            (0, 255, 0),
            2
        )
        cv2.putText(
            img_np,
            label_text,
            (box_coords[0], box_coords[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        
        detections.append({
            "label": label_text,
            "score": score.item(),
            "box": box_coords
        })
    
    end = time.time()
    
    return {
        "annotated_image": img_np,
        "detections": detections,
        "inference_time_ms": (end - start) * 1000
    }
