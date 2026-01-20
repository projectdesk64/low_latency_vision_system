# API Reference

## Classification Module

**Location**: `src/models/classifier.py`

### `load_classifier(model_id: str) -> Tuple[Processor, Model]`

Load a HuggingFace classification model.

**Parameters:**
- `model_id` (str): HuggingFace model identifier

**Returns:**
- Tuple of (processor, model)

**Example:**
```python
processor, model = load_classifier("google/mobilenet_v2_1.0_224")
```

---

### `run_classification(image, processor, model, size=224) -> dict`

Run image classification.

**Parameters:**
- `image` (PIL.Image): Input image
- `processor`: HuggingFace processor
- `model`: Classification model
- `size` (int): Input resolution (default: 224)

**Returns:**
```python
{
    "prediction": str,       # Class label
    "confidence": float,     # 0.0 to 1.0
    "inference_time_ms": float
}
```

---

## Detection Module

**Location**: `src/models/detector.py`

### `load_detector() -> Tuple[Processor, Model]`

Load the DETR object detection model.

**Returns:**
- Tuple of (processor, model)

---

### `run_detection(image, processor, model, threshold=0.7) -> dict`

Run object detection with bounding boxes.

**Parameters:**
- `image` (PIL.Image): Input image
- `processor`: HuggingFace processor
- `model`: Detection model
- `threshold` (float): Confidence threshold (default: 0.7)

**Returns:**
```python
{
    "annotated_image": np.ndarray,  # Image with boxes
    "detections": [
        {
            "label": str,
            "score": float,
            "box": [x1, y1, x2, y2]
        }
    ],
    "inference_time_ms": float
}
```

---

## Configuration

**Location**: `src/config.py`

| Variable | Type | Description |
|----------|------|-------------|
| `CLASSIFICATION_MODELS` | dict | Model name → HuggingFace ID |
| `DETECTION_MODEL` | str | DETR model ID |
| `DETECTION_THRESHOLD` | float | Default 0.7 |
| `RESOLUTION_OPTIONS` | list | Available resolutions |
| `RESOLUTION_MAP` | dict | Resolution name → int |
