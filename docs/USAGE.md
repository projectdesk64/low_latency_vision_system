# Usage Guide

## Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## User Interface

### 1. Model Selection

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| MobileNetV2 | ~80ms | Good | Real-time, edge devices |
| ResNet-50 | ~200ms | Better | Accuracy-critical tasks |

### 2. Resolution Selection

| Resolution | Speed | Quality |
|------------|-------|---------|
| 224×224 (High) | Baseline | Best quality |
| 160×160 (Medium) | Faster | Good quality |
| 128×128 (Low) | Fastest | Acceptable |

### 3. Object Detection Toggle

- **Enabled**: Runs DETR detection before classification (~200ms added)
- **Disabled**: Classification only (faster)

### 4. Image Upload

Supported formats: JPG, PNG, JPEG

### 5. Run Inference

Click "Run Inference" to process the image. Results show:
- Predicted class label
- Confidence score (0-1)
- Latency breakdown

## Command Line Scripts

### Single Model Test

```bash
python -m scripts.test_inference
```

### Model Comparison

```bash
python -m scripts.compare_models
```

## API Usage (Programmatic)

```python
from src.models import load_classifier, run_classification
from PIL import Image

# Load model
processor, model = load_classifier("google/mobilenet_v2_1.0_224")

# Load image
image = Image.open("path/to/image.jpg").convert("RGB")

# Run inference
result = run_classification(image, processor, model, size=224)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Time: {result['inference_time_ms']:.2f}ms")
```

## Performance Tips

1. **Use MobileNetV2** for fastest inference
2. **Lower resolution** (128×128) for speed
3. **Disable detection** when not needed
4. **GPU acceleration** significantly improves performance
