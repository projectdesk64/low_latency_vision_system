# Low-Latency Vision System

A production-ready image classification and object detection system optimized for edge devices.

## Project Structure

```
low_latency_vision_system/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/                      # Source code package
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── models/               # ML model modules
│   │   ├── __init__.py
│   │   ├── classifier.py     # Image classification
│   │   └── detector.py       # Object detection
│   └── ui/                   # UI components
│       ├── __init__.py
│       └── components.py     # Reusable Streamlit components
├── scripts/                  # Utility scripts
│   ├── __init__.py
│   ├── test_inference.py     # Single model test
│   └── compare_models.py     # Model comparison benchmark
└── assets/                   # Static assets
    └── test.png              # Test image
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

### 3. Run Test Scripts
```bash
# Single model inference test
python -m scripts.test_inference

# Compare MobileNetV2 vs ResNet-50
python -m scripts.compare_models
```

## Features

- **Fast Classification**: MobileNetV2 for low-latency inference
- **Accurate Classification**: ResNet-50 for higher accuracy
- **Object Detection**: DETR (DEtection TRansformer) with bounding boxes
- **Resolution Scaling**: 224x224, 160x160, or 128x128 input options
- **Latency Tracking**: Detailed breakdown of detection and classification times

## Configuration

Edit `src/config.py` to customize:
- Classification models
- Detection model and threshold
- Resolution options
- UI settings

## Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| MobileNetV2 | Fast classification | `google/mobilenet_v2_1.0_224` |
| ResNet-50 | Accurate classification | `microsoft/resnet-50` |
| DETR | Object detection | `facebook/detr-resnet-50` |
