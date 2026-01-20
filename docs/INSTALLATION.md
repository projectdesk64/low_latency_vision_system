# Installation Guide

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (optional)

## Quick Installation

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd low_latency_vision_system
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -m scripts.test_inference
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| transformers | >=4.30.0 | HuggingFace models |
| streamlit | >=1.28.0 | Web UI framework |
| opencv-python | >=4.8.0 | Image processing |
| pillow | >=9.0.0 | Image loading |
| numpy | >=1.24.0 | Numerical operations |

## GPU Support (Optional)

For CUDA GPU acceleration:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

### Common Issues

1. **Model download fails**: Check internet connection and HuggingFace access
2. **Out of memory**: Try lower resolution (128x128) or disable detection
3. **Slow inference**: Ensure PyTorch is using GPU if available
