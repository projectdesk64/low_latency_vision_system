# Architecture Overview

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                    │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │ Model Select │ Resolution   │ Detection    │ File Upload  │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      src/ Package                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    config.py                             │    │
│  │  • Model configurations                                  │    │
│  │  • Resolution mappings                                   │    │
│  │  • UI settings                                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐              │
│         ▼                    ▼                    ▼              │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │
│  │ classifier  │      │  detector   │      │    ui/      │      │
│  │    .py      │      │    .py      │      │ components  │      │
│  ├─────────────┤      ├─────────────┤      ├─────────────┤      │
│  │ MobileNetV2 │      │    DETR     │      │ Streamlit   │      │
│  │  ResNet-50  │      │ ResNet-50   │      │  widgets    │      │
│  └─────────────┘      └─────────────┘      └─────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HuggingFace Transformers                      │
│  • AutoImageProcessor                                            │
│  • AutoModelForImageClassification                               │
│  • AutoModelForObjectDetection                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Image Upload
     │
     ▼
┌─────────────┐
│ PIL.Image   │
│  (RGB)      │
└─────────────┘
     │
     ├──────────────────────────────────┐
     │                                  │
     ▼ (if detection enabled)           ▼
┌─────────────┐                  ┌─────────────┐
│  Detection  │                  │Classification│
│    DETR     │                  │ MobileNet/   │
│             │                  │  ResNet      │
└─────────────┘                  └─────────────┘
     │                                  │
     ▼                                  ▼
┌─────────────┐                  ┌─────────────┐
│ Bounding    │                  │ Class Label │
│   Boxes     │                  │ Confidence  │
└─────────────┘                  └─────────────┘
     │                                  │
     └──────────────┬───────────────────┘
                    ▼
            ┌─────────────┐
            │  Results +  │
            │  Latency    │
            │  Breakdown  │
            └─────────────┘
```

## Key Design Decisions

### 1. Modular Architecture
- Separation of concerns (models, UI, config)
- Easy to extend with new models
- Testable components

### 2. Cached Model Loading
- `@st.cache_resource` for models
- Load once, reuse across sessions
- Reduces latency on subsequent runs

### 3. Resolution Flexibility
- Trade-off between speed and accuracy
- User-controlled optimization
- Edge device friendly

### 4. Graceful Format Handling
- Try/except for different model processor formats
- Compatible with MobileNet and ResNet variants
