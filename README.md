# Low-Latency Vision System

An **edge-oriented computer vision system** for image classification with optional object detection, designed to study **latency–accuracy trade-offs under CPU-only constraints**.

This project focuses on **efficient inference**, not model training.

---

## Project Overview

Modern vision systems often rely on large pretrained models that are difficult to deploy on resource-constrained devices.
This project demonstrates how **pretrained models from Hugging Face** can be used efficiently on **CPU-only environments**, while maintaining transparency and usability.

Key objectives:

* Compare **lightweight vs heavy** vision models
* Measure **inference latency** on CPU
* Study the impact of **input resolution scaling**
* Improve interpretability using **object detection**
* Avoid training or fine-tuning entirely

---

## Project Structure

```
low_latency_vision_system/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py     # Image classification logic
│   │   └── detector.py       # Object detection logic
│   └── ui/
│       ├── __init__.py
│       └── components.py     # Reusable UI components
├── scripts/
│   ├── __init__.py
│   ├── test_inference.py     # Single model inference test
│   └── compare_models.py     # Model comparison benchmark
└── assets/
    └── test.png              # Sample image
```

---

## Environment Setup (Recommended)

This project uses a **Python virtual environment (venv)** to isolate dependencies.
Using a virtual environment is **strongly recommended**, especially for first-time setup.

---

## First-Time Setup

### Prerequisites

* Python **3.9 or higher**
* `pip` installed
* Internet connection (for downloading pretrained models)

---

## Create and Activate Virtual Environment

### ▶ Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
```

If you encounter a script execution error, run this **once**:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then activate again:

```powershell
venv\Scripts\activate
```

You should see `(venv)` in the terminal.

---

### ▶ macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in the terminal prompt.

---

## Install Dependencies

After activating the virtual environment:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⚠️ This project uses **CPU-only inference**.
No GPU or CUDA installation is required.

---

## Running the Application

Ensure the virtual environment is active:

```bash
streamlit run app.py
```

A browser window will open with the interactive interface.

---

## Running Utility Scripts

```bash
# Single model inference test
python -m scripts.test_inference

# Compare MobileNetV2 vs ResNet-50
python -m scripts.compare_models
```

---

## Features

* **Fast Image Classification**

  * MobileNetV2 optimized for low-latency CPU inference

* **High-Accuracy Image Classification**

  * ResNet-50 for improved prediction confidence

* **Optional Object Detection**

  * DETR (DEtection TRansformer) for visual explanation
  * Detection is used for **interpretability only**, not to modify classification

* **Resolution Scaling**

  * 224×224, 160×160, 128×128 input options

* **Top-3 Predictions**

  * Displays multiple likely classes to reflect uncertainty

* **Latency Breakdown**

  * Detection time
  * Classification time
  * Total pipeline latency

---

## Models Used

| Model       | Role                     | Source                        |
| ----------- | ------------------------ | ----------------------------- |
| MobileNetV2 | Low-latency classifier   | `google/mobilenet_v2_1.0_224` |
| ResNet-50   | High-accuracy classifier | `microsoft/resnet-50`         |
| DETR        | Object detection         | `facebook/detr-resnet-50`     |

All models are **pretrained** and used in **inference-only mode**.

---

## Design Decisions

* **No training or fine-tuning**

  * Training is intentionally avoided to reflect edge deployment constraints.

* **CPU-only execution**

  * Ensures portability across low-resource systems.

* **Detection decoupled from classification**

  * Improves interpretability without increasing model complexity.

* **Top-3 predictions**

  * Reduces misleading single-label outputs in ambiguous cases.

---

## Limitations

* Models are trained on generic datasets (ImageNet / COCO), which may lead to misclassification on domain-specific images.
* Object detection increases latency and is therefore optional.
* The system prioritizes efficiency and interpretability over maximum accuracy.

These are **expected trade-offs** in edge-oriented inference systems.

---

## Use Cases

* Edge AI deployment demonstrations
* Latency–accuracy analysis for vision models
* Educational projects on efficient inference
* CPU-based vision system prototyping

---

## Deactivating the Environment

When finished:

```bash
deactivate
```
---
## Conclusion

This project presents a practical and transparent approach to deploying pretrained vision models under strict computational constraints, highlighting the trade-offs between **speed**, **accuracy**, and **interpretability**.
