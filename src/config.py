"""
Configuration settings for the Low-Latency Vision System.
"""

# Classification Models
CLASSIFICATION_MODELS = {
    "Fast Model (MobileNetV2)": "google/mobilenet_v2_1.0_224",
    "Accurate Model (ResNet-50)": "microsoft/resnet-50"
}

# Object Detection Model
DETECTION_MODEL = "facebook/detr-resnet-50"
DETECTION_THRESHOLD = 0.7

# Resolution Options
RESOLUTION_OPTIONS = ["224 x 224 (High)", "160 x 160 (Medium)", "128 x 128 (Low)"]

RESOLUTION_MAP = {
    "224 x 224 (High)": 224,
    "160 x 160 (Medium)": 160,
    "128 x 128 (Low)": 128
}

# Supported Image Formats
SUPPORTED_FORMATS = ["jpg", "png", "jpeg"]

# UI Settings
PAGE_TITLE = "Low-Latency Vision System"
PAGE_LAYOUT = "centered"
APP_TITLE = "Low-Latency Image Classification System for Edge Devices"
