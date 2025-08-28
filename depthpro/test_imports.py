#!/usr/bin/env python3

print("Testing imports...")

try:
    import numpy as np
    print("OK: numpy imported successfully")
except ImportError as e:
    print(f"ERROR: numpy import failed: {e}")

try:
    import cv2
    print("OK: opencv imported successfully")
    print(f"  OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"ERROR: opencv import failed: {e}")

try:
    import torch
    print("OK: torch imported successfully")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"ERROR: torch import failed: {e}")

try:
    from ultralytics import YOLO
    print("OK: ultralytics imported successfully")
except ImportError as e:
    print(f"ERROR: ultralytics import failed: {e}")

try:
    import depth_pro
    print("OK: depth_pro imported successfully")
except ImportError as e:
    print(f"ERROR: depth_pro import failed: {e}")

print("\nAll imports tested.")