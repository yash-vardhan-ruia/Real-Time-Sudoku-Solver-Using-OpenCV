#!/usr/bin/env python
"""Quick test to verify model loads and inference works."""

import torch
import numpy as np
import cv2
from pathlib import Path

# Model constants and architecture from main.py
MODEL_IMAGE_SIZE = 28
MODEL_PTH_PATH = Path(__file__).with_name("digit_cnn.pth")

class DigitCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Model Load & Inference Pipeline")
    print("=" * 60)
    
    # Check model file exists
    if not MODEL_PTH_PATH.exists():
        print(f"✗ Model not found: {MODEL_PTH_PATH}")
        exit(1)
    print(f"✓ Model file found: {MODEL_PTH_PATH} ({MODEL_PTH_PATH.stat().st_size} bytes)")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # Load model
    try:
        model = DigitCNN().to(device)
        model.load_state_dict(torch.load(str(MODEL_PTH_PATH), map_location=device, weights_only=True))
        model.eval()
        print(f"✓ Model loaded and in eval mode")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        exit(1)
    
    # Test inference on dummy data
    try:
        dummy_batch = torch.randn(4, 1, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE).to(device)
        with torch.no_grad():
            output = model(dummy_batch)
        print(f"✓ Test inference successful")
        print(f"  Input shape:  {dummy_batch.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")
        
        # Check output is sensible (logits, no NaNs)
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"✗ Output contains NaN or Inf!")
            exit(1)
        print(f"✓ Output contains valid logits")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        exit(1)
    
    print("=" * 60)
    print("✓ All tests passed! Model is ready for inference.")
    print("=" * 60)
