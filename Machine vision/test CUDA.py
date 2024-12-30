import torch
import cv2
import numpy as np
import pandas as pd

print("PyTorch version:", torch.__version__)

print("OpenCV version:", cv2.__version__)

print("NumPy version:", np.__version__)

print("Pandas version:", pd.__version__)

# Check if CUDA is supported
if torch.cuda.is_available():
    print(f"CUDA is available. Version: {torch.version.cuda}")
else:
    print("CUDA is not available.")
