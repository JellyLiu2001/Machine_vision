import torch
print("Is CUDA available?:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
