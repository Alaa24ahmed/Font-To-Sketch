# import torch

# if torch.cuda.is_available():
#     print("CUDA is available")
#     print(f"CUDA device count: {torch.cuda.device_count()}")
#     print(f"Current CUDA device: {torch.cuda.current_device()}")
#     print(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# else:
#     print("CUDA is not available")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
