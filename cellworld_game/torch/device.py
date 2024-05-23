import torch

if torch.cuda.is_available():
    default_device = torch.device("cuda")  # Set device to GPU
    print("GPU is available")
else:
    default_device = torch.device("cpu")  # Set device to CPU
    print("GPU is not available, using CPU instead")
