import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device()}")
