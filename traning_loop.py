import torch
print(f"Device : {f"GPU ({torch.cuda.get_device_name(0)})pip list --format=freeze > requirements.txt" if torch.cuda.is_available() else "cpu"}")