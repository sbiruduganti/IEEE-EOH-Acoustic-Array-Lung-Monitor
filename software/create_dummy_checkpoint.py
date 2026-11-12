import os
import torch
from src.model import ICBHIModel

os.makedirs("checkpoints", exist_ok=True)
model = ICBHIModel(num_classes=4)

torch.save({
    'epoch': 0,
    'model_state_dict': model.state_dict(),
    'val_acc': 0.0,
}, "checkpoints/best_icbhi_model.pth")

print("Dummy checkpoint created at checkpoints/best_icbhi_model.pth!")