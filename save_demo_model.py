import torch
from modules.deepfake_efficientnet_model import EfficientNetDeepFake

model = EfficientNetDeepFake()
torch.save(model.state_dict(), "deepfake_detector_demo.pth")
