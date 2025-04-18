import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetDeepFake(nn.Module):
    def __init__(self):
        super(EfficientNetDeepFake, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
