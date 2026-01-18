import torch
import torch.nn as nn
from torchvision import models

class ForensicNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Stream 1: Spatial (Visual Clues)
        self.spatial = models.efficientnet_b0(weights='DEFAULT')
        num_ft = self.spatial.classifier[1].in_features
        self.spatial.classifier = nn.Identity()

        # Stream 2: Frequency (Math Clues)
        self.freq_stream = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_ft + 32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_rgb, x_dct):
        return self.classifier(torch.cat((self.spatial(x_rgb), self.freq_stream(x_dct)), dim=1))