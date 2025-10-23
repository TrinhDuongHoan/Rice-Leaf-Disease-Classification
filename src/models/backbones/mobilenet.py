from ..attentions.eca import ECA
import torch.nn as nn
import torchvision.models as models
import timm

class MobileNetV3_Small_ECA(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout=0.0, k=3):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            num_classes=0
        )
        self.out_channels = self.backbone.num_features
        self.eca = ECA(k)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.eca(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)