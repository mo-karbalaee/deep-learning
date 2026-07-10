import torch
import torch.nn as nn
import torchvision.models as models


def _resnet50():
    try:
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception:
        pass
    try:
        return models.resnet50(pretrained=True)
    except Exception:
        pass
    return models.resnet50()


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _resnet50()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 2),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.backbone(x))


class Ensemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        outputs = [m(x) for m in self.members]
        return torch.stack(outputs, dim=0).mean(dim=0)
