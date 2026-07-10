import torch.nn as nn
import torchvision.models as models


def _resnet18():
    try:
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        pass
    try:
        return models.resnet18(pretrained=True)
    except Exception:
        pass
    return models.resnet18()


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _resnet18()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.backbone(x))
