import torch
import torch.nn as nn
import torchvision.models as models


def _pretrained(name, weights_enum):
    fn = getattr(models, name)
    try:
        return fn(weights=getattr(models, weights_enum).DEFAULT)
    except Exception:
        pass
    try:
        return fn(pretrained=True)
    except Exception:
        pass
    return fn()


class _Net(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 2),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.backbone(x))


class ResNet(_Net):
    def __init__(self):
        super().__init__(_pretrained('resnet50', 'ResNet50_Weights'))


def build_resnet101():
    return _Net(_pretrained('resnet101', 'ResNet101_Weights'))


def build_resnext50():
    return _Net(_pretrained('resnext50_32x4d', 'ResNeXt50_32X4D_Weights'))


class Ensemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2])]
        preds = []
        for m in self.members:
            for v in views:
                preds.append(m(v))
        return torch.stack(preds, dim=0).mean(dim=0)
