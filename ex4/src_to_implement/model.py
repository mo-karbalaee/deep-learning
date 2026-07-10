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


def _head_fc(backbone):
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 2))
    return backbone


def _head_classifier_seq(backbone):
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 2))
    return backbone


def _head_convnext(backbone):
    in_features = backbone.classifier[2].in_features
    backbone.classifier[2] = nn.Linear(in_features, 2)
    return backbone


class _Model(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.backbone(x))


class ResNet(_Model):
    def __init__(self):
        super().__init__(_head_fc(_pretrained('resnet50', 'ResNet50_Weights')))


def build_resnext50():
    return _Model(_head_fc(_pretrained('resnext50_32x4d', 'ResNeXt50_32X4D_Weights')))


def build_efficientnet():
    return _Model(_head_classifier_seq(_pretrained('efficientnet_b3', 'EfficientNet_B3_Weights')))


def build_convnext():
    return _Model(_head_convnext(_pretrained('convnext_tiny', 'ConvNeXt_Tiny_Weights')))


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
