import torch.nn as nn
import torchvision.models as models

from config import num_classes, hidden_units, dropout_rate, device


class BrainTumorClassifier(nn.Module):
    def __init__(self):
        super(BrainTumorClassifier, self).__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for param in backbone.parameters():
            param.requires_grad = False

        for param in backbone.layer4.parameters():
            param.requires_grad = True

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, num_classes)
        )

        self.network = backbone

    def forward(self, x):
        return self.network(x)


def get_model():
    model = BrainTumorClassifier().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    return model
