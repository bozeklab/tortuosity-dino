import torch
from torch import nn

class LinearClassifier(nn.Module):
    def __init__(
            self,
            feature_extractor: nn.Module,
            embedding_dim: int,
            num_classes: int
    ):
        super(LinearClassifier, self).__init__()
        # Encoder.
        self.feature_extractor = feature_extractor

        # Classifier.
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.feature_extractor(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.classifier(feature)
        return out
