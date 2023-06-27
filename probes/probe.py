import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, feat_dim) -> None:
        super().__init__()
        self.w1 = nn.Linear(feat_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.w1(x)
        return self.sigmoid(x)
