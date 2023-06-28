import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, feat_dim, outdim=2) -> None:
        super().__init__()
        self.w1 = nn.Linear(feat_dim, outdim, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.w1(x)
        return self.softmax(x)
