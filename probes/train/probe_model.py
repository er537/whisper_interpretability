import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, feat_dim, outdim=2) -> None:
        super().__init__()
        self.w1 = nn.Linear(feat_dim, outdim, bias=False)
        # self.w2 = nn.Linear(1500, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.w1(x)
        # x = self.w2(x.permute(0, 2, 1)).permute(0, 2, 1)  # bsz, outdim, seq_len
        return self.softmax(x)
