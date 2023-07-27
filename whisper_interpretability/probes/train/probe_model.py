import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, feat_dim, seq_len, outdim=2) -> None:
        super().__init__()
        self.w1 = nn.Linear(feat_dim, outdim, bias=False)
        if seq_len is not None:
            self.w2 = nn.Linear(seq_len, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (bsz, seq_len, d_model)
        x = self.w1(x)
        if hasattr(self, "w2"):
            x = self.w2(x.permute(0, 2, 1)).permute(0, 2, 1) # optionally 'convolve' along sequence dimension
        return self.softmax(x)
