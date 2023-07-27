import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, activation_size, n_dict_components):
        super(AutoEncoder, self).__init__()
        self.tied = True  # tie encoder and decoder weights
        self.activation_size = activation_size
        self.n_dict_components = n_dict_components

        # Only defining the decoder layer, encoder will share its weights
        self.decoder = nn.Linear(n_dict_components, activation_size, bias=False)
        # Create a bias layer
        self.encoder_bias = nn.Parameter(torch.zeros(n_dict_components))

        # Initialize the decoder weights orthogonally
        nn.init.orthogonal_(self.decoder.weight)

        # Encoder is a Sequential with the ReLU activation
        # No need to define a Linear layer for the encoder as its weights are tied with the decoder
        self.encoder = nn.Sequential(nn.ReLU())

    def forward(self, x):
        # x: (bsz, seq_len, feat_dim)
        # Apply unit norm constraint to the decoder weights
        self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)
        c = self.encoder(x @ self.decoder.weight + self.encoder_bias)

        # Decoding step as before
        x_hat = self.decoder(c)
        return x_hat, c
