import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as func


class ConvVAE(nn.Module):
    def __init__(self, ):
        super(ConvVAE, self).__init__()

        # Encoder input convolutional layers
        self.encoder_cv_i = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)  # input
        self.encoder_cv_h = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)  # hidden

        # Bottleneck
        self.dense_mean = nn.Linear(22 * 22 * 20, 128)    # dense layer for mean
        self.dense_logvar = nn.Linear(22 * 22 * 20, 128)  # dense layer for variance
        self.dense_out = nn.Linear(128, 22 * 22 * 20)     # output dense layer

        # Decoder output convolutional layers
        self.decoder_cv_h = nn.ConvTranspose2d(20, 10, 3)  # hidden
        self.decoder_cv_o = nn.ConvTranspose2d(10, 1, 5)  # output

    @staticmethod
    def reparameterize(mean, log_var):
        std = torch.exp(log_var / 2)
        return mean + std * torch.randn_like(std)

    def encode(self, x):
        x = func.relu(self.encoder_cv_i(x))  # Feed to 5x5 input conv layer -> 24x24x10
        x = func.relu(self.encoder_cv_h(x))  # Feed to 3x3 hidden conv layer -> 22x22x20
        x = x.view(-1, 22 * 22 * 20)  # Flatten to dense layer dimensions

        # Fork: Get mean from mean dense layer, Get log_var from log_var dense layer
        return self.dense_mean(x), self.dense_logvar(x)

    def decode(self, z):
        x = func.relu(self.dense_out(z))  # Feed from latent representation to flattened x
        x = x.view(-1, 20, 22, 22)  # Unflatten X to image
        x = func.relu(self.decoder_cv_h(x))  # Feed to 3x3 hidden conv layer -> 24x24x10
        x = torch.sigmoid(self.decoder_cv_o(x))  # Feed to 5x5 output conv layer -> 28x28x1
        return x

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_out = self.decode(z)
        return x_out, mean, log_var
