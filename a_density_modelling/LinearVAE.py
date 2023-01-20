import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as func


class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc1a = nn.Linear(400, 100)
        self.fc21 = nn.Linear(100, 2)  # Latent space of 2D
        self.fc22 = nn.Linear(100, 2)  # Latent space of 2D
        self.fc3 = nn.Linear(2, 100)   # Latent space of 2D
        self.fc3a = nn.Linear(100, 400)
        self.fc4 = nn.Linear(400, 784)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = x.view(-1, 28*28)
        h1 = func.relu(self.fc1(x))
        h2 = func.relu(self.fc1a(h1))
        return self.fc21(h2), self.fc22(h2)

    def decode(self, z):
        h3 = func.relu(self.fc3(z))
        h4 = func.relu(self.fc3a(h3))
        return torch.sigmoid(self.fc4(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1, 28, 28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
