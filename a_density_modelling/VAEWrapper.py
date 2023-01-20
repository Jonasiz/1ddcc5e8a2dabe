import torch
from torch import optim
from torch.nn import functional as func
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAEWrapper:
    def __init__(self, vae_model, batch_size=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae_model = vae_model.to(self.device)
        self.batch_size = batch_size
        self.scheduler = ExponentialLR(optimizer=optim.Adam(vae_model.parameters(), lr=0.001),
                                       gamma=0.5)

        data_dir = './data'

        train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        self.train_loader = DataLoader(train_data, batch_size=batch_size,
                                       shuffle=True, num_workers=1, pin_memory=True)

        test_data = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor())
        self.test_loader = DataLoader(test_data, batch_size=batch_size,
                                      shuffle=True, num_workers=1, pin_memory=True)

    @staticmethod
    def loss_function(recon_x, x, mu, log_var):
        bce = func.binary_cross_entropy(recon_x.view(-1, 1, 28, 28), x.view(-1, 1, 28, 28), reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return bce + kld

    def fit(self, epochs):
        for epoch in range(epochs):
            mean_train_loss = self._train_model()
            print(f'Epoch: {epoch + 1}. Average train set loss: {mean_train_loss:.4f}')

            mean_test_loss = self._test_model()
            print(f'Epoch: {epoch + 1}. Average test set loss: {mean_test_loss:.4f}')

            with torch.no_grad():
                if isinstance(self.vae_model, LinearVAE):
                    sample = torch.randn(64, 2).to(self.device)
                    sample = self.vae_model.decode(sample).cpu()
                    save_image(sample.view(64, 1, 28, 28), f'./data/sample_{epoch + 1}.png')
                else:
                    sample = torch.randn(256, 128).to(self.device)
                    sample = self.vae_model.decode(sample).cpu()
                    save_image(sample.view(256, 1, 28, 28), f'./data/sample_{epoch + 1}.png')

    def gen_samples(self):
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                x_out, mean, log_var = self.vae_model(data)
                return x_out, data
                break

    def _train_model(self):
        self.vae_model.train()
        total_loss = 0

        for data, _ in self.train_loader:
            data = data.to(self.device)
            self.scheduler.optimizer.zero_grad()
            recon_batch, mu, log_var = self.vae_model(data)

            loss = self.loss_function(recon_batch, data, mu, log_var)
            loss.backward()  # calc gradients
            total_loss += loss.item()

            self.scheduler.optimizer.step()  # backpropagation

        self.scheduler.step()

        return total_loss / len(self.train_loader.dataset)

    def _test_model(self):
        self.vae_model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                recon_batch, mu, log_var = self.vae_model(data)
                total_loss += self.loss_function(recon_batch, data, mu, log_var).item()

        return total_loss / len(self.test_loader.dataset)