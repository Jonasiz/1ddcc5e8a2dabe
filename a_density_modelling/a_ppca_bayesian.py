from datetime import time

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torchvision
from matplotlib import pyplot as plt
from pyro.infer import Trace_ELBO, SVI
from pyro.optim.clipped_adam import ClippedAdam


def ppca_model(x, target_dims):
    sigma = pyro.param('sigma', lambda: torch.tensor(1.0))
    mean = pyro.param('mean', lambda: torch.zeros(x.shape[1], x.shape[0]))

    w = pyro.sample('w', dist.Normal(torch.zeros(x.shape[1], target_dims),
                                     torch.ones(x.shape[1], target_dims)).to_event(1))

    with pyro.plate('data', x.shape[0], subsample_size=5) as ind:
        mean_batch = mean[:, ind]
        x_batch = x[ind, :]

        z = pyro.sample('z', dist.Normal(torch.zeros(target_dims, x_batch.shape[0]),
                                         torch.ones(target_dims, x_batch.shape[0])).to_event(1))

        pyro.sample('x', dist.Normal(z.T @ w.T + mean_batch.T,
                                     sigma * torch.eye(x_batch.shape[0])),
                    obs=x_batch)


def ppca_guide(x, target_dims):
    w_mean = pyro.param('w_mean', torch.randn(x.shape[1], target_dims))
    w_scale = pyro.param('w_scale', torch.ones(x.shape[1], target_dims))

    z_mean = pyro.param('z_mean', torch.randn(target_dims, x.shape[0]))
    z_scale = pyro.param('z_scale', torch.ones(target_dims, x.shape[0]))

    w = pyro.sample('w', dist.Normal(w_mean, w_scale).to_event(0))

    with pyro.plate('data_g', x.shape[0], subsample_size=5) as ind:
        mean_batch = z_mean[:, ind]
        scale_batch = z_scale[:, ind]
        z = pyro.sample('z', dist.Normal(mean_batch, scale_batch).to_event(1))


def train_SVI(model, guide, x, target_dims, lr=0.001, iters=1000, print_every=100):
    optimizer = ClippedAdam({'lr': lr, 'betas': (0.95, 0.999), 'lrd': 0.1 ** (1 / iters)})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses = []
    time_start = time.time()

    for i in range(iters):
        loss_val = svi.step(x, target_dims)
        losses.append(loss_val)

        if i % print_every == 0:
            print(f'Iter {i}/{iters}. Loss: {loss_val}')

    print(f'Final Loss: {loss_val}')
    print(f'Total training time: {np.round(time.time() - time_start, 2)} seconds')

    return losses


def plot_losses(losses):
    fig, ax = plt.subplots()
    fig.set_size_inches(16 / 2, 9 / 2)

    ax.plot(list(range(len(losses))), losses)
    ax.set_xlabel('# of iterations')
    ax.set_ylabel('ELBO')
    fig.suptitle('ELBO curve')

    plt.show()


if __name__ == '__main__':
    data_dir = './data'

    train_set = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    x_train = train_set.data.float()[:10000]
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] ** 2)
    x_test = train_set.data.float()[:10000]
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] ** 2)

    losses = train_SVI(ppca_model, ppca_guide, x_train, 87, print_every=1)
    plot_losses()
