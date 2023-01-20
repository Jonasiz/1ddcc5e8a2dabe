import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms


def analyse_variance(x):
    eigenvalues, eigenvectors = torch.linalg.eig(torch.cov(x.T))
    var_cumsum = torch.cumsum(eigenvalues / eigenvalues.sum(), dim=0).float()

    print('Dimensions needed to capture 90% of the dataset variance:', int(torch.where(var_cumsum > 0.90)[0][0] + 1))
    print('Dimensions needed to capture 95% of the dataset variance:', int(torch.where(var_cumsum > 0.95)[0][0] + 1))
    print()

    fig, ax = plt.subplots()
    fig.set_size_inches((16 / 2, 9 / 2))
    fig.set_dpi(100)

    ax.plot(range(1, len(var_cumsum) + 1), var_cumsum * 100)
    ax.set_title('Cumulative sum', size=18)
    ax.set_xlabel('PCs (dimensions) used')
    ax.set_ylabel('% of dataset variance captured')

    plt.show()


def ppca(dataset, target_dims):
    dataset = dataset.reshape(-1, 28*28)
    # Calculate the variance estimate
    eigenvalues, eigenvectors = torch.linalg.eig(torch.cov(dataset.T))
    est_var = eigenvalues[target_dims:].mean()

    # Use the variance estimate to estimate the factor loading matrix W
    eig_vecs = eigenvectors[:, :target_dims]
    eig_vals = eigenvalues[:target_dims].diag()
    ppca_cov = est_var * torch.eye(target_dims)

    est_w = eig_vecs @ torch.sqrt((eig_vals - ppca_cov))

    # Compute the estimate of the datasets covariance matrix (C)
    dataset_cov = est_w.T @ est_w + ppca_cov

    # Compute the mean and center the dataset
    dataset_cntr = dataset - dataset.float().mean(axis=0)

    # Sample the analytical max. lik. estimates of factors z
    return (torch.inverse(dataset_cov) @ est_w.T @ dataset_cntr.to(torch.complex64).T).T


def plot_pc(data, plt_pc, ax_row, title):
    for target in torch.unique(data.targets):
        target_indexes = torch.where(data.targets == target)
        target_factors = plt_pc[target_indexes].double()

        ax_row[0].scatter(target_factors[:, 0], target_factors[:, 1], label=int(target), s=1.5)
        ax_row[1].scatter(target_factors[:, 1], target_factors[:, 2], label=int(target), s=1.5)
        ax_row[2].scatter(target_factors[:, 0], target_factors[:, 2], label=int(target), s=1.5)

        ax_row[0].set_xlabel('PC0')
        ax_row[0].set_ylabel('PC1')
        ax_row[1].set_xlabel('PC1')
        ax_row[1].set_ylabel('PC2')
        ax_row[2].set_xlabel('PC0')
        ax_row[2].set_ylabel('PC2')

        for ax in ax_row:
            ax.set_title(title)


def run_ppca(train_set, test_set):
    plt_train = ppca(train_set.data, target_dims=3)
    plt_test = ppca(test_set.data, target_dims=3)

    fig, ax = plt.subplots(2, 3, constrained_layout=True)
    fig.set_size_inches(16 * 1.2, 9 * 1.2)
    plot_pc(train_set, plt_train, ax[0], 'Train set')
    plot_pc(test_set, plt_test, ax[1], 'Test set')
    plt.show()


if __name__ == '__main__':
    data_dir = './data'
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())

    run_ppca(train_set, test_set)
