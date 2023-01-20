import matplotlib.pyplot as plt

from ConvVAE import ConvVAE
from LinearVAE import LinearVAE
from VAEWrapper import VAEWrapper


def plot_main():
    x_cov, data_cov = conv_vae.gen_samples()
    x_cov_p = x_cov.swapaxes(1, 3).squeeze()
    data_cov_p = data_cov.swapaxes(1, 3).squeeze()

    x_lin, data_lin = linear_vae.gen_samples()
    x_lin_p = x_lin.reshape(-1, 28, 28)
    data_lin_p = data_lin.swapaxes(1, 3).swapaxes(1, 2).squeeze()

    fig, ax = plt.subplots(3, 4, tight_layout=True)
    fig.set_size_inches(16, 9)

    for i in range(3):
        if i == 0:
            ax[i][0].set_title('sampled (conv)')
            ax[i][1].set_title('real (conv)')
            ax[i][2].set_title('sampled (lin)')
            ax[i][3].set_title('real (lin)')

        ax[i][0].imshow(x_cov_p[i].cpu(), cmap='gray')
        ax[i][1].imshow(data_cov_p[i].cpu(), cmap='gray')
        ax[i][2].imshow(x_lin_p[i].cpu(), cmap='gray')
        ax[i][3].imshow(data_lin_p[i].cpu(), cmap='gray')

    plt.show()


def main():
    print('FITTING LINEAR VAE')
    linear_model = LinearVAE()
    linear_vae = VAEWrapper(linear_model, batch_size=256)
    linear_vae.fit(5)
    print('DONE')

    print('FITTING CONVOLUTIONAL VAE')
    conv_model = ConvVAE()
    conv_vae = VAEWrapper(conv_model, batch_size=32)
    conv_vae.fit(5)
    print('DONE')

    print('lin params:', sum(param.numel() for param in linear_vae.vae_model.parameters()))
    print('conv params:', sum(param.numel() for param in conv_vae.vae_model.parameters()))


if __name__ == '__main__':
    main()
