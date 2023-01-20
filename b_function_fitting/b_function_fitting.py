import time

import arviz
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.nn
import torch
from pyro.contrib.gp.kernels import RBF, Periodic, Product
from pyro.contrib.gp.models import GPRegression
from pyro.infer import NUTS, MCMC

plt.rcParams.update({'font.size': 14})


def f(x):
    return np.sin(20 * x) + 2 * np.cos(14 * x) - 2 * np.sin(6 * x)


def generate_dataset():
    return torch.tensor([[x, f(x)] for x in [-1, -0.5, 0, 0.5, 1]])


def plot(x, y, ax, std=None, color='black',
         label=None, scatter=False, loglog=False):

    if scatter:
        ax.scatter(x, y, marker='x', color=color, label=label, linewidth=3, s=100)
    else:
        ax.plot(x, y, label=label, color=color)

    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if label is not None:
        ax.legend(loc='lower right')

    if std is not None:
        ax.plot(x, y - 2.0 * std, color=color, alpha=0.25)
        ax.plot(x, y + 2.0 * std, color=color, alpha=0.25)
        ax.fill_between(x,
                        y - 2.0 * std,
                        y + 2.0 * std,
                        color=color, alpha=0.1)


def plot_base(x_train, y_train, x_range, ax):
    plot(x_train, y_train, ax, scatter=True, color='blue', label='Data D')
    plot(x_range, f(x_range), ax, color='blue', label='f(x)')


def fit_regressor(x, y, alt_kernel=False):
    pyro.clear_param_store()

    if alt_kernel:
        k0 = Periodic(input_dim=1)
        k0.lengthscale = pyro.nn.PyroSample(dist.LogNormal(torch.tensor(-1.0), torch.tensor(1.0)))
        k0.variance = pyro.nn.PyroSample(dist.LogNormal(torch.tensor(0.0), torch.tensor(2.0)))
        k0.period = pyro.nn.PyroSample(dist.LogNormal(torch.tensor(0.0), torch.tensor(2.0)))

        k1 = RBF(input_dim=1)
        k1.lengthscale = pyro.nn.PyroSample(dist.LogNormal(torch.tensor(-1.0), torch.tensor(1.0)))
        k1.variance = pyro.nn.PyroSample(dist.LogNormal(torch.tensor(0.0), torch.tensor(2.0)))

        kernel = Product(k0, k1)
        regressor = GPRegression(x, y, kernel, noise=torch.tensor(0.0001))
        mcmc = MCMC(NUTS(regressor.model, jit_compile=True, ignore_jit_warnings=True),
                    num_samples=100, num_chains=5, warmup_steps=300)
        mcmc.run()

        k0_lscale = mcmc.get_samples()['kernel.kern0.lengthscale'].mean()
        k0_variance = mcmc.get_samples()['kernel.kern0.variance'].mean()
        k0 = Periodic(input_dim=1, lengthscale=k0_lscale, variance=k0_variance)

        k1_lscale = mcmc.get_samples()['kernel.kern1.lengthscale'].mean()
        k1_variance = mcmc.get_samples()['kernel.kern1.variance'].mean()
        k1 = RBF(input_dim=1, lengthscale=k1_lscale, variance=k1_variance)

        kernel = Product(k0, k1)
        regressor = GPRegression(x, y, kernel, noise=torch.tensor(0.0001))
    else:
        kernel = RBF(input_dim=1)
        kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(torch.tensor(-1.0), torch.tensor(1.0)))
        kernel.variance = pyro.nn.PyroSample(dist.LogNormal(torch.tensor(0.0), torch.tensor(2.0)))

        regressor = GPRegression(x, y, kernel, noise=torch.tensor(0.0001))
        mcmc = MCMC(NUTS(regressor.model, jit_compile=True, ignore_jit_warnings=True),
                    num_samples=100, num_chains=5, warmup_steps=300)
        mcmc.run()

        lscale = mcmc.get_samples()['kernel.lengthscale'].mean()
        variance = mcmc.get_samples()['kernel.variance'].mean()

        kernel = RBF(input_dim=1, lengthscale=lscale, variance=variance)
        regressor = GPRegression(x, y, kernel, noise=torch.tensor(0.0001))

    return regressor, mcmc


def mcmc_diag(mcmc):
    arviz_data = arviz.from_pyro(mcmc)
    mcmc.summary()
    print()
    print(arviz.summary(arviz_data))
    print()
    arviz.plot_trace(arviz_data, figsize=(16, 9), rug=True, rug_kwargs={'alpha': 0.2})
    arviz.plot_autocorr(arviz_data)
    plt.show()


def plot_loglog(mcmc):
    samples = mcmc.get_samples()
    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
    fig.suptitle('Log-log scatter plot of N = 500 from the posterior')
    ax.set_xlabel('lengthscale')
    ax.set_ylabel('variance')
    plot(samples['kernel.lengthscale'],
         samples['kernel.variance'],
         ax, scatter=True, loglog=True, color='Blue', label='Sample pair')
    plt.show()


def plot_deliverable(x_train, y_train, x_range, regressor):
    with torch.no_grad():
        mean, vars = regressor(x_range, noiseless=True)

    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    fig.suptitle('Plot of f(x), data D and m(x*) ± std(x*)')
    ax.set_xlabel('x*')
    ax.set_ylabel('f')
    plot_base(x_train, y_train, x_range, ax)
    plot(x_range, mean, ax, std=vars.sqrt(), color='red', label='m(x*) ± std(x*)')
    plt.show()


def bayesian_opt(x, y, x_range, orig_reg, alt_kernel=False, iters=10):
    x_opt, y_opt = x.tolist(), y.tolist()

    with torch.no_grad():
        mean_start, var_start = orig_reg(x_range,  noiseless=True)

    return_dict = {}

    for iteration in range(1, iters + 1):
        if iteration != 1:
            print('Iteration', iteration)

            x_opt_tnsr = torch.tensor(x_opt)
            y_opt_tnsr = torch.tensor(y_opt)

            regressor, _ = fit_regressor(x_opt_tnsr, y_opt_tnsr, alt_kernel)

            with torch.no_grad():
                mean, var = regressor(x_range, noiseless=False)
        else:
            mean = mean_start
            var = var_start

        new_y = dist.MultivariateNormal(mean_start, covariance_matrix=torch.diag(var)).sample()

        min_index = torch.argmin(new_y)
        pair = (x_range[min_index], f(x_range[min_index]))

        x_opt.append(pair[0])
        y_opt.append(pair[1])

        return_dict[f'iters_{iteration}'] = {
            'pair': pair,
            'mean': mean,
            'std': var.sqrt()
        }

        if iteration == iters:
            break

    return return_dict


def plot_bayesian_opt(iter_data, x_train, y_train, x_range, orig_reg):
    with torch.no_grad():
        mean, vars = orig_reg(x_range, noiseless=True)

    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    fig.suptitle('Bayesian optimization plots (green for k = 10)')
    ax.set_xlabel('x*')
    ax.set_ylabel('f')

    plot_base(x_train, y_train, x_range, ax)

    plot(x_range, mean, ax, std=vars.sqrt(), color='red', label='m(x*) ± std(x*)')

    plot(iter_data['iters_5']['pair'][0],
         iter_data['iters_5']['pair'][1], ax,
         scatter=True, color='lightskyblue',
         label=f'Pair (x*p, f(x*p)), k = 5')

    plot(iter_data['iters_10']['pair'][0],
         iter_data['iters_10']['pair'][1], ax,
         scatter=True, color='forestgreen',
         label=f'Pair (x*p, f(x*p)), k = 10')

    plot(x_range, iter_data['iters_5']['mean'], ax,
         std=iter_data['iters_5']['std'],
         color='lightskyblue',
         label=f'm(x*), k = 5')

    plot(x_range, iter_data['iters_10']['mean'], ax,
         std=iter_data['iters_10']['std'],
         color='forestgreen',
         label=f'm(x*), k = 10')

    plt.show()


def main():
    x_range = torch.linspace(-1, 1, 200)
    dataset = generate_dataset()
    x_train, y_train = dataset[:, 0].float(), dataset[:, 1].float()

    # Fit model
    # orig_reg = joblib.load('orig_reg_gold.o')
    orig_reg, mcmc = fit_regressor(x_train, y_train)
    # joblib.dump(orig_reg, f'./orig_reg{time.time()}.o')

    # MCMC diagnostics
    mcmc_diag(mcmc)

    return
    # Log-log plot
    plot_loglog(mcmc)

    # B1 deliverable plot
    plot_deliverable(x_train, y_train, x_range, orig_reg)

    # Bayesian optimization
    # iter_data = joblib.load('iter_data_final.o')
    iter_data = bayesian_opt(x_train, y_train, x_range, orig_reg)
    joblib.dump(iter_data, f'./iter_data_{time.time()}.o')

    # B2 deliverable plot
    plot_bayesian_opt(iter_data, x_train, y_train, x_range, orig_reg)

    print('DONE')


def main_periodic():
    x_range = torch.linspace(-1, 1, 200)
    dataset = generate_dataset()
    x_train, y_train = dataset[:, 0].float(), dataset[:, 1].float()

    # Fit model
    # orig_reg = joblib.load('orig_reg_periodic.o')
    orig_reg, mcmc = fit_regressor(x_train, y_train, alt_kernel=True)
    joblib.dump(orig_reg, f'./orig_reg_periodic_{time.time()}.o')

    plot_deliverable(x_train, y_train, x_range, orig_reg)

    iter_data = bayesian_opt(x_train, y_train, x_range, orig_reg)
    joblib.dump(iter_data, f'./iter_data_periodic_{time.time()}.o')

    plot_bayesian_opt(iter_data, x_train, y_train, x_range, orig_reg)


if __name__ == '__main__':
    main()
    # main_periodic()
