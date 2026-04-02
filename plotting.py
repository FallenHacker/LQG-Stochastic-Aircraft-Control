import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_timeseries(t, x, xhat, u, outdir, tag='lqg'):
    ensure_dir(outdir)
    theta, q = x[:, 0], x[:, 1]
    theta_h, q_h = xhat[:, 0], xhat[:, 1]

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t, theta, label='theta (true)')
    ax[0].plot(t, theta_h, '--', label='theta_hat')
    ax[0].set_ylabel('rad')
    ax[0].legend()

    ax[1].plot(t, q, label='q (true)')
    ax[1].plot(t, q_h, '--', label='q_hat')
    ax[1].set_ylabel('rad/s')
    ax[1].legend()

    ax[2].plot(t, u, label='u')
    ax[2].set_ylabel('rad')
    ax[2].set_xlabel('time (s)')
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'timeseries_{tag}.png'), dpi=200)
    plt.close(fig)


def plot_mc_envelope(t, theta_mat, outdir, tag='mc'):
    ensure_dir(outdir)
    mean = np.mean(theta_mat, axis=0)
    std = np.std(theta_mat, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, mean, label='mean')
    ax.fill_between(t, mean - 2*std, mean + 2*std, alpha=0.3, label='±2σ')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('theta (rad)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'envelope_{tag}.png'), dpi=200)
    plt.close(fig)
