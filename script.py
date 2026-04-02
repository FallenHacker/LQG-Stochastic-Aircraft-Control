import os, textwrap, json, numpy as np

files = {}

files['README.md'] = r'''# Stochastic Aircraft Pitch Control (LQG) — Reproducible Research Code

This repo implements **aircraft pitch stabilization under turbulence** using an LQG controller:
- Plant: continuous-time linear pitch dynamics with additive stochastic turbulence
- Estimator: discrete-time Kalman filter
- Controller: continuous-time LQR gain applied to estimated state
- Validation: Monte Carlo runs + metrics + plots

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

python run_experiment.py --config configs/baseline.yaml
```

Outputs go to `outputs/<run_id>/`.

## What to run
- `run_experiment.py`: single run (deterministic seed) + plots + metrics
- `monte_carlo.py`: batch Monte Carlo + summary CSV + plots

## Reproducibility
All scripts accept `--seed`. Monte Carlo uses a controlled seed sequence.
'''

files['requirements.txt'] = r'''numpy>=1.23
scipy>=1.10
matplotlib>=3.7
pyyaml>=6.0
pandas>=2.0
'''

files['configs/baseline.yaml'] = r'''run_id: baseline
seed: 1

sim:
  dt: 0.01
  T: 20.0

plant:
  a: 0.5
  b: 0.8
  c: 1.0
  sigma: 0.10

measurement:
  C: [1.0, 0.0]
  sigma_v: 0.05

lqr:
  Q: [[10.0, 0.0], [0.0, 1.0]]
  Ru: 1.0

limits:
  u_max: 0.52   # ~30 deg in rad

monte_carlo:
  N: 200
'''

files['src/parameters.py'] = r'''import numpy as np


def as_array(x, shape=None, dtype=float):
    a = np.array(x, dtype=dtype)
    if shape is not None and tuple(a.shape) != tuple(shape):
        raise ValueError(f"Expected shape {shape}, got {a.shape}")
    return a


def load_config(path):
    import yaml
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
'''

files['src/dynamics.py'] = r'''import numpy as np
from scipy.linalg import expm


def continuous_matrices(a, b, c, sigma):
    A = np.array([[0.0, 1.0],
                  [-b, -a]])
    B = np.array([[0.0],
                  [c]])
    G = np.array([[0.0],
                  [sigma]])
    return A, B, G


def discretize_zoh(A, B, dt):
    """Zero-order-hold discretization for (A,B) using matrix exponential.

    Returns Ad, Bd.
    """
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


def euler_maruyama_step(x, u, A, B, G, dt, rng):
    """One Euler–Maruyama step for dx = (A x + B u) dt + G dW.

    dW ~ N(0, dt).
    """
    dW = np.sqrt(dt) * rng.standard_normal(size=(G.shape[1],))
    drift = (A @ x + (B @ np.atleast_1d(u))).reshape(-1)
    x_next = x + drift * dt + (G @ dW).reshape(-1)
    return x_next


def process_noise_cov_discrete(G, dt):
    """Simple approximation: Qd ≈ G G^T dt, consistent with dW variance dt."""
    return (G @ G.T) * dt
'''

files['src/control.py'] = r'''import numpy as np
from scipy.linalg import solve_continuous_are


def lqr_continuous(A, B, Q, Ru):
    """Continuous-time LQR: minimize ∫ (x^T Q x + u^T Ru u) dt."""
    Q = np.array(Q, dtype=float)
    Ru = np.array([[Ru]], dtype=float) if np.isscalar(Ru) else np.array(Ru, dtype=float)
    P = solve_continuous_are(A, B, Q, Ru)
    K = np.linalg.solve(Ru, B.T @ P)  # Ru^{-1} B^T P
    return K, P


def saturate(u, u_max):
    if u_max is None:
        return u
    return float(np.clip(u, -u_max, u_max))
'''

files['src/estimation.py'] = r'''import numpy as np


class DiscreteKalmanFilter:
    def __init__(self, Ad, Bd, C, Qd, R):
        self.Ad = np.array(Ad, dtype=float)
        self.Bd = np.array(Bd, dtype=float)
        self.C = np.array(C, dtype=float).reshape(1, -1)
        self.Qd = np.array(Qd, dtype=float)
        self.R = float(R)

        n = self.Ad.shape[0]
        self.xhat = np.zeros(n)
        self.P = np.eye(n)

    def reset(self, xhat0=None, P0=None):
        if xhat0 is not None:
            self.xhat = np.array(xhat0, dtype=float).reshape(-1)
        if P0 is not None:
            self.P = np.array(P0, dtype=float)

    def predict(self, u):
        u = float(u)
        self.xhat = (self.Ad @ self.xhat + (self.Bd.reshape(-1) * u))
        self.P = self.Ad @ self.P @ self.Ad.T + self.Qd
        return self.xhat, self.P

    def update(self, y):
        y = float(y)
        C = self.C
        S = float(C @ self.P @ C.T + self.R)
        K = (self.P @ C.T).reshape(-1) / S  # (n,)

        innovation = y - float(C @ self.xhat)
        self.xhat = self.xhat + K * innovation

        I = np.eye(self.P.shape[0])
        self.P = (I - np.outer(K, C.reshape(-1))) @ self.P
        return self.xhat, self.P, K, innovation
'''

files['src/metrics.py'] = r'''import numpy as np


def rms(x):
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.square(x))))


def settling_time(t, theta, band=0.05, hold_time=1.0):
    """First time after which |theta| <= band for at least hold_time."""
    t = np.asarray(t)
    theta = np.asarray(theta)
    if len(t) < 2:
        return np.nan
    dt = t[1] - t[0]
    hold_n = int(np.ceil(hold_time / dt))
    for k in range(len(t) - hold_n):
        if np.all(np.abs(theta[k:k+hold_n]) <= band):
            return float(t[k])
    return np.nan


def summarize_run(t, x_hist, u_hist):
    theta = x_hist[:, 0]
    q = x_hist[:, 1]
    out = {
        'rms_theta': rms(theta),
        'rms_q': rms(q),
        'rms_u': rms(u_hist),
        'settling_time_0p05': settling_time(t, theta, band=0.05, hold_time=1.0),
    }
    return out
'''

files['src/plotting.py'] = r'''import os
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
'''

files['src/simulate.py'] = r'''import numpy as np

from .dynamics import continuous_matrices, discretize_zoh, euler_maruyama_step, process_noise_cov_discrete
from .control import lqr_continuous, saturate
from .estimation import DiscreteKalmanFilter


def simulate_lqg(cfg, seed=1, controller_mode='lqg'):
    """Run one simulation.

    controller_mode:
      - 'open': u = 0
      - 'lqr_perfect': u = -K x (true state)
      - 'lqg': u = -K xhat (Kalman estimate)
    """
    rng = np.random.default_rng(seed)

    dt = float(cfg['sim']['dt'])
    T = float(cfg['sim']['T'])
    N = int(np.round(T / dt))
    t = np.arange(N) * dt

    a = float(cfg['plant']['a'])
    b = float(cfg['plant']['b'])
    c = float(cfg['plant']['c'])
    sigma = float(cfg['plant']['sigma'])

    A, B, G = continuous_matrices(a, b, c, sigma)
    Ad, Bd = discretize_zoh(A, B, dt)
    Qd = process_noise_cov_discrete(G, dt)

    C = np.array(cfg['measurement']['C'], dtype=float)
    sigma_v = float(cfg['measurement']['sigma_v'])
    R = sigma_v**2

    Q = np.array(cfg['lqr']['Q'], dtype=float)
    Ru = float(cfg['lqr']['Ru'])
    K, _ = lqr_continuous(A, B, Q, Ru)
    K = K.reshape(1, -1)

    u_max = cfg.get('limits', {}).get('u_max', None)

    kf = DiscreteKalmanFilter(Ad, Bd, C, Qd, R)

    # Initial conditions
    x = np.array([0.2, 0.0], dtype=float)  # rad, rad/s
    kf.reset(xhat0=np.zeros(2), P0=np.eye(2))

    x_hist = np.zeros((N, 2))
    xhat_hist = np.zeros((N, 2))
    u_hist = np.zeros(N)
    y_hist = np.zeros(N)

    for k in range(N):
        # measurement from current state (sample v_k)
        v = sigma_v * rng.standard_normal()
        y = float(C @ x + v)

        # control
        if controller_mode == 'open':
            u = 0.0
        elif controller_mode == 'lqr_perfect':
            u = float(-K @ x)
        elif controller_mode == 'lqg':
            u = float(-K @ kf.xhat)
        else:
            raise ValueError('Unknown controller_mode')

        u = saturate(u, u_max)

        # log
        x_hist[k] = x
        xhat_hist[k] = kf.xhat
        u_hist[k] = u
        y_hist[k] = y

        # plant step
        x = euler_maruyama_step(x, u, A, B, G, dt, rng)

        # KF update: predict then update using y
        kf.predict(u)
        kf.update(y)

    return {
        't': t,
        'x': x_hist,
        'xhat': xhat_hist,
        'u': u_hist,
        'y': y_hist,
        'K_lqr': K,
        'cfg': cfg,
        'mode': controller_mode,
    }
'''

files['run_experiment.py'] = r'''import argparse
import os
import json

import numpy as np

from src.parameters import load_config
from src.simulate import simulate_lqg
from src.metrics import summarize_run
from src.plotting import plot_timeseries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--mode', type=str, default='lqg', choices=['open', 'lqr_perfect', 'lqg'])
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get('seed', 1) if args.seed is None else args.seed
    run_id = cfg.get('run_id', 'run')

    outdir = os.path.join('outputs', f"{run_id}_{args.mode}_seed{seed}")
    os.makedirs(outdir, exist_ok=True)

    sim = simulate_lqg(cfg, seed=seed, controller_mode=args.mode)
    t, x, xhat, u = sim['t'], sim['x'], sim['xhat'], sim['u']

    metrics = summarize_run(t, x, u)

    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_timeseries(t, x, xhat, u, outdir, tag=args.mode)

    print('Saved to:', outdir)
    print('Metrics:', metrics)


if __name__ == '__main__':
    main()
'''

files['monte_carlo.py'] = r'''import argparse
import os
import json

import numpy as np
import pandas as pd

from src.parameters import load_config
from src.simulate import simulate_lqg
from src.metrics import summarize_run
from src.plotting import plot_mc_envelope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--mode', type=str, default='lqg', choices=['open', 'lqr_perfect', 'lqg'])
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    Nmc = int(cfg.get('monte_carlo', {}).get('N', 200))
    run_id = cfg.get('run_id', 'mc')

    outdir = os.path.join('outputs', f"{run_id}_MC_{args.mode}_N{Nmc}")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    theta_mat = []

    # deterministic seed sequence
    base_rng = np.random.default_rng(args.seed)
    seeds = base_rng.integers(low=0, high=2**31-1, size=Nmc, dtype=np.int64)

    for i in range(Nmc):
        sim = simulate_lqg(cfg, seed=int(seeds[i]), controller_mode=args.mode)
        t, x, u = sim['t'], sim['x'], sim['u']
        theta_mat.append(x[:, 0])

        m = summarize_run(t, x, u)
        m['trial'] = i
        m['seed'] = int(seeds[i])
        rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'mc_summary.csv'), index=False)

    theta_mat = np.array(theta_mat)
    plot_mc_envelope(t, theta_mat, outdir, tag=f"{args.mode}")

    with open(os.path.join(outdir, 'config_used.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    print('Saved to:', outdir)
    print(df.describe(include='all'))


if __name__ == '__main__':
    main()
'''

# Write all files
for path, content in files.items():
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

print('Wrote files:', len(files))
print('\n'.join(sorted(files.keys())[:10]))
