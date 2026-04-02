import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulate import simulate_lqg
from metrics import summarize_run
from plotting import plot_mc_envelope

# =============================================================
# EXPERIMENT CONFIGURATION — Edit all parameters here directly
# =============================================================
CFG = {
    'run_id': 'baseline',
    'seed': 1,

    'sim': {
        'dt': 0.01,
        'T': 20.0,
    },

    'plant': {
        'a': 0.5,       # pitch damping coefficient
        'b': 0.8,       # pitch stiffness coefficient
        'c': 1.0,       # control effectiveness
        'sigma': 0.10,  # turbulence intensity
    },

    'measurement': {
        'C': [1.0, 0.0],
        'sigma_v': 0.05,  # measurement noise std dev (rad)
    },

    'lqr': {
        # 'Q': [[1.0, 0.0], [0.0, 1]],  # baseline state cost weights
        # 'Ru': 1.0,                     # baseline control effort weight
        
        'Q': [[100.0, 0.0], [0.0, 25.0]],  # state cost weights (Bryson rule)
        'Ru': 3.7,                           # control effort weight
    },

    'pd': {
        'Kp': 3.16,   # proportional gain for pitch angle
        'Kd': 1.5,    # derivative gain for pitch rate
    },

    'limits': {
        'u_max': 0.52,  # ~30 deg actuator saturation limit (rad)
    },

    'monte_carlo': {
        'N': 1000,       # number of Monte Carlo trials
    },
}
# =============================================================


def run_monte_carlo(cfg, mode='lqg', seed=1, estimator='kf'):
    """
    Runs Monte Carlo simulations for a given controller mode.
    cfg   : dict of parameters (use CFG above)
    mode  : 'open' | 'lqr_perfect' | 'pd_perfect' | 'lqg' | 'pd_noisy'
    seed  : base random seed for reproducibility
    Returns a pandas DataFrame of per-trial metrics.
    """
    Nmc = int(cfg.get('monte_carlo', {}).get('N', 200))
    run_id = cfg.get('run_id', 'mc')

    outdir = os.path.join('outputs', f"{run_id}_MC_{mode}_N{Nmc}")
    os.makedirs(outdir, exist_ok=True)

    Q = np.array(cfg['lqr']['Q'], dtype=float)
    Ru = float(cfg['lqr']['Ru'])

    rows = []
    theta_mat = []

    base_rng = np.random.default_rng(seed)
    seeds = base_rng.integers(low=0, high=2**31-1, size=Nmc, dtype=np.int64)

    for i in range(Nmc):
        sim = simulate_lqg(cfg, seed=int(seeds[i]), controller_mode=mode, estimator=estimator)
        t, x, u = sim['t'], sim['x'], sim['u']
        theta_mat.append(x[:, 0])

        m = summarize_run(t, x, u, Q=Q, Ru=Ru)
        m['trial'] = i
        m['seed'] = int(seeds[i])
        m['mode'] = mode
        rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, f'mc_summary_{mode}.csv'), index=False)

    theta_mat = np.array(theta_mat)
    plot_mc_envelope(t, theta_mat, outdir, tag=f"{mode}")

    with open(os.path.join(outdir, 'config_used.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"[{mode.upper()}] MC done. Saved to: {outdir}")
    return df


def compare_metrics(dfs, modes, outdir='outputs/comparison',
                    filename='metrics_comparison.png'):
    """Side-by-side boxplots for RMS pitch error, pitch rate, control effort."""
    os.makedirs(outdir, exist_ok=True)

    metrics_to_plot = ['rms_theta', 'rms_q', 'rms_u']
    titles = ['RMS Pitch Error (rad)', 'RMS Pitch Rate (rad/s)', 'RMS Control Effort (rad)']

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 4))

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        data = [df[metric].dropna() for df in dfs]
        ax.boxplot(data, tick_labels=modes, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='black'),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    plot_path = os.path.join(outdir, filename)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Metrics plot saved to: {plot_path}")


def plot_quadratic_cost(dfs, modes, outdir='outputs/comparison',
                        filename='cost_comparison.png'):
    """Boxplot of total quadratic cost J across controllers."""
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    data = [df['lqr_cost'].dropna() for df in dfs]

    ax.boxplot(data, tick_labels=modes, patch_artist=True,
               boxprops=dict(facecolor='lightgreen', color='black'),
               medianprops=dict(color='red', linewidth=2))

    ax.set_ylabel('Cost (J)')
    ax.grid(True, alpha=0.3, axis='y')

    max_val = max([d.max() for d in data])
    min_val = min([d.median() for d in data])
    if max_val > 10 * min_val:
        ax.set_yscale('log')
        ax.set_title('Total Quadratic Cost J (Log Scale)')
    else:
        ax.set_title('Total Quadratic Cost J')

    fig.tight_layout()
    plot_path = os.path.join(outdir, filename)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Cost plot saved to: {plot_path}")


def print_summary(dfs, modes):
    all_dfs = pd.concat(dfs)
    summary = (all_dfs.groupby('mode')[['rms_theta', 'rms_q', 'rms_u', 'lqr_cost']]
               .mean().reset_index())

    order = {m: i for i, m in enumerate(
        ['open', 'lqr_perfect', 'pd_perfect', 'lqg', 'pd_noisy'])}
    summary['sort'] = summary['mode'].map(order).fillna(99)
    summary = (summary.sort_values('sort').drop('sort', axis=1).round(4))

    print("\n" + "=" * 62)
    print("               SUMMARY TABLE (MEANS)                   ")
    print("=" * 62)
    print(summary.to_string(index=False))
    print("=" * 62 + "\n")


if __name__ == '__main__':
    seed = CFG['seed']

    print("Running Monte Carlo Simulations...")
    # Note: Currently there are 2 estimator options (KF vs EKF), but in this baseline we only run KF for all modes.
    # KF : Kalman Filter with linear plant model (used for all modes in this baseline)
    # EKF: Extended Kalman Filter with nonlinear plant model (not used in this baseline, but can be added for comparison in future experiments)
    df_open     = run_monte_carlo(CFG, mode='open',         seed=seed, estimator='kf')
    df_lqr      = run_monte_carlo(CFG, mode='lqr_perfect',  seed=seed, estimator='kf')
    df_pd_perf  = run_monte_carlo(CFG, mode='pd_perfect',   seed=seed, estimator='kf')
    df_lqg      = run_monte_carlo(CFG, mode='lqg',          seed=seed, estimator='kf')
    df_pd_noisy = run_monte_carlo(CFG, mode='pd_noisy',     seed=seed, estimator='kf')

    # --- Perfect state comparison ---
    print("\n--- Perfect State Feedback ---")
    compare_metrics(
        dfs=[df_lqr, df_pd_perf, df_open],
        modes=['LQR (True)', 'PD (True)', 'Open Loop'],
        filename='comparison_perfect_state.png',
    )
    plot_quadratic_cost(
        dfs=[df_lqr, df_pd_perf, df_open],
        modes=['LQR (True)', 'PD (True)', 'Open Loop'],
        filename='cost_perfect_state.png',
    )

    # --- Noisy / estimated state comparison ---
    print("\n--- Noisy Estimated Feedback ---")
    compare_metrics(
        dfs=[df_lqg, df_pd_noisy, df_open],
        modes=['LQG (Est)', 'PD (Est)', 'Open Loop'],
        filename='comparison_noisy_state.png',
    )
    plot_quadratic_cost(
        dfs=[df_lqg, df_pd_noisy, df_open],
        modes=['LQG (Est)', 'PD (Est)', 'Open Loop'],
        filename='cost_noisy_state.png',
    )

    # --- Summary table ---
    print_summary(
        dfs=[df_open, df_lqr, df_pd_perf, df_lqg, df_pd_noisy],
        modes=['open', 'lqr_perfect', 'pd_perfect', 'lqg', 'pd_noisy'],
    )
