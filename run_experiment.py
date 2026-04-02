import os
import json

import numpy as np

from simulate import simulate_lqg
from metrics import summarize_run
from plotting import plot_timeseries
from monte_carlo import CFG

def run_single(cfg=CFG, mode='lqg', seed=None):
    """Run one deterministic simulation and save plot + metrics."""
    if seed is None:
        seed = cfg.get('seed', 1)

    run_id = cfg.get('run_id', 'run')
    outdir = os.path.join('outputs', f"{run_id}_{mode}_seed{seed}")
    os.makedirs(outdir, exist_ok=True)

    Q  = cfg['lqr']['Q']
    Ru = cfg['lqr']['Ru']

    sim = simulate_lqg(cfg, seed=seed, controller_mode=mode, estimator='ekf')
    t, x, xhat, u = sim['t'], sim['x'], sim['xhat'], sim['u']

    metrics = summarize_run(t, x, u, Q=Q, Ru=Ru)

    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_timeseries(t, x, xhat, u, outdir, tag=mode)

    print(f'Saved to: {outdir}')
    print(f'Metrics : {metrics}')
    return sim, metrics


if __name__ == '__main__':
    sim, metrics = run_single(mode='lqg', seed=42)