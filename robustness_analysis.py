import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from simulate import _f_nonlinear, _jacobian_f, _h, _jacobian_h
from dynamics import continuous_matrices, discretize_zoh, euler_maruyama_step, process_noise_cov_discrete
from control import lqr_continuous, saturate
from estimation import DiscreteKalmanFilter, ExtendedKalmanFilter
from metrics import summarize_run
from monte_carlo import CFG

def simulate_robustness(cfg_nom, cfg_true, seed=1, mode='lqg', estimator='kf'):
    rng = np.random.default_rng(seed)
    dt = float(cfg_nom['sim']['dt'])
    T = float(cfg_nom['sim']['T'])
    N = int(np.round(T / dt))
    t = np.arange(N) * dt
    a_nom, b_nom, c_nom = cfg_nom['plant']['a'], cfg_nom['plant']['b'], cfg_nom['plant']['c']
    A_nom, B_nom, G_nom = continuous_matrices(a_nom, b_nom, c_nom, cfg_nom['plant']['sigma'])
    Ad_nom, Bd_nom = discretize_zoh(A_nom, B_nom, dt)
    Qd_nom = process_noise_cov_discrete(G_nom, dt)
    a_true, b_true, c_true = cfg_true['plant']['a'], cfg_true['plant']['b'], cfg_true['plant']['c']
    A_true, B_true, G_true = continuous_matrices(a_true, b_true, c_true, cfg_true['plant']['sigma'])
    C = np.array(cfg_nom['measurement']['C'], dtype=float)
    sigma_v = float(cfg_nom['measurement']['sigma_v'])
    R_meas = sigma_v**2
    Q_lqr = np.array(cfg_nom['lqr']['Q'], dtype=float)
    Ru = float(cfg_nom['lqr']['Ru'])
    K_lqr, _ = lqr_continuous(A_nom, B_nom, Q_lqr, Ru)
    K_lqr = np.asarray(K_lqr).flatten()
    Kp, Kd = cfg_nom.get('pd', {}).get('Kp', 2.0), cfg_nom.get('pd', {}).get('Kd', 1.5)
    K_pd = np.array([Kp, Kd])
    u_max = cfg_nom.get('limits', {}).get('u_max', None)
    if estimator == 'kf':
        filt = DiscreteKalmanFilter(Ad_nom, Bd_nom, C, Qd_nom, R_meas)
    else:
        def f(x, u, dt_): return _f_nonlinear(x, u, dt_, a_nom, b_nom, c_nom)
        def jf(x, u, dt_): return _jacobian_f(x, u, dt_, a_nom, b_nom, c_nom)
        filt = ExtendedKalmanFilter(f, _h, jf, _jacobian_h, Qd_nom, R_meas, 2)
    x = np.array([0.2, 0.0])
    filt.reset(xhat0=np.zeros(2), P0=np.eye(2))
    x_hist, u_hist = np.zeros((N, 2)), np.zeros(N)
    for k in range(N):
        v = sigma_v * rng.standard_normal()
        y = float(C @ x + v)
        u = -(K_lqr @ filt.xhat) if mode == 'lqg' else -(K_pd @ filt.xhat)
        u = saturate(u, u_max)
        x_hist[k], u_hist[k] = x, u
        x = euler_maruyama_step(x, u, A_true, B_true, G_true, dt, rng)
        if estimator == 'kf': filt.predict(u)
        else: filt.predict(u, dt)
        filt.update(y)
    return t, x_hist, u_hist

def main():
    variations = np.linspace(0.5, 1.5, 11)
    results = {'lqg': [], 'pd_noisy': []}
    for var in variations:
        cfg_true = copy.deepcopy(CFG)
        cfg_true['plant']['b'] *= var
        costs_lqg, costs_pd = [], []
        for i in range(20):
            t, x, u = simulate_robustness(CFG, cfg_true, seed=i, mode='lqg')
            costs_lqg.append(summarize_run(t, x, u, Q=CFG['lqr']['Q'], Ru=CFG['lqr']['Ru'])['lqr_cost'])
            t, x, u = simulate_robustness(CFG, cfg_true, seed=i, mode='pd_noisy')
            costs_pd.append(summarize_run(t, x, u, Q=CFG['lqr']['Q'], Ru=CFG['lqr']['Ru'])['lqr_cost'])
        results['lqg'].append(np.mean(costs_lqg))
        results['pd_noisy'].append(np.mean(costs_pd))
    plt.figure(figsize=(10, 6))
    plt.plot(variations, results['lqg'], 'o-', label='LQG (Est)')
    plt.plot(variations, results['pd_noisy'], 's-', label='PD (Est)')
    plt.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Nominal')
    plt.xlabel('Stiffness Scale Factor (True b / Nominal b)')
    plt.ylabel('Mean Quadratic Cost J')
    plt.title('Robustness Analysis: Sensitivity to Pitch Stiffness Mismatch')
    plt.legend()
    plt.grid(True)
    os.makedirs('outputs/robustness', exist_ok=True)
    plt.savefig('outputs/robustness/stiffness_sweep.png')
    print("Results saved to outputs/robustness/stiffness_sweep.png")

if __name__ == '__main__':
    main()