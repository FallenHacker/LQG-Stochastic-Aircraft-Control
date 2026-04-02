import numpy as np

from dynamics import (continuous_matrices, discretize_zoh,
                           euler_maruyama_step, process_noise_cov_discrete)
from control import lqr_continuous, saturate
from estimation import DiscreteKalmanFilter, ExtendedKalmanFilter


# ---------------------------------------------------------------------------
# Nonlinear pitch model for EKF
# ---------------------------------------------------------------------------

def _f_nonlinear(x, u, dt, a, b, c):
    """Euler-discretized nonlinear pitch: sin(theta) restoring moment."""
    theta, q = x
    theta_next = theta + dt * q
    q_next     = q + dt * (-a * q - b * np.sin(theta) + c * u)
    return np.array([theta_next, q_next])


def _jacobian_f(x, u, dt, a, b, c):
    """df/dx at current estimate (linearization for EKF predict)."""
    theta = x[0]
    return np.array([
        [1.0,          dt],
        [-dt * b * np.cos(theta), 1.0 - dt * a]
    ])


def _h(x):
    """Measurement: observe pitch angle only."""
    return np.array([x[0]])


def _jacobian_h(x):
    """dh/dx (constant for this model)."""
    return np.array([[1.0, 0.0]])


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

def simulate_lqg(cfg, seed=1, controller_mode='lqg', estimator='kf'):
    """Run one closed-loop simulation.

    controller_mode
    ---------------
    'open'        : u = 0  (no control)
    'lqr_perfect' : u = -K_lqr @ x_true
    'pd_perfect'  : u = -K_pd  @ x_true
    'lqg'         : u = -K_lqr @ xhat  (uses estimator)
    'pd_noisy'    : u = -K_pd  @ xhat  (uses estimator)

    estimator
    ---------
    'kf'  : linear Discrete Kalman Filter  (linear plant model)
    'ekf' : Extended Kalman Filter         (nonlinear sin(theta) model)
    """
    rng = np.random.default_rng(seed)

    dt = float(cfg['sim']['dt'])
    T  = float(cfg['sim']['T'])
    N  = int(np.round(T / dt))
    t  = np.arange(N) * dt

    a     = float(cfg['plant']['a'])
    b     = float(cfg['plant']['b'])
    c     = float(cfg['plant']['c'])
    sigma = float(cfg['plant']['sigma'])

    A, B, G = continuous_matrices(a, b, c, sigma)
    Ad, Bd  = discretize_zoh(A, B, dt)
    Qd      = process_noise_cov_discrete(G, dt)

    C       = np.array(cfg['measurement']['C'], dtype=float)
    sigma_v = float(cfg['measurement']['sigma_v'])
    R_meas  = sigma_v ** 2

    Q_lqr = np.array(cfg['lqr']['Q'], dtype=float)
    Ru    = float(cfg['lqr']['Ru'])
    K_lqr, _ = lqr_continuous(A, B, Q_lqr, Ru)
    K_lqr = np.asarray(K_lqr, dtype=float).reshape(-1)

    Kp   = float(cfg.get('pd', {}).get('Kp', 2.0))
    Kd   = float(cfg.get('pd', {}).get('Kd', 1.0))
    K_pd = np.array([Kp, Kd], dtype=float)

    u_max = cfg.get('limits', {}).get('u_max', None)

    # --- Build estimator ---
    if estimator == 'kf':
        filt = DiscreteKalmanFilter(Ad, Bd, C, Qd, R_meas)
    elif estimator == 'ekf':
        # bind plant parameters into closures
        def f(x, u, dt_): return _f_nonlinear(x, u, dt_, a, b, c)
        def jf(x, u, dt_): return _jacobian_f(x, u, dt_, a, b, c)
        filt = ExtendedKalmanFilter(
            f=f, h=_h,
            jacobian_f=jf, jacobian_h=_jacobian_h,
            Qd=Qd, R=R_meas, n=2
        )
    else:
        raise ValueError(f"Unknown estimator: '{estimator}'. Use 'kf' or 'ekf'.")

    x = np.array([0.2, 0.0], dtype=float)
    filt.reset(xhat0=np.zeros(2), P0=np.eye(2))

    x_hist    = np.zeros((N, 2))
    xhat_hist = np.zeros((N, 2))
    u_hist    = np.zeros(N)
    y_hist    = np.zeros(N)

    for k in range(N):
        v = sigma_v * rng.standard_normal()
        y = float(C @ x + v)

        # --- control law ---
        if controller_mode == 'open':
            u = 0.0
        elif controller_mode == 'lqr_perfect':
            u = -(K_lqr @ x)
        elif controller_mode == 'pd_perfect':
            u = -(K_pd @ x)
        elif controller_mode == 'lqg':
            u = -(K_lqr @ filt.xhat)
        elif controller_mode == 'pd_noisy':
            u = -(K_pd @ filt.xhat)
        else:
            raise ValueError(f"Unknown controller_mode: '{controller_mode}'")

        u = saturate(float(np.asarray(u).squeeze()), u_max)

        x_hist[k]    = x
        xhat_hist[k] = filt.xhat
        u_hist[k]    = u
        y_hist[k]    = y

        # --- true plant: always nonlinear SDE ---
        x = euler_maruyama_step(x, u, A, B, G, dt, rng)

        # --- filter step ---
        if estimator == 'kf':
            filt.predict(u)
        else:
            filt.predict(u, dt)
        filt.update(y)

    return {
        't':        t,
        'x':        x_hist,
        'xhat':     xhat_hist,
        'u':        u_hist,
        'y':        y_hist,
        'K_lqr':    K_lqr,
        'K_pd':     K_pd,
        'cfg':      cfg,
        'mode':     controller_mode,
        'estimator': estimator,
    }
