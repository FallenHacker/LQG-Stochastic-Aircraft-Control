import numpy as np


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

def compute_lqr_cost(x_hist, u_hist, dt, Q, Ru):
    """Compute discrete approximation of continuous LQR cost J."""
    Q = np.asarray(Q, dtype=float)
    Ru = float(Ru)
    
    J = 0.0
    for k in range(len(u_hist)):
        x_k = x_hist[k]
        u_k = u_hist[k]
        state_cost = x_k.T @ Q @ x_k
        control_cost = Ru * (u_k ** 2)
        J += (state_cost + control_cost) * dt
        
    return float(J)

def summarize_run(t, x_hist, u_hist, Q=None, Ru=None):
    theta = x_hist[:, 0]
    q = x_hist[:, 1]
    
    out = {
        'rms_theta': rms(theta),
        'rms_q': rms(q),
        'rms_u': rms(u_hist),
        'settling_time_0p05': settling_time(t, theta, band=0.05, hold_time=1.0),
    }
    
    if Q is not None and Ru is not None and len(t) > 1:
        dt = t[1] - t[0]
        out['lqr_cost'] = compute_lqr_cost(x_hist, u_hist, dt, Q, Ru)
        
    return out
