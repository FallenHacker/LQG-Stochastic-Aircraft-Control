import numpy as np
from scipy.linalg import solve_continuous_are


def lqr_continuous(A, B, Q, Ru):
    """Continuous-time LQR: minimizes J = integral(x'Qx + u'Ru) dt"""
    Q = np.array(Q, dtype=float)
    Ru = np.array([[Ru]], dtype=float) if np.isscalar(Ru) else np.array(Ru, dtype=float)
    P = solve_continuous_are(A, B, Q, Ru)
    K = np.linalg.solve(Ru, B.T @ P)  # Ru^{-1} * B^T * P . Similar to solve RK = B^T P but more stable if Ru is ill-conditioned.
    return K, P


def saturate(u, u_max):
    if u_max is None:
        return u
    return float(np.clip(u, -u_max, u_max))
