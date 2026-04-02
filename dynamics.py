import numpy as np
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
    """
    Zero-order-hold discretization for (A,B) using matrix exponential.
    This is used to compute the discrete-time system matrices Ad, Bd from the continuous-time matrices A, B and the time step dt.
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
