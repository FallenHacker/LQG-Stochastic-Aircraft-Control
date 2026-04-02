import numpy as np


class DiscreteKalmanFilter:
    """
    Discrete-time Linear Kalman Filter.
    x_{k+1} = Ad x_k + Bd u_k + w_k,   w_k ~ N(0, Qd)
    y_k     = C x_k + v_k,              v_k ~ N(0, R)
    """
    def __init__(self, Ad, Bd, C, Qd, R):
        self.Ad = np.array(Ad, dtype=float)
        self.Bd = np.array(Bd, dtype=float)
        self.C  = np.array(C,  dtype=float).reshape(1, -1)
        self.Qd = np.array(Qd, dtype=float)
        self.R  = float(R)

        n = self.Ad.shape[0]
        self.xhat = np.zeros(n)
        self.P    = np.eye(n)

    def reset(self, xhat0=None, P0=None):
        if xhat0 is not None:
            self.xhat = np.array(xhat0, dtype=float).reshape(-1)
        if P0 is not None:
            self.P = np.array(P0, dtype=float)

    def predict(self, u):
        u = float(np.asarray(u).squeeze())
        self.xhat = self.Ad @ self.xhat + self.Bd.reshape(-1) * u
        self.P    = self.Ad @ self.P @ self.Ad.T + self.Qd
        return self.xhat, self.P

    def update(self, y):
        y = float(np.asarray(y).squeeze())
        C = self.C
        S = (C @ self.P @ C.T + self.R).item()
        K = (self.P @ C.T / S).reshape(-1)

        innovation = y - (C @ self.xhat).item()
        self.xhat  = self.xhat + K * innovation
        self.P     = (np.eye(self.P.shape[0]) - np.outer(K, C.reshape(-1))) @ self.P
        return self.xhat, self.P, K, innovation


class ExtendedKalmanFilter:
    """
    Discrete-time Extended Kalman Filter for nonlinear systems.

    x_{k+1} = f(x_k, u_k, dt) + w_k,   w_k ~ N(0, Qd)
    y_k     = h(x_k) + v_k,             v_k ~ N(0, R)

    The EKF linearizes f and h around the current estimate at every
    time step using user-supplied Jacobian functions.

    Parameters
    ----------
    f          : callable(x, u, dt) -> ndarray(n,)   nonlinear dynamics
    h          : callable(x) -> ndarray(m,)           nonlinear measurement
    jacobian_f : callable(x, u, dt) -> ndarray(n,n)  df/dx at current x
    jacobian_h : callable(x) -> ndarray(m,n)         dh/dx at current x
    Qd         : ndarray(n,n)  process noise covariance
    R          : float         measurement noise variance (scalar)
    n          : int           state dimension
    """
    def __init__(self, f, h, jacobian_f, jacobian_h, Qd, R, n):
        self.f          = f
        self.h          = h
        self.jacobian_f = jacobian_f
        self.jacobian_h = jacobian_h
        self.Qd         = np.array(Qd, dtype=float)
        self.R          = float(R)

        self.xhat = np.zeros(n)
        self.P    = np.eye(n)

    def reset(self, xhat0=None, P0=None):
        if xhat0 is not None:
            self.xhat = np.array(xhat0, dtype=float).reshape(-1)
        if P0 is not None:
            self.P = np.array(P0, dtype=float)

    def predict(self, u, dt):
        """Propagate estimate through nonlinear dynamics and linearize."""
        u  = float(np.asarray(u).squeeze())
        F  = np.array(self.jacobian_f(self.xhat, u, dt), dtype=float)
        self.xhat = np.array(self.f(self.xhat, u, dt), dtype=float).reshape(-1)
        self.P    = F @ self.P @ F.T + self.Qd
        return self.xhat, self.P

    def update(self, y):
        """Correct estimate using noisy measurement."""
        y    = float(np.asarray(y).squeeze())
        H    = np.array(self.jacobian_h(self.xhat), dtype=float).reshape(1, -1)
        yhat = float(np.asarray(self.h(self.xhat)).squeeze())

        S = (H @ self.P @ H.T + self.R).item()
        K = (self.P @ H.T / S).reshape(-1)

        innovation = y - yhat
        self.xhat  = self.xhat + K * innovation
        self.P     = (np.eye(self.P.shape[0]) - np.outer(K, H.reshape(-1))) @ self.P
        return self.xhat, self.P, K, innovation
