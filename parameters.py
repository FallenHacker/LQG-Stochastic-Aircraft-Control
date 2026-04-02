import numpy as np


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
