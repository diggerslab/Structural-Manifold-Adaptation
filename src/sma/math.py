"""Coverage energy and bounded probability allocation."""
import numpy as np

def bounded_scale(values, total=1.0, lo=0.0, hi=1.0):
    """KL projection onto a capped simplex by a single multiplicative factor."""
    a = np.asarray(values, np.float64)
    if not (np.isfinite(a).all() and (a >= 0).all()):
        raise ValueError('Invalid SMA inputs or state.')
    if not (len(a) * lo <= total + 1e-12 and len(a) * hi >= total - 1e-12):
        raise ValueError('Invalid SMA inputs or state.')
    a = np.maximum(a, 1e-15)
    left = 0.0
    right = max(1.0, total / a.min())
    for _ in range(100):
        mid = (left + right) / 2
        if np.clip(a * mid, lo, hi).sum() < total:
            left = mid
        else:
            right = mid
    return np.clip(a * ((left + right) / 2), lo, hi)

def diffusion_kernel(W):
    W = np.asarray(W, np.float64)
    if not (np.allclose(W, W.T, atol=1e-12) and (W >= 0).all()):
        raise ValueError('Invalid SMA inputs or state.')
    degree = W.sum(1)
    inv = np.zeros_like(degree)
    nz = degree > 0
    inv[nz] = 1 / np.sqrt(degree[nz])
    L = np.diag(nz.astype(float)) - inv[:, None] * W * inv[None, :]
    eigen, Q = np.linalg.eigh(L)
    if not eigen.min() > -1e-08:
        raise ValueError('Invalid SMA inputs or state.')
    K = Q * np.exp(-np.maximum(eigen, 0)) @ Q.T
    if not (np.allclose(K, K.T, atol=1e-12) and np.linalg.eigvalsh(K).min() > 0):
        raise ValueError('Invalid SMA inputs or state.')
    return (L, K)

def coverage_coefficients(p, q, K, eta=1 / 2048, mean_weights=None):
    """Exact E(p)-E((1-eta)p+eta e_c), converted to bounded mean-one factors."""
    p, q = (np.asarray(p, np.float64), np.asarray(q, np.float64))
    d = p - q
    Kd = K @ d
    Kp = K @ p
    marginal = -2 * eta * (Kd - p @ Kd) - eta ** 2 * (np.diag(K) - 2 * Kp + p @ Kp)
    scale = max(float(np.median(np.abs(marginal))), 1e-12)
    raw = np.clip(1 + marginal / scale, 0.5, 2.0)
    average = q if mean_weights is None else np.asarray(mean_weights, np.float64)
    if not abs(average.sum() - 1) < 1e-10:
        raise ValueError('Invalid SMA inputs or state.')
    low, high = (0.0, 10.0)
    for _ in range(80):
        mid = (low + high) / 2
        if average @ np.clip(mid * raw, 0.5, 2.0) < 1:
            low = mid
        else:
            high = mid
    return (np.clip((low + high) / 2 * raw, 0.5, 2.0), marginal)
