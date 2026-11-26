"""Gaussians between reconstructed frames, for slow motion and freeze-time (works on NumPy or PyTorch arrays).

    nearest  repeat the closest frame (what plain frame-repeat slow motion does)
    linear   straight-line blend of positions and colours, normalised quaternion blend of rotations (v6)
    cubic    Catmull-Rom curve through four neighbouring frames for positions and rotations (new): it follows
             curved paths (a swinging bat, a ball arc) instead of cutting corners between frames
`keys` lists the frames that may be used (all frames by default); the interpolation experiment hides frames by
passing only every g-th frame.
"""
import numpy as np


def _normalize(q):
    if hasattr(q, 'detach'):  # torch
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def _align(q, ref):
    """flip q where it is on the other hemisphere from ref (q and -q are the same rotation)"""
    s = (q * ref).sum(-1, keepdims=True) if not hasattr(q, 'detach') else (q * ref).sum(-1, keepdim=True)
    return q * ((s >= 0) * 2.0 - 1.0)


def _catmull(p0, p1, p2, p3, u):
    u2, u3 = u * u, u * u * u
    return 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (3 * p1 - p0 - 3 * p2 + p3) * u3)


def _bracket(t, keys):
    keys = np.asarray(keys); i = int(np.clip(np.searchsorted(keys, t, side='right') - 1, 0, len(keys) - 1))
    if i == len(keys) - 1 or keys[i] == t: return i, i, 0.0, keys
    return i, i + 1, float((t - keys[i]) / (keys[i + 1] - keys[i])), keys


def state_at(p, t, method='linear', keys=None):
    """Returns (means, colours, unit rotations) of every Gaussian at (fractional) frame t."""
    T = p['means3D'].shape[0]
    keys = np.arange(T) if keys is None else np.asarray(sorted(keys))
    i1, i2, u, keys = _bracket(float(t), keys)
    k1, k2 = int(keys[i1]), int(keys[i2])
    M, C, Q = p['means3D'], p['rgb_colors'], p['unnorm_rotations']
    q1 = _normalize(Q[k1])
    if u == 0.0 or k1 == k2: return M[k1], C[k1], q1
    q2 = _align(_normalize(Q[k2]), q1)
    if method == 'nearest':
        k = k1 if u < 0.5 else k2; return M[k], C[k], (q1 if k == k1 else q2)
    col = C[k1] * (1 - u) + C[k2] * u
    if method == 'linear':
        return M[k1] * (1 - u) + M[k2] * u, col, _normalize(q1 * (1 - u) + q2 * u)
    if method == 'cubic':
        k0 = int(keys[max(i1 - 1, 0)]); k3 = int(keys[min(i2 + 1, len(keys) - 1)])
        q0 = _align(_normalize(Q[k0]), q1); q3 = _align(_normalize(Q[k3]), q2)
        return _catmull(M[k0], M[k1], M[k2], M[k3], u), col, _normalize(_catmull(q0, q1, q2, q3, u))
    raise ValueError(method)
