"""Panoptic 3D skeletons in the PanopticSports (D3G) world, plus the cues the replay director uses.

The raw capture 161029_sports1 has one body3DScene_<frame>.json per HD frame (COCO-19 joints, cm, Panoptic
world). PanopticSports clips are 150-frame cuts of the same capture with their own world frame. We map one to the
other with a similarity transform fitted on the 31 HD camera centres (same cameras, same names), and find each
clip's start frame by checking that projected joints land on the clip's foreground masks.
NumPy only (no GPU).
"""
import glob, json, os
import numpy as np

# Panoptic COCO-19 joint order
NECK, NOSE, PELVIS, L_SH, L_EL, L_WR, L_HIP, L_KN, L_AN, R_SH, R_EL, R_WR, R_HIP, R_KN, R_AN, L_EYE, L_EAR, R_EYE, R_EAR = range(19)
WRISTS = (L_WR, R_WR)
CONF = 0.2


# ------------------------------------------------------------------ geometry
def umeyama(src, dst):
    """Similarity s, R, T minimising |dst - (s R src + T)|^2 (both (N,3))."""
    mu_s, mu_d = src.mean(0), dst.mean(0); A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def apply_sim(sim, X):
    return sim['s'] * X @ np.asarray(sim['R']).T + np.asarray(sim['T'])


def project(X, K, w2c):
    """(N,3) world -> (N,2) pixels and depth, D3G convention (w2c 4x4, K 3x3)."""
    M = np.asarray(w2c); K = np.asarray(K)
    Xc = X @ M[:3, :3].T + M[:3, 3]; z = Xc[:, 2]
    uv = (Xc @ K.T)[:, :2] / np.maximum(z[:, None], 1e-6)
    return uv, z


def horiz(v, up):
    v = np.asarray(v, float); return v - (v @ up)[..., None] * up


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True); return v / np.maximum(n, 1e-9)


def gsmooth(x, sigma):
    """Gaussian smoothing along axis 0 that ignores NaNs."""
    x = np.asarray(x, float)
    if sigma <= 0: return x
    r = int(3 * sigma); g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    flat = x.reshape(len(x), -1); out = np.empty_like(flat)
    for i in range(flat.shape[1]):
        v = flat[:, i]; ok = ~np.isnan(v); vv = np.where(ok, v, 0.0)
        num = np.convolve(np.pad(vv, r, mode='edge'), g, 'valid'); den = np.convolve(np.pad(ok.astype(float), r, mode='edge'), g, 'valid')
        out[:, i] = np.where(den > 1e-6, num / np.maximum(den, 1e-9), np.nan)
    return out.reshape(x.shape)


# ------------------------------------------------------------------ raw Panoptic files
def load_calib(raw_dir):
    f = glob.glob(os.path.join(raw_dir, 'calibration_*.json'))[0]
    cams = {c['name']: c for c in json.load(open(f))['cameras'] if c.get('type') == 'hd'}
    return cams


def pose_frames(raw_dir):
    files = glob.glob(os.path.join(raw_dir, 'hdPose3d_stage1_coco19', 'body3DScene_*.json'))
    return sorted(int(os.path.basename(f)[12:20]) for f in files)


def load_bodies(raw_dir, frame):
    p = os.path.join(raw_dir, 'hdPose3d_stage1_coco19', f'body3DScene_{frame:08d}.json')
    if not os.path.exists(p): return []
    return [(int(b['id']), np.array(b['joints19'], float).reshape(19, 4)) for b in json.load(open(p)).get('bodies', [])]


def d3g_camera_centres(md, tmd):
    """node id (int, from 'node/frame.jpg') -> camera centre in the D3G world, for train + test cameras."""
    out = {}
    for m in (md, tmd):
        for fn, w2c in zip(m['fn'][0], m['w2c'][0]):
            out[int(fn.split('/')[0])] = np.linalg.inv(np.asarray(w2c))[:3, 3]
    return out


def fit_similarity(calib, md, tmd):
    """Panoptic world (cm) -> D3G world, fitted on the HD camera centres. Returns dict + rms residual."""
    cen = d3g_camera_centres(md, tmd); src, dst = [], []
    for node, c in cen.items():
        cam = calib.get(f'00_{node:02d}')
        if cam is None: continue
        R, t = np.asarray(cam['R']), np.asarray(cam['t']).reshape(3)
        src.append(-R.T @ t); dst.append(c)
    src, dst = np.array(src), np.array(dst)
    s, R, T = umeyama(src, dst)
    res = np.sqrt(((s * src @ R.T + T - dst) ** 2).sum(1).mean())
    return {'s': float(s), 'R': R.tolist(), 'T': T.tolist(), 'n_cams': len(src), 'rms': float(res)}


# ------------------------------------------------------------------ loading an aligned clip
def load_aligned(path):
    """results/skeletons/<seq>.json -> dict(ids, joints (T, B, 19, 4) with NaN where missing, meta)."""
    d = json.load(open(path)); frames = d['frames']; T = len(frames)
    ids = sorted({b['id'] for fr in frames for b in fr})
    J = np.full((T, len(ids), 19, 4), np.nan)
    for t, fr in enumerate(frames):
        for b in fr: J[t, ids.index(b['id'])] = np.array(b['j'], float)
    J[..., :3][J[..., 3] < CONF] = np.nan
    return {'ids': ids, 'joints': J, 'meta': {k: v for k, v in d.items() if k != 'frames'}}


# ------------------------------------------------------------------ director cues
def director_cues(sk, up, T=None, window=(0.2, 0.8)):
    """Key athlete, key moment and facing direction from the aligned skeletons.
    key athlete = the body whose wrists move fastest inside the window; key moment = that peak (release of a
    throw, contact of a swing, top of a lift); facing = horizontal torso normal, sign fixed by the face joints."""
    J = sk['joints']; T = T or J.shape[0]; up = unit(np.asarray(up, float))
    lo, hi = int(window[0] * T), int(window[1] * T)
    best = None
    for b in range(J.shape[1]):
        W = J[:, b, list(WRISTS), :3]                                     # (T, 2, 3)
        if np.isnan(W[:, :, 0]).all(axis=1).mean() > 0.5: continue        # body missing most of the clip
        sp = np.linalg.norm(np.diff(W, axis=0), axis=2)                   # (T-1, 2) per-frame wrist speed
        sp = np.nanmax(np.where(np.isnan(sp), -np.inf, sp), axis=1); sp[np.isinf(sp)] = np.nan
        sp = gsmooth(np.concatenate([[sp[0]], sp]), 2.0)
        win = sp[lo:hi]
        if np.isnan(win).all(): continue
        k = lo + int(np.nanargmax(win))
        if best is None or sp[k] > best[2]: best = (b, k, sp[k], sp)
    if best is None: return None
    b, key, _, speed = best
    P = J[:, b, :, :3]
    across = (P[:, R_SH] - P[:, L_SH]) + 0.5 * (P[:, R_HIP] - P[:, L_HIP])
    n = unit(horiz(np.cross(up, across), up))                             # one of the two torso normals
    face = np.nanmean(P[:, [NOSE, L_EYE, R_EYE]], axis=1) - P[:, NECK]
    sgn = np.sign(np.einsum('ij,ij->i', n, np.nan_to_num(horiz(face, up))))
    for t in range(T):                                                    # keep the sign when the face is not seen
        if not np.isfinite(sgn[t]) or sgn[t] == 0: sgn[t] = sgn[t - 1] if t else 1.0
    facing = gsmooth(n * sgn[:, None], 4.0)
    for t in range(T):
        if np.isnan(facing[t]).any(): facing[t] = facing[t - 1] if t else np.array([1.0, 0, 0])
    centre = gsmooth(P[:, PELVIS], 2.0)
    for t in range(T):
        if np.isnan(centre[t]).any(): centre[t] = centre[t - 1] if t else np.nanmean(P[:, PELVIS], 0)
    return {'key_body': sk['ids'][b], 'key_frame': int(key), 'speed': speed, 'facing': unit(horiz(facing, up)),
            'centre': centre, 'n_bodies': J.shape[1]}
