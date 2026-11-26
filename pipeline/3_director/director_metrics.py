"""Numbers that describe how well a planned replay camera behaves (NumPy only), used by the ablation study.

All distances are judged against the floater-filtered players, whatever the variant did itself.
  clip_pct        video frames where the camera is closer than the clearance to a player point
  min_clear_R     smallest camera-player distance, in dome radii
  at_limit_pct    frames pushed to the edge of the allowed space (radius and height both at their maximum)
  occluded_pct    frames where a player blocks the line of sight to the aim point
  in_frame        mean share of the action's points inside the image; cut_pct = frames with < 95 % in frame
  side_on_cos     |cos| between view direction and main motion during the freeze (0 = perfectly side-on)
  behind_pct      frames where the camera is > 110 deg away from where the key athlete faces (needs skeletons)
  eye_jerk_*      third difference of the camera position per video frame, in dome radii x 1000
  rot_jerk_*      third difference of the view direction per video frame, in degrees
"""
import numpy as np

BEHIND_COS = np.cos(np.radians(110.0))


def _fwd(eye, target):
    f = target - eye; return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)


def path_jerk(eye, target, R):
    j = np.linalg.norm(np.diff(eye, n=3, axis=0), axis=1) / R * 1000
    f = _fwd(eye, target); ang = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', f[1:], f[:-1]), -1, 1)))
    rj = np.abs(np.diff(ang, n=2))
    return dict(eye_jerk_med=float(np.median(j)), eye_jerk_p95=float(np.percentile(j, 95)), eye_jerk_max=float(j.max()),
                rot_jerk_med=float(np.median(rj)), rot_jerk_p95=float(np.percentile(rj, 95)), rot_jerk_max=float(rj.max()))


def evaluate(ct, eye, target, fov, up, players, S, W, H, clear, freeze_frames=None, motion_dir=None, cues=None,
             action=None, rf=None, q=None):
    """players: (T, N, 3) true player points; action: optional (T, N) mask of the points that must stay in frame."""
    T = players.shape[0]; N = len(ct); R = S['radius']; pts_i = np.clip(np.round(ct).astype(int), 0, T - 1)
    dmin = np.array([np.min(np.linalg.norm(players[pts_i[i]] - eye[i], axis=1)) for i in range(N)])
    occ_r = 0.06 * R; occ = np.zeros(N, bool); inf = np.zeros(N)
    f = _fwd(eye, target)
    for i in range(N):
        P = players[pts_i[i]]; e = eye[i]; v = target[i] - e; L = np.linalg.norm(v); v /= L; s = (P - e) @ v
        m = (s > 0.05 * L) & (s < 0.6 * L)
        if m.any(): occ[i] = np.sum(np.linalg.norm((P[m] - e) - np.outer(s[m], v), axis=1) < occ_r) > 15
        A = P if action is None else P[action[pts_i[i]]]
        z = f[i]; x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x); Q = A - e; zc = Q @ z
        t = np.tan(np.radians(fov[i]) / 2); inside = (zc > 1e-3) & (np.abs(Q @ x) <= t * zc) & (np.abs(Q @ y) <= t * zc * H / W)
        inf[i] = inside.mean() if len(A) else 1.0
    out = dict(n_frames=N, clip_pct=100 * float(np.mean(dmin < clear)), min_clear_R=float(dmin.min() / R),
               occluded_pct=100 * float(occ.mean()), in_frame=float(inf.mean()), cut_pct=100 * float(np.mean(inf < 0.95)))
    if rf is not None and q is not None: out['at_limit_pct'] = 100 * float(np.mean((rf >= 0.895) & (q >= 0.745)))
    if freeze_frames is not None and len(freeze_frames) and motion_dir is not None:
        fh = f[freeze_frames] - np.outer(f[freeze_frames] @ up, up); fh /= np.linalg.norm(fh, axis=1, keepdims=True)
        out['side_on_cos'] = float(np.mean(np.abs(fh @ motion_dir)))
    if cues is not None:
        d = eye - cues['centre'][pts_i]; d -= np.outer(d @ up, up); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        out['behind_pct'] = 100 * float(np.mean(np.einsum('ij,ij->i', d, cues['facing'][pts_i]) < BEHIND_COS))
    out.update(path_jerk(eye, target, R))
    return out
