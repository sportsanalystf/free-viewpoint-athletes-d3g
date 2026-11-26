"""Camera-path planning for render_cinematic.py (pure NumPy, testable without a GPU).

plan(W2C, fg_pts, fg_rgb, T, ...) returns per-video-frame (clip_time, eye, target, fov) plus info.
  * scene frame (up, focus, dome radius, height band) estimated from the real cameras
  * freeze frame = highest point of the ball (red foreground Gaussians), unless given
  * target = smoothed centre of the players' bounding box, pulled toward the ball
  * clearance: the eye is pushed outward until it is >= `clear` away from every player point
  * auto field of view: widened so all players + ball stay inside the frame with a margin
Everything that changes per frame is smoothed again after the constraints, so it stays fluid."""
import numpy as np


def scene_frame(W2C):
    Cs = np.array([-M[:3, :3].T @ M[:3, 3] for M in W2C]); Fs = np.array([M[2, :3] for M in W2C])
    up = -np.mean([M[1, :3] for M in W2C], 0); up /= np.linalg.norm(up)
    A = sum(np.eye(3) - np.outer(f, f) for f in Fs); b = sum((np.eye(3) - np.outer(f, f)) @ c for f, c in zip(Fs, Cs))
    focus = np.linalg.solve(A, b)
    e1 = Cs.mean(0) - focus; e1 -= (e1 @ up) * up
    if np.linalg.norm(e1) < 1e-6: e1 = np.cross(up, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    rel = Cs - focus; horiz = rel - np.outer(rel @ up, up)
    return dict(up=up, focus=focus, e1=e1, e2=e2, radius=float(np.median(np.linalg.norm(horiz, axis=1))),
                h_lo=float(np.percentile(rel @ up, 20)), h_hi=float(np.percentile(rel @ up, 80)),
                az_real=np.arctan2(rel @ e2, rel @ e1))


def gsmooth(x, sigma):
    if sigma <= 0: return x
    r = int(3 * sigma); g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); g /= g.sum()
    x = np.asarray(x, float); pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode='edge')
    if x.ndim == 1: return np.convolve(pad, g, 'valid')
    return np.stack([np.convolve(pad[:, i], g, 'valid') for i in range(x.shape[1])], 1)


def ball_track(fg_pts, fg_rgb, up):
    """Red foreground Gaussians = the ball. Returns per-frame centre (or None) and height along `up`."""
    rgb = fg_rgb
    red = (rgb[..., 0] > 0.45) & (rgb[..., 0] > 1.6 * rgb[..., 1]) & (rgb[..., 0] > 1.6 * rgb[..., 2])
    T = fg_pts.shape[0]; c = np.full((T, 3), np.nan)
    for t in range(T):
        P = fg_pts[t][red[t]]
        if len(P) >= 20: c[t] = np.median(P, 0)
    ok = ~np.isnan(c[:, 0])
    if ok.sum() < T * 0.5: return None, None
    for i in range(3): c[:, i] = np.interp(np.arange(T), np.nonzero(ok)[0], c[ok, i])
    c = gsmooth(c, 2.0); return c, c @ up


def plan(W2C, fg_pts, fg_rgb, W, H, fps=30, freeze=None, clear_frac=0.22, speed=0.4):
    S = scene_frame(W2C); up, focus, e1, e2, R = S['up'], S['focus'], S['e1'], S['e2'], S['radius']
    T = fg_pts.shape[0]
    # Stray Gaussians flagged "foreground" can sit anywhere in the dome (even on the walls); if they are
    # treated as players the camera gets shoved into the walls. Players are DENSE clusters inside the camera
    # ring; strays are sparse. Keep points inside 0.85 x the ring that are dense in most sampled frames
    # (this keeps every player, including one standing near the wall, which v3 wrongly dropped).
    from scipy.spatial import cKDTree
    keep = np.zeros((3, fg_pts.shape[1]), bool)
    for j, t in enumerate((T // 4, T // 2, 3 * T // 4)):
        P = fg_pts[t]; d = P - focus; dh = np.linalg.norm(d - np.outer(d @ up, up), axis=1)
        kd = cKDTree(P).query(P, k=9)[0][:, -1]
        keep[j] = (dh <= 0.85 * R) & (kd <= 2.5 * np.median(kd))
    cols = keep.sum(0) >= 2                                     # dense in at least 2 of the 3 sampled frames
    raw_n = fg_pts.shape[1]; fg_pts = fg_pts[:, cols]; fg_rgb = fg_rgb[:, cols]
    ball, bh = ball_track(fg_pts, fg_rgb, up)
    # Peak-motion analysis: speed of the fastest 5% of player points (hands, bat, racket, ball) per frame.
    V = np.linalg.norm(np.diff(fg_pts, axis=0), axis=2)                       # (T-1, N)
    peak = gsmooth(np.percentile(V, 95, axis=1), 2.0)
    lo, hi = int(0.2 * T), int(0.8 * T)
    if freeze is None:
        if bh is not None: freeze, why = int(lo + np.argmax(bh[lo:hi])), 'ball apex'
        else: freeze, why = int(lo + np.argmax(peak[lo:hi])), 'peak motion'
    else: why = 'user'
    F = float(freeze)
    # Main horizontal direction of that fast motion around the freeze -> look ACROSS it (perpendicular),
    # so swings / throws / passes are seen side-on during the bullet-time sweep.
    w0, w1 = int(max(0, F - 12)), int(min(T - 2, F + 12))
    D = np.diff(fg_pts, axis=0)[w0:w1 + 1]; sp = np.linalg.norm(D, axis=2)
    fast = sp > np.percentile(sp, 95)
    Dh = D[fast]; Dh = Dh - np.outer(Dh @ up, up)
    if len(Dh) >= 20:
        Cm = (Dh[:, :, None] * Dh[:, None, :]).sum(0); m = np.linalg.eigh(Cm)[1][:, -1]
        az_motion = np.degrees(np.arctan2(m @ e2, m @ e1))
    else:
        az_motion = np.degrees(S['az_real'][0])
    az_view = az_motion + 90.0                                                 # perpendicular to the motion
    # players' box centre per frame (robust), pulled toward the ball, then smoothed
    # aim at the middle of everything that must be in shot (players AND ball), so the lens only has to
    # cover the real extent symmetrically
    # Framing follows the PLAY: the ball and whoever is within reach of it (like a broadcast operator),
    # rather than squeezing every player into a wide shot. Clearance / occlusion still use all players.
    if ball is not None:   # soft weights -> the aim glides from player to player as the ball travels
        wgt = np.exp(-(np.linalg.norm(fg_pts - ball[:, None, :], axis=2) / (0.5 * R)) ** 2) + 1e-3
        box_c = (wgt[..., None] * fg_pts).sum(1) / wgt.sum(1, keepdims=True)
        box_c = 0.6 * box_c + 0.4 * ball
        near = wgt > 0.3
        near |= ~near.any(1, keepdims=True)
    else:
        near = np.ones(fg_pts.shape[:2], bool)
        box_c = 0.5 * (np.percentile(fg_pts, 2, axis=1) + np.percentile(fg_pts, 98, axis=1))
    tgt = 0.9 * gsmooth(box_c, 20.0) + 0.1 * focus
    # shot list; clip time runs at `speed` x real time except the freeze
    az0 = az_view - 100.0                                   # the freeze sweep (az0+35 .. az0+165) is centred on az_view
    t3 = F + 0.65 * (T - 1 - F)
    clip_secs = lambda a, b: max(2.5, (b - a) / (speed * fps))
    SHOTS = [  # (seconds, clip from->to, azimuth deg, radius frac, height frac, base fov)
        (clip_secs(0, F), (0, F), (az0, az0 + 35), (1.00, 0.85), (0.55, 0.45), (50, 46)),
        (5.0, (F, F), (az0 + 35, az0 + 165), (0.85, 0.80), (0.45, 0.35), (46, 46)),
        (clip_secs(F, t3), (F, t3), (az0 + 165, az0 + 225), (0.80, 0.84), (0.35, 0.62), (46, 48)),
        (clip_secs(t3, T - 1), (t3, T - 1), (az0 + 225, az0 + 250), (0.84, 0.80), (0.62, 0.45), (48, 46)),
    ]
    ease = lambda u: u * u * (3 - 2 * u); eio = lambda u: 0.5 - 0.5 * np.cos(np.pi * u)
    rows = []
    for dur, (c0, c1), (a0, a1), (r0, r1), (q0, q1), (f0, f1) in SHOTS:
        n = int(round(dur * fps))
        for i in range(n):
            u = i / n
            rows.append((c0 + (c1 - c0) * u, a0 + (a1 - a0) * eio(u), r0 + (r1 - r0) * ease(u), q0 + (q1 - q0) * ease(u), f0 + (f1 - f0) * ease(u)))
    rows = np.array(rows); N = len(rows)
    for col in (1, 2, 3, 4): rows[:, col] = gsmooth(rows[:, col], 8.0)
    ct = rows[:, 0]

    def at(arr, t):  # linear interpolation over clip time
        t0 = np.clip(np.floor(t).astype(int), 0, T - 1); t1 = np.minimum(t0 + 1, T - 1); w = (t - t0)[:, None]
        return arr[t0] * (1 - w) + arr[t1] * w
    target = gsmooth(at(tgt, ct), 14.0)                          # smooth again in VIDEO time (slow-mo stretches clip time)
    pts_i = np.clip(np.round(ct).astype(int), 0, T - 1)
    h = lambda q: S['h_lo'] + (S['h_hi'] - S['h_lo']) * q
    def eye_of(az, rf, q):
        th = np.radians(az); return focus + R * rf[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2) + h(q)[:, None] * up

    # ---- clearance + framing. The camera must stay inside the dome (rf <= RF_MAX: the real cameras are
    # mounted on its wall), keep >= `clear` from every player point, and see all players + ball with a
    # Only clearance moves the camera (outward, then up); framing is handled by the lens, so the camera
    # stays as close to the action as the shot asks for.
    clear = clear_frac * R; RF_MAX, Q_MAX, FOV_CAP, FOV_HARD = 0.90, 0.75, 56.0, 64.0
    def need_fov(e, tg, P):
        z = tg - e; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
        Q = P - e; zc = Q @ z; m = zc > 1e-3
        if m.mean() < 0.98: return 179.0
        tx = np.percentile(np.abs(Q[m] @ x) / zc[m], 98); ty = np.percentile(np.abs(Q[m] @ y) / zc[m], 98) * W / H
        return 2 * np.degrees(np.arctan(1.15 * max(tx, ty)))
    rf_req = np.zeros(N); q_req = np.zeros(N)
    for i in range(N):
        P = fg_pts[pts_i[i]]; best, best_f = None, 1e9
        cands = [(r_, rows[i, 3]) for r_ in np.arange(min(rows[i, 2], RF_MAX), RF_MAX + 1e-9, 0.02)] + \
                [(RF_MAX, q_) for q_ in np.arange(rows[i, 3], Q_MAX + 1e-9, 0.05)]
        for r_, q_ in cands:
            e = eye_of(rows[i:i + 1, 1], np.array([r_]), np.array([q_]))[0]
            if np.min(np.linalg.norm(P - e, axis=1)) < clear: continue
            best = (r_, q_); break                                 # closest position that is safe: stay near the action
        rf_req[i], q_req[i] = best if best is not None else (RF_MAX, Q_MAX)
    # line of sight: if a player stands between the camera and the aim point (the "back of a head fills the
    # frame" shots), raise the camera until it looks over them
    occ_r = 0.06 * R
    def blocked(e, tg, P):
        v = tg - e; L = np.linalg.norm(v); v = v / L; s_ = (P - e) @ v
        m = (s_ > 0.05 * L) & (s_ < 0.6 * L)
        if not m.any(): return False
        d = np.linalg.norm((P[m] - e) - np.outer(s_[m], v), axis=1); return np.sum(d < occ_r) > 15
    for i in range(N):
        P = fg_pts[pts_i[i]]; r_ = rf_req[i]
        for q_ in np.arange(max(rows[i, 3], q_req[i]), Q_MAX + 1e-9, 0.05):
            e = eye_of(rows[i:i + 1, 1], np.array([r_]), np.array([q_]))[0]
            if not blocked(e, target[i], P) and np.min(np.linalg.norm(P - e, axis=1)) >= clear: q_req[i] = q_; break
        else: q_req[i] = Q_MAX
    rf = np.minimum(np.maximum(rows[:, 2], rf_req), RF_MAX); q = np.minimum(np.maximum(rows[:, 3], q_req), Q_MAX)
    for _ in range(3): rf = np.maximum(gsmooth(rf, 10.0), rf_req); q = np.maximum(gsmooth(q, 10.0), q_req)
    rf = np.minimum(gsmooth(rf, 3.0), RF_MAX); q = np.minimum(gsmooth(q, 3.0), Q_MAX)
    eye = gsmooth(eye_of(rows[:, 1], rf, q), 6.0)                 # final pass removes kinks from the constraints
    fov_req = np.array([need_fov(eye[i], target[i], fg_pts[pts_i[i]][near[pts_i[i]]]) for i in range(N)])
    fov = np.minimum(np.maximum(rows[:, 4], fov_req), FOV_HARD)
    for _ in range(3): fov = np.minimum(np.maximum(gsmooth(fov, 10.0), np.minimum(fov_req, FOV_HARD)), FOV_HARD)
    fov = gsmooth(fov, 3.0)
    min_clear = np.array([np.min(np.linalg.norm(fg_pts[pts_i[i]] - eye[i], axis=1)) for i in range(N)])
    info = dict(freeze=F, freeze_reason=why, motion_az_deg=round(float(az_motion), 1), n_frames=N, seconds=N / fps, radius=R, clear=clear, min_clearance=float(min_clear.min()),
                fov_min=float(fov.min()), fov_max=float(fov.max()), ball_found=ball is not None,
                player_pts_kept=f'{fg_pts.shape[1]}/{raw_n}', rf_range=(float(rf.min()), float(rf.max())),
                height_frac_range=(float(q.min()), float(q.max())), frames_pushed=int(np.sum(rf_req > rows[:, 2] + 1e-6)), frames_raised=int(np.sum(q_req > rows[:, 3] + 1e-6)))
    info['_debug'] = dict(eye=eye, target=target, cams=np.array([-M[:3, :3].T @ M[:3, 3] for M in W2C]), pts=fg_pts[int(F)],
                          e1=e1, e2=e2, focus=focus)
    return ct, eye, target, fov, S['up'], info


def w2c_from(eye, target, up):
    z = target - eye; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    Rm = np.stack([x, y, z]); M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = -Rm @ eye; return M
