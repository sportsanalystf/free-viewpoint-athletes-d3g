"""Camera-path planning for render_cinematic.py: the replay "director" (pure NumPy, testable without a GPU).

plan(W2C, fg_pts, fg_rgb, W, H, ...) returns per-video-frame (clip_time, eye, target, fov, up, info).
  * scene frame (up, focus, dome radius, height band) estimated from the real cameras
  * players = dense clusters of foreground Gaussians (floater filter)
  * key moment: ball apex (red Gaussians) / peak motion / skeleton (peak wrist speed of the key athlete)
  * view: side-on to the main motion; with skeleton cues, the side and sweep direction that keep the camera in
    front of (or beside) the athlete are chosen ("facing-aware")
  * clearance, occlusion lift, auto field of view, two-pass smoothing
Every part can be switched off through `opts` (see VARIANTS) for the ablation study. opts={} reproduces v6.
"""
import numpy as np

DEFAULTS = dict(floater=True, clearance=True, occlusion=True, auto_fov=True, smooth='double', view='side',
                key='ball', facing=False)
FULL = dict(key='skeleton', facing=True)                      # v7 = v6 + skeleton key moment + facing-aware view
VARIANTS = {
    'v6': {},                                                  # mid-term director
    'full': FULL,
    'no_floater': dict(FULL, floater=False),
    'no_clearance': dict(FULL, clearance=False),
    'no_occlusion': dict(FULL, occlusion=False),
    'no_autofov': dict(FULL, auto_fov=False),
    'smooth_single': dict(FULL, smooth='single'),
    'smooth_none': dict(FULL, smooth='none'),
    'fixed_view': dict(FULL, view='fixed', facing=False),
    'no_facing': dict(FULL, facing=False),
    'key_ball': dict(FULL, key='ball'),
    'key_motion': dict(FULL, key='motion'),
}
BEHIND_COS = np.cos(np.radians(110.0))                         # camera > 110 deg away from where the athlete faces


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
                az_real=np.arctan2(rel @ e2, rel @ e1), cams=Cs, fwd=Fs)


def gsmooth(x, sigma):
    if sigma <= 0: return np.asarray(x, float)
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


def player_filter(fg_pts, S):
    """Players are DENSE clusters inside the camera ring; stray "floaters" are sparse. Keep points inside 0.85 x the
    ring that are dense in at least 2 of 3 sampled frames."""
    from scipy.spatial import cKDTree
    up, focus, R = S['up'], S['focus'], S['radius']; T = fg_pts.shape[0]
    keep = np.zeros((3, fg_pts.shape[1]), bool)
    for j, t in enumerate((T // 4, T // 2, 3 * T // 4)):
        P = fg_pts[t]; d = P - focus; dh = np.linalg.norm(d - np.outer(d @ up, up), axis=1)
        kd = cKDTree(P).query(P, k=9)[0][:, -1]
        keep[j] = (dh <= 0.85 * R) & (kd <= 2.5 * np.median(kd))
    return keep.sum(0) >= 2


SHOT_AZ = [(-100, -65), (-65, 65), (65, 125), (125, 150)]       # azimuth of each shot relative to the side-on view


def plan(W2C, fg_pts, fg_rgb, W, H, fps=30, freeze=None, clear_frac=0.22, speed=0.4, opts=None, cues=None):
    o = dict(DEFAULTS, **(opts or {}))
    S = scene_frame(W2C); up, focus, e1, e2, R = S['up'], S['focus'], S['e1'], S['e2'], S['radius']
    T = fg_pts.shape[0]; raw_n = fg_pts.shape[1]
    cols = player_filter(fg_pts, S)
    true_players = fg_pts[:, cols]                              # what the metrics judge against, in every variant
    if o['floater']: fg_pts = fg_pts[:, cols]; fg_rgb = fg_rgb[:, cols]
    ball, bh = ball_track(fg_pts, fg_rgb, up)
    V = np.linalg.norm(np.diff(fg_pts, axis=0), axis=2)                       # (T-1, N)
    peak = gsmooth(np.percentile(V, 95, axis=1), 2.0)
    lo, hi = int(0.2 * T), int(0.8 * T)
    if freeze is not None: why = 'user'
    elif o['key'] == 'skeleton' and cues is not None: freeze, why = int(cues['key_frame']), 'skeleton'
    elif o['key'] == 'ball' and bh is not None: freeze, why = int(lo + np.argmax(bh[lo:hi])), 'ball apex'
    else: freeze, why = int(lo + np.argmax(peak[lo:hi])), 'peak motion'
    F = float(freeze)
    # main horizontal direction of the fast motion around the freeze; we look ACROSS it (side-on)
    w0, w1 = int(max(0, F - 12)), int(min(T - 2, F + 12))
    D = np.diff(fg_pts, axis=0)[w0:w1 + 1]; sp = np.linalg.norm(D, axis=2)
    fast = sp > np.percentile(sp, 95)
    Dh = D[fast]; Dh = Dh - np.outer(Dh @ up, up)
    if len(Dh) >= 20:
        Cm = (Dh[:, :, None] * Dh[:, None, :]).sum(0); m = np.linalg.eigh(Cm)[1][:, -1]
        az_motion = np.degrees(np.arctan2(m @ e2, m @ e1))
    else:
        az_motion = np.degrees(S['az_real'][0])
    motion_dir = np.cos(np.radians(az_motion)) * e1 + np.sin(np.radians(az_motion)) * e2
    # aim: the ball and whoever is near it (soft weights), else the players' box centre
    if ball is not None:
        wgt = np.exp(-(np.linalg.norm(fg_pts - ball[:, None, :], axis=2) / (0.5 * R)) ** 2) + 1e-3
        box_c = (wgt[..., None] * fg_pts).sum(1) / wgt.sum(1, keepdims=True)
        box_c = 0.6 * box_c + 0.4 * ball
        near = wgt > 0.3
        near |= ~near.any(1, keepdims=True)
    else:
        near = np.ones(fg_pts.shape[:2], bool)
        box_c = 0.5 * (np.percentile(fg_pts, 2, axis=1) + np.percentile(fg_pts, 98, axis=1))
    sm = {'double': 1.0, 'single': 1.0, 'none': 0.0}[o['smooth']]
    tgt = 0.9 * gsmooth(box_c, 20.0 * sm) + 0.1 * focus
    t3 = F + 0.65 * (T - 1 - F)
    clip_secs = lambda a, b: max(2.5, (b - a) / (speed * fps))
    SHOTS = [  # (seconds, clip from->to, shot index, radius frac, height frac, base fov)
        (clip_secs(0, F), (0, F), 0, (1.00, 0.85), (0.55, 0.45), (50, 46)),
        (5.0, (F, F), 1, (0.85, 0.80), (0.45, 0.35), (46, 46)),
        (clip_secs(F, t3), (F, t3), 2, (0.80, 0.84), (0.35, 0.62), (46, 48)),
        (clip_secs(t3, T - 1), (t3, T - 1), 3, (0.84, 0.80), (0.62, 0.45), (48, 46)),
    ]
    ease = lambda u: u * u * (3 - 2 * u); eio = lambda u: 0.5 - 0.5 * np.cos(np.pi * u)
    h = lambda q: S['h_lo'] + (S['h_hi'] - S['h_lo']) * q

    def eye_of(az, rf, q):
        th = np.radians(az); return focus + R * rf[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2) + h(q)[:, None] * up

    def shot_rows(az_view, sdir):
        rows = []
        for dur, (c0, c1), k, (r0, r1), (q0, q1), (f0, f1) in SHOTS:
            n = int(round(dur * fps)); a0, a1 = SHOT_AZ[k]
            for i in range(n):
                u = i / n
                rows.append((c0 + (c1 - c0) * u, az_view + sdir * (a0 + (a1 - a0) * eio(u)), r0 + (r1 - r0) * ease(u),
                             q0 + (q1 - q0) * ease(u), f0 + (f1 - f0) * ease(u)))
        rows = np.array(rows)
        for col in (1, 2, 3, 4): rows[:, col] = gsmooth(rows[:, col], 8.0 * sm)
        return rows

    def back_fraction(rows):
        if cues is None: return 0.0
        ti = np.clip(np.round(rows[:, 0]).astype(int), 0, T - 1)
        e = eye_of(rows[:, 1], rows[:, 2], rows[:, 3]); d = e - cues['centre'][ti]; d -= np.outer(d @ up, up)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        return float(np.mean(np.einsum('ij,ij->i', d, cues['facing'][ti]) < BEHIND_COS))

    if o['view'] == 'fixed':
        az_view, sdir = float(np.degrees(S['az_real'][0])), 1.0
    elif o['facing'] and cues is not None:       # pick the side and sweep direction that keep the athlete's face in view
        cands = [(az_motion + s90, sd) for s90 in (90.0, -90.0) for sd in (1.0, -1.0)]
        az_view, sdir = min(cands, key=lambda c: back_fraction(shot_rows(*c)))
    else:
        az_view, sdir = az_motion + 90.0, 1.0
    rows = shot_rows(az_view, sdir); N = len(rows); ct = rows[:, 0]

    def at(arr, t):  # linear interpolation over clip time
        t0 = np.clip(np.floor(t).astype(int), 0, T - 1); t1 = np.minimum(t0 + 1, T - 1); w = (t - t0)[:, None]
        return arr[t0] * (1 - w) + arr[t1] * w
    target = gsmooth(at(tgt, ct), 14.0 if o['smooth'] == 'double' else 0.0)   # second pass, in VIDEO time
    pts_i = np.clip(np.round(ct).astype(int), 0, T - 1)

    # ---- clearance + framing: the camera stays inside the dome (rf <= RF_MAX), keeps >= `clear` from every player
    # point (moving out, then up), lifts over a player blocking the aim, and the lens widens to fit the play
    clear = clear_frac * R; RF_MAX, Q_MAX, FOV_CAP, FOV_HARD = 0.90, 0.75, 56.0, 64.0

    def need_fov(e, tg, P):
        z = tg - e; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
        Q = P - e; zc = Q @ z; m_ = zc > 1e-3
        if m_.mean() < 0.98: return 179.0
        tx = np.percentile(np.abs(Q[m_] @ x) / zc[m_], 98); ty = np.percentile(np.abs(Q[m_] @ y) / zc[m_], 98) * W / H
        return 2 * np.degrees(np.arctan(1.15 * max(tx, ty)))
    rf_req = rows[:, 2].copy(); q_req = rows[:, 3].copy()
    if o['clearance']:
        for i in range(N):
            P = fg_pts[pts_i[i]]; best = None
            cands = [(r_, rows[i, 3]) for r_ in np.arange(min(rows[i, 2], RF_MAX), RF_MAX + 1e-9, 0.02)] + \
                    [(RF_MAX, q_) for q_ in np.arange(rows[i, 3], Q_MAX + 1e-9, 0.05)]
            for r_, q_ in cands:
                e = eye_of(rows[i:i + 1, 1], np.array([r_]), np.array([q_]))[0]
                if np.min(np.linalg.norm(P - e, axis=1)) < clear: continue
                best = (r_, q_); break
            rf_req[i], q_req[i] = best if best is not None else (RF_MAX, Q_MAX)
    occ_r = 0.06 * R

    def blocked(e, tg, P):
        v = tg - e; L_ = np.linalg.norm(v); v = v / L_; s_ = (P - e) @ v
        m_ = (s_ > 0.05 * L_) & (s_ < 0.6 * L_)
        if not m_.any(): return False
        d = np.linalg.norm((P[m_] - e) - np.outer(s_[m_], v), axis=1); return np.sum(d < occ_r) > 15
    if o['occlusion']:
        for i in range(N):
            P = fg_pts[pts_i[i]]; r_ = rf_req[i]
            for q_ in np.arange(max(rows[i, 3], q_req[i]), Q_MAX + 1e-9, 0.05):
                e = eye_of(rows[i:i + 1, 1], np.array([r_]), np.array([q_]))[0]
                if not blocked(e, target[i], P) and (not o['clearance'] or np.min(np.linalg.norm(P - e, axis=1)) >= clear):
                    q_req[i] = q_; break
            else: q_req[i] = Q_MAX
    rf = np.minimum(np.maximum(rows[:, 2], rf_req), RF_MAX); q = np.minimum(np.maximum(rows[:, 3], q_req), Q_MAX)
    if sm > 0:
        for _ in range(3): rf = np.maximum(gsmooth(rf, 10.0), rf_req); q = np.maximum(gsmooth(q, 10.0), q_req)
        rf = np.minimum(gsmooth(rf, 3.0), RF_MAX); q = np.minimum(gsmooth(q, 3.0), Q_MAX)
    eye = gsmooth(eye_of(rows[:, 1], rf, q), 6.0 if o['smooth'] == 'double' else 0.0)   # final pass removes kinks
    if o['auto_fov']:
        fov_req = np.array([need_fov(eye[i], target[i], fg_pts[pts_i[i]][near[pts_i[i]]]) for i in range(N)])
        fov = np.minimum(np.maximum(rows[:, 4], fov_req), FOV_HARD)
        if sm > 0:
            for _ in range(3): fov = np.minimum(np.maximum(gsmooth(fov, 10.0), np.minimum(fov_req, FOV_HARD)), FOV_HARD)
            fov = gsmooth(fov, 3.0)
    else:
        fov = rows[:, 4].copy()
    min_clear = np.array([np.min(np.linalg.norm(fg_pts[pts_i[i]] - eye[i], axis=1)) for i in range(N)])
    info = dict(freeze=F, freeze_reason=why, motion_az_deg=round(float(az_motion), 1), view_az_deg=round(float(az_view), 1),
                sweep_dir=int(sdir), n_frames=N, seconds=N / fps, radius=R, clear=clear, min_clearance=float(min_clear.min()),
                fov_min=float(fov.min()), fov_max=float(fov.max()), ball_found=ball is not None,
                player_pts_kept=f'{int(cols.sum())}/{raw_n}', rf_range=(float(rf.min()), float(rf.max())),
                height_frac_range=(float(q.min()), float(q.max())), frames_pushed=int(np.sum(rf_req > rows[:, 2] + 1e-6)),
                frames_raised=int(np.sum(q_req > rows[:, 3] + 1e-6)), opts=o)
    info['_debug'] = dict(eye=eye, target=target, cams=S['cams'], pts=fg_pts[int(F)], e1=e1, e2=e2, focus=focus,
                          true_players=true_players, near=near if o['floater'] else None, rf=rf, q=q, ct=ct,
                          motion_dir=motion_dir, scene=S, freeze_frames=np.nonzero(np.abs(ct - F) < 1e-6)[0])
    return ct, eye, target, fov, S['up'], info


def naive_orbit(W2C, T, N, K_fov_deg=50.0):
    """Baseline: one loop through the real camera poses (sorted by azimuth), blending positions and view
    directions between neighbours. Returns ct, eye, target, fov like plan()."""
    S = scene_frame(W2C); order = np.argsort(S['az_real']); C = S['cams'][order]; Fw = S['fwd'][order]
    x = np.linspace(0, len(C), N, endpoint=False); i = np.floor(x).astype(int) % len(C); j = (i + 1) % len(C); f = (x - np.floor(x))[:, None]
    eye = (1 - f) * C[i] + f * C[j]; fw = (1 - f) * Fw[i] + f * Fw[j]; fw /= np.linalg.norm(fw, axis=1, keepdims=True)
    dist = np.linalg.norm(eye - S['focus'], axis=1, keepdims=True)
    return np.linspace(0, T - 1, N), eye, eye + fw * dist, np.full(N, K_fov_deg), S['up']


def w2c_from(eye, target, up):
    z = target - eye; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    Rm = np.stack([x, y, z]); M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = -Rm @ eye; return M
