"""Director ablation (CPU only): plan every variant in cine_path.VARIANTS for every clip, plus the naive orbit
baseline, and score each plan with director_metrics. No rendering needed; videos of selected variants are made
with render_cinematic.py. Run from inside Dynamic3DGaussians/.

    python ablate_director.py --exp n27_allcam        -> results/ablation/director_<exp>.csv / .json
"""
import argparse, csv, json, os, time, numpy as np, torch
import cine_path as CP
import director_metrics as DM
import skeleton as SK

ap = argparse.ArgumentParser()
ap.add_argument('--exp', default='n27_allcam'); ap.add_argument('--seqs', default='basketball,boxes,football,juggle,softball,tennis')
ap.add_argument('--variants', default=','.join(CP.VARIANTS)); ap.add_argument('--out', default='results/ablation')
ap.add_argument('--scale', type=float, default=2.0); ap.add_argument('--fps', type=int, default=30)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rows = []
for seq in a.seqs.split(','):
    md = json.load(open(f'./data/{seq}/train_meta.json')); tmd = json.load(open(f'./data/{seq}/test_meta.json'))
    W2C = [np.array(w) for w in md['w2c'][0]] + [np.array(w) for w in tmd['w2c'][0]]
    W, H = int(md['w'] * a.scale), int(md['h'] * a.scale)
    z = np.load(f'./output/{a.exp}/{seq}/params.npz'); seg = z['seg_colors']; T = z['means3D'].shape[0]
    fg = torch.nonzero(torch.tensor(seg[:, 0] > 0.5)).squeeze(1)
    rgb_mid = torch.tensor(z['rgb_colors'][T // 2])[fg]
    redm = (rgb_mid[:, 0] > 0.45) & (rgb_mid[:, 0] > 1.6 * rgb_mid[:, 1]) & (rgb_mid[:, 0] > 1.6 * rgb_mid[:, 2])
    g = torch.Generator(device='cpu').manual_seed(0)                       # same point sample as render_cinematic.py
    pick = torch.cat([fg[torch.randperm(len(fg), generator=g)[:6000]], fg[redm]]).unique().numpy()
    fg_pts = z['means3D'][:, pick].astype(np.float64); fg_rgb = z['rgb_colors'][:, pick].astype(np.float64)
    S = CP.scene_frame(W2C)
    skp = f'results/skeletons/{seq}.json'
    cues = SK.director_cues(SK.load_aligned(skp), S['up'], T) if os.path.exists(skp) else None
    if cues is None: print(f'WARNING no skeleton cues for {seq}; skeleton variants fall back to the old rules', flush=True)
    players = fg_pts[:, CP.player_filter(fg_pts, S)]
    action = None
    if cues is not None:                                                   # points that must stay in frame: around the key athlete
        action = np.linalg.norm(players - cues['centre'][:, None, :], axis=2) < 0.35 * S['radius']
    for v in a.variants.split(','):
        t0 = time.time()
        ct, eye, tgt, fov, up, info = CP.plan(W2C, fg_pts, fg_rgb, W, H, a.fps, None, 0.35, 0.4, opts=CP.VARIANTS[v], cues=cues)
        d = info['_debug']
        m = DM.evaluate(ct, eye, tgt, fov, up, players, S, W, H, info['clear'], d['freeze_frames'], d['motion_dir'], cues,
                        action=action, rf=d['rf'], q=d['q'])
        rows.append(dict(seq=seq, variant=v, freeze=info['freeze'], freeze_reason=info['freeze_reason'],
                         view_az=info['view_az_deg'], sweep_dir=info['sweep_dir'], **{k: round(x, 4) for k, x in m.items()}))
        print(f'{seq:10s} {v:14s} {time.time() - t0:5.1f}s  ' + '  '.join(f'{k}={x:.3g}' for k, x in m.items() if k != 'n_frames'), flush=True)
    N = rows[-1]['n_frames']; fx = md['k'][0][0][0][0]; fov_real = float(np.degrees(2 * np.arctan(md['w'] / 2 / fx)))
    ct, eye, tgt, fov, up = CP.naive_orbit(W2C, T, N, fov_real)
    m = DM.evaluate(ct, eye, tgt, fov, up, players, S, W, H, 0.35 * S['radius'], cues=cues, action=action)
    rows.append(dict(seq=seq, variant='naive_orbit', freeze=np.nan, freeze_reason='-', view_az=np.nan, sweep_dir=0, **{k: round(x, 4) for k, x in m.items()}))
    if cues is not None: rows.append(dict(seq=seq, variant='_skeleton_key_frame', freeze=cues['key_frame'], freeze_reason=f"body {cues['key_body']}"))
    del z, fg_pts, fg_rgb, players
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('seq', 'variant', 'freeze', 'freeze_reason'), k))
with open(f'{a.out}/director_{a.exp}.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
json.dump(rows, open(f'{a.out}/director_{a.exp}.json', 'w'), indent=1, default=float)
print(f'wrote {a.out}/director_{a.exp}.csv ({len(rows)} rows)')
