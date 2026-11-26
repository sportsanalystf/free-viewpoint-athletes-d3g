"""Align the Panoptic 3D skeletons to every PanopticSports clip (CPU only). Run from inside Dynamic3DGaussians/.
For each clip: fit Panoptic->D3G similarity on the camera centres, search the clip's start frame in the raw
capture (projected joints must land on the clip's foreground masks), and save the 150 frames of joints in the D3G
world to results/skeletons/<seq>.json, plus an overlay image to check by eye.

    python align_skeletons.py --raw ../161029_sports1             # all six clips
"""
import argparse, json, os, numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
import skeleton as SK

ap = argparse.ArgumentParser()
ap.add_argument('--raw', default='../161029_sports1'); ap.add_argument('--seqs', default='basketball,boxes,football,juggle,softball,tennis')
ap.add_argument('--out', default='results/skeletons')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
calib = SK.load_calib(a.raw)
frames = SK.pose_frames(a.raw); fset = set(frames)
print(f'{len(frames)} pose frames, {frames[0]}..{frames[-1]}', flush=True)
cache = {}


def joints_at(f, sim):
    if f not in cache:
        B = SK.load_bodies(a.raw, f)
        X = np.concatenate([J[J[:, 3] > SK.CONF, :3] for _, J in B]) if B else np.zeros((0, 3))
        cache[f] = SK.apply_sim(sim, X) if len(X) else X
    return cache[f]


def mask(seq, fn):
    m = np.array(Image.open(f'./data/{seq}/seg/{fn.replace(".jpg", ".png")}')); m = m[..., 0] if m.ndim == 3 else m
    return binary_dilation(m > 0, iterations=3)


summary = {}
for seq in a.seqs.split(','):
    md = json.load(open(f'./data/{seq}/train_meta.json')); tmd = json.load(open(f'./data/{seq}/test_meta.json'))
    T = len(md['fn']); sim = SK.fit_similarity(calib, md, tmd); cache.clear()
    print(f'\n{seq}: similarity scale {sim["s"]:.5f} D3G units per cm, camera-centre rms {sim["rms"]:.4f} ({sim["n_cams"]} cams)', flush=True)
    ncam = len(md['fn'][0]); cams_c = [0, ncam // 3, 2 * ncam // 3]; cams_f = list(range(0, ncam, max(1, ncam // 6)))[:6]
    masks = {}

    def score(start, ts, cams):
        hit = tot = 0
        for t in ts:
            X = joints_at(start + t, sim)
            if not len(X): continue
            for c in cams:
                key = (t, c)
                if key not in masks: masks[key] = mask(seq, md['fn'][t][c])
                M = masks[key]; uv, z = SK.project(X, md['k'][t][c], md['w2c'][t][c])
                ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < M.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < M.shape[0])
                u = uv[ok].astype(int); hit += int(M[u[:, 1], u[:, 0]].sum()); tot += int(ok.sum())
        return hit / (tot + 20), hit / max(tot, 1), tot

    ts_c = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    cand = [f for f in frames if all((f + t) in fset for t in ts_c)]
    coarse = sorted(((score(s, ts_c, cams_c), s) for s in cand[::2]), reverse=True)[:5]
    best = None
    for _, s0 in coarse:
        for s in range(s0 - 3, s0 + 4):
            if not all((s + t) in fset for t in range(0, T, 10)): continue
            sc = score(s, list(range(0, T, 10)), cams_f)
            if best is None or sc[0] > best[1][0]: best = (s, sc)
    start, (_, rate, n) = best
    print(f'{seq}: clip starts at raw frame {start}; {100 * rate:.1f}% of {n} projected joints on the athlete masks', flush=True)
    out_frames = []
    for t in range(T):
        bodies = []
        for bid, J in SK.load_bodies(a.raw, start + t):
            P = SK.apply_sim(sim, J[:, :3])
            bodies.append({'id': bid, 'j': [[round(float(x), 5) for x in p] + [round(float(c), 3)] for p, c in zip(P, J[:, 3])]})
        out_frames.append(bodies)
    json.dump({'seq': seq, 'raw_start': start, 'mask_hit_rate': rate, 'n_joints_checked': n, 'sim': sim, 'frames': out_frames},
              open(f'{a.out}/{seq}.json', 'w'))
    # overlay on a held-out camera at mid-clip
    t = T // 2; fn = tmd['fn'][t][0]; im = Image.open(f'./data/{seq}/ims/{fn}').convert('RGB'); d = ImageDraw.Draw(im)
    for b in out_frames[t]:
        P = np.array(b['j'])[:, :3]; c = np.array(b['j'])[:, 3]; uv, z = SK.project(P, tmd['k'][t][0], tmd['w2c'][t][0])
        for (u, v), cc, zz in zip(uv, c, z):
            if cc > SK.CONF and zz > 0: d.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(255, 40, 40))
    im.save(f'{a.out}/{seq}_overlay.jpg', quality=90)
    summary[seq] = {'raw_start': start, 'mask_hit_rate': round(rate, 4), 'scale_d3g_per_cm': sim['s'], 'cam_rms': sim['rms'],
                    'bodies_mid': len(out_frames[T // 2])}
json.dump(summary, open(f'{a.out}/summary.json', 'w'), indent=1)
print('\n' + json.dumps(summary, indent=1))
