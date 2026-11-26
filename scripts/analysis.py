"""Analysis figures (CPU). Run from inside Dynamic3DGaussians/ after training.
  * error vs motion speed: per-frame held-out PSNR (output/<exp>/<seq>/per_frame.csv) against how fast the
    athlete moves in that frame (95th-percentile Gaussian speed, cm/frame via the skeleton scale)
  * failure gallery: for each sport, the worst frame: real held-out view | our render, cropped to the athlete
    python analysis.py --exp n27_allcam      -> results/analysis/*
"""
import argparse, csv, json, os, numpy as np
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument('--exp', default='n27_allcam')
ap.add_argument('--seqs', default='basketball,boxes,football,juggle,softball,tennis'); ap.add_argument('--out', default='results/analysis')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows, gallery = [], []; fig, ax = plt.subplots(figsize=(6.5, 3.6))
for seq in a.seqs.split(','):
    base = f'./output/{a.exp}/{seq}'
    if not os.path.exists(f'{base}/per_frame.csv'): print('skip', seq); continue
    pf = np.loadtxt(f'{base}/per_frame.csv', delimiter=',', skiprows=1); psnr = pf[:, 1]
    z = np.load(f'{base}/params.npz'); fg = z['seg_colors'][:, 0] > 0.5; X = z['means3D'][:, fg][:, ::20]
    sp = np.percentile(np.linalg.norm(np.diff(X, axis=0), axis=2), 95, axis=1); sp = np.concatenate([[sp[0]], sp])
    skp = f'results/skeletons/{seq}.json'; s = json.load(open(skp))['sim']['s'] if os.path.exists(skp) else 0.01
    sp_cm = sp / s; T = min(len(psnr), len(sp_cm))
    r = float(np.corrcoef(sp_cm[1:T], psnr[1:T])[0, 1])          # frame 0 is the static fit; leave it out
    ax.scatter(sp_cm[1:T], psnr[1:T], s=6, label=f'{seq} (r={r:+.2f})')
    worst = int(np.argmin(psnr[1:T])) + 1; best = int(np.argmax(psnr[1:T])) + 1
    rows.append(dict(seq=seq, pearson_r_speed_psnr=round(r, 3), mean_speed_cm=round(float(sp_cm[1:T].mean()), 2), max_speed_cm=round(float(sp_cm[1:T].max()), 2),
                     worst_frame=worst, worst_psnr=round(float(psnr[worst]), 2), best_frame=best, best_psnr=round(float(psnr[best]), 2)))
    tmd = json.load(open(f'./data/{seq}/test_meta.json')); fn = tmd['fn'][worst][0]
    real = Image.open(f'./data/{seq}/ims/{fn}').convert('RGB'); ren = Image.open(f'{base}/test_renders/c0_{worst:03d}.png').convert('RGB')
    m = np.array(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}")); m = (m[..., 0] if m.ndim == 3 else m) > 0
    ys, xs = np.nonzero(m)
    if len(ys):
        pad = 12; box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(real.width, xs.max() + pad), min(real.height, ys.max() + pad))
        real, ren = real.crop(box), ren.crop(box)
    hh = 220; real = real.resize((int(real.width * hh / real.height), hh)); ren = ren.resize(real.size)
    tile = Image.new('RGB', (real.width * 2 + 4, hh), (255, 255, 255)); tile.paste(real, (0, 0)); tile.paste(ren, (real.width + 4, 0)); gallery.append(tile)
    print(rows[-1], flush=True); del z, X
ax.set_xlabel('athlete speed (cm / frame, 95th percentile)'); ax.set_ylabel('held-out PSNR (dB)'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{a.out}/psnr_vs_speed_{a.exp}.png', dpi=160)
if rows:
    with open(f'{a.out}/analysis_{a.exp}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if gallery:
    Wd = sum(g.width for g in gallery) + 8 * len(gallery); G = Image.new('RGB', (Wd, gallery[0].height), (255, 255, 255)); xo = 0
    for g in gallery: G.paste(g, (xo, 0)); xo += g.width + 8
    G.save(f'{a.out}/worst_frames_{a.exp}.jpg', quality=92)
print('done')
