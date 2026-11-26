"""3D motion accuracy against the Panoptic skeletons (CPU). Run from inside Dynamic3DGaussians/.
For every skeleton joint, the athlete Gaussians within 6 cm of it at its first visible frame are its "anchors".
We follow the anchors through the 4D model and compare their displacement with the joint's true displacement
(displacement, not position, because the anchors sit on the skin and clothes, not at the joint centre).
Baseline: a static model (no motion), whose error equals how far the joint moves.

    python eval_tracks.py --exps n27_allcam                 -> results/tracks/tracks_<exps>.csv, curve png
    python eval_tracks.py --exps n27_allcam,no_rigid,no_rot,no_iso --seqs basketball      (prior ablation)
"""
import argparse, csv, json, os, numpy as np
from scipy.spatial import cKDTree
import skeleton as SK

ap = argparse.ArgumentParser()
ap.add_argument('--exps', default='n27_allcam'); ap.add_argument('--seqs', default='basketball,boxes,football,juggle,softball,tennis')
ap.add_argument('--radius_cm', type=float, default=6.0); ap.add_argument('--out', default='results/tracks')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
GROUPS = {'torso': [SK.NECK, SK.PELVIS, SK.L_SH, SK.R_SH, SK.L_HIP, SK.R_HIP], 'head': [SK.NOSE, SK.L_EYE, SK.R_EYE, SK.L_EAR, SK.R_EAR],
          'arms': [SK.L_EL, SK.R_EL, SK.L_WR, SK.R_WR], 'legs': [SK.L_KN, SK.R_KN, SK.L_AN, SK.R_AN]}
rows, curves = [], {}
for exp in a.exps.split(','):
    for seq in a.seqs.split(','):
        skp = f'results/skeletons/{seq}.json'; pp = f'./output/{exp}/{seq}/params.npz'
        if not (os.path.exists(skp) and os.path.exists(pp)): print(f'skip {exp}/{seq}: missing {skp if not os.path.exists(skp) else pp}'); continue
        sk = SK.load_aligned(skp); s = sk['meta']['sim']['s']; J = sk['joints']          # (T, B, 19, 4), D3G units; cm = units / s
        z = np.load(pp); fg = z['seg_colors'][:, 0] > 0.5; X = z['means3D'][:, fg].astype(np.float64)   # (T, N, 3)
        T = min(X.shape[0], J.shape[0]); err = {g: [] for g in GROUPS}; stat = {g: [] for g in GROUPS}; per_t = np.zeros(T); cnt = np.zeros(T)
        trees = {}
        for b in range(J.shape[1]):
            for g, idx in GROUPS.items():
                for j in idx:
                    vis = np.nonzero(~np.isnan(J[:T, b, j, 0]))[0]
                    if len(vis) < 10: continue
                    t0 = vis[0]
                    if t0 not in trees: trees[t0] = cKDTree(X[t0])
                    anc = trees[t0].query_ball_point(J[t0, b, j, :3], a.radius_cm * s)
                    if len(anc) < 5: continue
                    anc = np.array(anc[:60]); A = X[:T, anc].mean(1)                   # (T, 3) anchor centre over time
                    for t in vis[1:]:
                        if t >= T: break
                        e = np.linalg.norm((A[t] - A[t0]) - (J[t, b, j, :3] - J[t0, b, j, :3])) / s
                        err[g].append(e); stat[g].append(np.linalg.norm(J[t, b, j, :3] - J[t0, b, j, :3]) / s)
                        per_t[t] += e; cnt[t] += 1
        allE = np.concatenate([np.array(v) for v in err.values() if v]) if any(err.values()) else np.array([np.nan])
        allS = np.concatenate([np.array(v) for v in stat.values() if v]) if any(stat.values()) else np.array([np.nan])
        row = dict(exp=exp, seq=seq, mean_cm=round(float(np.mean(allE)), 2), median_cm=round(float(np.median(allE)), 2),
                   pct_under_5cm=round(100 * float(np.mean(allE < 5)), 1), static_mean_cm=round(float(np.mean(allS)), 2), n=len(allE))
        for g in GROUPS: row[f'{g}_mean_cm'] = round(float(np.mean(err[g])), 2) if err[g] else None
        rows.append(row); curves[f'{exp}/{seq}'] = (per_t / np.maximum(cnt, 1)).tolist()
        print(row, flush=True); del z, X
tag = a.exps.replace(',', '+')
if rows:
    with open(f'{a.out}/tracks_{tag}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(curves, open(f'{a.out}/tracks_{tag}_per_frame.json', 'w'))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for k, v in curves.items(): ax.plot(v, label=k, lw=1.3)
    ax.set_xlabel('frame'); ax.set_ylabel('track error (cm)'); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(f'{a.out}/tracks_{tag}.png', dpi=160)
    print(f'wrote {a.out}/tracks_{tag}.csv')
