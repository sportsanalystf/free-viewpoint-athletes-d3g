"""Plot held-out PSNR vs number of cameras from results/sweep_frame0_<seq>.csv, and a grid of renders.
  python plot_sweep.py --seq basketball
"""
import argparse, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--seq', default='basketball'); ap.add_argument('--out', default='results')
a = ap.parse_args()
df = pd.read_csv(f'{a.out}/sweep_frame0_{a.seq}.csv')
g = df.groupby(['init', 'iters', 'n_cams'])['psnr'].agg(['mean', 'std', 'count']).reset_index()
print(g.to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 4.2))
for (init, it), d in g.groupby(['init', 'iters']):
    ax.errorbar(d['n_cams'], d['mean'], yerr=d['std'].fillna(0), marker='o', capsize=3,
                ls='-' if init == 'random' else '--', label=f'{init} init, {it} iters')
ax.set_xscale('log', base=2); ax.set_xticks(sorted(df.n_cams.unique())); ax.set_xticklabels(sorted(df.n_cams.unique()))
ax.set_xlabel('training cameras'); ax.set_ylabel('held-out PSNR (dB)'); ax.set_title(f'{a.seq}, frame 0: quality vs. cameras')
ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f'{a.out}/psnr_vs_cams_{a.seq}.png', dpi=150)
# render grid: rows = init, columns = camera count, at the largest iteration count, seed 0
it = df.iters.max(); ns = sorted(df.n_cams.unique()); rows = []
for init in ('allcam', 'random'):
    ims = []
    for n in ns:
        f = glob.glob(f'{a.out}/frame0_renders/{a.seq}_n{n:02d}_{init}_it{it}_s0.png')
        ims.append(np.array(Image.open(f[0]))[..., :3] if f else np.zeros((360, 640, 3), np.uint8))
    rows.append(np.hstack([np.array(Image.fromarray(i).resize((320, 180))) for i in ims]))
Image.fromarray(np.vstack(rows)).save(f'{a.out}/renders_grid_{a.seq}.png')
print('saved', f'{a.out}/psnr_vs_cams_{a.seq}.png', f'{a.out}/renders_grid_{a.seq}.png')
