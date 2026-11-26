"""In-between frame experiment (GPU). Run from inside Dynamic3DGaussians/.
Keep only every g-th reconstructed frame, rebuild the frames in between with each interpolation method
(nearest = frame repeat, linear = v6, cubic = new), render the 4 held-out cameras and compare with the real images.
The model's own frame (all frames kept) is the upper bound. PSNR is reported on the whole image and on the athlete
only (the model's foreground render at the true frame, dilated; held-out cameras have no masks), since the
static background hides most of the difference.

    python eval_interp.py --exp n27_allcam --gaps 2,4,8      -> results/interp/interp_<exp>.csv, crops_<exp>.jpg
"""
import argparse, csv, os, json, numpy as np, torch
from PIL import Image
import sparse_lib as L
import interp as IP

ap = argparse.ArgumentParser()
ap.add_argument('--exp', default='n27_allcam'); ap.add_argument('--seqs', default='basketball,boxes,football,juggle,softball,tennis')
ap.add_argument('--gaps', default='2,4,8'); ap.add_argument('--out', default='results/interp')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
METHODS = ['nearest', 'linear', 'cubic']


def rv(p, means, cols, rots):
    return {'means3D': means, 'colors_precomp': cols, 'rotations': rots, 'opacities': torch.sigmoid(p['logit_opacities']),
            'scales': torch.exp(p['log_scales']), 'means2D': torch.zeros_like(p['means3D'][0])}


def psnr(x, y, m=None):
    d = (x - y) ** 2
    mse = d.mean() if m is None else (d * m).sum() / (m.sum() * 3).clamp_min(1)
    return float(10 * torch.log10(1 / mse.clamp_min(1e-10)))


rows, crops = [], []
for seq in a.seqs.split(','):
    md, tmd = L.load_meta(seq)
    p = {k: torch.tensor(v).cuda().float() for k, v in np.load(f'./output/{a.exp}/{seq}/params.npz').items()}
    T = p['means3D'].shape[0]; ncam = len(tmd['fn'][0])
    gt, msk = {}, {}

    def real(t, c):
        if (t, c) not in gt:
            fn = tmd['fn'][t][c]
            gt[(t, c)] = torch.tensor(np.array(Image.open(f'./data/{seq}/ims/{fn}'))[..., :3]).float().cuda().permute(2, 0, 1) / 255
            means, _, rots = IP.state_at(p, t, 'linear', None)                  # true frame t, same mask for every method
            sg, _ = L.render(rv(p, means, p['seg_colors'], rots), tmd['w'], tmd['h'], tmd['k'][t][c], tmd['w2c'][t][c], is_rendervar=True)
            m = (sg[0:1] > 0.5).float()[None]
            msk[(t, c)] = torch.nn.functional.max_pool2d(m, 11, stride=1, padding=5)[0]
        return gt[(t, c)], msk[(t, c)]

    # speed of the fastest athlete points per frame, to find the hardest frame for the crop figure
    fgm = p['seg_colors'][:, 0] > 0.5
    spd = torch.quantile(torch.linalg.norm(p['means3D'][1:, fgm] - p['means3D'][:-1, fgm], dim=-1)[:, ::50], 0.95, dim=1).cpu().numpy()
    for g in [int(x) for x in a.gaps.split(',')]:
        keys = list(range(0, T, g))
        hidden = [t for t in range(keys[1], keys[-2]) if t % g]
        acc = {m: [[], []] for m in METHODS + ['oracle']}
        for t in hidden:
            for c in range(ncam):
                y, m = real(t, c); K, M = tmd['k'][t][c], tmd['w2c'][t][c]
                for meth in METHODS + ['oracle']:
                    st = IP.state_at(p, t, 'linear', None) if meth == 'oracle' else IP.state_at(p, t, meth, keys)
                    im, _ = L.render(rv(p, *st), tmd['w'], tmd['h'], K, M, is_rendervar=True)
                    acc[meth][0].append(psnr(im, y)); acc[meth][1].append(psnr(im, y, m))
        for meth, (full, fg) in acc.items():
            rows.append(dict(seq=seq, gap=g, method=meth, psnr=round(float(np.mean(full)), 3), psnr_fg=round(float(np.mean(fg)), 3), n=len(full)))
            print(f'{seq:10s} gap {g}  {meth:8s} PSNR {np.mean(full):6.2f}  athlete-only {np.mean(fg):6.2f}  ({len(full)} renders)', flush=True)
        if g == max(int(x) for x in a.gaps.split(',')):          # crop at the fastest hidden frame, held-out camera 0
            t = max(hidden, key=lambda tt: spd[min(tt, len(spd) - 1)]); y, m = real(t, 0)
            ys, xs = torch.nonzero(m[0], as_tuple=True)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            tiles = [y[:, y0:y1, x0:x1]]
            for meth in METHODS:
                im, _ = L.render(rv(p, *IP.state_at(p, t, meth, keys)), tmd['w'], tmd['h'], tmd['k'][t][0], tmd['w2c'][t][0], is_rendervar=True)
                tiles.append(im[:, y0:y1, x0:x1])
            hh = 200; ims = [Image.fromarray(L.to_uint8(x)).resize((max(1, int(x.shape[2] * hh / x.shape[1])), hh)) for x in tiles]
            row = Image.new('RGB', (sum(i.width for i in ims) + 6 * len(ims), hh), (255, 255, 255)); xo = 0
            for i in ims: row.paste(i, (xo, 0)); xo += i.width + 6
            crops.append(row); print(f'{seq}: crop at frame {t} (real | nearest | linear | cubic), gap {g}', flush=True)
    del p; torch.cuda.empty_cache()
with open(f'{a.out}/interp_{a.exp}.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if crops:
    Wd = max(c.width for c in crops); G = Image.new('RGB', (Wd, sum(c.height + 6 for c in crops)), (255, 255, 255)); yo = 0
    for c in crops: G.paste(c, (0, yo)); yo += c.height + 6
    G.save(f'{a.out}/crops_{a.exp}.jpg', quality=92)
print(f'wrote {a.out}/interp_{a.exp}.csv')
