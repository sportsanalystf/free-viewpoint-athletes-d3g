"""Experiment 2b: full 150-frame Dynamic 3D Gaussians with N training cameras.
Same schedule as the paper (10000 iterations on frame 0, 2000 per later frame). After every frame it
scores the 4 held-out cameras and saves the test-camera renders, so you get per-frame PSNR and a video.

  python train_sparse.py --seq basketball --cams 27 --init allcam --exp n27_allcam
  python train_sparse.py --seq basketball --cams 4  --init random --exp n04_random
"""
import argparse, os, json, time, numpy as np, torch
from PIL import Image
import sparse_lib as L

ap = argparse.ArgumentParser()
ap.add_argument('--seq', default='basketball'); ap.add_argument('--cams', type=int, default=27)
ap.add_argument('--init', choices=['allcam', 'random'], default='allcam'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--exp', required=True); ap.add_argument('--frames', type=int, default=0, help='0 = all 150')
ap.add_argument('--iters0', type=int, default=10000); ap.add_argument('--iters', type=int, default=2000)
a = ap.parse_args()
np.random.seed(a.seed); torch.manual_seed(a.seed)
out = f'./output/{a.exp}/{a.seq}'; os.makedirs(f'{out}/test_renders', exist_ok=True)
md, tmd = L.load_meta(a.seq)
T = len(md['fn']) if a.frames == 0 else min(a.frames, len(md['fn']))
keep = L.pick_cams(md, a.cams, a.seed); sub = L.subset(md, keep)
seq_dir = a.seq if a.init == 'allcam' else L.make_random_init(a.seq, a.seed)
json.dump(dict(vars(a), train_cams=keep, n_train_cams=len(keep)), open(f'{out}/config.json', 'w'), indent=1)
params, variables, opt = L.init(seq_dir, sub, md)
log = open(f'{out}/per_frame.csv', 'w'); log.write('t,psnr,ssim,n_gaussians,seconds\n')
saved = []; t0 = time.time()
for t in range(T):
    data = L.get_dataset(t, sub, seq_dir); first = t == 0
    if not first: params, variables = L.initialize_per_timestep(params, variables, opt)
    params, variables = L.train_timestep(params, variables, opt, data, a.iters0 if first else a.iters, first)
    saved.append(L.params2cpu(params, first))
    if first: variables = L.initialize_post_first_timestep(params, variables, opt)
    psnr, ssim, ims = L.eval_test(params, tmd, a.seq, t)
    for c, im in enumerate(ims): Image.fromarray(L.to_uint8(im)).save(f'{out}/test_renders/c{c}_{t:03d}.png')
    log.write(f'{t},{psnr:.3f},{ssim:.4f},{params["means3D"].shape[0]},{time.time() - t0:.0f}\n'); log.flush()
    print(f'frame {t:3d}/{T}  held-out PSNR {psnr:6.2f}  SSIM {ssim:.3f}  gaussians {params["means3D"].shape[0]}  '
          f'elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
    del data
os.makedirs(f'./output/{a.exp}', exist_ok=True)
L.save_params(saved, a.seq, a.exp)                     # -> output/<exp>/<seq>/params.npz (same format as the repo)
r = np.loadtxt(f'{out}/per_frame.csv', delimiter=',', skiprows=1)
print(f'RESULT exp={a.exp} seq={a.seq} cams={len(keep)} init={a.init} frames={T}  mean held-out PSNR {r[:, 1].mean():.2f}  '
      f'SSIM {r[:, 2].mean():.3f}  time {(time.time() - t0) / 60:.0f} min')
