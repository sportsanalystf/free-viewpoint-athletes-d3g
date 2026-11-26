"""Experiment 2a (fast): held-out quality vs number of cameras, first frame only.
Grid: cameras x init (all-camera cloud vs random) x iterations x seeds. ~1 min per run.
Writes results/sweep_frame0.csv and renders of one test camera for every setting.

  python sweep_frame0.py --seq basketball --cams 2,3,4,6,8,12,16,27 --iters 500,3000 --seeds 0,1
"""
import argparse, os, csv, time, numpy as np, torch
from PIL import Image
import sparse_lib as L

ap = argparse.ArgumentParser()
ap.add_argument('--seq', default='basketball'); ap.add_argument('--cams', default='2,3,4,6,8,12,16,27')
ap.add_argument('--iters', default='500,3000'); ap.add_argument('--seeds', default='0,1')
ap.add_argument('--out', default='results')
a = ap.parse_args()
os.makedirs(f'{a.out}/frame0_renders', exist_ok=True)
md, tmd = L.load_meta(a.seq)
csv_path = f'{a.out}/sweep_frame0_{a.seq}.csv'; new = not os.path.exists(csv_path)
f = open(csv_path, 'a', newline=''); w = csv.writer(f)
if new: w.writerow(['seq', 'n_cams', 'cams', 'init', 'iters', 'seed', 'psnr', 'ssim', 'n_gaussians', 'seconds'])
iters_list = sorted(int(x) for x in a.iters.split(','))
for seed in [int(s) for s in a.seeds.split(',')]:
    for n in [int(c) for c in a.cams.split(',')]:
        keep = L.pick_cams(md, n, seed); sub = L.subset(md, keep)
        for init in ('allcam', 'random'):
            np.random.seed(seed); torch.manual_seed(seed)
            seq_dir = a.seq if init == 'allcam' else L.make_random_init(a.seq, seed)
            params, variables, opt = L.init(seq_dir, sub, md)
            data = L.get_dataset(0, sub, seq_dir)
            t0 = time.time(); done = 0
            for it in iters_list:                       # one run, evaluated at each checkpoint
                params, variables = L.train_timestep(params, variables, opt, data, it - done, True, start=done)
                done = it
                psnr, ssim, ims = L.eval_test(params, tmd, a.seq, 0)
                w.writerow([a.seq, len(keep), ' '.join(map(str, keep)), init, it, seed, f'{psnr:.3f}', f'{ssim:.4f}',
                            params['means3D'].shape[0], f'{time.time() - t0:.1f}']); f.flush()
                Image.fromarray(L.to_uint8(ims[0])).save(f'{a.out}/frame0_renders/{a.seq}_n{len(keep):02d}_{init}_it{it}_s{seed}.png')
                print(f'RESULT n={len(keep):2d} init={init:6s} iters={it:5d} seed={seed}  PSNR {psnr:6.2f}  SSIM {ssim:.3f}  '
                      f'gaussians {params["means3D"].shape[0]}', flush=True)
            del params, variables, opt, data; torch.cuda.empty_cache()
f.close()
