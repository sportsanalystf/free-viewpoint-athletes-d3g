"""Smooth, cinematic free-viewpoint replay from a trained Dynamic 3D Gaussians model (v2).

v2 over v1: the camera keeps a safe distance from the players, the field of view widens
automatically so both players and the ball stay in frame, and the freeze-time moment is the
ball's highest point. Path planning lives in cine_path.py (NumPy only).
Slow motion / freeze are rendered by interpolating the Gaussians between frames (no repeated frames).

  python render_cinematic.py --seq basketball --exp n27_allcam            # auto freeze at ball apex
  python render_cinematic.py --seq basketball --exp n27_allcam --freeze 75
"""
import argparse, os, subprocess, numpy as np, torch
from PIL import Image
import sparse_lib as L
import cine_path as CP

ap = argparse.ArgumentParser()
ap.add_argument('--seq', default='basketball'); ap.add_argument('--exp', default='n27_allcam')
ap.add_argument('--freeze', type=float, default=None); ap.add_argument('--scale', type=float, default=2.0)
ap.add_argument('--clear', type=float, default=0.35, help='min camera-to-player distance, fraction of dome radius')
ap.add_argument('--speed', type=float, default=0.4, help='slow-motion factor outside the freeze')
ap.add_argument('--fps', type=int, default=30); ap.add_argument('--out', default='results/videos')
ap.add_argument('--preview', action='store_true'); ap.add_argument('--tag', default='v6')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
md, tmd = L.load_meta(a.seq)
W2C = [np.array(w) for w in md['w2c'][0]] + [np.array(w) for w in tmd['w2c'][0]]

p = {k: torch.tensor(v).cuda().float() for k, v in np.load(f'./output/{a.exp}/{a.seq}/params.npz').items()}
T = p['means3D'].shape[0]
fg = torch.nonzero(p['seg_colors'][:, 0] > 0.5).squeeze(1)
rgb_mid = p['rgb_colors'][T // 2, fg]
redm = (rgb_mid[:, 0] > 0.45) & (rgb_mid[:, 0] > 1.6 * rgb_mid[:, 1]) & (rgb_mid[:, 0] > 1.6 * rgb_mid[:, 2])
g = torch.Generator(device='cpu').manual_seed(0)
pick = torch.cat([fg[torch.randperm(len(fg), generator=g)[:6000].to(fg.device)], fg[redm]]).unique()
fg_pts = p['means3D'][:, pick].cpu().numpy(); fg_rgb = p['rgb_colors'][:, pick].cpu().numpy()

W, H = int(md['w'] * a.scale), int(md['h'] * a.scale)
ct, eye, target, fov, up, info = CP.plan(W2C, fg_pts, fg_rgb, W, H, a.fps, a.freeze, a.clear, a.speed)
dbg = info.pop('_debug')
print('plan:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}, flush=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt   # top-down map of the camera path
pr = lambda X: np.stack([(X - dbg['focus']) @ dbg['e1'], (X - dbg['focus']) @ dbg['e2']], -1)
fig, axp = plt.subplots(figsize=(6, 6)); c2 = pr(dbg['cams']); pp = pr(dbg['pts']); ee = pr(dbg['eye']); tt = pr(dbg['target'])
axp.scatter(pp[:, 0], pp[:, 1], s=1, c='0.6', label='players (freeze frame)'); axp.scatter(c2[:, 0], c2[:, 1], marker='^', c='k', label='real cameras')
sc = axp.scatter(ee[:, 0], ee[:, 1], s=4, c=np.arange(len(ee)), cmap='viridis', label='virtual camera'); axp.plot(tt[:, 0], tt[:, 1], 'r-', lw=1, label='aim point')
axp.set_aspect('equal'); axp.legend(fontsize=7, loc='upper right'); plt.colorbar(sc, label='video frame'); fig.tight_layout()
fig.savefig(f'{a.out}/cine_path_{a.seq}_{a.exp}_{a.tag}.png', dpi=130); print('saved path map', flush=True)

def K_for(f_deg):
    f = 0.5 * W / np.tan(np.radians(f_deg) / 2); return [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]
def gaussians_at(t):
    t0 = int(np.clip(np.floor(t), 0, T - 1)); t1 = min(t0 + 1, T - 1); w = float(t - t0)
    q0 = torch.nn.functional.normalize(p['unnorm_rotations'][t0], dim=-1); q1 = torch.nn.functional.normalize(p['unnorm_rotations'][t1], dim=-1)
    q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
    return {'means3D': p['means3D'][t0] * (1 - w) + p['means3D'][t1] * w,
            'colors_precomp': p['rgb_colors'][t0] * (1 - w) + p['rgb_colors'][t1] * w,
            'rotations': torch.nn.functional.normalize(q0 * (1 - w) + q1 * w, dim=-1),
            'opacities': torch.sigmoid(p['logit_opacities']), 'scales': torch.exp(p['log_scales']),
            'means2D': torch.zeros_like(p['means3D'][0])}

tmp = f'{a.out}/_cine_{a.seq}_{a.exp}_{a.tag}'; os.makedirs(tmp, exist_ok=True)
for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
idx = list(range(0, len(ct), 10)) if a.preview else list(range(len(ct)))
for n, i in enumerate(idx):
    M = CP.w2c_from(eye[i], target[i], up)
    im, _ = L.render(gaussians_at(ct[i]), W, H, K_for(fov[i]), M.tolist(), is_rendervar=True)
    Image.fromarray(L.to_uint8(im)).save(f'{tmp}/{n:04d}.png')
    if n % 50 == 0: print(f'rendered {n}/{len(idx)}', flush=True)
try:
    import imageio_ffmpeg
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'imageio-ffmpeg'], check=True); import imageio_ffmpeg
name = f'{a.out}/cinematic_{a.seq}_{a.exp}_{a.tag}{"_preview" if a.preview else ""}.mp4'
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-framerate', str(3 if a.preview else a.fps), '-i', f'{tmp}/%04d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'slow', '-movflags', '+faststart', name], check=True)
print(f'wrote {name}  ({len(idx)} frames, {W}x{H})')
