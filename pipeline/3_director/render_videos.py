"""Make the videos you look at:
 1. compare_<seq>.mp4  : held-out camera, side by side: REAL | each experiment (label + PSNR per frame)
 2. orbit_<exp>.mp4    : a virtual camera gliding around the dome while the clip plays (pure novel views)
The orbit path interpolates between the real camera poses (sorted by azimuth), so it needs no
assumption about the world axes.

  python render_videos.py --seq basketball --exps n27_allcam,n04_random --test-cam 0
"""
import argparse, os, json, subprocess, numpy as np, torch, cv2
from PIL import Image
import sparse_lib as L

ap = argparse.ArgumentParser()
ap.add_argument('--seq', default='basketball'); ap.add_argument('--exps', required=True)
ap.add_argument('--test-cam', type=int, default=0); ap.add_argument('--fps', type=int, default=30)
ap.add_argument('--out', default='results/videos'); ap.add_argument('--orbit-only', action='store_true')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
exps = a.exps.split(',')
md, tmd = L.load_meta(a.seq)
W, H = md['w'], md['h']

def label(img, text, sub=''):
    img = img.copy(); cv2.rectangle(img, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if sub: cv2.putText(img, sub, (W - 150, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 255), 1, cv2.LINE_AA)
    return img

def ffmpeg_exe():
    # the conda env's ffmpeg is missing libvmaf on this cluster; use the self-contained imageio-ffmpeg binary
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        subprocess.run(['pip', 'install', '-q', 'imageio-ffmpeg'], check=True)
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()

def encode(frames_dir, path):
    subprocess.run([ffmpeg_exe(), '-v', 'error', '-y', '-framerate', str(a.fps), '-i', f'{frames_dir}/%04d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', path], check=True)

# ---- 1. side-by-side on a held-out camera (skipped with --orbit-only)
c = a.test_cam; tmp = f'{a.out}/_cmp'; os.makedirs(tmp, exist_ok=True)
psnr = {e: np.loadtxt(f'./output/{e}/{a.seq}/per_frame.csv', delimiter=',', skiprows=1)[:, 1] for e in exps}
T = min(len(v) for v in psnr.values())
for t in (range(T) if not a.orbit_only else []):
    real = np.array(Image.open(f"./data/{a.seq}/ims/{tmd['fn'][t][c]}"))[..., :3]
    tiles = [label(real, 'REAL (held-out camera)')]
    for e in exps:
        cfg = json.load(open(f'./output/{e}/{a.seq}/config.json'))
        im = np.array(Image.open(f'./output/{e}/{a.seq}/test_renders/c{c}_{t:03d}.png'))[..., :3]
        tiles.append(label(im, f"{cfg['n_train_cams']} cams, {cfg['init']} init", f'{psnr[e][t]:.1f} dB'))
    while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    Image.fromarray(grid).save(f'{tmp}/{t:04d}.png')
if not a.orbit_only: encode(tmp, f'{a.out}/compare_{a.seq}_cam{c}.mp4'); print('wrote', f'{a.out}/compare_{a.seq}_cam{c}.mp4')

# ---- 2. orbit: slerp/lerp between consecutive real cameras around the ring
def quat(Rm):
    q = np.empty(4); tr = np.trace(Rm)
    if tr > 0:
        s = np.sqrt(tr + 1) * 2; q[:] = [0.25 * s, (Rm[2, 1] - Rm[1, 2]) / s, (Rm[0, 2] - Rm[2, 0]) / s, (Rm[1, 0] - Rm[0, 1]) / s]
    else:
        i = np.argmax(np.diag(Rm)); j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + Rm[i, i] - Rm[j, j] - Rm[k, k]) * 2; q[0] = (Rm[k, j] - Rm[j, k]) / s
        q[1 + i] = 0.25 * s; q[1 + j] = (Rm[j, i] + Rm[i, j]) / s; q[1 + k] = (Rm[k, i] + Rm[i, k]) / s
    return q / np.linalg.norm(q)
def rotm(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def slerp(q0, q1, u):
    d = np.dot(q0, q1)
    if d < 0: q1, d = -q1, -d
    if d > 0.9995: q = q0 + u * (q1 - q0); return q / np.linalg.norm(q)
    th = np.arccos(d); return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)
order = np.argsort(L.ring_angles(md))
c2w = [np.linalg.inv(np.array(md['w2c'][0][i])) for i in order]
K = md['k'][0][order[0]]
def orbit_pose(u):                      # u in [0,1) around the full ring
    x = u * len(c2w); i = int(x) % len(c2w); j = (i + 1) % len(c2w); f = x - int(x)
    R = rotm(slerp(quat(c2w[i][:3, :3]), quat(c2w[j][:3, :3]), f)); C = (1 - f) * c2w[i][:3, 3] + f * c2w[j][:3, 3]
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = C; return np.linalg.inv(M).tolist()
for e in exps:
    p = dict(np.load(f'./output/{e}/{a.seq}/params.npz'))
    p = {k: torch.tensor(v).cuda().float() for k, v in p.items()}
    tmp = f'{a.out}/_orbit_{e}'; os.makedirs(tmp, exist_ok=True)
    cfg = json.load(open(f'./output/{e}/{a.seq}/config.json'))
    nT = p['means3D'].shape[0]
    for t in range(nT):
        rv = {'means3D': p['means3D'][t], 'colors_precomp': p['rgb_colors'][t],
              'rotations': torch.nn.functional.normalize(p['unnorm_rotations'][t]),
              'opacities': torch.sigmoid(p['logit_opacities']), 'scales': torch.exp(p['log_scales']),
              'means2D': torch.zeros_like(p['means3D'][0])}
        im, _ = L.render(rv, W, H, K, orbit_pose(t / nT), is_rendervar=True)
        Image.fromarray(label(L.to_uint8(im), f"novel view orbit - {cfg['n_train_cams']} cams, {cfg['init']} init")).save(f'{tmp}/{t:04d}.png')
    encode(tmp, f'{a.out}/orbit_{a.seq}_{e}.mp4'); print('wrote', f'{a.out}/orbit_{a.seq}_{e}.mp4')
    del p; torch.cuda.empty_cache()
