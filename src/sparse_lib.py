"""Shared code for the "how many cameras?" experiments. Run from inside Dynamic3DGaussians/.
Wraps the original train.py functions; nothing in the original repo is modified except the
SciPy KNN patch in helpers.py."""
import json, os, numpy as np, torch
from PIL import Image
from train import get_dataset, initialize_params, initialize_optimizer, get_loss, \
    initialize_per_timestep, initialize_post_first_timestep
from helpers import setup_camera, params2rendervar, params2cpu, save_params
from external import calc_psnr, calc_ssim, densify
from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def load_meta(seq):
    return json.load(open(f'./data/{seq}/train_meta.json')), json.load(open(f'./data/{seq}/test_meta.json'))


def cam_centers(md, t=0):
    return np.array([np.linalg.inv(np.array(w))[:3, 3] for w in md['w2c'][t]])


def ring_angles(md):
    """Azimuth of each camera around the dome, in the dome's own plane (PCA), so no world-axis assumption."""
    C = cam_centers(md); c0 = C.mean(0)
    _, _, Vt = np.linalg.svd(C - c0)
    a, b = (C - c0) @ Vt[0], (C - c0) @ Vt[1]
    return np.arctan2(b, a)


def pick_cams(md, n, seed=0):
    """n training cameras spread evenly in azimuth; seed rotates which cameras are picked."""
    ncam = len(md['fn'][0])
    if n >= ncam: return list(range(ncam))
    order = np.argsort(ring_angles(md))
    start = (seed * 7) % ncam
    idx = (start + np.floor(np.arange(n) * ncam / n)).astype(int) % ncam
    return sorted(order[idx].tolist())


def subset(md, keep):
    """Keep only cameras `keep` at every timestep (any per-camera list field is subset)."""
    ncam = len(md['fn'][0]); out = {}
    for k, v in md.items():
        if isinstance(v, list) and len(v) == len(md['fn']) and isinstance(v[0], list) and len(v[0]) == ncam:
            out[k] = [[vt[i] for i in keep] for vt in v]
        else:
            out[k] = v
    return out


def make_random_init(seq, seed=0, n=100000):
    """Init cloud that uses NO camera information: uniform points in the box of the real cloud,
    grey colour, same foreground ratio. Removes the all-camera leak in init_pt_cld.npz."""
    name = f'{seq}_rand{seed}'; d = f'./data/{name}'
    if not os.path.exists(f'{d}/init_pt_cld.npz'):
        os.makedirs(d, exist_ok=True)
        real = np.load(f'./data/{seq}/init_pt_cld.npz')['data']; rng = np.random.default_rng(seed)
        lo, hi = np.percentile(real[:, :3], 1, 0), np.percentile(real[:, :3], 99, 0)
        rnd = np.zeros((n, 7)); rnd[:, :3] = rng.uniform(lo, hi, (n, 3)); rnd[:, 3:6] = 0.5
        rnd[:, 6] = rng.random(n) < real[:, 6].mean()
        for f in ('ims', 'seg'):
            if not os.path.exists(f'{d}/{f}'): os.symlink(os.path.abspath(f'./data/{seq}/{f}'), f'{d}/{f}')
        np.savez(f'{d}/init_pt_cld.npz', data=rnd)
    return name


def init(seq_dir, md_sub, md_full):
    params, variables = initialize_params(seq_dir, md_sub)
    # Fix: scene radius must not depend on how many cameras we keep (densification uses it).
    C = cam_centers(md_full); variables['scene_radius'] = 1.1 * np.max(np.linalg.norm(C - C.mean(0), axis=-1))
    return params, variables, initialize_optimizer(params, variables)


def train_timestep(params, variables, optimizer, data, iters, first, start=0):
    """`start` = absolute iteration index of the first step (densify() schedule depends on it)."""
    todo = []
    for i in range(start, start + iters):
        if not todo: todo = data.copy()
        cur = todo.pop(np.random.randint(len(todo)))
        loss, variables = get_loss(params, cur, variables, first); loss.backward()
        with torch.no_grad():
            if first: params, variables = densify(params, variables, optimizer, i)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return params, variables


@torch.no_grad()
def render(params_or_rv, w, h, k, w2c, is_rendervar=False):
    cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
    rv = params_or_rv if is_rendervar else params2rendervar(params_or_rv)
    im, _, depth = Renderer(raster_settings=cam)(**rv)
    return im.clamp(0, 1), depth


@torch.no_grad()
def eval_test(params, tmd, seq, t):
    """PSNR / SSIM on every held-out test camera at timestep t (images from the original clip)."""
    ps, ss, ims = [], [], []
    for c in range(len(tmd['fn'][t])):
        im, _ = render(params, tmd['w'], tmd['h'], tmd['k'][t][c], tmd['w2c'][t][c])
        gt = torch.tensor(np.array(Image.open(f"./data/{seq}/ims/{tmd['fn'][t][c]}"))).float().cuda().permute(2, 0, 1) / 255
        ps.append(calc_psnr(im, gt).mean().item()); ss.append(calc_ssim(im, gt).item()); ims.append(im)
    return float(np.mean(ps)), float(np.mean(ss)), ims


def to_uint8(im): return (im.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
