"""Shared code for the "how many cameras?" experiments. Run from inside Dynamic3DGaussians/.
Wraps the original train.py functions; nothing in the original repo is modified except the
SciPy KNN patch in helpers.py."""
import json, os, numpy as np, torch
from PIL import Image
from train import get_dataset, initialize_params, initialize_optimizer, get_loss, \
    initialize_per_timestep, initialize_post_first_timestep
from helpers import setup_camera, params2rendervar, params2cpu, save_params, l1_loss_v1, l1_loss_v2, \
    weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult
from external import calc_psnr, calc_ssim, densify, build_rotation
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

# D3G's loss weights (train.py). The prior ablation sets some of them to 0 (train_sparse.py --weights rigid=0).
WEIGHTS = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0, 'soft_col_cons': 0.01}


def get_loss_w(params, curr_data, variables, is_initial_timestep, weights=None):
    """Same as D3G's train.get_loss, but with configurable weights (and terms with weight 0 are skipped)."""
    w = WEIGHTS if weights is None else weights
    losses = {}
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']
    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))
    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]; fg_rot = rendervar['rotations'][is_fg]
        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"]); rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        if w['rigid'] > 0:
            curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
            losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"], variables["neighbor_weight"])
        if w['rot'] > 0:
            losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None], variables["neighbor_weight"])
        if w['iso'] > 0:
            curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
            losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])
        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()
        bg_pts = rendervar['means3D'][~is_fg]; bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])
        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])
    loss = sum([w[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


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
        loss, variables = get_loss_w(params, cur, variables, first); loss.backward()
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
