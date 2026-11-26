"""Figures of the intermediate output after every pipeline step (for the mid-term report / slides).
Run from inside Dynamic3DGaussians/ (jobs.slurm does this):   python stage_figs.py
Writes results/stage_figs/*.png|jpg|json|csv and results/stage_figs.zip. Every figure is independent:
if one fails, the error is printed and the others still get made.

  01_inputs_basketball.jpg   all 31 synced camera views at one instant (red border = held-out test camera)
  01_inputs_allsports.jpg    4 views of each of the 6 sports
  02_rig_basketball.png      camera rig in 3D (train / test / a 4-camera subset) around the players
  03_masks_basketball.jpg    image | foreground mask | overlay, 3 cameras
  04_init_basketball.png     initial point cloud: released all-camera cloud vs random cloud
  05_frame0_basketball.jpg   frame-0 training: init -> 500 -> 1k -> 3k -> 10k iterations vs real
  06_layers_<seq>.jpg        trained model at t=75: real | colour render | depth | fg/bg segmentation
  07_tracks_<seq>.jpg        3D Gaussian trajectories over the clip, projected into a test camera
  08_eval.csv / 08_eval.json / 08_psnr_per_frame.png   held-out PSNR/SSIM for every sport, every frame
  09_heldout_allsports.jpg   real vs rendered held-out view, all sports
  10_planner_<seq>.png       key-moment curves (ball height, peak motion) + floater filter, + 10_planner.json
"""
import os, json, csv, traceback, zipfile, numpy as np, torch
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sparse_lib as L
import cine_path as CP

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument('--models', default='mixed', choices=['mixed', 'ours', 'released'],
    help="mixed = our basketball + released others (mid-term); ours = n27_allcam for all six; released = pretrained for all")
_a = _ap.parse_args()
OUT = 'results/stage_figs' + ('' if _a.models == 'mixed' else f'_{_a.models}'); os.makedirs(OUT, exist_ok=True)
SEQS = ['basketball', 'boxes', 'football', 'juggle', 'softball', 'tennis']
EXP = {s: {'mixed': 'n27_allcam' if s == 'basketball' else 'pretrained', 'ours': 'n27_allcam', 'released': 'pretrained'}[_a.models] for s in SEQS}
TMID = 75


def step(name):
    def deco(fn):
        def run(*a, **k):
            try:
                fn(*a, **k); print('OK  ', name, flush=True)
            except Exception:
                print('FAIL', name, flush=True); traceback.print_exc()
        return run
    return deco


def img(seq, fn): return Image.open(f'./data/{seq}/ims/{fn}').convert('RGB')


def mask(seq, fn):
    s = np.array(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}")).astype(np.float32)
    if s.ndim == 3: s = s[..., 0]
    return s / 255.0 if s.max() > 1 else s


def label(im, text, color=(255, 255, 255), border=None):
    im = im.copy(); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0)); d.text((6, 5), text, fill=color)
    if border: d.rectangle([0, 0, im.width - 1, im.height - 1], outline=border, width=5)
    return im


def grid(tiles, cols):
    w, h = tiles[0].size; rows = (len(tiles) + cols - 1) // cols
    G = Image.new('RGB', (cols * w, rows * h), (20, 24, 31))
    for i, t in enumerate(tiles): G.paste(t, ((i % cols) * w, (i // cols) * h))
    return G


def load_params(seq):
    return {k: torch.tensor(v).cuda().float() for k, v in np.load(f'./output/{EXP[seq]}/{seq}/params.npz').items()}


def rendervar(p, t, colors=None):
    return {'means3D': p['means3D'][t], 'colors_precomp': p['rgb_colors'][t] if colors is None else colors,
            'rotations': torch.nn.functional.normalize(p['unnorm_rotations'][t], dim=-1),
            'opacities': torch.sigmoid(p['logit_opacities']), 'scales': torch.exp(p['log_scales']),
            'means2D': torch.zeros_like(p['means3D'][0])}


# ------------------------------------------------------------------ 1. inputs
@step('01 inputs')
def fig_inputs():
    md, tmd = L.load_meta('basketball'); t = TMID; tiles = []
    for c, fn in enumerate(md['fn'][t]):
        tiles.append(label(img('basketball', fn).resize((320, 180)), f'train cam {fn.split("/")[0]}'))
    for c, fn in enumerate(tmd['fn'][t]):
        tiles.append(label(img('basketball', fn).resize((320, 180)), f'TEST cam {fn.split("/")[0]}', (255, 120, 120), (230, 60, 60)))
    grid(tiles, 8).save(f'{OUT}/01_inputs_basketball.jpg', quality=90)
    rows = []
    for s in SEQS:
        md, _ = L.load_meta(s); n = len(md['fn'][t]); pick = [0, n // 4, n // 2, 3 * n // 4]
        rows += [label(img(s, md['fn'][t][c]).resize((320, 180)), f'{s}  cam {md["fn"][t][c].split("/")[0]}') for c in pick]
    grid(rows, 4).save(f'{OUT}/01_inputs_allsports.jpg', quality=90)


# ------------------------------------------------------------------ 2. rig
@step('02 rig')
def fig_rig():
    md, tmd = L.load_meta('basketball')
    C = L.cam_centers(md); Ct = L.cam_centers(tmd)
    F = np.array([np.array(w)[2, :3] for w in md['w2c'][0]]); Ft = np.array([np.array(w)[2, :3] for w in tmd['w2c'][0]])
    sub = L.pick_cams(md, 4, 0)
    pc = np.load('./data/basketball/init_pt_cld.npz')['data']; fgp = pc[pc[:, 6] > 0.5]
    fgp = fgp[np.random.default_rng(0).choice(len(fgp), min(4000, len(fgp)), replace=False)]
    S = CP.scene_frame([np.array(w) for w in md['w2c'][0]])
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*fgp[:, :3].T, s=0.5, c=fgp[:, 3:6].clip(0, 1), alpha=0.6)
    ax.scatter(*C.T, marker='^', s=40, c='#2B6CB0', label='27 training cameras')
    ax.scatter(*Ct.T, marker='^', s=60, c='#E8772E', label='4 held-out test cameras')
    ax.scatter(*C[sub].T, marker='o', s=160, facecolors='none', edgecolors='k', linewidths=1.5, label='a 4-camera subset')
    L_ = 0.25 * S['radius']
    for c, f in zip(np.vstack([C, Ct]), np.vstack([F, Ft])): ax.plot(*np.stack([c, c + L_ * f]).T, c='0.5', lw=0.6)
    ax.legend(loc='upper left', fontsize=8); ax.set_box_aspect((1, 1, 0.6))
    u = S['up']; ax.view_init(elev=25, azim=-60)
    for a_ in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels): a_([])
    fig.tight_layout(); fig.savefig(f'{OUT}/02_rig_basketball.png', dpi=160); plt.close(fig)
    json.dump(dict(up=u.tolist(), radius=S['radius'], n_train=len(C), n_test=len(Ct), subset4=sub),
              open(f'{OUT}/02_rig_basketball.json', 'w'), indent=1)


# ------------------------------------------------------------------ 3. masks
@step('03 masks')
def fig_masks():
    md, _ = L.load_meta('basketball'); t = TMID; n = len(md['fn'][t]); tiles = []
    for c in (0, n // 3, 2 * n // 3):
        fn = md['fn'][t][c]; im = np.array(img('basketball', fn)); m = mask('basketball', fn)
        ov = im.copy(); ov[m > 0.5] = (0.45 * ov[m > 0.5] + 0.55 * np.array([255, 140, 40])).astype(np.uint8)
        tiles += [label(Image.fromarray(im).resize((320, 180)), f'cam {fn.split("/")[0]}: image'),
                  label(Image.fromarray((m * 255).astype(np.uint8)).convert('RGB').resize((320, 180)), 'foreground mask'),
                  label(Image.fromarray(ov).resize((320, 180)), 'overlay')]
    grid(tiles, 3).save(f'{OUT}/03_masks_basketball.jpg', quality=90)


# ------------------------------------------------------------------ 4. init clouds
@step('04 init')
def fig_init():
    real = np.load('./data/basketball/init_pt_cld.npz')['data']
    rnd = np.load(f"./data/{L.make_random_init('basketball', 0)}/init_pt_cld.npz")['data']
    fig = plt.figure(figsize=(11, 5))
    for k, (P, title) in enumerate([(real, f'released init cloud ({len(real):,} pts, built from ALL cameras)'),
                                    (rnd, f'random init ({len(rnd):,} pts, no camera info)')]):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        Q = P[np.random.default_rng(0).choice(len(P), min(30000, len(P)), replace=False)]
        ax.scatter(*Q[:, :3].T, s=0.3, c=Q[:, 3:6].clip(0, 1)); ax.set_title(title, fontsize=9)
        ax.view_init(elev=25, azim=-60); ax.set_box_aspect((1, 1, 0.6))
        for a_ in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels): a_([])
    fig.tight_layout(); fig.savefig(f'{OUT}/04_init_basketball.png', dpi=150); plt.close(fig)
    json.dump(dict(real_pts=int(len(real)), real_fg_frac=float(real[:, 6].mean()), random_pts=int(len(rnd))),
              open(f'{OUT}/04_init_basketball.json', 'w'), indent=1)


# ------------------------------------------------------------------ 5. frame-0 training progression
@step('05 frame0')
def fig_frame0():
    md, tmd = L.load_meta('basketball')
    params, variables, opt = L.init('basketball', md, md)
    im0, _ = L.render(params, tmd['w'], tmd['h'], tmd['k'][0][0], tmd['w2c'][0][0])
    n_init = params['means3D'].shape[0]
    gauss = {}
    if os.path.exists('results/sweep_frame0_basketball.csv'):
        for r in csv.DictReader(open('results/sweep_frame0_basketball.csv')):
            if r['n_cams'] == '27' and r['init'] == 'allcam' and r['seed'] == '0':
                gauss[int(r['iters'])] = (int(r['n_gaussians']), float(r['psnr']))
    tiles = [label(img('basketball', tmd['fn'][0][0]).resize((320, 180)), 'real (held-out camera)'),
             label(Image.fromarray(L.to_uint8(im0)).resize((320, 180)), f'init, 0 it ({n_init:,} G)')]
    for it in (500, 1000, 3000, 10000):
        f = f'results/frame0_renders/basketball_n27_allcam_it{it}_s0.png'
        if os.path.exists(f):
            g, ps = gauss.get(it, (0, 0))
            tiles.append(label(Image.open(f).convert('RGB').resize((320, 180)), f'{it} it  {ps:.1f} dB  {g:,} G'))
    grid(tiles, 3).save(f'{OUT}/05_frame0_basketball.jpg', quality=90)
    del params, variables, opt; torch.cuda.empty_cache()


# ------------------------------------------------------------------ 6/7/8/9. trained models, every sport
@step('06-09 models')
def fig_models():
    rows, per_frame, held = [], {}, []
    for s in SEQS:
        try:
            md, tmd = L.load_meta(s); p = load_params(s); T = p['means3D'].shape[0]
            W, H, K, M = tmd['w'], tmd['h'], tmd['k'][TMID][0], tmd['w2c'][TMID][0]
            fg = p['seg_colors'][:, 0] > 0.5
            # 6. layers
            rgb, depth = L.render(rendervar(p, TMID), W, H, K, M, is_rendervar=True)
            seg, _ = L.render(rendervar(p, TMID, p['seg_colors']), W, H, K, M, is_rendervar=True)
            d = depth[0].cpu().numpy(); v = d[d > 0]; lo, hi = np.percentile(v, 2), np.percentile(v, 98)
            dcol = (plt.cm.turbo(1 - np.clip((d - lo) / (hi - lo + 1e-6), 0, 1))[..., :3] * 255).astype(np.uint8)
            real = img(s, tmd['fn'][TMID][0])
            tiles = [label(real.resize((320, 180)), f'{s}: real held-out view'),
                     label(Image.fromarray(L.to_uint8(rgb)).resize((320, 180)), 'rendered colour'),
                     label(Image.fromarray(dcol).resize((320, 180)), 'rendered depth'),
                     label(Image.fromarray(L.to_uint8(seg)).resize((320, 180)), 'rendered fg (red) / bg (blue)')]
            grid(tiles, 4).save(f'{OUT}/06_layers_{s}.jpg', quality=90)
            held += [label(real.resize((320, 180)), f'{s}: real'), label(Image.fromarray(L.to_uint8(rgb)).resize((320, 180)), f'{s}: rendered')]
            # 7. tracks (projected Gaussian centres over the whole clip)
            X = p['means3D'][:, fg].cpu().numpy()                          # (T, Nfg, 3)
            mv = np.linalg.norm(X[-1] - X[0], axis=1); cand = np.nonzero(mv > np.percentile(mv, 50))[0]
            idx = np.random.default_rng(0).choice(cand, min(700, len(cand)), replace=False); X = X[:, idx]
            Mw = np.array(tmd['w2c'][T - 1][0]); Kk = np.array(tmd['k'][T - 1][0])
            Xc = X @ Mw[:3, :3].T + Mw[:3, 3]; z = Xc[..., 2]; uv = (Xc @ Kk.T)[..., :2] / np.maximum(z[..., None], 1e-6)
            fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150); ax.imshow(np.array(img(s, tmd['fn'][T - 1][0])))
            cmap = plt.cm.hsv
            for j in range(uv.shape[1]):
                if (z[:, j] > 0).all():
                    segs = np.stack([uv[:-1, j], uv[1:, j]], 1)
                    ax.add_collection(LineCollection(segs, colors=[cmap(j / uv.shape[1])], linewidths=0.7, alpha=0.85))
            ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off'); fig.subplots_adjust(0, 0, 1, 1)
            fig.savefig(f'{OUT}/07_tracks_{s}.jpg', dpi=150); plt.close(fig)
            # 8. held-out PSNR / SSIM for every frame
            ps_t, ss_t = [], []
            for t in range(T):
                ps_c, ss_c = [], []
                for c in range(len(tmd['fn'][t])):
                    im, _ = L.render(rendervar(p, t), W, H, tmd['k'][t][c], tmd['w2c'][t][c], is_rendervar=True)
                    gt = torch.tensor(np.array(img(s, tmd['fn'][t][c]))).float().cuda().permute(2, 0, 1) / 255
                    ps_c.append(L.calc_psnr(im, gt).mean().item()); ss_c.append(L.calc_ssim(im, gt).item())
                ps_t.append(np.mean(ps_c)); ss_t.append(np.mean(ss_c))
            per_frame[s] = ps_t
            rows.append(dict(seq=s, model=EXP[s], frames=T, gaussians=int(p['means3D'].shape[1]), fg_gaussians=int(fg.sum().item()),
                             psnr=round(float(np.mean(ps_t)), 2), ssim=round(float(np.mean(ss_t)), 3),
                             psnr_min=round(float(np.min(ps_t)), 2), psnr_max=round(float(np.max(ps_t)), 2)))
            print('EVAL', rows[-1], flush=True)
            del p; torch.cuda.empty_cache()
        except Exception:
            print('FAIL model', s, flush=True); traceback.print_exc()
    if rows:
        with open(f'{OUT}/08_eval.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        json.dump(dict(rows=rows, per_frame=per_frame), open(f'{OUT}/08_eval.json', 'w'))
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for s, v in per_frame.items(): ax.plot(v, label=s, lw=1.4)
        ax.set_xlabel('frame'); ax.set_ylabel('held-out PSNR (dB)'); ax.grid(alpha=0.3); ax.legend(ncol=3, fontsize=8)
        fig.tight_layout(); fig.savefig(f'{OUT}/08_psnr_per_frame.png', dpi=160); plt.close(fig)
    if held: grid(held, 4).save(f'{OUT}/09_heldout_allsports.jpg', quality=90)


# ------------------------------------------------------------------ 10. planner analysis
@step('10 planner')
def fig_planner():
    from scipy.spatial import cKDTree
    allinfo = {}
    for s in SEQS:
        try:
            md, tmd = L.load_meta(s)
            W2C = [np.array(w) for w in md['w2c'][0]] + [np.array(w) for w in tmd['w2c'][0]]
            p = np.load(f'./output/{EXP[s]}/{s}/params.npz')
            T = p['means3D'].shape[0]; fg = np.nonzero(p['seg_colors'][:, 0] > 0.5)[0]
            rgb_mid = p['rgb_colors'][T // 2, fg]
            redm = (rgb_mid[:, 0] > 0.45) & (rgb_mid[:, 0] > 1.6 * rgb_mid[:, 1]) & (rgb_mid[:, 0] > 1.6 * rgb_mid[:, 2])
            g = torch.Generator(device='cpu').manual_seed(0)
            pick = np.unique(np.concatenate([fg[torch.randperm(len(fg), generator=g)[:6000].numpy()], fg[redm]]))
            P = p['means3D'][:, pick]; C = p['rgb_colors'][:, pick]
            ct, eye, target, fov, up, info = CP.plan(W2C, P, C, int(md['w'] * 2), int(md['h'] * 2), 30, None, 0.35, 0.4)
            info.pop('_debug'); allinfo[s] = {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in info.items()}
            # re-derive the curves the planner used
            S = CP.scene_frame(W2C); focus, R = S['focus'], S['radius']
            keep = np.zeros((3, P.shape[1]), bool)
            for j, t in enumerate((T // 4, T // 2, 3 * T // 4)):
                Q = P[t]; d = Q - focus; dh = np.linalg.norm(d - np.outer(d @ up, up), axis=1)
                kd = cKDTree(Q).query(Q, k=9)[0][:, -1]; keep[j] = (dh <= 0.85 * R) & (kd <= 2.5 * np.median(kd))
            cols = keep.sum(0) >= 2
            ball, bh = CP.ball_track(P[:, cols], C[:, cols], up)
            V = np.linalg.norm(np.diff(P[:, cols], axis=0), axis=2); peak = CP.gsmooth(np.percentile(V, 95, axis=1), 2.0)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
            nz = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
            ax[0].plot(np.arange(1, T), nz(peak), label='peak player speed', c='#2B6CB0')
            if bh is not None: ax[0].plot(nz(bh), label='ball height', c='#E8772E')
            ax[0].axvline(info['freeze'], c='k', ls='--', lw=1); ax[0].axvspan(0, 0.2 * T, color='0.9'); ax[0].axvspan(0.8 * T, T, color='0.9')
            ax[0].set_title(f"{s}: key moment = frame {info['freeze']:.0f} ({info['freeze_reason']})", fontsize=9)
            ax[0].set_xlabel('frame'); ax[0].legend(fontsize=8)
            F = int(info['freeze']); Q = P[F] - focus
            pr = lambda X: np.stack([X @ S['e1'], X @ S['e2']], -1)
            a_, b_ = pr(Q[~cols]), pr(Q[cols]); cams = pr(np.array([-M[:3, :3].T @ M[:3, 3] for M in W2C]) - focus)
            ax[1].scatter(*a_.T, s=2, c='#E8772E', label=f'removed as floaters ({(~cols).sum()})')
            ax[1].scatter(*b_.T, s=1, c='#2B6CB0', label=f'kept as players ({cols.sum()})')
            ax[1].scatter(*cams.T, marker='^', c='k', s=14, label='real cameras'); ax[1].set_aspect('equal')
            ax[1].set_title('player points, top-down', fontsize=9); ax[1].legend(fontsize=7, loc='upper right')
            fig.tight_layout(); fig.savefig(f'{OUT}/10_planner_{s}.png', dpi=150); plt.close(fig)
            print('PLAN', s, allinfo[s], flush=True)
        except Exception:
            print('FAIL planner', s, flush=True); traceback.print_exc()
    json.dump(allinfo, open(f'{OUT}/10_planner.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    fig_inputs(); fig_rig(); fig_masks(); fig_init(); fig_frame0(); fig_planner(); fig_models()
    with zipfile.ZipFile(f'{OUT}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(OUT)): z.write(f'{OUT}/{f}', f'{os.path.basename(OUT)}/{f}')
    print(f'wrote {OUT}.zip with', len(os.listdir(OUT)), 'files')
