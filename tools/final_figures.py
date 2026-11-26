"""Figures for the final report, from the cluster outputs kept in media/final/ (see docs/experiments.md).
    python tools/final_figures.py        -> report/final/figures/*
Needs Pillow and ffmpeg. Mid-term figures are made by tools/compose_figures.py.
"""
import os, subprocess, shutil
from PIL import Image, ImageDraw
from compose_figures import tile, save as _save, font

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f'{REPO}/media/final'; OUT = f'{REPO}/report/final/figures'; FR = f'{M}/frames'
SPORTS = ['basketball', 'boxes', 'football', 'juggle', 'softball', 'tennis']
os.makedirs(OUT, exist_ok=True); os.makedirs(FR, exist_ok=True)
F = font(20)


def save(tiles, cols, name, pad=6):
    import compose_figures as C; old = C.OUT; C.OUT = OUT; _save(tiles, cols, name, pad); C.OUT = old


def frames(video, fracs, stem):
    n = int(subprocess.check_output(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0', '-show_entries',
                                     'stream=nb_read_packets', '-of', 'csv=p=0', video]).decode().strip().split(',')[0])
    out = []
    for f in fracs:
        k = int(f * (n - 1)); p = f'{FR}/{stem}_{k}.jpg'
        if not os.path.exists(p):
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-vf', f'select=eq(n\\,{k})', '-frames:v', '1', '-q:v', '3', p], check=True)
        out.append(p)
    return out


def main():
    # teaser: six sports x three moments of the final replay (push-in, freeze sweep, crane)
    t = []
    for s in SPORTS: t.append(frames(f'{M}/videos/cinematic_{s}_n27_allcam_full.mp4', (0.28, 0.42, 0.66), f'{s}_full'))
    save([tile(Image.open(t[c][r]), 320, 180) for r in range(3) for c in range(6)], 6, 'teaser.jpg', pad=0)
    # held-out views + per-frame PSNR, our models
    h = Image.open(f'{M}/stage_figs_ours/09_heldout_allsports.jpg').convert('RGB'); p = Image.open(f'{M}/stage_figs_ours/08_psnr_per_frame.png').convert('RGB')
    p = p.resize((h.width, int(p.height * h.width / p.width)))
    G = Image.new('RGB', (h.width, h.height + p.height + 6), (255, 255, 255)); G.paste(h, (0, 0)); G.paste(p, (0, h.height + 6)); G.save(f'{OUT}/heldout_ours.jpg', quality=92)
    save([tile(Image.open(f'{M}/stage_figs_ours/07_tracks_{s}.jpg'), 480, 270, s) for s in ['basketball', 'football', 'softball']], 3, 'tracks_ours.jpg')
    # skeleton overlays
    save([tile(Image.open(f'{M}/skeleton_overlays/{s}_overlay.jpg'), 480, 270, s) for s in SPORTS], 3, 'skeletons.jpg')
    # interpolation crops: rows = sports, columns = real | nearest | linear | cubic (gap 8)
    C = Image.open(f'{M}/crops_n27_allcam.jpg').convert('RGB')
    rows, y = [], 0
    import numpy as np
    a = np.array(C).astype(int); white = (a.min(axis=2) > 245).all(axis=1)
    edges, inrow = [], False
    for yy, w in enumerate(white):
        if not w and not inrow: start = yy; inrow = True
        if w and inrow: edges.append((start, yy)); inrow = False
    if inrow: edges.append((start, len(white)))
    keep = {0: 'basketball', 2: 'football', 5: 'tennis'}
    strips = []
    for k, (y0, y1) in enumerate(edges):
        if k not in keep: continue
        strip = C.crop((0, y0, C.width, y1)); cols = (np.array(strip).astype(int).min(axis=2) > 245).all(axis=0)
        xs = np.nonzero(~cols)[0]; strip = strip.crop((0, 0, xs.max() + 1, strip.height)); strips.append(strip)
    ww = 1600; strips = [s.resize((ww, int(s.height * ww / s.width))) for s in strips]   # same width, so the four columns line up
    Wd = max(s.width for s in strips); lab = 32
    G = Image.new('RGB', (Wd, lab + sum(s.height + 6 for s in strips)), (255, 255, 255)); d = ImageDraw.Draw(G)
    q = strips[0].width // 4
    for i, name in enumerate(['real held-out view', 'frame repeat', 'straight-line blend', 'cubic (ours)']): d.text((i * q + 6, 6), name, fill=(20, 24, 31), font=F)
    yo = lab
    for s in strips: G.paste(s, (0, yo)); yo += s.height + 6
    G.save(f'{OUT}/interp_crops.jpg', quality=92)
    for src, dst in [('psnr_vs_speed_n27_allcam.png', 'psnr_vs_speed.png'), ('worst_frames_n27_allcam.jpg', 'worst_frames.jpg'),
                     ('cine_paths/cine_path_basketball_n27_allcam_full.png', 'path_full.png')]:
        shutil.copy(f'{M}/{src}', f'{OUT}/{dst}')
    for f in ['inspiration.jpg', 'stages.jpg', 'director_versions.jpg']:
        shutil.copy(f'{REPO}/report/midterm/figures/{f}', f'{OUT}/{f}')
    print('final figures in', OUT)


if __name__ == '__main__':
    main()
