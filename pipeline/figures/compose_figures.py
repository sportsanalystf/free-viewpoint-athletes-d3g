"""Build the mid-term report figures from the cluster outputs kept in outputs/midterm/.

    python pipeline/figures/compose_figures.py            # extract video frames, then write report/midterm/figures/*
    python pipeline/figures/compose_figures.py --no-frames  # reuse frames already in outputs/midterm/frames

Inputs (not in git, see .gitignore):
  outputs/midterm/stage_figs/   stage figures from the cluster (cluster/jobs.slurm figs)
  outputs/midterm/videos/            cinematic_v6/<sport>_v6.mp4 and director_versions/basketball_v1..v4.mp4
  outputs/midterm/cine_paths/        cine_path_basketball_n27_allcam_v6.png
  outputs/midterm/inspiration/ atlas.png, arcturus.png, mirrash7.png, sdmiddlecamp.png (screenshots)
Needs Pillow and ffmpeg. The workflow diagram is pipeline/figures/workflow.tex.
"""
import os, sys, subprocess
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M = f'{REPO}/outputs/midterm'; SF = f'{M}/stage_figs'; FR = f'{M}/frames'; OUT = f'{REPO}/report/midterm/figures'
SPORTS = ['basketball', 'boxes', 'football', 'juggle', 'softball', 'tennis']
os.makedirs(FR, exist_ok=True); os.makedirs(OUT, exist_ok=True)


def extract_frames():
    def grab(video, n, out):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-vf', f'select=eq(n\\,{n})',
                        '-frames:v', '1', '-q:v', '3', out], check=True)
    for s in SPORTS:
        for n in (60, 150, 200, 260, 330, 420):
            grab(f'{M}/videos/cinematic_v6/{s}_v6.mp4', n, f'{FR}/{s}_{n}.jpg')
    for k, v in enumerate(['basketball_v1', 'basketball_v2', 'basketball_v3']):
        grab(f'{M}/videos/director_versions/{v}.mp4', 200, f'{FR}/ver{k + 1}.jpg')
    grab(f'{M}/videos/cinematic_v6/basketball_v6.mp4', 200, f'{FR}/ver4.jpg')


def font(sz):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/Library/Fonts/Arial Bold.ttf',
              '/System/Library/Fonts/Supplemental/Arial Bold.ttf']:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()
F = font(20)


def tile(im, w, h, title=None, fit='cover', bg=(255, 255, 255)):
    im = im.convert('RGB'); r = max(w / im.width, h / im.height) if fit == 'cover' else min(w / im.width, h / im.height)
    im2 = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
    T = Image.new('RGB', (w, h), bg); T.paste(im2, ((w - im2.width) // 2, (h - im2.height) // 2))
    if title:
        d = ImageDraw.Draw(T); tw = d.textlength(title, font=F); d.rectangle([0, 0, tw + 16, 30], fill=(20, 24, 31))
        d.text((8, 4), title, fill=(255, 255, 255), font=F)
    return T


def cell(path, cols, rows, i):
    im = Image.open(path); w, h = im.width // cols, im.height // rows; r, c = divmod(i, cols)
    return im.crop((c * w, r * h, (c + 1) * w, (r + 1) * h))


def bbox_crop(im, pad=10):
    b = ImageOps.invert(im.convert('L')).point(lambda v: 255 if v > 12 else 0).getbbox()
    return im.crop((max(0, b[0] - pad), max(0, b[1] - pad), min(im.width, b[2] + pad), min(im.height, b[3] + pad))) if b else im


def save(tiles, cols, name, pad=6):
    w, h = tiles[0].size; rows = (len(tiles) + cols - 1) // cols
    G = Image.new('RGB', (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), (255, 255, 255))
    for k, t in enumerate(tiles): G.paste(t, ((k % cols) * (w + pad), (k // cols) * (h + pad)))
    G.save(f'{OUT}/{name}', quality=92); print('wrote', name, G.size)


def main():
    if '--no-frames' not in sys.argv: extract_frames()
    TW, TH = 480, 270
    # teaser: columns = sports, rows = three moments of the planned shot
    save([tile(Image.open(f'{FR}/{s}_{n}.jpg'), 320, 180) for n in (150, 200, 260) for s in SPORTS], 6, 'teaser.jpg', pad=0)
    # one tile per pipeline step
    inputs = f'{SF}/01_inputs_basketball.jpg'; I = Image.open(inputs); iw, ih = I.width // 8, I.height // 4
    mont = Image.new('RGB', (iw * 3, ih * 3))
    for k, (r, c) in enumerate([(0, 0), (0, 3), (0, 6), (1, 1), (1, 4), (1, 7), (2, 2), (3, 3), (3, 5)]):
        mont.paste(I.crop((c * iw, r * ih, (c + 1) * iw, (r + 1) * ih)), ((k % 3) * iw, (k // 3) * ih))
    held = Image.new('RGB', (iw * 2, ih * 2))
    for k, c in enumerate([3, 4, 5, 6]): held.paste(I.crop((c * iw, 3 * ih, (c + 1) * iw, 4 * ih)), ((k % 2) * iw, (k // 2) * ih))
    init = Image.open(f'{SF}/04_init_basketball.png').convert('RGB'); pl = Image.open(f'{SF}/10_planner_basketball.png').convert('RGB')
    st = [tile(mont, TW, TH, '1 Synced input views'),
          tile(cell(f'{SF}/03_masks_basketball.jpg', 3, 3, 5), TW, TH, '2 Foreground mask'),
          tile(held, TW, TH, '3 Held-out test views'),
          tile(bbox_crop(init.crop((0, int(init.height * 0.12), init.width // 2, init.height))), TW, TH, '4 Starting point cloud', fit='contain'),
          tile(cell(f'{SF}/05_frame0_basketball.jpg', 3, 2, 1), TW, TH, '5 Frame 0, before training'),
          tile(cell(f'{SF}/05_frame0_basketball.jpg', 3, 2, 5), TW, TH, '5 Frame 0, after 10k it'),
          tile(Image.open(f'{SF}/07_tracks_basketball.jpg'), TW, TH, '6-7 Gaussian tracks'),
          tile(cell(f'{SF}/06_layers_basketball.jpg', 4, 1, 2), TW, TH, '4D model: depth'),
          tile(cell(f'{SF}/06_layers_basketball.jpg', 4, 1, 1), TW, TH, '8 Held-out render'),
          tile(bbox_crop(pl.crop((0, 0, pl.width // 2, pl.height))), TW, TH, '10 Key moment', fit='contain'),
          tile(Image.open(f'{M}/cine_paths/cine_path_basketball_n27_allcam_v6.png'), TW, TH, '11-12 Planned path', fit='contain'),
          tile(Image.open(f'{FR}/basketball_200.jpg'), TW, TH, '13 Final replay frame')]
    save(st, 6, 'stages.jpg')
    names = ['v1: too close', 'v2: pushed into a wall', 'v3: lost a player', 'v6: frames the play']
    save([tile(Image.open(f'{FR}/ver{i + 1}.jpg'), TW, TH, n) for i, n in enumerate(names)], 4, 'director_versions.jpg')
    # inspiration (2x2 screenshots)
    tl = []
    for t, f in [('(a) World Labs Atlas', 'atlas.png'), ('(b) Arcturus', 'arcturus.png'),
                 ('(c) @mirrash7: one-camera 3D replay', 'mirrash7.png'), ('(d) @sdmiddlecamp: Gaussian splat', 'sdmiddlecamp.png')]:
        T = tile(Image.open(f'{M}/inspiration/{f}'), 600, 380, None, fit='contain', bg=(245, 245, 245))
        d = ImageDraw.Draw(T); d.rectangle([0, 350, 600, 380], fill=(20, 24, 31)); d.text((8, 354), t, fill=(255, 255, 255), font=F); tl.append(T)
    save(tl, 2, 'inspiration.jpg')
    # held-out views for all sports + per-frame PSNR
    h = Image.open(f'{SF}/09_heldout_allsports.jpg').convert('RGB'); p = Image.open(f'{SF}/08_psnr_per_frame.png').convert('RGB')
    p = p.resize((h.width, int(p.height * h.width / p.width)))
    G = Image.new('RGB', (h.width, h.height + p.height + 6), (255, 255, 255)); G.paste(h, (0, 0)); G.paste(p, (0, h.height + 6))
    G.save(f'{OUT}/heldout_q1.jpg', quality=92); print('wrote heldout_q1.jpg', G.size)
    save([tile(Image.open(f'{SF}/07_tracks_{s}.jpg'), TW, TH, s) for s in ['basketball', 'football', 'softball']], 3, 'tracks.jpg')


if __name__ == '__main__':
    main()
