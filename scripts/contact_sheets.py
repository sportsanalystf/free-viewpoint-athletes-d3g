"""Contact sheets for labelling each clip's key moment by eye (CPU). Run from inside Dynamic3DGaussians/.
Every 3rd frame of one held-out camera, numbered, plus the skeleton wrist-speed curve with its peak marked.
    python contact_sheets.py      -> results/keyframes/sheet_<seq>.jpg, speed_<seq>.png
"""
import argparse, json, os, numpy as np
from PIL import Image, ImageDraw
import skeleton as SK
import cine_path as CP

ap = argparse.ArgumentParser(); ap.add_argument('--seqs', default='basketball,boxes,football,juggle,softball,tennis')
ap.add_argument('--step', type=int, default=3); ap.add_argument('--cam', type=int, default=0); ap.add_argument('--out', default='results/keyframes')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
for seq in a.seqs.split(','):
    md = json.load(open(f'./data/{seq}/train_meta.json')); tmd = json.load(open(f'./data/{seq}/test_meta.json')); T = len(tmd['fn'])
    ts = list(range(0, T, a.step)); tw, th, cols = 256, 144, 10
    G = Image.new('RGB', (cols * tw, ((len(ts) + cols - 1) // cols) * th), (20, 24, 31))
    for k, t in enumerate(ts):
        im = Image.open(f"./data/{seq}/ims/{tmd['fn'][t][a.cam]}").convert('RGB').resize((tw, th)); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 44, 16], fill=(0, 0, 0)); d.text((4, 2), str(t), fill=(255, 255, 0)); G.paste(im, ((k % cols) * tw, (k // cols) * th))
    G.save(f'{a.out}/sheet_{seq}.jpg', quality=88)
    skp = f'results/skeletons/{seq}.json'
    if os.path.exists(skp):
        W2C = [np.array(w) for w in md['w2c'][0]] + [np.array(w) for w in tmd['w2c'][0]]
        c = SK.director_cues(SK.load_aligned(skp), CP.scene_frame(W2C)['up'], T)
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 2.4)); ax.plot(c['speed'], label=f"wrist speed, body {c['key_body']}")
        ax.axvline(c['key_frame'], ls='--', c='k', label=f"skeleton key frame {c['key_frame']}"); ax.set_xlabel('frame'); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f'{a.out}/speed_{seq}.png', dpi=130); plt.close(fig)
    print('wrote', seq, flush=True)
