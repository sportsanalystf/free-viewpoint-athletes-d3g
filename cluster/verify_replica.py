"""Check that the replica reproduces our recorded results (standard library only).
Compares the replica's fresh stage-figure run with results/ in the repo:
held-out PSNR/SSIM per sport and the director's freeze frame per sport.
    source cluster/config.sh && python3 cluster/verify_replica.py "$D3G"
"""
import csv, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D3G = sys.argv[1] if len(sys.argv) > 1 else '.'
new_dir = os.path.join(D3G, 'results', 'stage_figs')
ok = True

ref = {r['seq']: r for r in csv.DictReader(open(f'{REPO}/results/midterm/heldout_eval.csv'))}
new_csv = os.path.join(new_dir, '08_eval.csv')
if not os.path.exists(new_csv):
    sys.exit(f'FAIL no {new_csv} yet (has the figs job finished?)')
new = {r['seq']: r for r in csv.DictReader(open(new_csv))}
print(f'{"clip":12s} {"PSNR ref":>9s} {"replica":>8s} {"SSIM ref":>9s} {"replica":>8s}')
for s, r in ref.items():
    n = new.get(s)
    if n is None: print(f'{s:12s} missing in replica'); ok = False; continue
    dp, ds = abs(float(n['psnr']) - float(r['psnr'])), abs(float(n['ssim']) - float(r['ssim']))
    good = dp <= 0.05 and ds <= 0.002; ok &= good
    print(f'{s:12s} {float(r["psnr"]):9.2f} {float(n["psnr"]):8.2f} {float(r["ssim"]):9.3f} {float(n["ssim"]):8.3f}  {"OK" if good else "DIFFERENT"}')

ref_p = json.load(open(f'{REPO}/results/midterm/planner_keymoments.json'))
new_p_path = os.path.join(new_dir, '10_planner.json')
if os.path.exists(new_p_path):
    new_p = json.load(open(new_p_path))
    for s, r in ref_p.items():
        a, b = r['freeze'], new_p.get(s, {}).get('freeze')
        good = b is not None and abs(float(a) - float(b)) < 0.5; ok &= good
        print(f'{s:12s} freeze frame  ref {a:5.0f}  replica {b if b is None else round(b)}  {"OK" if good else "DIFFERENT"}')
else:
    print('FAIL no 10_planner.json'); ok = False

print('\nREPLICA REPRODUCES OUR RESULTS' if ok else '\nSOME RESULTS DIFFER (see above)')
sys.exit(0 if ok else 1)
