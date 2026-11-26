"""Zip the repo for upload to the cluster (OOD Files -> home folder), skipping media/ and hpc_pull/.
Writes MANIFEST.sha256 first so hpc/check_sync.sh can confirm both copies match.
    python tools/make_bundle.py      ->  cv-project_upload.zip  (unzips to cv-project/)
"""
import os, zipfile, hashlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'media', 'hpc_pull', '__pycache__', '.git'}
os.chdir(REPO)
files = []
for d, ds, fs in os.walk('.'):
    ds[:] = [x for x in ds if x not in SKIP_DIRS]
    for f in fs:
        if f == '.DS_Store' or f.endswith(('.zip', '.pyc')) or f == 'MANIFEST.sha256': continue
        files.append(os.path.relpath(os.path.join(d, f), '.'))
files.sort()
with open('MANIFEST.sha256', 'w') as m:
    for f in files: m.write(hashlib.sha256(open(f, 'rb').read()).hexdigest() + '  ' + f + '\n')
with zipfile.ZipFile('cv-project_upload.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files + ['MANIFEST.sha256']:
        zi = zipfile.ZipInfo.from_file(f, 'cv-project/' + f)
        zi.external_attr = (0o755 if f.endswith('.sh') else 0o644) << 16
        z.writestr(zi, open(f, 'rb').read(), zipfile.ZIP_DEFLATED)
print(f'cv-project_upload.zip: {len(files) + 1} files, {os.path.getsize("cv-project_upload.zip") / 1e6:.1f} MB')
