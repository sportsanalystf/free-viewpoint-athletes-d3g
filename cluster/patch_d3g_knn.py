"""One-time patch for the Dynamic3DGaussians checkout: replace the Open3D k-NN in helpers.py
(it segfaults on the UVA cluster) with a SciPy KD-tree that returns the same arrays.
   python cluster/patch_d3g_knn.py $D3G   (see cluster/config.sh)
Safe to run twice."""
import sys, re, pathlib
p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '.') / 'helpers.py'
s = p.read_text()
if 'cKDTree' in s:
    print('already patched:', p); sys.exit(0)
new = '''def o3d_knn(pts, num_knn):
    # SciPy KD-tree instead of Open3D (Open3D crashed on the cluster); same return format.
    from scipy.spatial import cKDTree
    pts = np.ascontiguousarray(pts, np.float64)
    d, i = cKDTree(pts).query(pts, k=num_knn + 1)
    return (d[:, 1:] ** 2).astype(np.float64), i[:, 1:].astype(np.int64)


'''
s2, n = re.subn(r'def o3d_knn\(pts, num_knn\):.*?(?=\ndef )', new.rstrip('\n') + '\n\n', s, flags=re.S)
assert n == 1, 'o3d_knn not found in helpers.py'
p.write_text(s2); print('patched:', p)
