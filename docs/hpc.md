# Running on UVA HPC

Slurm account `cs6501_fall26`, partitions `gpu` and `standard`. All paths come from `hpc/config.sh`.
The project lives in one folder on scratch (scratch is not backed up and purges files unused for 90 days,
so the repo copy on GitHub/OneDrive is the source of truth):

```
/scratch/$USER/cv_project_v2/              $CVP_ROOT
  cv-project/                              this repo; submit every job from here
  envs/d3g_v2/                             conda env (prefix env)
  Dynamic3DGaussians/                      D3G at commit 7dbbd4d + hpc/patches/helpers_knn.diff;
                                           jobs copy src/*.py and scripts/*.py in here and run from it
    data/<seq>/                            PanopticSports (ims/, seg/, train_meta.json, test_meta.json, init_pt_cld.npz)
    output/n27_allcam/basketball/          our trained basketball model (params.npz, per_frame.csv, config.json)
    output/pretrained -> ../../pretrained  the D3G authors' released models, one params.npz per sport
    results/                               videos/, stage_figs/, frame0_renders/, CSVs
  diff-gaussian-rasterization-w-depth/     rasterizer at commit cb65e4b, compiled into the env
  161029_sports1/, panoptic-toolbox/       Panoptic calibration + 3D skeletons (no HD videos)
  logs/                                    every job writes logs/<jobname>_<jobid>.log
```

The original working folder `/scratch/$USER/cv_project` (and `~/exp`) is `$CVP_OLD`: scripts only read from it.

## Building the replica

1. Get the code: `mkdir -p /scratch/$USER/cv_project_v2/logs && cd /scratch/$USER/cv_project_v2 &&
   git clone <repo url> cv-project` (or upload and unzip `cv-project_upload.zip` from `tools/make_bundle.py`).
2. `cd cv-project && bash hpc/setup_replica.sh` submits three chained jobs:
   - `setup_v2` (CPU, ~1 h): clones D3G and the rasterizer at our commits, applies the k-NN patch, creates the
     env from `hpc/env/pip_freeze_d3g.txt` (PyTorch 2.11 + CUDA 12.8), copies data and models from `$CVP_OLD`
   - `rebuild` (GPU, ~10 min): compiles the rasterizer for A100/A40/A6000/B200/RTX Pro 6000
   - `figs` (GPU, ~25 min): re-makes every stage figure and held-out score
3. `source hpc/config.sh && python3 hpc/verify_replica.py "$D3G"` compares the new scores and freeze frames with
   `results/` and prints `REPLICA REPRODUCES OUR RESULTS` if they match.
4. `bash hpc/check_sync.sh` writes `$CVP_ROOT/sync_report.txt` (repo commit, models, results).

Known issues: V100 nodes can't run the rasterizer (the scripts exclude them); during maintenance reservations the
`gpu` partition may need `--gres=gpu:1`; `unzip` rejects the released `output.zip` (`hpc/fetch_pretrained.slurm`
uses Python's zipfile).

## Jobs (`hpc/jobs.slurm`, submit from the repo root)

| Role | Command | Time |
|---|---|---|
| rebuild | `sbatch hpc/jobs.slurm rebuild` | ~10 min |
| full | `sbatch hpc/jobs.slurm full <seq> 27 allcam` | ~3 h |
| cine | `sbatch hpc/jobs.slurm cine <seq> <exp>` or `bash hpc/submit_v6.sh` (all six) | ~20 min per clip |
| videos | `sbatch hpc/jobs.slurm videos <seq> <exp,...>` | ~30 min |
| figs | `sbatch hpc/jobs.slurm figs` | ~25 min |

Keep jobs off V100s with `${V100:+--exclude=$V100}`, where
`V100=$(sinfo -h -p gpu -o "%N %G" | grep -i v100 | awk '{print $1}' | paste -sd, -)`.

## Keeping local and cluster in sync

- **Local → cluster:** commit and `git push` locally, then `cd $CVP_ROOT/cv-project && git pull` on the cluster.
  Without git access, `python tools/make_bundle.py` writes `cv-project_upload.zip` + `MANIFEST.sha256` to upload instead.
- **Cluster → local:** `bash hpc/snapshot_from_hpc.sh` packs code, logs, results and the env into
  `/scratch/$USER/cv_project_v2_snapshot_<date>.tar.gz`; download it into `hpc_pull/`.
- **Check:** `bash hpc/check_sync.sh`.
