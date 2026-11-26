# Running on UVA HPC

Slurm account `cs6501_fall26`, partitions `gpu` and `standard`. All paths come from `cluster/config.sh`.
The project lives in one folder on scratch (scratch is not backed up and purges files unused for 90 days,
so the repo copy on GitHub/OneDrive is the source of truth):

```
/scratch/$USER/cv_project_v2/              $CVP_ROOT
  cv-project/                              this repo; submit every job from here
  envs/d3g_v2/                             conda env (prefix env)
  Dynamic3DGaussians/                      D3G at commit 7dbbd4d + cluster/patches/helpers_knn.diff;
                                           jobs copy pipeline/*/*.py in here and run from it
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
   git clone <repo url> cv-project`.
2. `cd cv-project && bash cluster/setup_replica.sh` submits three chained jobs:
   - `setup_v2` (CPU, ~1 h): clones D3G and the rasterizer at our commits, applies the k-NN patch, creates the
     env from `cluster/env/pip_freeze_d3g.txt` (PyTorch 2.11 + CUDA 12.8), copies data and models from `$CVP_OLD`
   - `rebuild` (GPU, ~10 min): compiles the rasterizer for A100/A40/A6000/B200/RTX Pro 6000
   - `figs` (GPU, ~25 min): re-makes every stage figure and held-out score
3. `source cluster/config.sh && python3 cluster/verify_replica.py "$D3G"` compares the new scores and freeze frames with
   `results/midterm/` and prints `REPLICA REPRODUCES OUR RESULTS` if they match.
4. `bash cluster/check_sync.sh` writes `$CVP_ROOT/sync_report.txt` (repo commit, models, results).

Known issues: V100 nodes can't run the rasterizer (the scripts exclude them); during maintenance reservations the
`gpu` partition may need `--gres=gpu:1`.

## Jobs (submit from the repo root)

`cluster/jobs.slurm` (GPU) and `cluster/cpu.slurm` (CPU) both run `cluster/run_role.sh`, which lists every role. Main ones:

| Role | Command | Time |
|---|---|---|
| rebuild | `sbatch cluster/jobs.slurm rebuild` | ~5 min |
| full | `sbatch cluster/jobs.slurm full <seq> 27 allcam` | ~3 h |
| cine | `sbatch cluster/jobs.slurm cine <seq> n27_allcam --variant full` | ~20 min |
| figs | `sbatch cluster/jobs.slurm figs ours` | ~5-25 min |
| interp | `sbatch cluster/jobs.slurm interp n27_allcam` | ~30 min |
| align / sheets / ablate / tracks / analysis | `sbatch cluster/cpu.slurm <role> ...` | minutes |
| everything for the final | `bash cluster/submit_final.sh` (see [experiments.md](experiments.md)) | ~4 h wall time |

Keep jobs off V100s with `${V100:+--exclude=$V100}`, where
`V100=$(sinfo -h -p gpu -o "%N %G" | grep -i v100 | awk '{print $1}' | paste -sd, -)` (the submit scripts do this).

## Keeping local and cluster in sync

- **Local → cluster:** commit and `git push` locally, then `cd $CVP_ROOT/cv-project && git pull` on the cluster.
- **Cluster → local:** `bash cluster/snapshot_from_hpc.sh` packs code, logs, results and the env into
  `/scratch/$USER/cv_project_v2_snapshot_<date>.tar.gz`; download it and unpack what you need into `outputs/`.
- **Check:** `bash cluster/check_sync.sh`.
