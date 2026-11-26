# Results (small files kept in git, grouped by pipeline stage)

| Folder | What it holds | Made by |
|---|---|---|
| `1_reconstruction/runs/<exp>/<seq>/` | `config.json` and `per_frame.csv` (held-out PSNR/SSIM, Gaussians, time per frame) for every training run: `n27_allcam` (six sports), `no_rigid`, `no_rot`, `no_iso` (prior ablation), `n27_allcam_rep` (identical repeat) | `pipeline/1_reconstruct/train_sparse.py` |
| `2_evaluation/heldout/` | Q1: held-out PSNR/SSIM per sport, and per frame | `pipeline/2_evaluate/stage_figs.py` |
| `2_evaluation/skeleton_alignment/` | start frame, scale and mask hit rate of the aligned skeletons | `pipeline/2_evaluate/align_skeletons.py` |
| `2_evaluation/tracks/` | Q2 and the prior ablation: 3D track error per clip and body group, and per frame | `pipeline/2_evaluate/eval_tracks.py` |
| `2_evaluation/interpolation/` | Q3: athlete and full-image PSNR of rebuilt in-between frames | `pipeline/2_evaluate/eval_interp.py` |
| `2_evaluation/analysis/` | error vs speed, worst frame per clip | `pipeline/2_evaluate/analysis.py` |
| `3_director/ablation_director.csv` | Q4 and Q5: every director variant on every sport | `pipeline/3_director/ablate_director.py` |
| `3_director/plans/` | the planned camera path and its metrics for every rendered replay | `pipeline/3_director/render_cinematic.py` |
| `3_director/keyframe_labels.json` | hand-labelled key moments used in Q4 | by hand, from contact sheets |
| `midterm/` | everything reported at the mid-term (basketball run, released-model scores, frame-0 sweep, replay smoothness) | |

Videos and full-size figures are in `outputs/` (not in git).
