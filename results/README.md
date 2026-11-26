# Results (small files kept in git)

| File | What it is |
|---|---|
| `heldout_eval.csv` | Held-out PSNR/SSIM per sport, mean over 150 frames and the 4 test cameras |
| `heldout_eval_per_frame.json` | The same, for every frame (`per_frame[sport]` = list of 150 PSNR values) |
| `planner_keymoments.json` | What the replay director decided for each clip: freeze frame, reason, motion direction, clearance, lens range |
| `replay_smoothness.csv` | Camera "jerk" of each replay video (change in global optical-flow motion per frame, px at 320 px width; lower is smoother) |
| `camera_rig_basketball.json`, `init_cloud_basketball.json` | Scene frame (up axis, dome radius) and starting point-cloud stats |
| `preflight_results.txt` | First cluster check: GPU, dataset, skeleton alignment |
| `runs/<exp>/` | `config.json` and `per_frame.csv` (held-out PSNR/SSIM per frame) of our own training runs on basketball |
| `extra/` | Earlier frame-0 experiment data, not used in the report |

Videos, frames, the full stage figures and all cluster job logs are in `media/` (not in git).
