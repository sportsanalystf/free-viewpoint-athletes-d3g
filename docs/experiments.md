# Final-project experiments

All of them run with one command on the cluster: `bash cluster/submit_final.sh` (see the job chain in that file).

## What is new in the method

| | Problem seen at mid-term | Fix | Code |
|---|---|---|---|
| A. Skeleton key moment | the red-object "ball" rule fired on red wrapping paper in *boxes* | key moment = peak wrist speed of the most active athlete (release of a throw, contact of a swing) | `src/skeleton.py` `director_cues`, `cine_path.plan(key='skeleton')` |
| B. Facing-aware camera | *football* and *juggle* replays opened behind the athlete | of the two side-on views and two sweep directions, pick the one that keeps the camera in front of / beside where the athlete faces | `cine_path.plan(facing=True)` |
| C. Cubic in-between frames | bats and rackets blur in slow motion | Catmull-Rom curves through four frames for Gaussian positions and rotations, instead of straight lines between two | `src/interp.py` |

The skeletons come from the raw Panoptic capture: `pipeline/2_evaluate/align_skeletons.py` maps them into each clip's world
(similarity fitted on the camera centres) and finds each clip's start frame (joints must land on the masks).

## Questions, tests and baselines

| Question | Test | Baseline | Output (in `Dynamic3DGaussians/results/`) |
|---|---|---|---|
| Q1 Is the 4D model accurate? | held-out PSNR/SSIM, all six sports trained by us | the D3G authors' released models | `stage_figs_ours/08_eval.csv` |
| Q2 Is the 3D motion real? | displacement error of Gaussian anchors vs skeleton joints (cm) | static model | `tracks/tracks_n27_allcam.csv` |
| Q3 Do cubic in-between frames help? | hide frames (keep every 2nd/4th/8th), rebuild, compare with real held-out images (whole image and athlete only) | frame repeat, linear blend; upper bound = real model frame | `interp/interp_n27_allcam.csv`, `crops_*.jpg` |
| Q4 Right key moment? | frame error vs hand-labelled key frames (`results/3_director/keyframe_labels.json`, from `keyframes/sheet_*.jpg`) | red-object rule, peak motion | `ablation/director_n27_allcam.csv` (`freeze` column) |
| Q5 Is the camera well behaved? | clearance, occlusion, framing, side-on, behind-the-athlete, path jerk | naive orbit through the real cameras; each part switched off | `ablation/director_n27_allcam.csv`, `videos/plan_*.json` |
| Prior ablation | train basketball without the rigidity / rotation / isometry losses | full D3G | `output/no_*/basketball/per_frame.csv`, `tracks/tracks_n27_allcam+no_rigid+no_rot+no_iso.csv` |

## Ablation variants (`cine_path.VARIANTS`)

`full` = v6 + A + B (rendered with C). Each other variant switches one thing off: `no_floater`, `no_clearance`,
`no_occlusion`, `no_autofov`, `smooth_single`, `smooth_none`, `fixed_view`, `no_facing`, `key_ball`, `key_motion`;
`v6` is the mid-term director. `naive_orbit` blends the real camera poses.

## Analysis

`analysis/psnr_vs_speed_*.png` (error vs athlete speed, with correlation per sport) and `analysis/worst_frames_*.jpg`
(the worst held-out frame of each sport, real vs render).
