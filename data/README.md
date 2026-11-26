# Data

Nothing in this folder is tracked by git. On the cluster the data lives in `/scratch/$USER/cv_project/`.

**PanopticSports** (released with Dynamic 3D Gaussians, Luiten et al., 3DV 2024): six 150-frame clips
(basketball, boxes, football, juggle, softball, tennis) from 31 synchronized HD cameras at 640×360,
27 for training and 4 held out, with camera calibration, foreground masks and a starting point cloud.

- Data: `https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip` (≈590 MB), unzip into
  `Dynamic3DGaussians/data/`
- Released trained models (≈10.8 GB): `.../Dynamic3DGaussians/output.zip`, fetched by `hpc/fetch_pretrained.slurm`

**CMU Panoptic Studio** raw capture `161029_sports1` (HD videos, calibration, 3D skeletons), downloaded with
the `panoptic-toolbox` scripts; used only to align the 3D skeletons to the clips.

Please follow the licences and terms of use of both datasets.
