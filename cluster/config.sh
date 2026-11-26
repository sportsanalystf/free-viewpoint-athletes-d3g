# Where everything lives on the cluster. Sourced by env.sh and every cluster/ script.
# Override for one command with e.g.  CVP_ROOT=/scratch/$USER/other bash cluster/check_sync.sh
# (if you change CVP_ROOT, also change the "#SBATCH -o" log paths in jobs.slurm / setup_replica.slurm)
CVP_ROOT=${CVP_ROOT:-/scratch/$USER/cv_project_v2}        # this replica: repo, data, models, outputs, logs, env
CVP_ENV=${CVP_ENV:-$CVP_ROOT/envs/d3g_v2}                 # conda env (a path = prefix env, a bare name = named env)
CVP_OLD=${CVP_OLD:-/scratch/$USER/cv_project}             # the original setup; only ever read from, never written
WD=$CVP_ROOT
D3G=$WD/Dynamic3DGaussians
RAST=$WD/diff-gaussian-rasterization-w-depth
LOGS=$WD/logs
D3G_COMMIT=7dbbd4dec404308524ff402756bdb8143a2589b0
RAST_COMMIT=cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110
SEQS="basketball boxes football juggle softball tennis"
