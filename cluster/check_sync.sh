#!/bin/bash
# Write an inventory of what the cluster holds, to compare with the local copy of the repo.
#   cd $CVP_ROOT/cv-project && bash cluster/check_sync.sh     ->  $CVP_ROOT/sync_report.txt
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/cluster/config.sh"; OUT=$WD/sync_report.txt
{
  echo "# sync report  $(date)  host $(hostname)  repo $REPO"
  echo; echo "## repo version"
  if [ -d "$REPO/.git" ]; then
    git -C "$REPO" fetch -q origin 2>/dev/null
    echo "commit $(git -C "$REPO" rev-parse --short HEAD)  ($(git -C "$REPO" log -1 --format=%cd --date=short))"
    echo "behind origin: $(git -C "$REPO" rev-list --count HEAD..@{u} 2>/dev/null || echo '?')   local changes:"; git -C "$REPO" status --short
  elif [ -f "$REPO/MANIFEST.sha256" ]; then
    ( cd "$REPO" && sha256sum -c --quiet MANIFEST.sha256 2>&1 && echo "all $(wc -l < MANIFEST.sha256) files match MANIFEST.sha256" )
  else echo "no git and no MANIFEST.sha256"; fi
  echo; echo "## repo code (sha256)"; ( cd "$REPO" && sha256sum pipeline/*/*.py cluster/* 2>/dev/null )
  echo; echo "## copies inside Dynamic3DGaussians that differ from the repo (jobs refresh these at start)"
  for f in "$REPO"/pipeline/common/*.py "$REPO"/pipeline/[123]_*/*.py; do n=$(basename "$f"); [ -e "$D3G/$n" ] && ! cmp -s "$f" "$D3G/$n" && echo "DIFF $n"; done; echo "(end)"
  echo; echo "## KNN patch"; grep -q cKDTree "$D3G/helpers.py" 2>/dev/null && echo "helpers.py patched" || echo "helpers.py NOT patched (python cluster/patch_d3g_knn.py $D3G)"
  echo; echo "## dataset"; ls "$D3G/data" 2>/dev/null | tr '\n' ' '; echo
  echo; echo "## models"; ls -la "$D3G"/output/*/*/params.npz 2>/dev/null || echo "none"
  echo; echo "## results"
  ( cd "$D3G/results" 2>/dev/null && find . -maxdepth 2 -type f \( -name '*.mp4' -o -name '*.csv' -o -name '*.zip' -o -name '*.png' -o -name '*.json' \) -printf '%10s  %p\n' | sort -k2 ) || echo "no results dir"
  echo; echo "## logs"; ls -1 "$WD"/logs 2>/dev/null | sed 's/_[0-9]*\.log$//' | sort | uniq -c
  echo; echo "## disk use"; du -sh "$WD"/* 2>/dev/null
  echo; echo "## conda env"; ls -d "$CVP_ENV" 2>/dev/null || echo "missing: $CVP_ENV"
} > "$OUT"
echo "wrote $OUT"
