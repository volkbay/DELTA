#!/usr/bin/env bash
# Download DELTA / ALED pretrained weights from the v1.0 GitHub release.
#   ./download_weights.sh                 # default: the M3ED-trained models
#   ./download_weights.sh DELTA_M3.pth    # specific file(s)
#   ./download_weights.sh all             # every released checkpoint
set -euo pipefail
BASE="https://github.com/heudiasyc/DELTA/releases/download/v1.0"
DST="$(cd "$(dirname "$0")" && pwd)/weights"
ALL=(ALED_M3 ALED_MV ALED_SL ALED_SL_M3 ALED_SL_MV \
     DELTA_M3 DELTA_MV DELTA_SL DELTA_SL_M3 DELTA_SL_MV)
if [ "$#" -eq 0 ]; then
  MODELS=(DELTA_M3.pth ALED_M3.pth)
elif [ "$1" = "all" ]; then
  MODELS=("${ALL[@]/%/.pth}")
else
  MODELS=("$@")
fi
mkdir -p "$DST"
for m in "${MODELS[@]}"; do
  echo "-> $m"
  curl -fSL --retry 5 --retry-delay 10 -o "$DST/$m" "$BASE/$m"
done
echo "done: $DST"
