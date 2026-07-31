#!/bin/bash
# Generate DELTA dense depth for ONE M3ED sequence, using sequence-agnostic
# staging folders and the single configs/DELTA/test_m3ed.json.
#
#   ./run_m3ed_depth.sh <sequence_name> [--set KEY=VALUE ...]
#   ./run_m3ed_depth.sh spot_indoor_building_loop
#   ./run_m3ed_depth.sh spot_outdoor_day_penno_short_loop --set dilate_lidar=5
#
# There is deliberately no per-sequence and no per-experiment config. The
# staging folders hold whatever sequence is being processed right now:
#
#   m3ed_raw/        symlinks to the current sequence's *_data.h5 / *_depth_gt.h5
#   m3ed_processed/  DELTA .pt shards for the current sequence
#   out/depth_m3ed.h5   generic output, filed into the sequence folder at the end
#
# The result is copied to $SET/<sequence>/dense_depth_delta_m3.h5 (override the
# final name with OUT_NAME=... if an experiment needs to keep several variants
# side by side, e.g. OUT_NAME=dense_depth_delta_m3_dilated.h5).
set -o pipefail

SEQ=${1:?usage: $0 <sequence_name> [--set KEY=VALUE ...]}
shift
SET=${SET:-/data/users/okbay.volkan/set/m3ed}
CKPT=${CKPT:-weights/DELTA_M3ED.pth}
OUT_NAME=${OUT_NAME:-dense_depth_delta_m3.h5}

cd "$(dirname "$0")" || exit 1
SRC="$SET/$SEQ"
[ -d "$SRC" ] || { echo "E - no such sequence: $SRC"; exit 1; }

# 1. stage the raw inputs (replace whatever the previous sequence left behind)
rm -f m3ed_raw/*.h5
mkdir -p m3ed_raw m3ed_processed out
ln -sf "$SRC/${SEQ}_data.h5"     m3ed_raw/
ln -sf "$SRC/${SEQ}_depth_gt.h5" m3ed_raw/
echo "I - staged $SEQ into m3ed_raw/"

# 2. preprocess into the shared processed folder (cleared first: the loader
#    consumes every shard it finds, so leftovers from another sequence would
#    silently be evaluated too)
rm -f m3ed_processed/*.pt
python dataset/preprocess/DELTA/preprocess_m3ed_dataset.py \
  test m3ed_raw m3ed_processed || exit 1

# 3. inference -- one config, everything else a parameter
rm -f out/depth_m3ed.h5
python test.py configs/DELTA/test_m3ed.json "$CKPT" "$@" || exit 1

# 4. file the generic output under the sequence it belongs to
mv out/depth_m3ed.h5 "$SRC/$OUT_NAME" || exit 1
echo "I - wrote $SRC/$OUT_NAME"
