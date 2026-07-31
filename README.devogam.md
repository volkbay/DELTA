# DELTA (devogam fork)

Fork of [heudiasyc/DELTA](https://github.com/heudiasyc/DELTA) used by *devogam*
to generate **dense event-frame depth** for M3ED sequences. Added as a submodule
at `dep/DELTA`.

M3ED's own GT depth is a projected LiDAR-map render, and how sparse it is
depends entirely on `depth_scan_aggregation` in M3ED's `dataset_list.yaml`:
**4** for every car sequence (measured 0.38% valid pixels on
`car_urban_day_city_hall` — sparser than the single-scan LiDAR *input* at
0.76%) but **400** for the spot sequences (~7% valid). DELTA was trained and
published on the car setting, so spot GT is a different target distribution on
top of the platform gap. DELTA densifies either to ~99%.

## Setup (single conda env — upstream uses two, one is enough)
```bash
conda env create -f environment.yml && conda activate delta
./download_weights.sh            # M3ED-trained DELTA_M3.pth + ALED_M3.pth
./download_weights.sh all        # every released checkpoint
```

## Generate dense depth for an M3ED sequence
```bash
# <raw> holds <seq>_data.h5 + <seq>_depth_gt.h5
python dataset/preprocess/DELTA/preprocess_m3ed_dataset.py test <raw> <processed>
# set dataset.path_test -> <processed> and dump_depth_h5 in configs/DELTA/test_m3ed.json
python test.py configs/DELTA/test_m3ed.json weights/DELTA_M3.pth
# -> out/dense_depth.h5 : (n_frames, H, W) float16, metric metres, event frame
```

`run_m3ed_depth.sh <sequence>` wraps all of the above with sequence-agnostic
staging. Preprocessing is CPU-only and slow (~40 min and ~113 GB of `.pt`
shards for a 5-minute car sequence, so point `m3ed_processed` at a big disk);
inference then takes ~35 min.

**GPU selection gotcha:** `test.py` takes the first CUDA device, and CUDA
enumerates *fastest first*, which need not match `nvidia-smi` indices. On the
local box the RTX 6000 Ada is `nvidia-smi` index 1 but CUDA index 0, so use
`CUDA_VISIBLE_DEVICES=0`. Picking the wrong one fails with "no kernel image is
available for execution on the device" (the 1080 Ti is sm_61, below the
minimum this torch build ships).

## Changes vs upstream
- `dataset/preprocess/DELTA/preprocess_m3ed_dataset.py`: stream events from HDF5
  instead of loading all ~6.85 B into RAM (89 GB); pandas-free `np.unique` event
  indexing (`pd.factorize` rejects a list of tuples on newer pandas); chunk the
  test set (bounded RAM) with a ceil so no tail frames are dropped; `int()` the
  event index-map values for h5py slicing.
- `trainer_tester/tester.py`: optional `dump_depth_h5` config key -> dump the
  bf-aligned dense-depth predictions (× `lidar_max_range`, metres) to an h5.
- `environment.yml`, `download_weights.sh`: single-env setup + weight fetch.
  `pytorch>=2.6` is pinned: the decoding head calls `nn.Buffer`, added in 2.5,
  and conda's unpinned solve lands on 2.4. `torchvision` must be upgraded in
  lockstep or the import dies with "operator torchvision::nms does not exist".
