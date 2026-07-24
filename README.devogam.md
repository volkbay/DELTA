# DELTA (devogam fork)

Fork of [heudiasyc/DELTA](https://github.com/heudiasyc/DELTA) used by *devogam*
to generate **dense event-frame depth** for M3ED sequences (the M3ED GT depth is
sparse LiDAR, ~7% valid; DELTA densifies it to ~99%). Added as a submodule at
`dep/DELTA`.

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

## Changes vs upstream
- `dataset/preprocess/DELTA/preprocess_m3ed_dataset.py`: stream events from HDF5
  instead of loading all ~6.85 B into RAM (89 GB); pandas-free `np.unique` event
  indexing (`pd.factorize` rejects a list of tuples on newer pandas); chunk the
  test set (bounded RAM) with a ceil so no tail frames are dropped; `int()` the
  event index-map values for h5py slicing.
- `trainer_tester/tester.py`: optional `dump_depth_h5` config key -> dump the
  bf-aligned dense-depth predictions (× `lidar_max_range`, metres) to an h5.
- `environment.yml`, `download_weights.sh`: single-env setup + weight fetch.
