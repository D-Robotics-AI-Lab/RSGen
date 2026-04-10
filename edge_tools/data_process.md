# HED

Set **`paths.py`** → `DATA_ROOT` (and mirror your data under `CC-Diff/dataset/...` / `dataset/HRSC2016/...` / `ckpts/...` as defined there).

## `hed_process/`

1. Edit `../paths.py` (`DATA_ROOT`).
2. Run from `data_tools/hed_process`:

```bash
python gen_hed_dota_ori.py      # DOTA: scan image folder
python gen_hed_hrsc_ori.py      # HRSC: scan image folder
```

## `edge/`

**Train reference (HED masked by boxes)** — after HED is done, paths come from `paths.py`, then:

```bash
cd data_tools/edge
python get_train_hed_reference_dota.py   # DOTA txt
python get_train_hed_reference_hrsc.py   # HRSC xml
python get_train_hed_reference_rsvg.py     # DIOR-RSVG xml
```

**Val reference** — in order:

1. `dota_run_crop.py` or `hrsc_run_crop.py`
2. `dota_val_edge.py` / `hrsc_val_edge.py` / `dior_val_edge.py`
