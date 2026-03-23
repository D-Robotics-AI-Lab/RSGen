# Training and Inference
This section uses FICGen as an example, with DOTA as the training target.

### 3.1 Enter the FICGen Directory

```bash
cd FICGen
```

### 3.2 Train the L2I Base Model on DOTA

```bash
bash dist_train_ficgen_dota.sh
```

### 3.3 Update the Checkpoint Path

After training, modify the checkpoint loading path in `train_ficgen_fgcontrol_dota.py` so that it points to the trained **L2I base model** checkpoint.

For example, update the relevant code in `train_ficgen_fgcontrol_dota.py` as follows:

```python
state_dict = load_file(
    os.path.join(
        '/data/checkpoint-5400',
        "unet/diffusion_pytorch_model.safetensors"
    )
)
```
### 3.4 Train FGControl

```bash
dist_train_ficgen_dota_fgcontrol.sh
```

### 3.5 Run Inference
```python
torchrun --nproc_per_node=8 infer_multi_ours_dota.py
```
