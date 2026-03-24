### Evaluation

### 1. FID
```bash
cd CC-Diff
cd eval
```
Install the required package:
```bash
pip install torch-fidelity
```
Then run:
```bash
bash fid.sh
```

### 2. YOLOScore
To evaluate YOLO Score, first convert the annotation format using `dota2yolo.py` in `eval/utils`:
```bash
python eval/utils/dota2yolo.py
```
After that, train the model according to the training process under the `yoloscore` directory.

Finally, replace the validation set (val) with your own generated validation results, and then run the evaluation.

### 3. Trainability
For the third metric, you need to first create the environment. You can refer to the environment setup under `mmdet/docker`.

Use `covert_dota_coco_format.py` to convert the annotations into COCO format.

For experiments with additional synthetic data, use `combine_syn_images_dota.py` to combine the data.

After the data preparation is complete, use the official tools provided by MMDetection (`tools`) for training and evaluation.
