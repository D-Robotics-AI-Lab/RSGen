import mmcv
import numpy as np
import os
from mmcv import Config
from mmdet.datasets import build_dataset
from mmrotate.core import imshow_det_rbboxes

# ---- 强制使用你的自定义类别顺序 ----
CUSTOM_CLASSES = (
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle",
    "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool"
)

config_path = 'configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
pkl_path = 'results_wo.pkl'
out_dir = '/data/vepfs/public/xianbao01.hou/new/ccdiff/vis/ficgen_gen_ori_obbox_dota_wo_fgcnet_wo_edge2edge_train'
score_thr = 0.6  # 置信度阈值
# ------------------------------

os.makedirs(out_dir, exist_ok=True)
cfg = Config.fromfile(config_path)
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)
results = mmcv.load(pkl_path)

# 使用你的 CLASSES 定制化 Label 名字
gt_class_names = [f" {c}" for c in dataset.CLASSES] 
pred_class_names = [f" {c}" for c in CUSTOM_CLASSES]

print(f" 开始生成 V3 修正版对比图 (共 {len(dataset)} 张图片)...")
prog_bar = mmcv.ProgressBar(len(dataset))

for i in range(len(dataset)):
    img_name = dataset.data_infos[i]['filename']
    img_path = os.path.join(dataset.img_prefix, img_name)
    
    img = mmcv.imread(img_path)
    if img is None:
        prog_bar.update()
        continue

    # 1. 绘制 GT (绿色)
    ann_info = dataset.get_ann_info(i)
    gt_bboxes = ann_info.get('bboxes', np.zeros((0, 5)))
    gt_labels = ann_info.get('labels', np.zeros((0,), dtype=np.int32))
    
    if len(gt_bboxes) > 0:
        img = imshow_det_rbboxes(
            img, gt_bboxes, gt_labels,
            class_names=gt_class_names,   # 强制传入自定义前缀和顺序
            bbox_color='green', text_color='green',
            thickness=2,
            show=False, out_file=None
        )

    # 2. 绘制 预测框 (红色)
    pred = results[i]
    pred_bboxes = np.vstack(pred)
    pred_labels = np.concatenate([
        np.full(bbox.shape[0], j, dtype=np.int32)
        for j, bbox in enumerate(pred)
    ])
    
    out_file = os.path.join(out_dir, img_name)
    
    if len(pred_bboxes) > 0:
        imshow_det_rbboxes(
            img, pred_bboxes, pred_labels,
            class_names=pred_class_names, 
            score_thr=score_thr,
            bbox_color='red', text_color='red',
            thickness=1,
            show=False, out_file=out_file
        )
    else:
        mmcv.imwrite(img, out_file)
        
    prog_bar.update()
        
print(f"\n可视化完成！类别顺序已强制更正。")
