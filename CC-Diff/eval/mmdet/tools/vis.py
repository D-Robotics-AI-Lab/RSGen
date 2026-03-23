import mmcv
import numpy as np
import os
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.rsvg import RSVGDataset  


config_path = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_rsvg.py' 
pkl_path = 'results_wo.pkl'
out_dir = '/data/vepfs/public/xianbao01.hou/new/ccdiff/vis/ficgen_gen_ori_obbox_rsvg_wo_fgcnet_wo_edge2edge_train'
score_thr = 0.6 # 置信度阈值
# ------------------------------

os.makedirs(out_dir, exist_ok=True)

cfg = Config.fromfile(config_path)
cfg.data.test.test_mode = True

# 构建数据集
dataset = build_dataset(cfg.data.test)
results = mmcv.load(pkl_path)

# 直接从你的 RSVGDataset 中动态获取类别和调色板！
# 这样即使你以后修改了 RSVGDataset，这里的代码也不用动
class_names = [f" {c}" for c in dataset.CLASSES]

# mmcv 的颜色格式默认是 BGR，如果你定义的 PALETTE 是 RGB，
# 建议在这里做个翻转 [::-1]，如果已经是 BGR 则不需要。这里假设原定义是 RGB。
palette_bgr = [tuple(color[::-1]) for color in dataset.PALETTE] 

print(f"开始生成 RSVGDataset 对比图 (共 {len(dataset)} 张图片)...")
prog_bar = mmcv.ProgressBar(len(dataset))

for i in range(len(dataset)):
    img_name = dataset.data_infos[i]['filename']
    img_path = os.path.join(dataset.img_prefix, img_name)
    img = mmcv.imread(img_path)
    
    if img is None:
        prog_bar.update()
        continue


    ann_info = dataset.get_ann_info(i)
    # COCO 格式的 GT 框是 4维的 (xmin, ymin, xmax, ymax)
    gt_bboxes = ann_info.get('bboxes', np.zeros((0, 4))) 
    gt_labels = ann_info.get('labels', np.zeros((0,), dtype=np.int32))
    
    if len(gt_bboxes) > 0:
        img = imshow_det_bboxes(
            img,
            gt_bboxes,
            gt_labels,
            class_names=class_names,
            bbox_color='green',  # GT 统一绿色
            text_color='green',
            thickness=2,         # 稍微画粗一点以示区分
            show=False,
            out_file=None 
        )

    pred = results[i]
    
    # 兼容处理：分离 bbox 结果
    if isinstance(pred, tuple):
        bbox_results = pred[0]
    else:
        bbox_results = pred

    pred_bboxes = np.vstack(bbox_results)
    pred_labels = np.concatenate([
        np.full(bbox.shape[0], j, dtype=np.int32)
        for j, bbox in enumerate(bbox_results)
    ])
    
    out_file = os.path.join(out_dir, img_name)
    
    if len(pred_bboxes) > 0:
        imshow_det_bboxes(
            img,
            pred_bboxes,
            pred_labels,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=palette_bgr,  #
            text_color=palette_bgr,  
            thickness=1,
            show=False,
            out_file=out_file
        )
    else:
        mmcv.imwrite(img, out_file)
        
    prog_bar.update()

print(f"\n可视化完成！预测框已启用 RSVG 专属多色彩调色板。")
