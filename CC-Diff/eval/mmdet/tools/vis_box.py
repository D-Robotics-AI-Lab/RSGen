import mmcv
import numpy as np
import os
from mmcv import Config
from mmdet.datasets import build_dataset

# 兼容前面的导入问题，推荐使用这个路径
from mmdet.core.visualization import imshow_det_bboxes 

config_path = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_rsvg.py' 
pkl_path = 'results_w.pkl'  # 如果你只画GT，这个其实可以不加载
out_dir = '/data/vepfs/public/xianbao01.hou/new/ccdiff/vis/black_bg_gt_vis_dior_rsvg'
score_thr = 0.6 
# ------------------------------

os.makedirs(out_dir, exist_ok=True)

cfg = Config.fromfile(config_path)
cfg.data.test.test_mode = True

# 构建数据集以获取真值
dataset = build_dataset(cfg.data.test)
results = mmcv.load(pkl_path) # 加载预测结果（备用）

class_names = [f" {c}" for c in dataset.CLASSES]
palette_bgr = [tuple(color[::-1]) for color in dataset.PALETTE] 

print(f"开始生成纯黑背景标注图 (共 {len(dataset)} 张图片)...")
prog_bar = mmcv.ProgressBar(len(dataset))

for i in range(len(dataset)):
    img_name = dataset.data_infos[i]['filename']
    img_path = os.path.join(dataset.img_prefix, img_name)
    
    # 1. 读取原图，主要是为了获取原图的高和宽
    img = mmcv.imread(img_path)
    if img is None:
        prog_bar.update()
        continue
        
    # 2. 【核心步骤】创建纯黑背景！形状和原图一模一样，但像素全为 0
    black_canvas = np.zeros_like(img)

    # ==========================================
    # 数据源选择：你要画 GT 还是 PKL 里的预测？
    # ==========================================
    
    # 选项 A：获取数据集里的真实标注 (Ground Truth) —— 推荐
    ann_info = dataset.get_ann_info(i)
    bboxes = ann_info.get('bboxes', np.zeros((0, 4))) 
    labels = ann_info.get('labels', np.zeros((0,), dtype=np.int32))

    # 选项 B：如果你其实是想把 results.pkl 里的框当做结果画出来，请取消下面这段注释：
    '''
    pred = results[i]
    if isinstance(pred, tuple):
        bbox_results = pred[0]
    else:
        bbox_results = pred
    
    # 过滤掉低于阈值的预测框
    bboxes_list, labels_list = [], []
    for j, bbox_cls in enumerate(bbox_results):
        if bbox_cls.shape[0] > 0:
            valid_inds = bbox_cls[:, 4] > score_thr
            bboxes_list.append(bbox_cls[valid_inds, :4]) # 只取坐标，不要分数
            labels_list.append(np.full(valid_inds.sum(), j, dtype=np.int32))
            
    bboxes = np.vstack(bboxes_list) if len(bboxes_list) > 0 else np.zeros((0, 4))
    labels = np.concatenate(labels_list) if len(labels_list) > 0 else np.zeros((0,), dtype=np.int32)
    '''
    # ==========================================

    out_file = os.path.join(out_dir, img_name)
    
    # 3. 在纯黑画布上进行绘制
    if len(bboxes) > 0:
        imshow_det_bboxes(
            black_canvas,       # 传入黑图，而不是原图 img
            bboxes,
            labels,
            class_names=class_names,
            bbox_color=palette_bgr,  # 继续使用你的专属多色彩调色板
            text_color=palette_bgr,
            thickness=2,             # 黑底上框可以稍微粗一点，更清晰
            show=False,
            out_file=out_file
        )
    else:
        # 如果这张图完全没有框，直接保存纯黑的图
        mmcv.imwrite(black_canvas, out_file)
        
    prog_bar.update()

print(f"\n生成完成！所有标注框已绘制在纯黑背景上，保存在: {out_dir}")
