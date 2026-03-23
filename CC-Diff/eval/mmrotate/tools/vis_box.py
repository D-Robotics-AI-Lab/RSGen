import mmcv
import numpy as np
import os
from mmcv import Config
from mmdet.datasets import build_dataset
from mmrotate.core import imshow_det_rbboxes

# ================= 基础配置 =================
config_path = 'configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
cfg = Config.fromfile(config_path)

# 定义 Train 和 Test 各自独立的输出文件夹
out_dir_train = '/data/vepfs/public/xianbao01.hou/new/ccdiff/vis/dota_black_bg_train'
out_dir_test = '/data/vepfs/public/xianbao01.hou/new/ccdiff/vis/dota_black_bg_test'

os.makedirs(out_dir_train, exist_ok=True)
os.makedirs(out_dir_test, exist_ok=True)
# ============================================

def generate_black_bg_gt(dataset_cfg, out_dir, split_name="Dataset"):
    """
    独立封装的生成器：根据传入的数据集配置，生成黑底真值图
    """
    print(f"\n=========================================")
    print(f"🚀 开始处理 {split_name} 数据集 ...")
    print(f"输出目录: {out_dir}")
    print(f"=========================================")
    
    # 强制不使用 test_mode 也可以加载，但为了严谨，如果是 Test 就设为 True
    if split_name == 'Test':
        dataset_cfg.test_mode = True
        
    dataset = build_dataset(dataset_cfg)
    
    # 提取类别和调色板
    class_names = [f" {c}" for c in dataset.CLASSES]
    if hasattr(dataset, 'PALETTE'):
        palette_bgr = [tuple(color[::-1]) for color in dataset.PALETTE]
    else:
        palette_bgr = 'green'

    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i in range(len(dataset)):
        img_name = dataset.data_infos[i]['filename']
        img_path = os.path.join(dataset.img_prefix, img_name)
        
        # 1. 读取原图获取尺寸
        img = mmcv.imread(img_path)
        if img is None:
            prog_bar.update()
            continue
            
        # 2. 创建纯黑背景
        black_canvas = np.zeros_like(img)
        
        # 3. 获取纯真值 GT (训练集和测试集都可以安全调用此方法获取原始标注)
        ann_info = dataset.get_ann_info(i)
        bboxes = ann_info.get('bboxes', np.zeros((0, 5)))
        labels = ann_info.get('labels', np.zeros((0,), dtype=np.int32))
        
        out_file = os.path.join(out_dir, img_name)
        
        # 4. 在纯黑画布上进行绘制
        if len(bboxes) > 0:
            imshow_det_rbboxes(
                black_canvas,            
                bboxes,
                labels,
                class_names=class_names,
                bbox_color=palette_bgr,
                text_color=palette_bgr,
                thickness=2,             
                show=False,
                out_file=out_file
            )
        else:
            # 如果没有框，直接保存纯黑图
            mmcv.imwrite(black_canvas, out_file)
            
        prog_bar.update()
        
    print(f"\n✅ {split_name} 数据集生成完成！")

# ================= 启动执行 =================
# 1. 先生训练集 (Train)
generate_black_bg_gt(cfg.data.train, out_dir_train, split_name="Train")

# 2. 再生成测试集 (Test)
# 如果你之前跑过 results.pkl 想画预测框，你可以修改上面的函数，或者这里仅画 GT
generate_black_bg_gt(cfg.data.test, out_dir_test, split_name="Test")
