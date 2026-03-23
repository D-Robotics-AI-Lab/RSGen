import os
import shutil
import json
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

# DOTA 类别定义 (确保顺序与现有 JSON 中的 category_id 对应)
CLASSES = ("plane","ship","storage-tank","baseball-diamond","tennis-court",
                "basketball-court","ground-track-field","harbor","bridge","large-vehicle",
                "small-vehicle","helicopter","roundabout","soccer-ball-field","swimming-pool")

def get_image_info(file_path):
    with Image.open(file_path) as img:
        width, height = img.size
    return width, height

def update_coco_with_new_jsonl_and_images_multi(existing_coco_json, meta_path, orig_img_dir, 
                                                syn_img_dirs_list, output_json, combined_img_dir,
                                                include_orig=True):
    """
    将原始 COCO JSON 与【多个】生成的合成数据合并。
    
    Args:
        existing_coco_json: 原始数据的 COCO 格式标注文件 (用于获取类别信息和原始数据)
        meta_path: 生成数据对应的 metadata.jsonl (包含 bbox/caption)
        orig_img_dir: 原始图片目录
        syn_img_dirs_list: List[str], 包含多个生成图片文件夹路径的列表
        output_json: 输出的合并后 JSON 路径
        combined_img_dir: 输出的合并后图片目录
        include_orig: bool, 是否包含原始真实数据。
                      True: 合并 (Real + Syn)
                      False: 仅生成数据 (Only Syn)
    """
    
    mode_str = "合并模式 (Real + Syn)" if include_orig else "仅生成数据模式 (Only Syn)"
    print(f"=== 开始处理 COCO 数据集 [{mode_str}] ===")
    print(f"源数据源数量: {len(syn_img_dirs_list)}")
    
    # 1. 加载现有的 COCO JSON
    # 即使是 Only Syn 模式，也需要加载它以获取 Categories 定义
    print(f"正在加载基础标注结构: {existing_coco_json}")
    with open(existing_coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # 2. 初始化 ID 和 列表容器
    images_list = []
    annotations_list = []
    current_image_id = 0
    current_ann_id = 0

    # 创建输出图片目录
    os.makedirs(combined_img_dir, exist_ok=True)

    # ==========================================
    # 3. 处理原始数据 (根据开关决定是否执行)
    # ==========================================
    if include_orig:
        print("配置为包含原始数据，正在初始化 ID 并准备复制...")
        
        # 继承原始列表
        images_list = coco_data.get('images', [])
        annotations_list = coco_data.get('annotations', [])

        # 获取当前最大的 image_id 和 annotation_id
        if images_list:
            current_image_id = max(img['id'] for img in images_list)
        if annotations_list:
            current_ann_id = max(ann['id'] for ann in annotations_list)

        print(f"原始数据包含 {len(images_list)} 张图片。")
        print(f"初始 Max Image ID: {current_image_id}")
        
        # 复制原始图片
        print(f"正在复制原始图片...")
        for image_info in tqdm(images_list, desc="Copying Originals"):
            src_img_path = os.path.join(orig_img_dir, image_info['file_name'])
            dst_img_path = os.path.join(combined_img_dir, image_info['file_name'])
            
            # 只有不存在时才复制，节省时间
            if not os.path.exists(dst_img_path):
                try:
                    shutil.copy(src_img_path, dst_img_path)
                except FileNotFoundError:
                    pass
    else:
        print("配置为【不包含】原始数据。ID 将从 0 开始计数，原始图片将不会被复制。")
        # 此时 images_list 和 annotations_list 保持为空
        # 但 coco_data 中的 categories 会被保留用于后续写入

    # ==========================================
    # 4. 读取 Metadata
    # ==========================================
    print(f"正在读取 Metadata: {meta_path}")
    meta_data = []
    with open(meta_path, 'r') as f:
        for line in f:
            try:
                meta_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        
    # ==========================================
    # 5. 循环处理每个生成数据源
    # ==========================================
    
    for batch_idx, current_syn_dir in enumerate(syn_img_dirs_list):
        # 确定后缀: 第一个文件夹 -> _1, 第二个 -> _2
        suffix_id = batch_idx + 1
        suffix_str = f"_{suffix_id}"
        
        print(f"\n[Source {suffix_id}] 处理生成文件夹: {current_syn_dir}")
        
        if not os.path.exists(current_syn_dir):
            print(f"警告: 路径不存在，跳过: {current_syn_dir}")
            continue
            
        added_count = 0
        
        for sample in tqdm(meta_data, desc=f"Source {suffix_id}"):
            filename = sample['file_name'] # e.g., P0001.png
            
            # 检查生成图片是否存在
            src_img_path = os.path.join(current_syn_dir, filename)
            if not os.path.exists(src_img_path):
                continue
                
            # 构造新文件名: P0001.png -> P0001_1.png
            name_part, ext_part = os.path.splitext(filename)
            new_filename = f"{name_part}{suffix_str}{ext_part}"
            
            dst_img_path = os.path.join(combined_img_dir, new_filename)
            
            # 获取图片尺寸 (这对 COCO 很重要)
            try:
                width, height = get_image_info(src_img_path)
            except Exception as e:
                print(f"Error reading image {src_img_path}: {e}")
                continue

            # --- 更新 Image Info ---
            current_image_id += 1 # 递增 image_id
            
            images_list.append({
                "id": current_image_id,
                "file_name": new_filename,
                "width": width,
                "height": height
            })
            
            # --- 复制图片 ---
            shutil.copy(src_img_path, dst_img_path)
            added_count += 1

            # --- 更新 Annotation Info ---
            labels = sample.get('caption', [])[1:] 
            bboxes = sample.get('bndboxes', [])
            
            for label, bbox in zip(labels, bboxes):
                if label == '' or label not in CLASSES:
                    continue
                
                x1, y1, x2, y2 = bbox
                
                # 反归一化坐标
                xmin = int(x1 * width)
                ymin = int(y1 * height)
                xmax = int(x2 * width)
                ymax = int(y2 * height)
                
                o_width = xmax - xmin
                o_height = ymax - ymin
                
                # 查找 category_id
                try:
                    category_id = CLASSES.index(label) + 1
                except ValueError:
                    print(f"Warning: Label '{label}' not in CLASSES definition.")
                    continue
                
                current_ann_id += 1 # 递增 annotation_id
                
                annotations_list.append({
                    "id": current_ann_id,
                    "image_id": current_image_id, # 对应当前的新图 ID
                    "category_id": category_id,
                    "bbox": [xmin, ymin, o_width, o_height], # COCO format: [x, y, w, h]
                    "area": o_width * o_height,
                    "segmentation": [],
                    "iscrowd": 0
                })

        print(f"Source {suffix_id} 完成: 添加了 {added_count} 张图片。")

    # 6. 保存最终结果
    coco_data["images"] = images_list
    coco_data["annotations"] = annotations_list
    
    print(f"\n正在保存 JSON 到: {output_json}")
    with open(output_json, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)
        
    print(f"全部完成! 最终 Image ID 计数到: {current_image_id}, Annotation ID 计数到: {current_ann_id}")

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # 配置区域
    # -------------------------------------------------------------------------
    
    # 1. 基础配置
    existing_coco_json = '/data/vepfs/users/xianbao01.hou/CC-Diff/dataset/filter_train/train_coco.json'
    meta_path = '/data/vepfs/users/xianbao01.hou/CC-Diff/dataset/filter_train/images/metadata.jsonl'
    orig_img_dir = '/data/vepfs/users/xianbao01.hou/CC-Diff/dataset/filter_train/images'
    
    # 2. 生成数据源列表 (List)
    syn_img_dirs_list = [
        '/data/vepfs/users/xianbao01.hou/CC-Diff/DOTA_train_ref_w_ours_w_edge2edge_fft_hf_out_train_seed43/raw',
        # '/data/vepfs/users/xianbao01.hou/CC-Diff/DOTA_train_ref_w_ours_w_edge2edge_fft_hf_out_train_seed44/raw',
        # '/data/vepfs/users/xianbao01.hou/CC-Diff/DOTA_train_ref_w_ours_w_edge2edge_fft_hf_out_train_seed45/raw',
    ]

    # -------------------------------------------------------------------------
    # 【核心开关】 是否包含原始真实数据
    # True:  合并 Real + Syn (Ids 累加)
    # False: 只有 Syn (Ids 从0开始, 不复制原图)
    # -------------------------------------------------------------------------
    INCLUDE_ORIGINAL_DATA = False 

    # 3. 输出配置
    # 建议根据开关修改路径名，以免混淆
    if INCLUDE_ORIGINAL_DATA:
        output_folder_name = 'mmcv_ccdiff_DOTA_train_ref_w_ours_w_edge2edge_fft_hf_out_train_seed43_44_200'
    else:
        output_folder_name = 'mmcv_ccdiff_DOTA_train_ref_w_ours_w_edge2edge_fft_hf_out_train_seed43_only_syn'

    output_root_dir = os.path.join('/data/vepfs/public/xianbao01.hou/dataset/CC-Diff/DOTA', output_folder_name)
    
    output_json = os.path.join(output_root_dir, 'combined_train_annotations.json')
    combined_img_dir = os.path.join(output_root_dir, 'combined_train')

    # 确保输出父目录存在
    os.makedirs(output_root_dir, exist_ok=True)

    # 执行
    update_coco_with_new_jsonl_and_images_multi(
        existing_coco_json=existing_coco_json, 
        meta_path=meta_path, 
        orig_img_dir=orig_img_dir, 
        syn_img_dirs_list=syn_img_dirs_list, 
        output_json=output_json, 
        combined_img_dir=combined_img_dir,
        include_orig=INCLUDE_ORIGINAL_DATA  # 传入开关
    )
