import os
import shutil
import json
from tqdm import tqdm

def combine_dota_datasets_multi(orig_img_dir, orig_label_dir, 
                                syn_meta_path, syn_img_dirs_list, 
                                out_img_dir, out_label_dir):
    """
    将原始DOTA数据和【多个】生成的合成数据合并。
    
    参数:
        orig_img_dir: 原始图片文件夹
        orig_label_dir: 原始标签文件夹
        syn_meta_path: 生成数据对应的 metadata.jsonl (用于获取文件名列表)
        syn_img_dirs_list: 包含多个生成图片文件夹路径的【列表】 (List[str])
        out_img_dir: 输出图片路径
        out_label_dir: 输出标签路径
        
    策略：
    1. 原始数据：直接复制。
    2. 生成数据 (Source 1)：复制并重命名为 *_1.png, 复用原始标签存为 *_1.txt
    3. 生成数据 (Source 2)：复制并重命名为 *_2.png, 复用原始标签存为 *_2.txt
    ...以此类推
    """
    
    # 创建输出目录
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    print(f"=== 开始合并数据集 (多源模式) ===")
    print(f"原始图片: {orig_img_dir}")
    print(f"原始标签: {orig_label_dir}")
    print(f"生成数据源数量: {len(syn_img_dirs_list)} 个")
    print(f"输出目录: {os.path.dirname(out_img_dir)}")

    # ==========================================
    # 1. 复制原始数据 (Original Data)
    # ==========================================
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    orig_imgs = [f for f in os.listdir(orig_img_dir) if f.lower().endswith(valid_exts)]
    
    print(f"\n[1/3] 正在复制原始数据 (共 {len(orig_imgs)} 张)...")
    for filename in tqdm(orig_imgs, desc="Originals"):
        # 1.1 复制图片
        src_img_path = os.path.join(orig_img_dir, filename)
        dst_img_path = os.path.join(out_img_dir, filename)
        shutil.copy2(src_img_path, dst_img_path)

        # 1.2 复制对应的 DOTA txt 标签
        name_part = os.path.splitext(filename)[0]
        label_filename = name_part + '.txt'
        src_label_path = os.path.join(orig_label_dir, label_filename)
        dst_label_path = os.path.join(out_label_dir, label_filename)

        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)

    # ==========================================
    # 2. 预读取 Metadata
    # ==========================================
    # 只需要读取一次 metadata，因为所有生成数据都是基于同一批文件名生成的
    print(f"\n[2/3] 正在读取 Metadata...")
    meta_data = []
    try:
        with open(syn_meta_path, 'r') as f:
            for line in f:
                try:
                    meta_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"错误: 找不到 Metadata 文件: {syn_meta_path}")
        return

    print(f"Metadata 包含 {len(meta_data)} 个样本条目。")

    # ==========================================
    # 3. 循环处理每个生成数据源
    # ==========================================
    total_syn_success = 0
    
    # 遍历列表中的每个文件夹路径
    for batch_idx, current_syn_img_dir in enumerate(syn_img_dirs_list):
        # batch_idx 从 0 开始，所以后缀设为 batch_idx + 1
        # 例如: 第一个文件夹 -> 后缀 _1, 第二个文件夹 -> 后缀 _2
        suffix_id = batch_idx + 1
        suffix_str = f"_{suffix_id}" 
        
        print(f"\n[3/3 - Source {suffix_id}] 处理生成文件夹: {current_syn_img_dir}")
        
        if not os.path.exists(current_syn_img_dir):
            print(f"警告: 路径不存在，跳过: {current_syn_img_dir}")
            continue

        current_success = 0
        current_missing_label = 0
        
        for sample in tqdm(meta_data, desc=f"Source {suffix_id}"):
            filename = sample['file_name']  # e.g., "P0001.png"
            
            # 当前生成源的图片路径
            syn_img_src = os.path.join(current_syn_img_dir, filename)
            
            if not os.path.exists(syn_img_src):
                continue

            # 构造新的文件名
            # "P0001.png" -> name="P0001", ext=".png" -> "P0001_1.png" (如果 suffix_id=1)
            name_part, ext_part = os.path.splitext(filename)
            new_filename = f"{name_part}{suffix_str}{ext_part}"
            new_label_filename = f"{name_part}{suffix_str}.txt"
            
            dst_img_path = os.path.join(out_img_dir, new_filename)
            dst_label_path = os.path.join(out_label_dir, new_label_filename)

            # 3.1 复制生成的图片
            shutil.copy2(syn_img_src, dst_img_path)
            
            # 3.2 复制对应的【原始】DOTA 标签
            # 逻辑：无论生成的是 _1 还是 _2，它们的 Ground Truth 都来自原始的无后缀文件名
            orig_label_path = os.path.join(orig_label_dir, f"{name_part}.txt")
            
            if os.path.exists(orig_label_path):
                shutil.copy2(orig_label_path, dst_label_path)
                current_success += 1
            else:
                current_missing_label += 1

        total_syn_success += current_success
        print(f"Source {suffix_id} 完成: 添加了 {current_success} 张图片 (标签缺失: {current_missing_label})")

    print(f"\n=== 全部处理完成 ===")
    print(f"累计添加生成样本: {total_syn_success}")
    print(f"最终数据集位置:\nImages: {out_img_dir}\nLabels: {out_label_dir}")

if __name__ == '__main__':

    
    # 1. 原始 DOTA 数据集路径
    orig_root = '/data/vepfs/users/xianbao01.hou/CC-Diff/dataset/filter_train'
    orig_img_dir = os.path.join(orig_root, 'images')
    orig_label_dir = os.path.join(orig_root, 'labelTxt')

    # 2. 生成的数据源列表 (在这里添加所有你想合并的生成数据文件夹)
    # 系统会自动按照列表顺序分配后缀：第一个文件夹 -> _1, 第二个 -> _2, 第三个 -> _3
    syn_img_dirs_list = [
        '/data/vepfs/users/xianbao01.hou/CC-Diff/DOTA_train_ref_w_ours_w_edge2edge_fft_hf_out_train_seed43/raw',
        
        '/data/vepfs/users/xianbao01.hou/CC-Diff/DOTA_train_ref_w_ours_w_edge2edge_fft_hf_out_train_seed44/raw',
        '/data/vepfs/users/xianbao01.hou/CC-Diff/DOTA_train_ref_w_ours_w_edge2edge_fft_hf_out_train_seed45/raw',
        
    ]
    
    syn_meta_path = '/data/vepfs/users/xianbao01.hou/CC-Diff/dataset/filter_train/images/metadata.jsonl'

    # 3. 输出路径
    out_root = '/data/vepfs/public/xianbao01.hou/dataset/CC-Diff/DOTA_obbox/ccdiff_DOTA_train_ref_w_ours_w_edge2edge_fft_hf_out_train_seed43_44_45_300'
    out_img_dir = os.path.join(out_root, 'images')
    out_label_dir = os.path.join(out_root, 'labelTxt')

    # 执行
    combine_dota_datasets_multi(
        orig_img_dir=orig_img_dir,
        orig_label_dir=orig_label_dir,
        syn_meta_path=syn_meta_path,
        syn_img_dirs_list=syn_img_dirs_list, # 传入列表
        out_img_dir=out_img_dir,
        out_label_dir=out_label_dir
    )
