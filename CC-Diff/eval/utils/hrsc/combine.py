import os
import shutil
import xml.etree.ElementTree as ET

def combine_hrsc_and_generated(orig_img_dir, orig_ann_dir, gen_img_dir, txt_path, output_dir):
    """
    联合 HRSC 原始数据集与生成数据集，1:1 合并到一个新目录中。
    """
    
    # 1. 创建输出目录结构
    out_img_dir = os.path.join(output_dir, 'AllImages')
    out_ann_dir = os.path.join(output_dir, 'Annotations')
    out_txt_dir = os.path.join(output_dir, 'ImageSets')
    
    for d in [out_img_dir, out_ann_dir, out_txt_dir]:
        os.makedirs(d, exist_ok=True)
        
    # 2. 读取原始 txt 列表中的合法名称
    with open(txt_path, 'r', encoding='utf-8') as f:
        valid_names = [line.strip() for line in f if line.strip()]

    # 辅助函数：根据无后缀的名称找到具体的文件（兼容 .bmp, .jpg, .png）
    def find_file_with_ext(directory, base_name):
        for ext in ['.bmp', '.jpg', '.png', '.jpeg']:
            if os.path.exists(os.path.join(directory, base_name + ext)):
                return base_name + ext, ext
        return None, None

    combined_names = []
    success_orig = 0
    success_gen = 0

    print("🚀 开始联合数据集，这可能需要一点时间，请稍候...")

    # 3. 遍历原 txt 里的每一张图片
    for name in valid_names:
        # ---- A. 处理原始 HRSC 数据 ----
        orig_img_file, orig_ext = find_file_with_ext(orig_img_dir, name)
        orig_xml_file = name + '.xml'
        orig_xml_path = os.path.join(orig_ann_dir, orig_xml_file)
        
        if not orig_img_file or not os.path.exists(orig_xml_path):
            print(f"⚠️ 警告: 原始数据缺失，跳过 {name}")
            continue
            
        # 复制原图和原标注到新目录
        shutil.copy(os.path.join(orig_img_dir, orig_img_file), os.path.join(out_img_dir, orig_img_file))
        shutil.copy(orig_xml_path, os.path.join(out_ann_dir, orig_xml_file))
        combined_names.append(name)
        success_orig += 1
        
        # ---- B. 处理对应的生成数据 ----
        gen_img_file, gen_ext = find_file_with_ext(gen_img_dir, name)
        # 只有当生成目录下存在这张图时，才进行 1:1 扩充，多余的生成图会被自动忽略
        if gen_img_file:
            new_name = f"{name}_1"
            new_img_filename = f"{new_name}{gen_ext}"
            new_xml_filename = f"{new_name}.xml"
            
            # 1. 复制并重命名生成图片
            shutil.copy(os.path.join(gen_img_dir, gen_img_file), 
                        os.path.join(out_img_dir, new_img_filename))
            
            # 2. 读取原 XML，修改内部的 filename 节点，然后另存为 _1.xml
            tree = ET.parse(orig_xml_path)
            root = tree.getroot()
            
            # 确保 XML 内部的 filename 指向新的生成图名称，防止 dataloader 找错图
            filename_node = root.find('filename')
            if filename_node is not None:
                filename_node.text = new_img_filename
                
            tree.write(os.path.join(out_ann_dir, new_xml_filename), encoding='utf-8')
            
            combined_names.append(new_name)
            success_gen += 1

    # 4. 写入合并后的新 txt 文件
    out_txt_path = os.path.join(out_txt_dir, os.path.basename(txt_path))
    with open(out_txt_path, 'w', encoding='utf-8') as f:
        for n in combined_names:
            f.write(n + '\n')

    # 5. 打印统计信息
    print(f"\n📊 数据集联合完成！")
    print(f"  - 📂 新数据集输出路径: {output_dir}")
    print(f"  - 🖼️ 成功复制原图: {success_orig} 张")
    print(f"  - ✨ 成功匹配并重命名生成图: {success_gen} 张")
    print(f"  - 📝 新 txt 列表总条目数: {len(combined_names)}")

# ================= 使用配置 =================
if __name__ == '__main__':
    # 1. HRSC 原始数据路径
    ORIGINAL_IMG_DIR = '/data/vepfs/public/xianbao01.hou/dataset/HRSC2016/HRSC2016/FullDataSet/AllImages/'
    ORIGINAL_ANN_DIR = '/data/vepfs/public/xianbao01.hou/dataset/HRSC2016/HRSC2016/FullDataSet/Annotations/'
    
    # 2. 你的生成图片目录
    GENERATED_IMG_DIR = '/data/vepfs/public/xianbao01.hou/new/ccdiff/ficgen_data/train_ficgen_gen_ori_obbox_hrsc_w_fgcnet_wo_edge2edge_epoch100_200/raw'
    
    TXT_PATH = '/data/vepfs/public/xianbao01.hou/dataset/HRSC2016/HRSC2016/ImageSets/trainval.txt'
    
    # 4. 合并后的全新数据集存放路径 (脚本会自动在这个目录下建子文件夹)
    OUTPUT_DATASET_DIR = '/data/vepfs/public/xianbao01.hou/new/ccdiff/ficgen_data/HRSC2016_Combined_w_fgcnet_wo_edge2edge_epoch100_200'

    # 执行合并
    combine_hrsc_and_generated(ORIGINAL_IMG_DIR, ORIGINAL_ANN_DIR, GENERATED_IMG_DIR, TXT_PATH, OUTPUT_DATASET_DIR)
