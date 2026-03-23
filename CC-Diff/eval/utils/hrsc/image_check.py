import os

def check_files_in_txt(dir_path, txt_path, ignore_extension=True):
    """
    检查目录下的所有文件是否都存在于 txt 列表中。
    (只要求目录文件是 txt 列表的子集，允许 txt 中有额外条目)
    
    参数:
        dir_path: 存放图片或文件的目录路径
        txt_path: txt 文件的路径
        ignore_extension: 是否忽略目录中文件的后缀名进行比对 (默认 True)
    """
    
    # 1. 读取 txt 文件中的名称
    with open(txt_path, 'r', encoding='utf-8') as f:
        txt_names = set(line.strip() for line in f if line.strip())
        
    # 2. 获取目录下的所有文件名称
    if not os.path.exists(dir_path):
        print(f"❌ 目录不存在: {dir_path}")
        return
        
    dir_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    
    # 3. 处理后缀名
    if ignore_extension:
        dir_files_to_check = set(os.path.splitext(f)[0] for f in dir_files)
    else:
        dir_files_to_check = set(dir_files)
        
    # 4. 核心逻辑：找出在目录中，但不在 txt 中的文件
    # 注意：这里只查目录里多出来的，txt 里多出来的会被自动忽略
    missing_in_txt = dir_files_to_check - txt_names
    
    # 5. 打印结果
    print(f"📊 统计信息:")
    print(f"  - 目录下需要检查的图片数: {len(dir_files_to_check)}")
    print(f"  - TXT 列表提供的总条目数: {len(txt_names)}")
    
    if not missing_in_txt:
        print(f"\n✅ 完美！目录中的 {len(dir_files_to_check)} 个文件全部都在 txt 列表中！")
        
        # 增加一个友好的提示，说明 txt 包含的内容更多
        if len(txt_names) > len(dir_files_to_check):
            extra_count = len(txt_names) - len(dir_files_to_check)
            print(f"  (注: txt 列表比当前目录多出 {extra_count} 个条目，这是符合预期的)")
        return True
    else:
        print(f"\n❌ 警告：目录中有 {len(missing_in_txt)} 个文件不在 txt 列表中！")
        print("以下是部分未在 txt 中的文件示例 (最多显示10个):")
        for item in list(missing_in_txt)[:10]:
            print(f"  -> {item}")
        return False

# ================= 使用示例 =================
if __name__ == '__main__':
    # 替换为你实际的路径
    directory_to_check = '/data/vepfs/public/xianbao01.hou/new/ccdiff/ficgen_data/train_ficgen_gen_ori_obbox_hrsc_wo_fgcnet_wo_edge2edge_epoch100_200/raw'
    txt_list = '/data/vepfs/public/xianbao01.hou/dataset/HRSC2016/HRSC2016/ImageSets/trainval.txt'
    
    # 执行检查
    check_files_in_txt(directory_to_check, txt_list, ignore_extension=True)
