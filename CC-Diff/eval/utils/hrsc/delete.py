import os

def delete_unlisted_files(dir_path, txt_path, ignore_extension=True, dry_run=False):
    """
    删除目录中不在 txt 列表里的所有文件。
    
    参数:
        dir_path: 存放图片的目录路径
        txt_path: txt 文件的路径
        ignore_extension: 是否忽略文件后缀名进行比对 (默认 True)
        dry_run: 是否为“试运行”模式。True: 只打印不删除；False: 真正执行删除 (默认 False)
    """
    
    # 1. 读取 txt 文件中的合法名称，存入 Set 加速查询
    with open(txt_path, 'r', encoding='utf-8') as f:
        valid_names = set(line.strip() for line in f if line.strip())
        
    # 2. 检查目录
    if not os.path.exists(dir_path):
        print(f"❌ 目录不存在: {dir_path}")
        return
        
    # 3. 获取目录下的所有文件
    dir_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    
    deleted_count = 0
    kept_count = 0
    
    print("🔍 开始扫描文件...")
    if dry_run:
        print("⚠️ 当前为 [试运行] 模式，不会真正删除文件。")
    else:
        print("⚠️ 当前为 [真实删除] 模式，正在清理文件！")

    # 4. 遍历并执行删除逻辑
    for filename in dir_files:
        # 提取用于比对的名称（带不带后缀）
        name_to_check = os.path.splitext(filename)[0] if ignore_extension else filename
        
        # 如果这个文件不在 txt 列表里，干掉它
        if name_to_check not in valid_names:
            file_path = os.path.join(dir_path, filename)
            if not dry_run:
                os.remove(file_path)  # 真正删除文件
                print(f"  🗑️ 已物理删除: {filename}")
            else:
                print(f"  [假装删除] 准备删除: {filename}")
            deleted_count += 1
        else:
            kept_count += 1

    # 5. 打印最终统计信息
    print(f"\n📊 清理完成统计:")
    print(f"  - ✅ 保留了 {kept_count} 个文件 (存在于 txt 中)")
    print(f"  - ❌ 删除了 {deleted_count} 个文件 (不在 txt 中)")
    
    if dry_run and deleted_count > 0:
        print("\n💡 提示: 如果确认上述文件可以删除，请将代码中的 dry_run=False 并重新运行。")

# ================= 使用示例 =================
if __name__ == '__main__':
    # 替换为你实际的路径
    directory_to_clean = '/data/vepfs/public/xianbao01.hou/new/ccdiff/ficgen_data/train_ficgen_gen_ori_obbox_hrsc_wo_fgcnet_wo_edge2edge_epoch100_200/raw'
    txt_list = '/data/vepfs/public/xianbao01.hou/dataset/HRSC2016/HRSC2016/ImageSets/trainval.txt'
    
    # ⚠️ 危险操作提醒：
    # 建议第一次运行时，保持 dry_run=True，看看终端打印出的要删除的文件对不对。
    # 确认没问题后，把 dry_run 改为 False，再次运行即可真正删除。
    delete_unlisted_files(directory_to_clean, txt_list, ignore_extension=True, dry_run=False)
