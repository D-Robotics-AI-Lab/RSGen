import os

file_path = '/root/miniconda3/lib/python3.9/site-packages/mmcv/parallel/_functions.py'

# 读取文件
with open(file_path, 'r') as f:
    content = f.read()

# 目标替换的旧代码
old_code = 'streams = [_get_stream(device) for device in target_gpus]'

# 新代码 (逻辑：如果是 int 就转为 torch.device，否则直接用)
new_code = 'streams = [_get_stream(torch.device(\'cuda\', device) if isinstance(device, int) else device) for device in target_gpus]'

if old_code in content:
    # 1. 替换核心逻辑
    content = content.replace(old_code, new_code)
    
    # 2. 确保文件头部有 import torch
    if'import torch\n' not in content and'import torch ' not in content:
        content = 'import torch\n' + content
        print('📦 已在文件头部添加 import torch')
    
    # 3. 写回文件
    with open(file_path, 'w') as f:
        f.write(content)
    print('✅ 文件修改成功！修复了 PyTorch 2.x 的 device 兼容问题。')
else:
    if new_code in content:
        print('⚠️ 代码看似已经修改过了，无需重复操作。')
    else:
        print('❌ 未找到目标代码，可能文件版本不同，请检查文件内容。')
