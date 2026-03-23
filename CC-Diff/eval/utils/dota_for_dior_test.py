import json
import os

# 1. 路径设置
dota_json_path = '/data/vepfs/users/xianbao01.hou/CC-Diff/dataset/filter_val/val_coco.json'
save_json_path = '/data/vepfs/users/xianbao01.hou/CC-Diff/dataset/filter_val/dota_mapped_to_dior.json'

# 这是你的模型输出层的顺序
dior_classes = [
    'vehicle','chimney','golffield','Expressway-toll-station','stadium',
    'groundtrackfield','windmill','trainstation','harbor','overpass',
    'baseballfield','tenniscourt','bridge','basketballcourt','airplane',
    'ship','storagetank','Expressway-Service-area','airport','dam'
]

# 建立 DIOR 类别名到 ID 的映射 (ID 从 1 开始，符合 COCO 标准)
dior_name2id = {name: i + 1 for i, name in enumerate(dior_classes)}

# 3. 定义 DOTA -> DIOR 的映射规则
# Key: DOTA 原名, Value: DIOR 目标名
mapping_rules = {
    'plane': 'airplane',
    'baseball-diamond': 'baseballfield',
    'bridge': 'bridge',
    'ground-track-field': 'groundtrackfield',
    'small-vehicle': 'vehicle',
    'large-vehicle': 'vehicle',
    'ship': 'ship',
    'tennis-court': 'tenniscourt',
    'basketball-court': 'basketballcourt',
    'storage-tank': 'storagetank',
    'harbor': 'harbor',
}

def convert_dataset():
    print(f"Loading {dota_json_path}...")
    with open(dota_json_path, 'r') as f:
        dota_data = json.load(f)

    new_annotations = []
    
    # 建立 DOTA category id 到 name 的查找表
    dota_id2name = {cat['id']: cat['name'] for cat in dota_data['categories']}
    
    print("Converting annotations...")
    for ann in dota_data['annotations']:
        dota_cat_id = ann['category_id']
        dota_cat_name = dota_id2name.get(dota_cat_id)
        
        # 检查这个类别是否是我们定义的公共类别
        if dota_cat_name in mapping_rules:
            target_dior_name = mapping_rules[dota_cat_name]
            target_dior_id = dior_name2id[target_dior_name]
            
            # 修改标注的 category_id 为 DIOR 的 ID
            ann['category_id'] = target_dior_id
            new_annotations.append(ann)
    
    # 构建新的 categories 列表 (完全使用 DIOR 的定义)
    new_categories = []
    for name, cid in dior_name2id.items():
        new_categories.append({'id': cid, 'name': name, 'supercategory': 'none'})

    # 替换数据
    dota_data['annotations'] = new_annotations
    dota_data['categories'] = new_categories
    
    print(f"Original annotations: {len(dota_data['annotations'])} (Not correct, overwritten)") 
    print(f"Filtered annotations: {len(new_annotations)}")
    
    with open(save_json_path, 'w') as f:
        json.dump(dota_data, f)
    
    print(f"Saved to {save_json_path}")
    print("Done! Now use this JSON for testing.")

if __name__ == '__main__':
    convert_dataset()
