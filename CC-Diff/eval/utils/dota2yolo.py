# import os
# import cv2
# from pathlib import Path

# def convert_dota_to_yolo(dota_label_dir, yolo_label_dir, image_dir, class_map):
#     """
#     Convert DOTA v1.0 annotations to YOLOv8 format.
    
#     Args:
#         dota_label_dir (str): Directory containing DOTA annotation files
#         yolo_label_dir (str): Directory to save YOLOv8 annotation files
#         image_dir (str): Directory containing images (to get dimensions)
#         class_map (dict): Dictionary mapping category names to class IDs
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(yolo_label_dir, exist_ok=True)
    
#     # Process each DOTA annotation file
#     for label_file in Path(dota_label_dir).glob("*.txt"):
#         with open(label_file, 'r') as f:
#             lines = f.readlines()
        
#         # Prepare output file
#         output_file = os.path.join(yolo_label_dir, label_file.name)
#         yolo_annotations = []
        
#         # Get image dimensions
#         image_name = label_file.stem + '.png'  # Assuming images are PNG
#         image_path = os.path.join(image_dir, image_name)
#         if not os.path.exists(image_path):
#             print(f"Image {image_path} not found, skipping...")
#             continue
#         img = cv2.imread(image_path)
#         img_height, img_width = img.shape[:2]
        
#         # Process each annotation line
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) < 10:  # Skip invalid lines
#                 continue
                
#             # Extract coordinates and category
#             x1, y1, x2, y2, x3, y3, x4, y4, category = parts[:9]
#             if category not in class_map:
#                 print(f"Category {category} not in class_map, skipping...")
#                 continue
                
#             # Convert to float
#             coords = [float(x) for x in [x1, y1, x2, y2, x3, y3, x4, y4]]
            
#             # Calculate bounding box
#             x_min = min(coords[0::2])
#             x_max = max(coords[0::2])
#             y_min = min(coords[1::2])
#             y_max = max(coords[1::2])
            
#             # Calculate YOLO format: center_x, center_y, width, height
#             center_x = (x_min + x_max) / 2
#             center_y = (y_min + y_max) / 2
#             width = x_max - x_min
#             height = y_max - y_min
            
#             # Normalize to [0, 1]
#             center_x /= img_width
#             center_y /= img_height
#             width /= img_width
#             height /= img_height
            
#             # Get class ID
#             class_id = class_map[category]
            
#             # Format YOLO annotation
#             yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
#             yolo_annotations.append(yolo_line)
        
#         # Write to output file
#         with open(output_file, 'w') as f:
#             f.write('\n'.join(yolo_annotations) + '\n')
#         print(f"Converted {label_file.name} to YOLOv8 format")

# def main():
#     # Example configuration
#     dota_label_dir = "/home/users/xianbao01.hou/CC-Diff/dataset/filter_train/labelTxt"  # Replace with your DOTA labels directory
#     yolo_label_dir = "/home/users/xianbao01.hou/CC-Diff/dataset/YOLO_DOTA/labels/train"  # Replace with your output directory
#     image_dir = "/home/users/xianbao01.hou/CC-Diff/dataset/filter_train/images"            # Replace with your images directory
    
#     # Define class mapping (modify according to your dataset)
#     # objects = {
#     #     "plane": 0, "ship": 1, "storage-tank": 2, "baseball-diamond": 3,
#     #     "tennis-court": 4, "basketball-court": 5, "ground-track-field": 6,
#     #     "harbor": 7, "bridge": 8, "large-vehicle": 9, "small-vehicle": 10,
#     #     "helicopter": 11, "roundabout": 12, "soccer-ball-field": 13,
#     #     "swimming-pool": 14
#     # }
#     class_map = {
#         'plane': 0,
#         'ship': 1,
#         'storage-tank': 2,
#         'baseball-diamond': 3,
#         'tennis-court': 4,
#         'basketball-court': 5,
#         'ground-track-field': 6,
#         'harbor': 7,
#         'bridge': 8,
#         'large-vehicle': 9,
#         'small-vehicle': 10,
#         'helicopter': 11,
#         'roundabout': 12,
#         'soccer-ball-field': 13,
#         'swimming-pool': 14,
#     }
    
#     convert_dota_to_yolo(dota_label_dir, yolo_label_dir, image_dir, class_map)

# if __name__ == "__main__":
#     main()



#ori_for_hbbox
# import os
# import numpy as np
# from tqdm import tqdm

# class_dict = {"plane": 0,
#               "ship": 1,
#               "storage-tank": 2,
#               "baseball-diamond": 3,
#               "tennis-court": 4,
#             "basketball-court": 5,
#             "ground-track-field": 6,
#             "harbor": 7,
#             "bridge": 8,
#             "large-vehicle": 9,
#             "small-vehicle": 10,
#             "helicopter": 11,
#             "roundabout": 12,
#             "soccer-ball-field": 13,
#             "swimming-pool": 14}

# def poly2hbb(polys):
#     """Convert polygons to horizontal bboxes.

#     Args:
#         polys (np.array): Polygons with shape (N:,
#          8)

#     Returns:
#         np.array: Horizontal bboxes.
#     """
#     shape = polys.shape
#     polys = polys.reshape(*shape[:-1], shape[-1] // 2, 2)
#     lt_point = np.min(polys, axis=-2)
#     rb_point = np.max(polys, axis=-2)
#     return np.concatenate([lt_point, rb_point], axis=-1)

# def _load_dota_txt(txtfile):
#     """Load DOTA's txt annotation.

#     Args:
#         txtfile (str): Filename of single txt annotation.

#     Returns:
#         dict: Annotation of single image.
#     """
#     gsd, bboxes, labels, diffs = None, [], [], []
#     if txtfile is None:
#         pass
#     elif not os.path.isfile(txtfile):
#         print(f"Can't find {txtfile}, treated as empty txtfile")
#     else:
#         with open(txtfile, 'r') as f:
#             for line in f:
#                 if line.startswith('gsd'):
#                     num = line.split(':')[-1]
#                     try:
#                         gsd = float(num)
#                     except ValueError:
#                         gsd = None
#                     continue

#                 items = line.split(' ')
#                 if len(items) >= 9:
#                     bboxes.append([float(i) for i in items[:8]])
#                     labels.append(items[8])
#                     diffs.append(int(items[9]) if len(items) == 10 else 0)

#     bboxes = np.array(bboxes, dtype=np.float32) if bboxes else \
#         np.zeros((0, 8), dtype=np.float32)
#     bboxes = poly2hbb(bboxes).tolist()
#     diffs = np.array(diffs, dtype=np.int64) if diffs else \
#         np.zeros((0,), dtype=np.int64)
#     ann = dict(bboxes=bboxes, labels=labels, filename=os.path.split(txtfile)[-1])
#     return ann



# anno_path = '/home/users/xianbao01.hou/CC-Diff/dataset/filter_val/labelTxt'
# annot_file_list = sorted([os.path.join(anno_path, i) for i in os.listdir(anno_path)])

# def convert_dict_to_yolo(data_dict: dict):
#     """
#     A function to convert the extracted data dict into a text file as per the YOLO format.
#     The final text file is saved in the directory "dior_data/yolo_annotations/data_dict['filename'].txt".
    
#     Parameters: data_dict: dict, A dict containing the data.
#     """
#     data = []
    
#     # Reading the bounding box data
#     for label, bbox in zip(data_dict['labels'], data_dict['bboxes']):
#         try:
#             class_id = class_dict[label]
#         except KeyError:
#             print(f'Invalid Class. Object class: "{label}" not present in the class list.')
            
#         # Transforming the bbox in Yolo format [X, Y, W, H]
#         img_w, img_h, = 512, 512 # Normalizing the bbox using image size
#         xmin, ymin, xmax, ymax = bbox
#         x_center = ((xmin + xmax) / 2) / img_w
#         y_center = ((ymin + ymax) / 2) / img_h
#         width = (xmax - xmin) / img_w 
#         height = (ymax - ymin) / img_h
        
#         # Writing the new data to the data list in Yolo format
#         data.append(f'{class_id} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}')
        
#     # File name for saving the text file(same as xml and jpg file name)
#     yolo_annot_dir = '/home/users/xianbao01.hou/CC-Diff/dataset/YOLO_DOTA/labels/val_test'
#     if not os.path.exists(yolo_annot_dir):
#         os.makedirs(yolo_annot_dir)
#     save_file_name = os.path.join(yolo_annot_dir, data_dict['filename'])
    
#     # Saving the yolo annotation in a text file
#     f = open(save_file_name, 'w+')
#     f.write('\n'.join(data))
#     f.close()
    
# print('[INFO] Annotation extraction and creation into Yolo has started.')
# for annot_file in tqdm(annot_file_list):
#     data_dict = _load_dota_txt(annot_file)
#     convert_dict_to_yolo(data_dict)
# print('[INFO] All the annotation are converted into Yolo format.')

#for obbox
import os
import numpy as np
from tqdm import tqdm

# 这里的类别字典保持不变，请确保与你的数据集实际类别一致
class_dict = {
    "plane": 0,
    "ship": 1,
    "storage-tank": 2,
    "baseball-diamond": 3,
    "tennis-court": 4,
    "basketball-court": 5,
    "ground-track-field": 6,
    "harbor": 7,
    "bridge": 8,
    "large-vehicle": 9,
    "small-vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer-ball-field": 13,
    "swimming-pool": 14
}

def _load_dota_txt(txtfile):
    """
    加载 DOTA 的 txt 标注。
    修改点：不再转换为水平框 (HBB)，直接返回原始的 8 点坐标。
    """
    gsd, bboxes, labels, diffs = None, [], [], []
    if txtfile is None:
        pass
    elif not os.path.isfile(txtfile):
        print(f"Can't find {txtfile}, treated as empty txtfile")
    else:
        with open(txtfile, 'r') as f:
            for line in f:
                if line.startswith('gsd'):
                    # 处理 GSD 信息，保持原有逻辑
                    num = line.split(':')[-1]
                    try:
                        gsd = float(num)
                    except ValueError:
                        gsd = None
                    continue

                items = line.split(' ')
                # DOTA 格式通常是: x1 y1 x2 y2 x3 y3 x4 y4 classname difficulty
                if len(items) >= 9:
                    # 读取前8个作为坐标
                    bboxes.append([float(i) for i in items[:8]])
                    # 第9个是类别
                    labels.append(items[8])
                    # 只有当有第10个元素时才读取 difficulty
                    diffs.append(int(items[9]) if len(items) == 10 else 0)

    # 转换为 numpy array，但不调用 poly2hbb
    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else \
        np.zeros((0, 8), dtype=np.float32)
    
    # 直接转为 list 返回，保留 (N, 8) 的形状
    bboxes = bboxes.tolist()
    
    diffs = np.array(diffs, dtype=np.int64) if diffs else \
        np.zeros((0,), dtype=np.int64)
    
    ann = dict(bboxes=bboxes, labels=labels, filename=os.path.split(txtfile)[-1])
    return ann


# 请修改为你实际的输入路径
anno_path = '/data/vepfs/users/xianbao01.hou/CC-Diff/dataset/filter_train/labelTxt'
annot_file_list = sorted([os.path.join(anno_path, i) for i in os.listdir(anno_path)])

def convert_dict_to_yolo_obb(data_dict: dict):
    """
    将字典数据转换为 YOLO OBB 格式 (class + 8个归一化坐标)。
    保存路径示例: ".../labels/val_test/filename.txt"
    """
    data = []
    
    # 警告：这里硬编码了 512。
    # 如果你的 DOTA 图片经过了裁切（split），请确保裁切大小确实是 512x512。
    # 如果是原始 DOTA 大图，尺寸是不固定的，这里会导致坐标错误。
    img_w, img_h = 512, 512 

    for label, bbox in zip(data_dict['labels'], data_dict['bboxes']):
        try:
            class_id = class_dict[label]
        except KeyError:
            # 如果遇到字典里没有的类（比如 DOTA 中的 container-crane 等），可以选择跳过或报错
            print(f'Skipping invalid Class: "{label}"')
            continue
            
        # bbox 此时是 [x1, y1, x2, y2, x3, y3, x4, y4]
        # 我们需要分别归一化 x 和 y
        
        normalized_poly = []
        for i in range(0, 8, 2):
            # 偶数索引是 x，奇数索引是 y
            x = bbox[i]
            y = bbox[i+1]
            
            # 归一化并限制在 [0, 1] 之间 (可选，YOLOv8 建议归一化)
            norm_x = x / img_w
            norm_y = y / img_h
            
            # 某些裁剪策略可能导致坐标略微越界，YOLO通常能处理，但最好clip一下或者保持原值
            # 这里直接归一化
            normalized_poly.append(norm_x)
            normalized_poly.append(norm_y)
        
        # 格式化字符串: class_id x1 y1 x2 y2 x3 y3 x4 y4
        # 使用 .6f 保证旋转框的精度
        poly_str = ' '.join([f'{x:.6f}' for x in normalized_poly])
        data.append(f'{class_id} {poly_str}')
        
    # 保存路径配置
    yolo_annot_dir = '/data/vepfs/users/xianbao01.hou/CC-Diff/dataset/YOLO_DOTA/labels/train'
    if not os.path.exists(yolo_annot_dir):
        os.makedirs(yolo_annot_dir)
    
    save_file_name = os.path.join(yolo_annot_dir, data_dict['filename'])
    
    with open(save_file_name, 'w+') as f:
        f.write('\n'.join(data))

print('[INFO] Annotation extraction and creation into YOLO OBB format has started.')
for annot_file in tqdm(annot_file_list):
    data_dict = _load_dota_txt(annot_file)
    convert_dict_to_yolo_obb(data_dict)
print('[INFO] All annotations are converted into YOLO OBB format.')
