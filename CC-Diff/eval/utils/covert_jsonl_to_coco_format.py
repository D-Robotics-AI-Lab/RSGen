import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

# 类别列表保持不变
CLASSES = ('vehicle','chimney','golffield','Expressway-toll-station','stadium',
           'groundtrackfield','windmill','trainstation','harbor','overpass',
           'baseballfield','tenniscourt','bridge','basketballcourt','airplane',
           'ship','storagetank','Expressway-Service-area','airport','dam')

def get_image_info(file_path):
    """辅助函数，用于在XML中没有尺寸信息时读取图片尺寸"""
    try:
        with Image.open(file_path) as img:
            width, height = img.size
        return width, height
    except FileNotFoundError:
        return None, None

def convert_specific_xml_to_coco(img_dir, xml_dir, output_json):
    """
    根据指定的图片文件夹，从总的XML标注文件夹中找到对应的标注，并转换为COCO格式。

    Args:
        img_dir (str): 目标图片集所在的目录 (例如 train/ 或 test/)。这是转换的基准。
        xml_dir (str): 存放所有XML标注文件的总目录。
        output_json (str): 输出的COCO JSON文件的保存路径。
    """
    annotations = []
    images_info = []
    
    # 初始化COCO数据结构
    coco_output = {
        "info": {
            "description": "Specific Dataset Conversion to COCO format",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": "2024-11-09"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 1. 以指定的图片目录为基准，获取所有图片文件名
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    image_filenames = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in supported_formats])
    
    if not image_filenames:
        print(f"错误：在目录 '{img_dir}' 中没有找到任何支持的图片文件。请检查路径。")
        return
        
    print(f"在 '{img_dir}' 目录中找到 {len(image_filenames)} 张图片，开始查找对应标注并转换...")

    image_id_counter = 1
    annotation_id_counter = 1

    # 2. 遍历图片列表，查找并处理对应的XML文件
    for filename in tqdm(image_filenames, desc="Processing images"):
        # 构建对应的XML文件名
        xml_filename = os.path.splitext(filename)[0] + '.xml'
        xml_path = os.path.join(xml_dir, xml_filename)
        
        # 检查对应的XML文件是否存在
        if not os.path.exists(xml_path):
            print(f"警告：图片 '{filename}' 对应的标注文件 '{xml_filename}' 不存在于 '{xml_dir}'，将跳过此图片。")
            continue
            
        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图片尺寸
        size_tag = root.find('size')
        if size_tag is not None:
            width = int(size_tag.find('width').text)
            height = int(size_tag.find('height').text)
        else:
            # 如果XML中没有尺寸信息，则从图片文件读取
            img_path = os.path.join(img_dir, filename)
            width, height = get_image_info(img_path)
            if width is None:
                print(f"警告：无法从XML和图片文件 '{filename}' 中获取尺寸信息，跳过。")
                continue
        
        # 添加图片信息 (只有当XML存在且有效时才添加)
        current_image_id = image_id_counter
        images_info.append({
            "id": current_image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })
        
        # 遍历XML中的所有<object>标签
        for obj in root.findall('object'):
            label = obj.find('name').text
            
            try:
                category_id = CLASSES.index(label) + 1
            except ValueError:
                print(f"警告：在文件 {xml_filename} 中发现未知类别 '{label}'，已跳过此标注。")
                continue
            
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            o_width = xmax - xmin
            o_height = ymax - ymin
            
            annotations.append({
                "id": annotation_id_counter,
                "image_id": current_image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "area": o_width * o_height,
                "segmentation": [],
                "iscrowd": 0
            })
            annotation_id_counter += 1
        
        image_id_counter += 1 # 只有成功处理一张图片后，ID才增加

    # 3. 生成categories字段
    categories = []
    for i, category in enumerate(CLASSES):
        categories.append({
            "id": i + 1,
            "name": category,
            "supercategory": "none"
        })
    
    coco_output["images"] = images_info
    coco_output["categories"] = categories
    coco_output["annotations"] = annotations

    # 写入JSON文件
    with open(output_json, 'w') as json_file:
        json.dump(coco_output, json_file, indent=4)
    
    print(f"\n转换完成！共处理了 {len(images_info)} 张图片。")
    print(f"COCO JSON 文件已保存至: {output_json}")

if __name__ == '__main__':
    
    # 1. 指定你要转换的图片集所在的目录 (例如，测试集图片目录)
    TARGET_IMAGE_DIR = '/home/users/xianbao01.hou/CC-Diff/dataset/DIOR_NOT_800/test'  

    # 2. 指定存放了 *全部* XML标注文件的总目录
    ALL_XML_ANNOTATIONS_DIR = '/home/users/xianbao01.hou/CC-Diff/dataset/DIOR_NOT_800/Annotations'

    # 3. 指定输出的COCO格式JSON文件的路径和名称
    OUTPUT_COCO_JSON = '/home/users/xianbao01.hou/CC-Diff/dataset/DIOR_NOT_800/test_annotations.json'
    
    # 调用转换函数
    convert_specific_xml_to_coco(TARGET_IMAGE_DIR, ALL_XML_ANNOTATIONS_DIR, OUTPUT_COCO_JSON)
