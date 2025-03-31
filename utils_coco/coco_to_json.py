import json
import os
# 读取文本文件并解析数据
txt_filename = "/home/zsc/ILISTO/fewshot/ITSDT/val_80.txt"
annotations = []
images = []
image_id = 1
annotation_id = 1

with open(txt_filename, "r") as txt_file:
    for line in txt_file:
        parts = line.strip().split(" ")
        image_path = parts[0]
        bboxes = parts[1:]
        annotations_for_image = []

        for bbox in bboxes:
            x_min, y_min, x_max, y_max, _ = map(int, bbox.split(","))
            width = x_max - x_min
            height = y_max - y_min
            annotation = {
                "segmentation": [[]],
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "category_id": 1,  # 假设只有一个类别，可以根据实际情况修改
                "id": annotation_id,
            }
            annotations_for_image.append(annotation)
            annotation_id += 1

        image_filename = os.path.basename(image_path)
        image_info = {
            "id": image_id,
            "file_name": image_path,  # 使用完整的图像路径
            "height": 480,  # 根据实际图像高度修改
            "width": 640,   # 根据实际图像宽度修改
        }
        images.append(image_info)
        annotations.extend(annotations_for_image)
        image_id += 1

# 构建COCO数据集结构
coco_data = {
    "images": images,
    "annotations": annotations,
}

# 创建文件夹和JSON文件
output_folder = "/home/zsc/"
os.makedirs(output_folder, exist_ok=True)
json_filename = os.path.join(output_folder, "val_80.json")

# 将COCO数据写入JSON文件
with open(json_filename, "w") as json_file:
    json.dump(coco_data, json_file, indent=4)

print(f"转换完成，已将{txt_filename}转换为{json_filename}")