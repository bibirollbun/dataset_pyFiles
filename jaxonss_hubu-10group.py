!pip install --upgrade pip --no-cache-dir
!pip install ultralytics --no-cache-dir

import os
import pandas as pd
import json
import shutil
import yaml
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# 配置参数 - 修改为Kaggle路径
BASE_DATA_DIR = '/kaggle/input/cowboyoutfits'  # 输入数据集根目录
TEST_CSV_PATH = os.path.join(BASE_DATA_DIR, 'test.csv')  # 测试集CSV
IMAGE_SRC_DIR = os.path.join(BASE_DATA_DIR, 'images')  # 图片目录
OUTPUT_DIR = '/kaggle/working'  # 输出目录

# 目标检测类别ID映射
CATEGORY_IDS = [87, 1034, 131, 318, 588]  # belt, sunglasses, boot, cowboy_hat, jacket
# 新增：类别名称到ID的映射字典（用于生成JSON）
CATEGORY_NAME_TO_ID = {
    'belt': 87,
    'sunglasses': 1034,
    'boot': 131,
    'cowboy_hat': 318,
    'jacket': 588
}
class_names = ['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket']  # 类别名称列表


# 数据准备函数
def prepare_data():
    # 加载训练标注
    train_json_path = os.path.join(BASE_DATA_DIR, 'train.json')
    with open(train_json_path, 'r') as f:
        train_info = json.load(f)
    temp_an = train_info["annotations"]
    all_images_info = train_info["images"]
    # 统计信息计算
    total_annotations = len(temp_an)
    totle_id = set(each["image_id"] for each in temp_an)
    print(f"[统计] 总标注数量: {total_annotations}")
    print(f"[统计] 包含目标类别的图片数: {len(totle_id)}")
    # 按类别统计图片数
    category = {i: [] for i in CATEGORY_IDS}
    for ann in temp_an:
        for cid in CATEGORY_IDS:
            if ann["category_id"] == cid:
                category[cid].append(ann["image_id"])
                break
    for cid in CATEGORY_IDS:
        print(f"[统计] 类别 {cid} 的样本数: {len(category[cid])}")
    # 找出有标注的图片文件名
    annotated_image_names = {img['file_name'] for img in all_images_info if img['id'] in totle_id}
    # 划分训练集和验证集
    train_image_names, valid_image_names = train_test_split(
        list(annotated_image_names), test_size=0.2, random_state=42
    )

    # 复制图片函数
    def copy_images(img_names, source_dir, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        copied = 0
        missing = 0
        for img_name in img_names:
            src = os.path.join(source_dir, img_name)
            dst = os.path.join(target_dir, img_name)
            if os.path.exists(src):
                shutil.copy(src, dst)
                copied += 1
            else:
                print(f"[警告] 图片不存在: {src}")
                missing += 1
        print(f"[复制统计] 成功: {copied}, 缺失: {missing}")
        return copied

    # 复制训练集和验证集图片
    train_copied = copy_images(
        train_image_names, IMAGE_SRC_DIR, os.path.join(OUTPUT_DIR, "data/train/images")
    )
    val_copied = copy_images(
        valid_image_names, IMAGE_SRC_DIR, os.path.join(OUTPUT_DIR, "data/val/images")
    )
    print(f"[划分] 训练集图片数: {train_copied}, 验证集图片数: {val_copied}")

    # 生成标签函数
    def generate_labels(annotations, img_names, img_info_list, label_dir):
        os.makedirs(label_dir, exist_ok=True)
        trans = {cid: idx for idx, cid in enumerate(CATEGORY_IDS)}
        name_to_info = {img['file_name']: img for img in img_info_list}
        generated_count = 0
        for img_name in img_names:
            if img_name in name_to_info:
                img_info = name_to_info[img_name]
                img_id = img_info['id']
                width = img_info['width']
                height = img_info['height']
                label_path = os.path.join(label_dir, f"{os.path.splitext(img_name)[0]}.txt")
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        if ann['image_id'] == img_id and ann['category_id'] in CATEGORY_IDS:
                            yolo_class_id = trans[ann['category_id']]
                            x_center = (ann['bbox'][0] + ann['bbox'][2] / 2) / width
                            y_center = (ann['bbox'][1] + ann['bbox'][3] / 2) / height
                            w = ann['bbox'][2] / width
                            h = ann['bbox'][3] / height
                            f.write(f"{yolo_class_id} {x_center:.4f} {y_center:.4f} {w:.4f} {h:.4f}\n")
                generated_count += 1
        print(f"[标签生成] 生成了 {generated_count} 个标签文件")

    # 生成训练集和验证集标签
    generate_labels(temp_an, train_image_names, all_images_info,
                    os.path.join(OUTPUT_DIR, 'data/train/labels'))
    generate_labels(temp_an, valid_image_names, all_images_info,
                    os.path.join(OUTPUT_DIR, 'data/val/labels'))
    return len(train_image_names), len(valid_image_names)


# 主程序
if __name__ == '__main__':
    # 数据准备与模型训练（保留原有逻辑）
    train_count, val_count = prepare_data()
    # 生成data.yaml
    data_yaml = {
        'train': os.path.abspath(os.path.join(OUTPUT_DIR, 'data/train/images')),
        'val': os.path.abspath(os.path.join(OUTPUT_DIR, 'data/val/images')),
        'nc': len(CATEGORY_IDS),
        'names': class_names
    }
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"[YAML] 数据集配置文件已生成")
    # 模型训练
    os.environ["YOLO_DOWNLOAD_FONT"] = "0"
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 需替换为实际训练好的模型
    model.train(
        data=os.path.join(OUTPUT_DIR, 'data.yaml'),
        epochs=10, imgsz=768, batch=12, lr0=0.004, lrf=0.01, device=0
    )

    # 重点：测试集处理部分（生成JSON）
    if os.path.exists(TEST_CSV_PATH):
        test_df = pd.read_csv(TEST_CSV_PATH)
        test_images = [os.path.join(IMAGE_SRC_DIR, fname) for fname in test_df['file_name']]
        test_images_dir = os.path.join(OUTPUT_DIR, 'data/test/images')
        os.makedirs(test_images_dir, exist_ok=True)

        # 复制测试集图片
        for src in test_images:
            if os.path.exists(src):
                shutil.copy(src, test_images_dir)

        # 模型预测
        results = model.predict(source=test_images_dir, save=False, save_txt=False)

        # 生成JSON结果
        kaggle_results = []
        # 建立文件名到image_id的映射
        filename_to_id = {row['file_name']: int(row['id']) for _, row in test_df.iterrows()}

        for result in results:
            img_path = result.path
            img_name = os.path.basename(img_path)
            image_id = filename_to_id.get(img_name)
            if not image_id:
                continue  # 跳过无image_id的图片

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue  # 跳过无检测结果的图片

            # 处理每个检测框
            for cls, conf, box in zip(boxes.cls, boxes.conf, boxes.xyxy):
                class_idx = int(cls)
                if 0 <= class_idx < len(class_names):
                    category_name = class_names[class_idx]
                    category_id = CATEGORY_NAME_TO_ID[category_name]
                    confidence = float(conf)

                    # 转换边界框格式：[x1, y1, x2, y2] -> [x1, y1, width, height]
                    x1, y1, x2, y2 = box.tolist()
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                    # 构造JSON记录（保留6位小数）
                    prediction = {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [float(f"{coord:.6f}") for coord in bbox],
                        "score": float(f"{confidence:.6f}")
                    }
                    kaggle_results.append(prediction)

        # 保存JSON文件
        json_path = os.path.join(OUTPUT_DIR, 'predictions.json')
        with open(json_path, 'w') as f:
            json.dump(kaggle_results, f)

        print(f"[Kaggle] 预测结果已保存至: {json_path}")
        print(f"[统计] 共生成 {len(kaggle_results)} 条预测记录")
    else:
        print(f"[警告] 测试集CSV不存在: {TEST_CSV_PATH}")

