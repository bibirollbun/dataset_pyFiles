import os
import numpy as np
import pandas as pd
from PIL import Image
import shutil
import time
import yaml
from pathlib import Path
from tqdm.notebook import tqdm  # Use tqdm.notebook for Jupyter/Kaggle environments
import glob

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# Set random seed for reproducibility
np.random.seed(42)


# Define Kaggle paths
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
train_dir = os.path.join(data_path, "train")

# Define YOLO dataset structure
yolo_dataset_dir = "/kaggle/working/yolo_dataset"
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# Create directories
for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)


# check submission style
submission_path = data_path+'sample_submission.csv'
submission_df = pd.read_csv(submission_path)
submission_df


# check submission style
train_labels_path = data_path+'train_labels.csv'
train_labels = pd.read_csv(train_labels_path)
train_labels.head(10)


def show_train_image(image_subdir,start_idx=0):
    image_pattern = '*.jpg'
    # グリッドの列数 (1行に何枚の画像を表示するか)
    # 必要に応じて調整してください
    cols = 5
    
    # 表示する画像の最大数 (多すぎると表示に時間がかかる場合やメモリを消費する場合があるため)
    # None にすると見つかった画像を全て表示
    max_images_to_display = 25
    
    # --- 処理 ---
    # 画像ファイルのパスリストを取得
    search_path = f"{train_dir}/{image_subdir}/{image_pattern}"
    image_paths = glob.glob(search_path)
    image_paths = sorted(image_paths)

    print(f"{len(image_paths)=}")
    
    # 取得したパスの数を制限
    if max_images_to_display is not None:
        image_paths = image_paths[start_idx:start_idx+max_images_to_display]
    
    num_images = len(image_paths)
    
    if num_images == 0:
        print(f"not found path: '{search_path}'")
    else:
        print(f"show {num_images} images")
        
        # 行数を計算
        rows = math.ceil(num_images / cols)
    
        # FigureとAxes（サブプロット）を作成
        # figsize は表示ウィンドウのサイズを調整します (幅, 高さ)
        # 画像数や cols の値に合わせて適宜調整してください
        plt.figure(figsize=(cols * 2.5, rows * 2.5)) # 1枚あたり約2.5x2.5インチとして計算
    
        # 各画像をサブプロットに表示
        for i, img_path in enumerate(image_paths):
            # サブプロットを指定 (行数, 列数, 現在のインデックス+1)
            plt.subplot(rows, cols, i + 1)
    
            try:
                # 画像を読み込み
                img = Image.open(img_path).convert('RGB')
    
                # 画像を表示
                plt.imshow(img)
    
                # タイトルを追加 (オプション)
                # plt.title(f"Image {i+1}") # シンプルな番号
                # ファイル名だけを表示したい場合 (パスから抽出)
                # import os
                plt.title(img_path[img_path.rfind('/')+1:], fontsize=8)
    
                # 軸の目盛りやラベルを非表示にして画像を大きく表示
                plt.axis('off')
    
            except Exception as e:
                print(f"エラー: {img_path} の読み込みまたは表示に失敗しました - {e}")
                # エラーが発生した場所にテキストで表示することも可能
                # plt.text(0.5, 0.5, 'Error', horizontalalignment='center', verticalalignment='center')
                plt.axis('off') # エラー表示でも軸はオフに
    
        # サブプロット間のスペースを調整して、タイトルなどが重ならないようにする
        plt.tight_layout()
    
        # ウィンドウを表示
        plt.show()


show_train_image('tomo_003acc')


show_train_image('tomo_00e047', start_idx=160)


def show_motor(img_path: str, x: float, y: float) -> None:
    """
    指定された画像を表示し、特定の座標を中心とした10pxの範囲を緑色の線で囲みます。

    Args:
        img_path (str): 表示する画像のパス。
        x (float): 囲む領域の中心となるx座標。
        y (float): 囲む領域の中心となるy座標。
    """

    plt.figure(figsize=(10, 10)) # 1枚あたり約2.5x2.5インチとして計算

    # 画像を読み込み
    img = Image.open(img_path).convert('RGB')

    # 画像を表示
    plt.imshow(img)

    # タイトルを追加 (オプション) - 元のコードの記述を維持
    # import os # 必要ならここでインポートするか関数の外でインポート
    plt.title(img_path[img_path.rfind('/')+1:], fontsize=8) # ファイル名だけを表示

    # 軸の目盛りやラベルを非表示にして画像を大きく表示
    plt.axis('off')

    # 緑色の矩形を描画
    # 中心 (x, y) からの10pxの範囲なので、左下は (x-10, y-10)、幅と高さは 20
    # Rectangle((x, y), width, height, ...) の (x, y) は左下隅の座標
    rect = patches.Rectangle(
        (x - 10, y - 10),  # 矩形の左下隅の座標
        20,                # 矩形の幅 (中心から左右に10pxずつなので合計20px)
        20,                # 矩形の高さ (中心から上下に10pxずつなので合計20px)
        linewidth=2,       # 線の太さ
        edgecolor='green', # 線の色を緑に設定
        facecolor='none'   # 塗りつぶしはなし
    )

    # 現在のaxes（描画領域）を取得し、矩形を追加
    ax = plt.gca()
    ax.add_patch(rect)

    # サブプロット間のスペースを調整して、タイトルなどが重ならないようにする
    plt.tight_layout()

    # ウィンドウを表示
    plt.show()

# 使用例 (画像のパス 'your_image.jpg' と座標 150, 200 を適切に変更してください)
# try:
#     show_motor('your_image.jpg', 150, 200)
# except FileNotFoundError:
#     print("エラー: 'your_image.jpg' が見つかりません。適切な画像パスを指定してください。")
# except Exception as e:
#     print(f"エラーが発生しました: {e}")


show_motor(f"{train_dir}/tomo_00e047/slice_0168.jpg",546.0,603.0)


show_motor(f"{train_dir}/tomo_00e047/slice_0167.jpg",546.0,603.0)


show_motor(f"{train_dir}/tomo_00e047/slice_0169.jpg",546.0,603.0)


show_motor(f"{train_dir}/tomo_00e047/slice_0168.jpg",603.0,546.0)


print("slice_0168.jpg-1(axis 0)")
show_motor(f"{train_dir}/tomo_00e047/slice_0167.jpg",603.0,546.0)


print("slice_0168.jpg-2(axis 0)")
show_motor(f"{train_dir}/tomo_00e047/slice_0166.jpg",603.0,546.0)


print("slice_0168.jpg-3(axis 0)")
show_motor(f"{train_dir}/tomo_00e047/slice_0165.jpg",603.0,546.0)


print("slice_0168.jpg-4(axis 0)")
show_motor(f"{train_dir}/tomo_00e047/slice_0164.jpg",603.0,546.0)


print("slice_0168.jpg-5(axis 0)")
show_motor(f"{train_dir}/tomo_00e047/slice_0164.jpg",603.0,546.0)


print("slice_0168.jpg+1(axis 0)")
show_motor(f"{train_dir}/tomo_00e047/slice_0169.jpg",603.0,546.0)


print("slice_0168.jpg+2(axis 0)")
show_motor(f"{train_dir}/tomo_00e047/slice_0170.jpg",603.0,546.0)


# Define constants
TRUST = 4  # Number of slices above and below center slice (total 2*TRUST + 1 slices)
SLICE_PERCENTILE = 2
BOX_SIZE = 24  # Bounding box size for annotations (in pixels)
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation


# Image processing functions
def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles
    """
    # Calculate percentiles
    p_small = np.percentile(slice_data, SLICE_PERCENTILE)
    p_large = np.percentile(slice_data, 100-SLICE_PERCENTILE)
    
    # Clip the data to the percentile range
    clipped_data = np.clip(slice_data, p_small, p_large)
    
    # Normalize to [0, 255] range
    normalized = 255 * (clipped_data - p_small) / (p_large - p_small)
    
    return np.uint8(normalized)

def prepare_yolo_dataset(trust=TRUST, train_split=TRAIN_SPLIT):
    """
    Extract slices containing motors from tomograms and save to YOLO structure with annotations
    """
    # Load the labels CSV
    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    
    # Count total number of motors
    total_motors = labels_df['Number of motors'].sum()
    print(f"Total number of motors in the dataset: {total_motors}")
    
    # Get unique tomograms that have motors
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].unique()
    
    print(f"Found {len(unique_tomos)} unique tomograms with motors")
    
    # Perform the train-val split at the tomogram level (not motor level)
    # This ensures all slices from a single tomogram go to either train or val
    np.random.shuffle(unique_tomos)  # Shuffle the tomograms
    split_idx = int(len(unique_tomos) * train_split)
    train_tomos = unique_tomos[:split_idx]
    val_tomos = unique_tomos[split_idx:]
    
    print(f"Split: {len(train_tomos)} tomograms for training, {len(val_tomos)} tomograms for validation")
    
    # Function to process a set of tomograms
    def process_tomogram_set(tomogram_ids, images_dir, labels_dir, set_name):
        motor_counts = []
        for tomo_id in tomogram_ids:
            # Get all motors for this tomogram
            tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                if pd.isna(motor['Motor axis 0']):
                    continue
                motor_counts.append(
                    (tomo_id, 
                     int(motor['Motor axis 0']), 
                     int(motor['Motor axis 1']), 
                     int(motor['Motor axis 2']),
                     int(motor['Array shape (axis 0)']))
                )
        
        print(f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")
        
        # Process each motor
        processed_slices = 0
        
        for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):
            # Calculate range of slices to include
            z_min = max(0, z_center - trust)
            z_max = min(z_max - 1, z_center + trust)
            
            # Process each slice in the range
            for z in range(z_min, z_max + 1):
                # Create slice filename
                slice_filename = f"slice_{z:04d}.jpg"
                
                # Source path for the slice
                src_path = os.path.join(train_dir, tomo_id, slice_filename)
                
                if not os.path.exists(src_path):
                    print(f"Warning: {src_path} does not exist, skipping.")
                    continue
                
                # Load and normalize the slice
                img = Image.open(src_path)
                img_array = np.array(img)
                
                # Normalize the image
                normalized_img = normalize_slice(img_array)
                
                # Create destination filename (with unique identifier)
                dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                
                # Save the normalized image
                Image.fromarray(normalized_img).save(dest_path)
                
                # Get image dimensions
                img_width, img_height = img.size
                
                # Create YOLO format label
                # YOLO format: <class> <x_center> <y_center> <width> <height>
                # Values are normalized to [0, 1]
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                box_width_norm = BOX_SIZE / img_width
                box_height_norm = BOX_SIZE / img_height
                
                # Write label file
                label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                
                processed_slices += 1
        
        return processed_slices, len(motor_counts)
    
    # Process training tomograms
    train_slices, train_motors = process_tomogram_set(train_tomos, yolo_images_train, yolo_labels_train, "training")
    
    # Process validation tomograms
    val_slices, val_motors = process_tomogram_set(val_tomos, yolo_images_val, yolo_labels_val, "validation")
    
    # Create YAML configuration file for YOLO
    yaml_content = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'motor'}
    }
    
    with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\nProcessing Summary:")
    print(f"- Train set: {len(train_tomos)} tomograms, {train_motors} motors, {train_slices} slices")
    print(f"- Validation set: {len(val_tomos)} tomograms, {val_motors} motors, {val_slices} slices")
    print(f"- Total: {len(train_tomos) + len(val_tomos)} tomograms, {train_motors + val_motors} motors, {train_slices + val_slices} slices")
    
    # Return summary info
    return {
        "dataset_dir": yolo_dataset_dir,
        "yaml_path": os.path.join(yolo_dataset_dir, 'dataset.yaml'),
        "train_tomograms": len(train_tomos),
        "val_tomograms": len(val_tomos),
        "train_motors": train_motors,
        "val_motors": val_motors,
        "train_slices": train_slices,
        "val_slices": val_slices
    }

# Run the preprocessing
summary = prepare_yolo_dataset(TRUST)
print(f"\nPreprocessing Complete:")
print(f"- Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
print(f"- Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
print(f"- Dataset directory: {summary['dataset_dir']}")
print(f"- YAML configuration: {summary['yaml_path']}")
print(f"\nReady for YOLO training!")


print(f"TRUST: {TRUST}")
print(f"SLICE_PERCENTILE: {SLICE_PERCENTILE}")
print(f"BOX_SIZE: {BOX_SIZE}")
print(f"TRAIN_SPLIT: {TRAIN_SPLIT}")

# Python側でf-stringを使って最終的なパス文字列を組み立てる
output_path = f"byu_yolo_dataset_{TRUST=}_{SLICE_PERCENTILE=}_{BOX_SIZE=}_{TRAIN_SPLIT=}"

# 組み立てた一つのPython変数をシェルコマンドで使う
# この場合、$output_path が Pythonの変数 output_path の値に置き換わります
!echo "$output_path".tar.gz



!tar -czvf "$output_path".tar.gz -C yolo_dataset .


# if you execute sequencely.

images_train_dir = "/kaggle/working/yolo_dataset/images/train/"
labels_train_dir = "/kaggle/working/yolo_dataset/labels/train/"

# # else
# images_train_dir = "/kaggle/input/parse-data/yolo_dataset/images/train/"
# labels_train_dir = "/kaggle/input/parse-data/yolo_dataset/labels/train/"

# BOX_SIZE = 24


import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import numpy as np
import glob

# Define base_dir - this was missing in the original code
# In Kaggle, we can use the working directory as base or remove it completely
# since we're using absolute paths
base_dir = "/kaggle/working"  # or simply use "" if using absolute paths


def visualize_random_training_samples(num_samples=4):
    """
    Visualize random training samples with YOLO annotations
    
    Args:
        num_samples (int): Number of random images to display
    """
    # Get all image files from the train directory
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(images_train_dir, "**", ext), recursive=True))
    
    # Make sure we have enough images
    if len(image_files) == 0:
        print("No image files found in the train directory!")
        return
        
    num_samples = min(num_samples, len(image_files))
    
    # Select random images
    random_images = random.sample(image_files, num_samples)
    
    # Create a figure with subplots
    rows = int(np.ceil(num_samples / 2))
    cols = min(num_samples, 2)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    
    # Handle the case of a single subplot
    if num_samples == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    
    # Process each selected image
    for i, img_path in enumerate(random_images):
        try:
            # Get corresponding label file
            # YOLO labels have same name but .txt extension instead of image extension
            relative_path = os.path.relpath(img_path, images_train_dir)
            label_path = os.path.join(labels_train_dir, os.path.splitext(relative_path)[0] + '.txt')
            
            # Load the image
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Normalize image using percentiles for better visualization
            img_array = np.array(img)
            p2 = np.percentile(img_array, 2)
            p98 = np.percentile(img_array, 98)
            normalized = np.clip(img_array, p2, p98)
            normalized = 255 * (normalized - p2) / (p98 - p2)
            img_normalized = Image.fromarray(np.uint8(normalized))
            
            # Convert image to RGB for colored box
            img_rgb = img_normalized.convert('RGB')
            
            # Create a transparent overlay
            overlay = Image.new('RGBA', img_rgb.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Load YOLO format annotations if they exist
            annotations = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        # YOLO format: class x_center y_center width height
                        # All values are normalized from 0 to 1
                        values = line.strip().split()
                        class_id = int(values[0])
                        x_center = float(values[1]) * img_width
                        y_center = float(values[2]) * img_height
                        width = float(values[3]) * img_width
                        height = float(values[4]) * img_height
                        
                        annotations.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
            
            # Draw all annotations
            for ann in annotations:
                x_center = ann['x_center']
                y_center = ann['y_center']
                width = ann['width']
                height = ann['height']
                
                # Calculate bounding box coordinates
                x1 = max(0, int(x_center - width/2))
                y1 = max(0, int(y_center - height/2))
                x2 = min(img_width, int(x_center + width/2))
                y2 = min(img_height, int(y_center + height/2))
                
                # Draw semi-transparent red rectangle
                draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 64), outline=(255, 0, 0, 200))
                
                # Draw label
                label_text = f"Class {ann['class_id']}"
                draw.text((x1, y1-10), label_text, fill=(255, 0, 0, 255))
            
            # If no annotations found, indicate this
            if not annotations:
                draw.text((10, 10), "No annotations found", fill=(255, 0, 0, 255))
            
            # Composite the overlay onto the original image
            img_rgb = Image.alpha_composite(img_rgb.convert('RGBA'), overlay).convert('RGB')
            
            # Display the image with annotations
            axes[i].imshow(np.array(img_rgb))
            img_name = os.path.basename(img_path)
            axes[i].set_title(f"Image: {img_name}\nAnnotations: {len(annotations)}")
            axes[i].axis('on')
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading image: {os.path.basename(img_path)}", 
                       horizontalalignment='center', verticalalignment='center')
            axes[i].axis('off')
    
    # Handle extra subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Displayed {num_samples} random images with YOLO annotations")

# Run the visualization
visualize_random_training_samples(4)













