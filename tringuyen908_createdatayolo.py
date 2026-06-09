#!/usr/bin/env python3
"""
Improved Kaggle Data Processor for YOLO Training
Processes ImageNet + COCO datasets and creates downloadable ZIP
"""

import pandas as pd
import shutil
import os
from PIL import Image
from pycocotools.coco import COCO
import random
import zipfile
from pathlib import Path
import json
import time

# Ä�Æ°á»�ng dáº«n dataset (Kaggle paths)
IMAGENET_DIR = '/kaggle/input/imagenet-object-localization-challenge'
COCO_DIR = '/kaggle/input/coco-2017-dataset/coco2017'
OUTPUT_DIR = '/kaggle/working/yolo_dataset'

# Mapping classes cho YOLO
IMAGENET_CLASSES = {
    'n02992529': 1,  # mobile phone -> phone
    'n04069434': 2,  # reflex camera
    'n03976467': 3   # Polaroid camera
}

COCO_CLASSES = {
    1: 0,   # person (COCO ID: 1) -> YOLO ID: 0
    77: 1   # cell phone (COCO ID: 77) -> YOLO ID: 1
}

CLASS_NAMES = ['person', 'phone', 'reflex_camera', 'polaroid_camera']

def setup_directories():
    """Táº¡o cáº¥u trÃºc thÆ° má»¥c"""
    print("ğŸ“� Setting up directories...")
    
    dirs = [
        Path(OUTPUT_DIR) / 'images' / 'train',
        Path(OUTPUT_DIR) / 'images' / 'val',
        Path(OUTPUT_DIR) / 'labels' / 'train', 
        Path(OUTPUT_DIR) / 'labels' / 'val',
        Path(OUTPUT_DIR) / 'temp'
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {dir_path}")
    
    return True

def debug_imagenet_format():
    """Debug format cá»§a ImageNet annotations"""
    print("ğŸ”� Debugging ImageNet annotation format...")
    
    try:
        annotations_file = Path(IMAGENET_DIR) / 'LOC_train_solution.csv'
        annotations = pd.read_csv(annotations_file)
        
        print(f"ğŸ“Š Total annotations: {len(annotations)}")
        print(f"ğŸ“‹ Columns: {list(annotations.columns)}")
        
        # Xem sample predictions
        print("\nğŸ“� Sample PredictionStrings:")
        for i in range(min(10, len(annotations))):
            pred_str = annotations.iloc[i]['PredictionString']
            pred_parts = pred_str.split()
            print(f"   {i+1}: {pred_str}")
            print(f"      Parts count: {len(pred_parts)}")
            print(f"      Parts: {pred_parts[:8]}...")  # First 8 parts
        
        # TÃ¬m pattern vá»›i classes cáº§n thiáº¿t
        desired_classes = list(IMAGENET_CLASSES.keys())
        for class_id in desired_classes:
            matching = annotations[annotations['PredictionString'].str.contains(class_id, na=False)]
            if len(matching) > 0:
                sample = matching.iloc[0]['PredictionString']
                parts = sample.split()
                print(f"\nğŸ�¯ Sample for {class_id}:")
                print(f"   Full string: {sample}")
                print(f"   Parts: {parts}")
                
                # TÃ¬m vá»‹ trÃ­ cá»§a class_id
                try:
                    idx = parts.index(class_id)
                    print(f"   Class at index: {idx}")
                    if idx + 5 < len(parts):
                        print(f"   Next 5 values: {parts[idx+1:idx+6]}")
                except ValueError:
                    print(f"   Class not found in parts")
                break
        
        return True
        
    except Exception as e:
        print(f"â�Œ Debug failed: {e}")
        return False

def process_imagenet():
    """Xá»­ lÃ½ ImageNet dataset vá»›i flexible parsing"""
    print("\nğŸ–¼ï¸�  Processing ImageNet dataset...")
    
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    processed_images = []
    
    try:
        # Debug format trÆ°á»›c
        if not debug_imagenet_format():
            return [], class_counts
        
        # Load annotations
        annotations_file = Path(IMAGENET_DIR) / 'LOC_train_solution.csv'
        annotations = pd.read_csv(annotations_file)
        
        # Lá»�c annotations 
        desired_classes = list(IMAGENET_CLASSES.keys())
        pattern = '|'.join(desired_classes)
        filtered_annotations = annotations[
            annotations['PredictionString'].str.contains(pattern, na=False)
        ]
        
        print(f"ğŸ“Š Found {len(filtered_annotations)} relevant annotations")
        
        processed_count = 0
        for idx, row in filtered_annotations.iterrows():
            if processed_count % 100 == 0:
                print(f"   Processing: {processed_count}/{len(filtered_annotations)}")
            
            image_id = row['ImageId']
            predictions = row['PredictionString'].split()
            
            labels_for_image = []
            image_copied = False
            
            # Parse predictions vá»›i flexible format
            i = 0
            while i < len(predictions):
                if predictions[i] in desired_classes:
                    try:
                        class_id = predictions[i]
                        
                        # Determine format dá»±a trÃªn sá»‘ values cÃ²n láº¡i
                        remaining = len(predictions) - i - 1
                        
                        if remaining >= 5:
                            # Format: class_id confidence xmin ymin xmax ymax
                            confidence = float(predictions[i+1])
                            coords = list(map(float, predictions[i+2:i+6]))
                            i += 6
                        elif remaining >= 4:
                            # Format: class_id xmin ymin xmax ymax (no confidence)
                            coords = list(map(float, predictions[i+1:i+5]))
                            confidence = 0.9  # Default confidence
                            i += 5
                        elif remaining >= 4:
                            # Format: class_id x_center y_center width height
                            coords = list(map(float, predictions[i+1:i+5]))
                            confidence = 0.9
                            i += 5
                        else:
                            print(f"   âš ï¸�  Not enough values for bbox: {remaining}")
                            i += 1
                            continue
                        
                        # Handle different coordinate formats
                        if len(coords) == 4:
                            # Check if coords are normalized (0-1) or absolute
                            if all(c <= 1.0 for c in coords):
                                # Normalized format (x_center, y_center, width, height)
                                x_center, y_center, width_norm, height_norm = coords
                            else:
                                # Absolute format (xmin, ymin, xmax, ymax)
                                xmin, ymin, xmax, ymax = coords
                                
                                # Cáº§n kÃ­ch thÆ°á»›c image Ä‘á»ƒ normalize
                                image_path = Path(IMAGENET_DIR) / 'ILSVRC/Data/CLS-LOC/train' / class_id / f'{image_id}.JPEG'
                                if not image_path.exists():
                                    continue
                                
                                with Image.open(image_path) as img:
                                    img_width, img_height = img.size
                                
                                x_center = (xmin + xmax) / 2 / img_width
                                y_center = (ymin + ymax) / 2 / img_height
                                width_norm = (xmax - xmin) / img_width
                                height_norm = (ymax - ymin) / img_height
                        else:
                            print(f"   âš ï¸�  Invalid coords length: {len(coords)}")
                            continue
                        
                        # Validate coordinates
                        if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 < width_norm <= 1 and 0 < height_norm <= 1):
                            
                            # Copy image if not done yet
                            if not image_copied:
                                image_path = Path(IMAGENET_DIR) / 'ILSVRC/Data/CLS-LOC/train' / class_id / f'{image_id}.JPEG'
                                if image_path.exists():
                                    output_image_path = Path(OUTPUT_DIR) / 'temp' / f'imagenet_{image_id}.jpg'
                                    shutil.copy(str(image_path), str(output_image_path))
                                    image_copied = True
                            
                            yolo_class_id = IMAGENET_CLASSES[class_id]
                            label = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                            labels_for_image.append(label)
                            class_counts[yolo_class_id] += 1
                        
                    except (ValueError, IndexError) as e:
                        print(f"   âš ï¸�  Error parsing prediction: {e}")
                        # Debug what we're trying to parse
                        if i < len(predictions):
                            debug_slice = predictions[i:min(i+8, len(predictions))]
                            print(f"      Trying to parse: {debug_slice}")
                        i += 1
                else:
                    i += 1
            
            # Save processed data
            if labels_for_image and image_copied:
                processed_images.append({
                    'image_file': f'imagenet_{image_id}.jpg',
                    'labels': labels_for_image,
                    'source': 'imagenet'
                })
            
            processed_count += 1
                
    except Exception as e:
        print(f"â�Œ Error processing ImageNet: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"âœ… ImageNet processed: {len(processed_images)} images")
    return processed_images, class_counts

def process_coco():
    """Xá»­ lÃ½ COCO dataset"""
    print("\nğŸ�·ï¸�  Processing COCO dataset...")
    
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    processed_images = []
    
    try:
        # Load COCO annotations
        ann_file = Path(COCO_DIR) / 'annotations/instances_train2017.json'
        if not ann_file.exists():
            print(f"â�Œ COCO annotations not found: {ann_file}")
            return [], class_counts
        
        print(f"ğŸ“‹ Loading COCO annotations...")
        coco = COCO(str(ann_file))
        
        # Láº¥y image IDs
        person_img_ids = coco.getImgIds(catIds=[1])  # person
        phone_img_ids = coco.getImgIds(catIds=[77])  # cell phone
        
        # Giá»›i háº¡n sá»‘ lÆ°á»£ng Ä‘á»ƒ cÃ¢n báº±ng
        max_person_images = 3000
        person_img_ids = random.sample(person_img_ids, min(len(person_img_ids), max_person_images))
        
        # Combine vÃ  remove duplicates
        all_img_ids = list(set(person_img_ids + phone_img_ids))
        random.shuffle(all_img_ids)
        
        print(f"ğŸ“Š Found {len(person_img_ids)} person images")
        print(f"ğŸ“Š Found {len(phone_img_ids)} phone images") 
        print(f"ğŸ“Š Total unique images: {len(all_img_ids)}")
        
        processed_count = 0
        for img_id in all_img_ids:
            if processed_count % 500 == 0:
                print(f"   Processing: {processed_count}/{len(all_img_ids)}")
            
            try:
                img_info = coco.loadImgs(img_id)[0]
                
                # Láº¥y annotations
                ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[1, 77])
                anns = coco.loadAnns(ann_ids)
                
                if not anns:
                    processed_count += 1
                    continue
                
                # Kiá»ƒm tra hÃ¬nh áº£nh tá»“n táº¡i
                image_path = Path(COCO_DIR) / 'train2017' / img_info['file_name']
                if not image_path.exists():
                    processed_count += 1
                    continue
                
                # Sao chÃ©p hÃ¬nh áº£nh
                output_image_path = Path(OUTPUT_DIR) / 'temp' / f'coco_{img_info["file_name"]}'
                shutil.copy(str(image_path), str(output_image_path))
                
                # Táº¡o labels
                labels_for_image = []
                for ann in anns:
                    if ann['category_id'] in COCO_CLASSES:
                        bbox = ann['bbox']  # [x, y, width, height]
                        x, y, w, h = bbox
                        
                        if w <= 0 or h <= 0:
                            continue
                        
                        # YOLO format conversion
                        x_center = (x + w/2) / img_info['width']
                        y_center = (y + h/2) / img_info['height']
                        width_norm = w / img_info['width']
                        height_norm = h / img_info['height']
                        
                        # Clamp values
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width_norm = max(0, min(1, width_norm))
                        height_norm = max(0, min(1, height_norm))
                        
                        yolo_class_id = COCO_CLASSES[ann['category_id']]
                        label = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                        labels_for_image.append(label)
                        class_counts[yolo_class_id] += 1
                
                if labels_for_image:
                    processed_images.append({
                        'image_file': f'coco_{img_info["file_name"]}',
                        'labels': labels_for_image,
                        'source': 'coco'
                    })
                
            except Exception as e:
                print(f"   âš ï¸�  Error processing image {img_id}: {e}")
            
            processed_count += 1
                
    except Exception as e:
        print(f"â�Œ Error processing COCO: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"âœ… COCO processed: {len(processed_images)} images")
    return processed_images, class_counts

def create_train_val_split(all_images, train_ratio=0.8):
    """Táº¡o train/val split"""
    print(f"\nğŸ“‚ Creating train/val split ({train_ratio:.0%} train)...")
    
    random.shuffle(all_images)
    split_idx = int(train_ratio * len(all_images))
    
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"ğŸ“Š Train: {len(train_images)} images")
    print(f"ğŸ“Š Val: {len(val_images)} images")
    
    return train_images, val_images

def save_dataset(train_images, val_images):
    """LÆ°u dataset vá»›i cáº¥u trÃºc YOLO"""
    print("\nğŸ’¾ Saving dataset...")
    
    # Táº¡o train set
    for img_data in train_images:
        # Copy image
        src_path = Path(OUTPUT_DIR) / 'temp' / img_data['image_file']
        dst_path = Path(OUTPUT_DIR) / 'images' / 'train' / img_data['image_file']
        shutil.move(str(src_path), str(dst_path))
        
        # Save labels
        label_file = dst_path.stem + '.txt'
        label_path = Path(OUTPUT_DIR) / 'labels' / 'train' / label_file
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(img_data['labels']) + '\n')
    
    # Táº¡o val set
    for img_data in val_images:
        # Copy image
        src_path = Path(OUTPUT_DIR) / 'temp' / img_data['image_file']
        dst_path = Path(OUTPUT_DIR) / 'images' / 'val' / img_data['image_file']
        shutil.move(str(src_path), str(dst_path))
        
        # Save labels
        label_file = dst_path.stem + '.txt'
        label_path = Path(OUTPUT_DIR) / 'labels' / 'val' / label_file
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(img_data['labels']) + '\n')
    
    # Clean temp directory
    shutil.rmtree(Path(OUTPUT_DIR) / 'temp')
    
    print("âœ… Dataset saved successfully")

def create_config_files():
    """Táº¡o cÃ¡c file cáº¥u hÃ¬nh cho YOLO"""
    print("\nâš™ï¸�  Creating configuration files...")
    
    # Táº¡o data.yaml
    yaml_content = f"""# YOLO Dataset Configuration
# Generated from ImageNet + COCO

train: images/train
val: images/val

nc: 4
names: {CLASS_NAMES}

# Augmentation settings
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
"""
    
    with open(Path(OUTPUT_DIR) / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    # Táº¡o README
    readme_content = f"""# YOLO Dataset - ImageNet + COCO

## Dataset Information
- **Classes**: {len(CLASS_NAMES)}
- **Class Names**: {', '.join(CLASS_NAMES)}
- **Sources**: ImageNet Object Localization Challenge + COCO 2017
- **Format**: YOLO (txt annotations)

## Directory Structure
```
yolo_dataset/
â”œâ”€â”€ data.yaml              # Dataset configuration
â”œâ”€â”€ images/
â”‚   â”œâ”€â”€ train/            # Training images  
â”‚   â””â”€â”€ val/              # Validation images
â”œâ”€â”€ labels/
â”‚   â”œâ”€â”€ train/            # Training labels (YOLO format)
â”‚   â””â”€â”€ val/              # Validation labels (YOLO format)
â””â”€â”€ README.md             # This file
```

## Usage
1. Extract this dataset
2. Update paths in data.yaml if needed
3. Train with: `yolo train data=data.yaml model=yolov8n.pt epochs=100`

## Class Mapping
- 0: person (from COCO)
- 1: phone (from COCO cell_phone + ImageNet mobile_phone)  
- 2: reflex_camera (from ImageNet)
- 3: polaroid_camera (from ImageNet)

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(Path(OUTPUT_DIR) / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print("âœ… Configuration files created")

def create_zip_archive():
    """Táº¡o file ZIP Ä‘á»ƒ download"""
    print("\nğŸ“¦ Creating ZIP archive...")
    
    zip_path = '/kaggle/working/yolo_dataset.zip'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, OUTPUT_DIR)
                zipf.write(file_path, arcname)
                
                if len(zipf.namelist()) % 1000 == 0:
                    print(f"   Added {len(zipf.namelist())} files to ZIP...")
    
    # Thá»‘ng kÃª ZIP
    zip_size = os.path.getsize(zip_path) / (1024*1024)  # MB
    print(f"âœ… ZIP created: {zip_path}")
    print(f"ğŸ“Š ZIP size: {zip_size:.1f} MB")
    
    return zip_path

def print_final_stats(total_counts):
    """In thá»‘ng kÃª cuá»‘i cÃ¹ng"""
    print("\n" + "="*50)
    print("ğŸ“Š FINAL DATASET STATISTICS")
    print("="*50)
    
    total_instances = sum(total_counts.values())
    
    for class_id, count in total_counts.items():
        percentage = count / total_instances * 100 if total_instances > 0 else 0
        print(f"{CLASS_NAMES[class_id]:>15}: {count:>6} instances ({percentage:.1f}%)")
    
    print(f"{'TOTAL':>15}: {total_instances:>6} instances")
    print("="*50)

def main():
    """Main processing function"""
    print("ğŸš€ KAGGLE YOLO DATASET PROCESSOR")
    print("="*50)
    
    start_time = time.time()
    
    try:
        # Setup
        setup_directories()
        
        # Process datasets
        imagenet_images, imagenet_counts = process_imagenet()
        coco_images, coco_counts = process_coco()
        
        # Combine results
        all_images = imagenet_images + coco_images
        total_counts = {k: imagenet_counts[k] + coco_counts[k] for k in imagenet_counts}
        
        if not all_images:
            print("â�Œ No images processed! Check dataset paths and availability.")
            return
        
        print(f"\nğŸ�¯ Total processed: {len(all_images)} images")
        
        # Create splits
        train_images, val_images = create_train_val_split(all_images)
        
        # Save dataset
        save_dataset(train_images, val_images)
        
        # Create config files  
        create_config_files()
        
        # Create ZIP
        zip_path = create_zip_archive()
        
        # Final statistics
        print_final_stats(total_counts)
        
        elapsed_time = time.time() - start_time
        print(f"\nâ�±ï¸�  Total processing time: {elapsed_time:.1f} seconds")
        
        print(f"\nğŸ�‰ SUCCESS! Dataset ready for download:")
        print(f"ğŸ“� ZIP file: {zip_path}")
        print(f"ğŸ’¾ Size: {os.path.getsize(zip_path)/(1024*1024):.1f} MB")
        
        print(f"\nğŸ’¡ Next steps:")
        print("1. Download the ZIP file from Kaggle")
        print("2. Extract and use with: yolo train data=data.yaml model=yolov8n.pt")
        
    except Exception as e:
        print(f"\nâ�Œ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()




