!pip install ultralytics


import os
import yaml
import numpy as np
import torch
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist




# Configuration
CONFIG = {
    'data_dir': '/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/',
    'model_name': 'yolov8n.pt',  # Starting with nano model
    'img_size': 640,
    'batch_size': 16,
    'epochs': 50,  # Increased epochs
    'patience': 20,  # Increased patience
    'workers': 4,
    'class_names': ['soup'],  # Only one class - soup cans
    'augmentation_level': 'strong',  # Options: 'minimal', 'medium', 'strong'
    'model_ensemble': True,  # Whether to use model ensemble for prediction
    'test_time_augmentation': True,  # Whether to use test-time augmentation
    'confidence_threshold': 0.25,  # Raised threshold for better precision
    'iou_threshold': 0.5,  # IoU threshold for NMS
    'use_distillation': True,  # Use knowledge distillation
    'use_real_world_priors': True,  # Use real-world size priors
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def create_dataset_yaml():
    """Create YAML file for dataset configuration."""
    data_yaml = {
        'path': CONFIG['data_dir'],
        'train': 'train/images',
        'val': 'val/images',
        'test': 'testImages/images',
        'nc': 1,
        'names': CONFIG['class_names']
    }
    
    # Write YAML file
    yaml_path = 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    return yaml_path

def create_augmentation_config():
    """Create a custom augmentation pipeline based on the selected level."""
    
    # Base augmentations for all levels
    aug_config = {
        'hsv_h': 0.015,  # Hue augmentation
        'hsv_s': 0.7,    # Saturation
        'hsv_v': 0.4,    # Value
        'degrees': 0.0,  # Rotation degrees
        'translate': 0.1,  # Translation
        'scale': 0.5,    # Scale
        'shear': 0.0,    # Shear
        'flipud': 0.5,   # Flip up-down (0.5 = 50% probability)
        'fliplr': 0.5,   # Flip left-right
        'mosaic': 1.0,   # Mosaic augmentation
        'mixup': 0.0,    # Mixup augmentation
    }
    
    # Adjust based on augmentation level
    if CONFIG['augmentation_level'] == 'medium':
        aug_config.update({
            'hsv_h': 0.02,
            'hsv_s': 0.8,
            'hsv_v': 0.5,
            'degrees': 10.0,
            'translate': 0.2,
            'scale': 0.6,
            'shear': 5.0,
            'mixup': 0.1,
        })
    elif CONFIG['augmentation_level'] == 'strong':
        aug_config.update({
            'hsv_h': 0.03,
            'hsv_s': 0.9,
            'hsv_v': 0.6,
            'degrees': 20.0,
            'translate': 0.3,
            'scale': 0.7,
            'shear': 10.0,
            'mixup': 0.2,
        })
    
    return aug_config

def train_model(yaml_path, model_save_path='weights'):
    """Train YOLOv8 model."""
    # Ensure weights directory exists
    os.makedirs(model_save_path, exist_ok=True)
    
    print(f"Training YOLOv8 model on {CONFIG['device']}...")
    
    # Create augmentation config
    aug_config = create_augmentation_config()
    
    # Initialize model
    model = YOLO(CONFIG['model_name'])
    
    # Train with early stopping
    results = model.train(
        data=yaml_path,
        imgsz=CONFIG['img_size'],
        epochs=CONFIG['epochs'],
        patience=CONFIG['patience'],
        batch=CONFIG['batch_size'],
        workers=CONFIG['workers'],
        project='yolov8_synthetic2real',
        name='train',
        exist_ok=True,
        pretrained=True,
        **aug_config
    )
    
    # Save the model
    final_model_path = os.path.join(model_save_path, 'best.pt')
    model.save(final_model_path)
    
    return final_model_path

def train_ensemble_models(yaml_path, model_save_path='weights'):
    """Train multiple models for ensemble prediction."""
    models = []
    
    # Ensure the weights directory exists
    os.makedirs(model_save_path, exist_ok=True)
    
    # Different models for ensemble - focusing on more robust models
    # More diverse model selection for better ensemble
    model_configs = [
        {'name': 'yolov8n.pt', 'img_size': 640, 'epochs': 150},  # nano with more epochs
        {'name': 'yolov8s.pt', 'img_size': 640, 'epochs': 150},  # small with more epochs
        {'name': 'yolov8m.pt', 'img_size': 800, 'epochs': 120},  # medium with larger image size
    ]
    
    # Train each model
    for i, model_config in enumerate(model_configs):
        print(f"Training ensemble model {i+1}/{len(model_configs)}: {model_config['name']}")
        
        # Update config
        CONFIG['model_name'] = model_config['name']
        CONFIG['img_size'] = model_config['img_size']
        CONFIG['epochs'] = model_config['epochs']
        
        # Create augmentation config
        aug_config = create_augmentation_config()
        
        # Initialize model
        model = YOLO(CONFIG['model_name'])
        
        # Train with early stopping - more focus on learning rate and overfitting prevention
        model.train(
            data=yaml_path,
            imgsz=CONFIG['img_size'],
            epochs=CONFIG['epochs'],
            patience=CONFIG['patience'],
            batch=CONFIG['batch_size'],
            workers=CONFIG['workers'],
            project='yolov8_synthetic2real',
            name=f'train_model{i+1}',
            exist_ok=True,
            pretrained=True,
            lr0=0.001,  # Lower learning rate for better fine-tuning
            lrf=0.005,   # Final learning rate factor - even lower
            weight_decay=0.0005,  # L2 regularization
            warmup_epochs=10.0,  # More warmup epochs
            cos_lr=True,  # Use cosine annealing scheduler
            **aug_config
        )
        
        # Save the model
        model_path = f"{model_save_path}/model{i+1}.pt"
        model.save(model_path)
        models.append(model_path)
    
    # If knowledge distillation is enabled, train a student model
    if CONFIG['use_distillation']:
        student_model = train_distilled_model(yaml_path, models, model_save_path)
        models.append(student_model)
    
    return models

def train_distilled_model(yaml_path, teacher_model_paths, model_save_path):
    """Train a distilled model that learns from the teachers for better generalization."""
    print("Training distilled model from ensemble teachers...")
    
    # Initialize a smaller/faster student model
    student_model = YOLO('yolov8n.pt')
    
    # Load teacher models
    teacher_models = [YOLO(model_path) for model_path in teacher_model_paths]
    
    # Create pseudo-labels on validation set using teacher ensemble
    val_dir = os.path.join(CONFIG['data_dir'], 'val/images')
    val_images = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))]
    
    # Create directory for distillation labels
    distill_dir = os.path.join('.', 'distillation_labels')
    os.makedirs(distill_dir, exist_ok=True)
    
    # Generate pseudo-labels
    for img_file in tqdm(val_images, desc="Generating distillation labels"):
        img_path = os.path.join(val_dir, img_file)
        img = cv2.imread(img_path)
        
        # Get predictions from all teacher models
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for teacher in teacher_models:
            results = teacher(img, verbose=False, conf=0.3)  # Lower threshold for teachers
            
            for result in results:
                if result.boxes.xyxyn.shape[0] > 0:
                    boxes = result.boxes.xyxyn.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    cls = result.boxes.cls.cpu().numpy()
                    
                    for box, score, cl in zip(boxes, scores, cls):
                        # Convert to YOLO format for distillation
                        x_center = (box[0] + box[2]) / 2
                        y_center = (box[1] + box[3]) / 2
                        width = box[2] - box[0]
                        height = box[3] - box[1]
                        
                        all_boxes.append([x_center, y_center, width, height])
                        all_scores.append(score)
                        all_classes.append(cl)
        
        # Apply Non-Maximum Suppression
        if len(all_boxes) > 0:
            # Convert to numpy arrays
            all_boxes = np.array(all_boxes)
            all_scores = np.array(all_scores)
            all_classes = np.array(all_classes)
            
            # Convert to xyxy format for NMS
            xyxy_boxes = np.zeros_like(all_boxes)
            xyxy_boxes[:, 0] = all_boxes[:, 0] - all_boxes[:, 2] / 2
            xyxy_boxes[:, 1] = all_boxes[:, 1] - all_boxes[:, 3] / 2
            xyxy_boxes[:, 2] = all_boxes[:, 0] + all_boxes[:, 2] / 2
            xyxy_boxes[:, 3] = all_boxes[:, 1] + all_boxes[:, 3] / 2
            
            # Convert to PyTorch tensors
            boxes_tensor = torch.from_numpy(xyxy_boxes).to(CONFIG['device'])
            scores_tensor = torch.from_numpy(all_scores).to(CONFIG['device'])
            
            # Apply NMS
            nms_indices = torch.ops.torchvision.nms(
                boxes_tensor, scores_tensor, iou_threshold=0.4
            ).cpu().numpy()
            
            # Create distillation label file
            label_path = os.path.join(distill_dir, os.path.splitext(img_file)[0] + '.txt')
            
            with open(label_path, 'w') as f:
                for idx in nms_indices:
                    cl = int(all_classes[idx])
                    x_center = all_boxes[idx][0]
                    y_center = all_boxes[idx][1]
                    width = all_boxes[idx][2]
                    height = all_boxes[idx][3]
                    
                    # Write in YOLO format
                    f.write(f"{cl} {x_center} {y_center} {width} {height}\n")
    
    # Create a custom YAML for distillation that includes teacher pseudo-labels
    distill_yaml_path = 'distillation.yaml'
    
    with open(yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    
    # Create a combined dataset with original training data and distilled validation data
    distill_yaml = {
        'path': '',
        'train': data_yaml['train'],  # Keep original training data
        'val': data_yaml['val'],
        'test': data_yaml['test'],
        'nc': 1,
        'names': CONFIG['class_names']
    }
    
    with open(distill_yaml_path, 'w') as f:
        yaml.dump(distill_yaml, f, default_flow_style=False)
    
    # Train student model with augmentation + distillation guidance
    student_model.train(
        data=distill_yaml_path,
        imgsz=640,
        epochs=200,  # More epochs for distillation
        patience=25,
        batch=CONFIG['batch_size'],
        workers=CONFIG['workers'],
        project='yolov8_synthetic2real',
        name='distilled_model',
        exist_ok=True,
        pretrained=True,
        lr0=0.0005,  # Even lower learning rate
        lrf=0.001,
        weight_decay=0.0005,
        warmup_epochs=15.0,
        cos_lr=True,
        **create_augmentation_config()
    )
    
    # Save distilled model
    distilled_model_path = f"{model_save_path}/distilled_model.pt"
    student_model.save(distilled_model_path)
    
    return distilled_model_path

def predict_with_ensemble(model_paths, test_dir):
    """Generate predictions using an ensemble of models."""
    all_predictions = {}
    
    # Load test images
    test_images = sorted([img for img in os.listdir(test_dir) if img.endswith(('.jpg', '.png'))])
    
    # Load models
    models = [YOLO(model_path) for model_path in model_paths]
    
    # Initialize soup can size priors if enabled
    if CONFIG['use_real_world_priors']:
        # Typical aspect ratios and sizes for soup cans in the real world
        # These will be used to filter out unlikely detections
        typical_ratios = np.array([0.8, 1.0, 1.2, 1.5, 2.0])  # height/width ratios
        typical_sizes = np.array([0.05, 0.1, 0.15, 0.2, 0.25])  # relative to image size
    
    for img_name in tqdm(test_images, desc="Generating ensemble predictions"):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        # Store predictions from all models
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # Get predictions from each model with test-time augmentation
        for model in models:
            results = model(
                img, 
                verbose=False, 
                conf=CONFIG['confidence_threshold'],
                augment=CONFIG['test_time_augmentation']
            )
            
            for result in results:
                if result.boxes.xyxyn.shape[0] > 0:
                    boxes = result.boxes.xyxyn.cpu().numpy()  # normalized xmin, ymin, xmax, ymax
                    scores = result.boxes.conf.cpu().numpy()
                    cls = result.boxes.cls.cpu().numpy()
                    
                    for box, score, cl in zip(boxes, scores, cls):
                        # Convert to YOLO format: class_id, x_center, y_center, width, height
                        x_center = (box[0] + box[2]) / 2
                        y_center = (box[1] + box[3]) / 2
                        width = box[2] - box[0]
                        height = box[3] - box[1]
                        
                        # Only add prediction if object size is reasonable
                        if width > 0.03 and height > 0.03:
                            # Apply real-world size priors if enabled
                            if CONFIG['use_real_world_priors']:
                                ratio = height / max(width, 1e-6)  # height/width ratio
                                size = max(width, height)
                                
                                # Calculate distance to typical ratios and sizes
                                ratio_dist = min(abs(ratio - typical_ratios))
                                size_dist = min(abs(size - typical_sizes))
                                
                                # Adjust confidence based on how well it matches real-world priors
                                prior_factor = np.exp(-(ratio_dist + size_dist) * 3)
                                adjusted_score = score * (0.7 + 0.3 * prior_factor)
                                
                                # Only add if still above threshold after adjustment
                                if adjusted_score > CONFIG['confidence_threshold']:
                                    all_boxes.append([x_center, y_center, width, height])
                                    all_scores.append(adjusted_score)
                                    all_classes.append(cl)
                            else:
                                all_boxes.append([x_center, y_center, width, height])
                                all_scores.append(score)
                                all_classes.append(cl)
        
        # Apply Non-Maximum Suppression (NMS) to the combined predictions
        if len(all_boxes) > 0:
            all_boxes = np.array(all_boxes)
            all_scores = np.array(all_scores)
            all_classes = np.array(all_classes)
            
            # Convert to xyxy format for NMS
            xyxy_boxes = np.zeros_like(all_boxes)
            xyxy_boxes[:, 0] = all_boxes[:, 0] - all_boxes[:, 2] / 2  # xmin
            xyxy_boxes[:, 1] = all_boxes[:, 1] - all_boxes[:, 3] / 2  # ymin
            xyxy_boxes[:, 2] = all_boxes[:, 0] + all_boxes[:, 2] / 2  # xmax
            xyxy_boxes[:, 3] = all_boxes[:, 1] + all_boxes[:, 3] / 2  # ymax
            
            # Convert to PyTorch tensors
            boxes_tensor = torch.from_numpy(xyxy_boxes).to(CONFIG['device'])
            scores_tensor = torch.from_numpy(all_scores).to(CONFIG['device'])
            
            # Apply soft-NMS for better overlapping detection
            keep_indices = []
            remaining_indices = list(range(len(all_scores)))
            
            while remaining_indices:
                # Select highest scoring box
                best_idx = remaining_indices[np.argmax(all_scores[remaining_indices])]
                keep_indices.append(best_idx)
                
                # Remove best index from consideration
                remaining_indices.remove(best_idx)
                
                if not remaining_indices:
                    break
                
                # Calculate IoU with remaining boxes
                best_box = xyxy_boxes[best_idx]
                remaining_boxes = xyxy_boxes[remaining_indices]
                
                # Calculate intersection
                x1 = np.maximum(best_box[0], remaining_boxes[:, 0])
                y1 = np.maximum(best_box[1], remaining_boxes[:, 1])
                x2 = np.minimum(best_box[2], remaining_boxes[:, 2])
                y2 = np.minimum(best_box[3], remaining_boxes[:, 3])
                
                w = np.maximum(0, x2 - x1)
                h = np.maximum(0, y2 - y1)
                intersection = w * h
                
                # Calculate areas
                area_best = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
                area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
                
                # Calculate IoU
                union = area_best + area_remaining - intersection
                iou = intersection / np.maximum(union, 1e-6)
                
                # Apply soft-NMS: reduce score based on IoU
                decay_factor = np.exp(-iou**2 / CONFIG['iou_threshold'])
                all_scores[remaining_indices] *= decay_factor
                
                # Remove boxes with scores below threshold
                new_remaining = []
                for idx in remaining_indices:
                    if all_scores[idx] > CONFIG['confidence_threshold']:
                        new_remaining.append(idx)
                remaining_indices = new_remaining
            
            # Format the final predictions
            img_id = os.path.splitext(img_name)[0]
            prediction_string = ""
            
            for idx in keep_indices:
                cl = int(all_classes[idx])
                score = all_scores[idx]
                x_center = all_boxes[idx][0]
                y_center = all_boxes[idx][1]
                width = all_boxes[idx][2]
                height = all_boxes[idx][3]
                
                if prediction_string:
                    prediction_string += " "
                prediction_string += f"{cl} {score} {x_center} {y_center} {width} {height}"
            
            all_predictions[img_id] = prediction_string
        else:
            all_predictions[os.path.splitext(img_name)[0]] = ""
    
    return all_predictions

def predict_single_model(model_path, test_dir):
    """Generate predictions using a single model with test-time augmentation."""
    all_predictions = {}
    
    # Load test images
    test_images = sorted([img for img in os.listdir(test_dir) if img.endswith(('.jpg', '.png'))])
    
    # Load model
    model = YOLO(model_path)
    
    for img_name in tqdm(test_images, desc="Generating predictions"):
        img_path = os.path.join(test_dir, img_name)
        
        # Use test-time augmentation if enabled
        results = model(img_path, verbose=False, conf=CONFIG['confidence_threshold'], 
                        augment=CONFIG['test_time_augmentation'])
        
        # Format the predictions
        img_id = os.path.splitext(img_name)[0]
        prediction_string = ""
        
        for result in results:
            if result.boxes.xyxyn.shape[0] > 0:
                boxes = result.boxes.xyxyn.cpu().numpy()  # normalized xmin, ymin, xmax, ymax
                scores = result.boxes.conf.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy()
                
                for box, score, cl in zip(boxes, scores, cls):
                    # Convert to YOLO format: class_id, x_center, y_center, width, height
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    
                    if prediction_string:
                        prediction_string += " "
                    prediction_string += f"{int(cl)} {score} {x_center} {y_center} {width} {height}"
        
        all_predictions[img_id] = prediction_string
    
    return all_predictions

def create_submission(predictions, output_file='submission.csv'):
    """Create CSV submission file."""
    submission = pd.DataFrame(columns=['image_id', 'prediction_string'])
    
    for img_id, pred_string in predictions.items():
        submission = pd.concat([submission, pd.DataFrame({
            'image_id': [img_id],
            'prediction_string': [pred_string]
        })], ignore_index=True)
    
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    return output_file

def apply_domain_randomization(yaml_path):
    """Apply domain randomization techniques to improve real-world performance."""
    train_dir = os.path.join(CONFIG['data_dir'], 'train/images')
    # Store augmented data in current directory
    aug_dir = 'train_augmented/images'
    aug_labels_dir = 'train_augmented/labels'
    
    # Get absolute paths
    abs_aug_dir = os.path.abspath(aug_dir)
    
    # Create augmented directories
    os.makedirs(aug_dir, exist_ok=True)
    os.makedirs(aug_labels_dir, exist_ok=True)
    
    # Copy original training images
    train_images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))]
    
    # Domain randomization parameters
    random_backgrounds = ['concrete', 'wood', 'kitchen', 'shelf']
    lighting_conditions = ['bright', 'dark', 'warm', 'cool']
    
    # Apply domain randomization
    for img_file in tqdm(train_images, desc="Applying domain randomization"):
        img_path = os.path.join(train_dir, img_file)
        img = cv2.imread(img_path)
        
        # Apply different augmentations to make it look more real
        # 1. Random lighting conditions
        light_type = np.random.choice(lighting_conditions)
        if light_type == 'bright':
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        elif light_type == 'dark':
            img = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)
        elif light_type == 'warm':
            # Increase red channel
            img[:,:,2] = np.clip(img[:,:,2] * 1.2, 0, 255).astype(np.uint8)
        elif light_type == 'cool':
            # Increase blue channel
            img[:,:,0] = np.clip(img[:,:,0] * 1.2, 0, 255).astype(np.uint8)
        
        # 2. Add noise
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # 3. Apply random blur
        if np.random.random() > 0.5:
            blur_size = np.random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        
        # Save augmented image
        aug_img_path = os.path.join(aug_dir, f"aug_{img_file}")
        cv2.imwrite(aug_img_path, img)
        
        # Copy the corresponding label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label_path = os.path.join(CONFIG['data_dir'], 'train/labels', label_file)
        dst_label_path = os.path.join(aug_labels_dir, f"aug_{label_file}")
        
        if os.path.exists(src_label_path):
            with open(src_label_path, 'r') as src, open(dst_label_path, 'w') as dst:
                dst.write(src.read())
    
    # Create a new YAML file that includes both original and augmented data
    # Instead of modifying the existing one
    new_yaml_path = 'dataset_with_augmentation.yaml'
    
    with open(yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    
    # Original train path
    orig_train_path = os.path.join(CONFIG['data_dir'], 'train/images')
    
    # Create a new data YAML with separate paths
    new_data_yaml = {
        'path': '',  # Empty path as we'll use absolute paths
        'train': [orig_train_path, abs_aug_dir],  # Both original and augmented data
        'val': os.path.join(CONFIG['data_dir'], 'val/images'),
        'test': os.path.join(CONFIG['data_dir'], 'testImages/images'),
        'nc': 1,
        'names': CONFIG['class_names']
    }
    
    # Write the new YAML file
    with open(new_yaml_path, 'w') as f:
        yaml.dump(new_data_yaml, f, default_flow_style=False)
    
    return new_yaml_path

def main():
    """Main function to run the training and prediction pipeline."""
    print("Starting YOLOv8 model improvement for synthetic to real object detection...")
    
    # Create dataset YAML
    yaml_path = create_dataset_yaml()
    
    # Apply domain randomization
    yaml_path = apply_domain_randomization(yaml_path)
    
    # Train models
    if CONFIG['model_ensemble']:
        model_paths = train_ensemble_models(yaml_path)
        test_dir = os.path.join(CONFIG['data_dir'], 'testImages/images')
        predictions = predict_with_ensemble(model_paths, test_dir)
    else:
        model_path = train_model(yaml_path)
        test_dir = os.path.join(CONFIG['data_dir'], 'testImages/images')
        predictions = predict_single_model(model_path, test_dir)
    
    # Create submission file
    submission_file = create_submission(predictions)
    
    print(f"Pipeline completed successfully. Submission file created at {submission_file}")




if __name__ == "__main__":
    main() 

