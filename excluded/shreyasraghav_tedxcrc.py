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




# Configuration
CONFIG = {
    'data_dir': '/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/',
    'model_name': 'yolov8n.pt',  # Starting with nano model
    'img_size': 640,
    'batch_size': 8,    # Smaller batch size for better generalization
    'epochs': 50,       # More moderate epochs
    'patience': 10,     # Early stopping patience
    'workers': 4,
    'class_names': ['soup'],  # Only one class - soup cans
    'augmentation_level': 'medium',  # More moderate augmentation
    'model_ensemble': True,  # Whether to use model ensemble for prediction
    'test_time_augmentation': True,  # Whether to use test-time augmentation
    'confidence_threshold': 0.2,  # More balanced confidence threshold
    'iou_threshold': 0.45,  # IoU threshold for NMS 
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
        lr0=0.001,  # Lower learning rate for better fine-tuning
        lrf=0.01,   # Learning rate final factor
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
    model_configs = [
        {'name': 'yolov8n.pt', 'img_size': 640},  # nano - reliable baseline
        {'name': 'yolov8s.pt', 'img_size': 640}   # small - more capacity
    ]
    
    # Train each model
    for i, model_config in enumerate(model_configs):
        print(f"Training ensemble model {i+1}/{len(model_configs)}: {model_config['name']}")
        
        # Update config
        CONFIG['model_name'] = model_config['name']
        CONFIG['img_size'] = model_config['img_size']
        
        # Create augmentation config
        aug_config = create_augmentation_config()
        
        # Initialize model
        model = YOLO(CONFIG['model_name'])
        
        # Train with reliable parameters for synthetic-to-real transfer
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
            lrf=0.01,   # Learning rate final factor
            weight_decay=0.0005,  # L2 regularization
            warmup_epochs=3.0,  # Warmup epochs
            **aug_config
        )
        
        # Save the model
        model_path = f"{model_save_path}/model{i+1}.pt"
        model.save(model_path)
        models.append(model_path)
    
    return models

def predict_with_ensemble(model_paths, test_dir):
    """Generate predictions using an ensemble of models."""
    all_predictions = {}
    
    # Load test images
    test_images = sorted([img for img in os.listdir(test_dir) if img.endswith(('.jpg', '.png'))])
    
    # Load models
    models = [YOLO(model_path) for model_path in model_paths]
    
    # Process each image
    for img_name in tqdm(test_images, desc="Generating ensemble predictions"):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        
        # Store predictions from all models
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # Get predictions from each model
        for i, model in enumerate(models):
            # Use test-time augmentation
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
                        
                        # Filter out too small detections
                        if width > 0.03 and height > 0.03:
                            # Weight ensemble models based on their size (larger models get higher weight)
                            model_weight = 1.0 if i == 0 else 1.2  # Give slight preference to the small model
                            weighted_score = score * model_weight
                            
                            all_boxes.append([x_center, y_center, width, height])
                            all_scores.append(weighted_score)
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
            
            # Apply standard NMS
            nms_indices = torch.ops.torchvision.nms(
                boxes_tensor, scores_tensor, iou_threshold=CONFIG['iou_threshold']
            ).cpu().numpy()
            
            # Format the final predictions
            img_id = os.path.splitext(img_name)[0]
            prediction_string = ""
            
            for idx in nms_indices:
                cl = int(all_classes[idx])
                # Normalizing score to the original range
                score = min(all_scores[idx], 1.0)  
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
                    
                    # Filter by size
                    if width > 0.03 and height > 0.03:
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
        
        # 2. Add noise - more moderate noise
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # 3. Apply random blur - more moderate blur
        if np.random.random() > 0.5:
            blur_size = np.random.choice([3, 5])
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

