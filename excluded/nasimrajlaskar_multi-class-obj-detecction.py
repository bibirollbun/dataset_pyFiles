# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2


!pip install ultralytics -q



from ultralytics import YOLO
import yaml


print("ğŸ“� Available input data:")
for item in os.listdir('/kaggle/input/'):
    print(f"  - {item}")


starter_data_path = "/kaggle/input/multi-class-object-detection-challenge/Starter_Dataset"
test_data_path = "/kaggle/input/multi-class-object-detection-challenge/testImages/images"
yolo_config_path = "/kaggle/input/multi-class-object-detection-challenge/yolo_params.yaml"
print(f"\nğŸ“Š Data paths:")
print(f"  ğŸ�‹ï¸� Training data: {starter_data_path}")
print(f"  ğŸ�¯ Test images: {test_data_path}")
print(f"  âš™ï¸� YOLO config: {yolo_config_path}")


def show_image(image_path, title="", figsize=(10, 6)):
    """Display image inline in notebook"""
    if os.path.exists(image_path):
        plt.figure(figsize=figsize)
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        print(f"â�Œ Image not found: {image_path}")


def show_images_grid(image_paths, titles=None, figsize=(15, 10), cols=2):
    """Display multiple images in a grid"""
    if not image_paths:
        return


def show_images_grid(image_paths, titles=None, figsize=(15, 10), cols=2):
    """Display multiple images in a grid"""
    if not image_paths:
        return
    
    rows = (len(image_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, image_path in enumerate(image_paths):
        if os.path.exists(image_path):
            img = mpimg.imread(image_path)
            axes[i].imshow(img)
            axes[i].set_title(titles[i] if titles else os.path.basename(image_path))
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(image_paths), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_sample_data():
    """Show sample training and test images"""
    print("ğŸ–¼ï¸� Sample Training Images:")
    
    # Show sample training images
    train_img_path = os.path.join(starter_data_path, "train", "images")
    if os.path.exists(train_img_path):
        train_images = [f for f in os.listdir(train_img_path)[:4] 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        train_paths = [os.path.join(train_img_path, img) for img in train_images]
        show_images_grid(train_paths, titles=[f"Training: {img}" for img in train_images])
    
    print("ğŸ�¯ Sample Test Images:")
    
    # Show sample test images
    if os.path.exists(test_data_path):
        test_images = [f for f in os.listdir(test_data_path)[:4] 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        test_paths = [os.path.join(test_data_path, img) for img in test_images]
        show_images_grid(test_paths, titles=[f"Test: {img}" for img in test_images])

visualize_sample_data()



def create_working_yolo_config():
    """Create a working YOLO config with absolute paths"""
    
    # First, try to use the provided config if it exists and is valid
    if os.path.exists(yolo_config_path):
        try:
            with open(yolo_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check if the config has valid paths
            if 'train' in config and config['train'] is not None:
                # Convert relative paths to absolute if needed
                if not os.path.isabs(config['train']):
                    config['train'] = os.path.join(starter_data_path, config['train'])
                if 'val' in config and config['val'] and not os.path.isabs(config['val']):
                    config['val'] = os.path.join(starter_data_path, config['val'])
                
                # Update the path to absolute
                config['path'] = starter_data_path
                
                # Save the corrected config
                working_config_file = "working_dataset.yaml"
                with open(working_config_file, 'w') as f:
                    yaml.dump(config, f)
                
                print(f"âœ… Using corrected provided config: {working_config_file}")
                return working_config_file
                
        except Exception as e:
            print(f"âš ï¸� Error reading provided config: {e}")
    
    # Create our own config if the provided one doesn't work
    print("ğŸ”§ Creating custom YOLO config...")
    
    # Define absolute paths
    train_path = os.path.join(starter_data_path, "train", "images")
    val_path = os.path.join(starter_data_path, "val", "images")
    
    # Verify these paths exist
    if not os.path.exists(train_path):
        print(f"â�Œ Training images not found at: {train_path}")
        # Try alternative structure
        alt_train_path = os.path.join(starter_data_path, "images", "train")
        if os.path.exists(alt_train_path):
            train_path = alt_train_path
            val_path = os.path.join(starter_data_path, "images", "val")
            print(f"âœ… Found alternative structure: {train_path}")
        else:
            print("â�Œ Cannot find training images in expected locations")
            return None
    
    config = {
        'path': starter_data_path,
        'train': train_path,
        'val': val_path,
        'test': '',
        'names': {
            0: 'cheerios',
            1: 'soup'
        },
        'nc': 2
    }
    
    config_file = "custom_dataset.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"âœ… Created custom YOLO config: {config_file}")
    
    # Verify the paths exist
    if os.path.exists(config['train']) and os.path.exists(config['val']):
        train_count = len([f for f in os.listdir(config['train']) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_count = len([f for f in os.listdir(config['val']) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"   âœ… Found {train_count} training images")
        print(f"   âœ… Found {val_count} validation images")
    
    return config_file

# Create working config
config_file = create_working_yolo_config()

if config_file is None:
    print("â�Œ Could not create valid config - stopping here")
    exit()



def train_and_visualize_model(config_path):
    """Train model and show results inline"""
    
    print("ğŸ�‹ï¸� Starting training...")
    
    # Load YOLOv8 nano
    model = YOLO("yolov8n.pt")
    
    try:
        # Train the model
        results = model.train(
            data=config_path,
            epochs=25,
            imgsz=640,
            batch=16,
            patience=8,
            save=True,
            verbose=True,
            project="runs",
            name="first_model",
            plots=True,
            val=True,
            save_period=5,
            device=0 if os.system('nvidia-smi') == 0 else 'cpu'
        )
        
        print("âœ… Training complete!")
        
        # Now visualize the results inline!
        results_dir = "runs/first_model"
        
        print("\nğŸ“Š TRAINING RESULTS VISUALIZATION:")
        print("="*50)
        
        # 1. Show training curves
        results_plot = os.path.join(results_dir, "results.png")
        if os.path.exists(results_plot):
            print("ğŸ“ˆ Training Curves:")
            show_image(results_plot, "Training Progress Over Time", figsize=(12, 8))
        
        # 2. Show confusion matrix
        confusion_plots = [
            os.path.join(results_dir, "confusion_matrix.png"),
            os.path.join(results_dir, "confusion_matrix_normalized.png")
        ]
        valid_confusion = [p for p in confusion_plots if os.path.exists(p)]
        if valid_confusion:
            print("ğŸ�¯ Model Performance:")
            show_images_grid(valid_confusion, 
                           titles=["Confusion Matrix", "Normalized Confusion Matrix"],
                           figsize=(12, 6))
        
        # 3. Show PR curves
        curve_plots = [
            os.path.join(results_dir, "BoxPR_curve.png"),
            os.path.join(results_dir, "BoxF1_curve.png")
        ]
        valid_curves = [p for p in curve_plots if os.path.exists(p)]
        if valid_curves:
            print("ğŸ“Š Performance Curves:")
            show_images_grid(valid_curves,
                           titles=["Precision-Recall Curve", "F1 Score Curve"],
                           figsize=(12, 6))
        
        # 4. Show label distribution
        labels_plot = os.path.join(results_dir, "labels.jpg")
        if os.path.exists(labels_plot):
            print("ğŸ�·ï¸� Label Distribution:")
            show_image(labels_plot, "Dataset Label Analysis", figsize=(10, 6))
        
        # 5. Show training batch samples
        print("ğŸ–¼ï¸� Training Batch Samples:")
        train_batches = [
            os.path.join(results_dir, f"train_batch{i}.jpg") 
            for i in range(3) 
            if os.path.exists(os.path.join(results_dir, f"train_batch{i}.jpg"))
        ]
        if train_batches:
            show_images_grid(train_batches[:2], 
                           titles=[f"Training Batch {i}" for i in range(len(train_batches[:2]))],
                           figsize=(15, 8))
        
        # 6. Show validation predictions vs ground truth
        print("âœ… Validation Results (Predictions vs Ground Truth):")
        val_images = []
        val_titles = []
        
        for i in range(2):  # Show first 2 validation batches
            labels_path = os.path.join(results_dir, f"val_batch{i}_labels.jpg")
            pred_path = os.path.join(results_dir, f"val_batch{i}_pred.jpg")
            
            if os.path.exists(labels_path):
                val_images.append(labels_path)
                val_titles.append(f"Ground Truth Batch {i}")
            if os.path.exists(pred_path):
                val_images.append(pred_path)
                val_titles.append(f"Predictions Batch {i}")
        
        if val_images:
            show_images_grid(val_images, titles=val_titles, figsize=(15, 12), cols=2)
        
        return os.path.join(results_dir, "weights", "best.pt")
        
    except Exception as e:
        print(f"â�Œ Training failed: {e}")
        return "yolov8n.pt"

# Train and visualize
if config_file:
    model_path = train_and_visualize_model(config_file)
    print(f"ğŸ�¯ Model ready at: {model_path}")
else:
    model_path = "yolov8n.pt"



def make_predictions_with_visualization(model_path, test_folder):
    """Generate predictions and show sample results"""
    
    print("ğŸ”® Making predictions...")
    
    if not os.path.exists(test_folder):
        print(f"â�Œ Test folder not found: {test_folder}")
        return []
    
    # Load model
    model = YOLO(model_path)
    
    predictions = []
    test_images = [f for f in os.listdir(test_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"ğŸ“Š Predicting on {len(test_images)} images...")
    
    # Store some sample predictions to visualize
    sample_predictions = []
    
    for i, image_file in enumerate(test_images):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(test_images)} images")
        
        try:
            image_path = os.path.join(test_folder, image_file)
            image_id = os.path.splitext(image_file)[0]
            
            # Run prediction
            results = model.predict(image_path, conf=0.25, verbose=False)
            
            # Store first few for visualization
            if len(sample_predictions) < 4:
                sample_predictions.append((image_path, results))
            
            # Format predictions
            pred_parts = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    for j in range(len(boxes)):
                        cls = int(boxes.cls[j])
                        conf = float(boxes.conf[j])
                        x_center, y_center, width, height = boxes.xywhn[j]
                        
                        pred_parts.append(f"{cls} {conf:.6f} {float(x_center):.6f} {float(y_center):.6f} {float(width):.6f} {float(height):.6f}")
            
            pred_string = " ".join(pred_parts)
            predictions.append([image_id, pred_string])
            
        except Exception as e:
            print(f"âš ï¸� Error processing {image_file}: {e}")
            image_id = os.path.splitext(image_file)[0]
            predictions.append([image_id, ""])
    
    # Visualize sample predictions
    print("\nğŸ�¯ Sample Predictions on Test Images:")
    print("="*50)
    
    if sample_predictions:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (image_path, results) in enumerate(sample_predictions):
            if idx >= 4:
                break
                
            # Load and display image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw predictions
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Draw rectangle
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    
                    # Add label
                    class_names = ['cheerios', 'soup']
                    label = f"{class_names[cls]}: {conf:.2f}"
                    cv2.putText(img, label, (int(x1), int(y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"Prediction {idx+1}: {os.path.basename(image_path)}")
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("âœ… Predictions complete!")
    return predictions

# Make predictions with visualization
if os.path.exists(test_data_path):
    predictions = make_predictions_with_visualization(model_path, test_data_path)
    print(f"ğŸ“Š Made predictions for {len(predictions)} images")
else:
    print("â�Œ Can't find test images")
    predictions = []



def create_and_analyze_submission(predictions, filename="submission.csv"):
    """Create submission and show analysis"""
    
    if not predictions:
        print("â�Œ No predictions to create submission")
        return None
    
    df = pd.DataFrame(predictions, columns=["image_id", "prediction_string"])
    df.to_csv(filename, index=False)
    
    print(f"ğŸ“� Submission saved as: {filename}")
    print(f"Shape: {df.shape}")
    
    # Analyze submission
    non_empty = df[df['prediction_string'] != ''].shape[0]
    empty = df[df['prediction_string'] == ''].shape[0]
    
    print(f"\nğŸ“Š Submission Analysis:")
    print(f"   Images with detections: {non_empty}")
    print(f"   Images with no detections: {empty}")
    print(f"   Detection rate: {non_empty/len(df)*100:.1f}%")
    
    # Show distribution of predictions
    pred_lengths = df['prediction_string'].str.split().str.len().fillna(0)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(pred_lengths, bins=20, alpha=0.7)
    plt.title('Distribution of Predictions per Image')
    plt.xlabel('Number of Detections')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    detection_status = ['No Detections', 'Has Detections']
    detection_counts = [empty, non_empty]
    plt.pie(detection_counts, labels=detection_status, autopct='%1.1f%%')
    plt.title('Detection Coverage')
    
    plt.tight_layout()
    plt.show()
    
    # Show first few predictions
    print("\nFirst few predictions:")
    display(df.head())
    
    return df

if predictions:
    submission_df = create_and_analyze_submission(predictions)





