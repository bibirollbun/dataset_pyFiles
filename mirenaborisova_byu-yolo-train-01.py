!pip install ultralytics -q


import numpy as np
import random
import torch

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


byu_yolo_dataset_path = '/kaggle/input/byu-yolo-datasets-01/BYU_YOLO_dataset'
yolo_weights_path = '/kaggle/working/yolo_weights'
yolo_pretrained_weights = 'yolov8n.pt'

dataset_yaml_path = '/kaggle/input/byu-yolo-datasets-01/BYU_YOLO_dataset/dataset.yaml'


import os
os.makedirs(yolo_weights_path, exist_ok=True)


from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NUM_IMAGES = 4
BOX_SIZE = 24
def generalization(model):
    
    validation_path = byu_yolo_dataset_path + '/images/val'
    validation_images = os.listdir(validation_path)
    
    img_files = random.sample(validation_images, NUM_IMAGES)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, img_file in enumerate(img_files):
            
        img_path = os.path.join(validation_path, img_file)
        
        results = model.predict(img_path, conf=0.25)[0]
        
        img = Image.open(img_path)
        axes[i].imshow(np.array(img), cmap='gray')
        
        try:
            img_file_split = img_file.split('_')
            y_startswith = [y for y in img_file_split if y.startswith('y')]
            x_startswith = [x for x in img_file_split if x.startswith('x')]
            
            if y_startswith and x_startswith:
                y_coordinate = int(y_startswith[0][1:])
                x_coordinate = int(x_startswith[0][1:].split('.')[0])
                
                rectangle = Rectangle((x_coordinate - BOX_SIZE // 2,
                                       y_coordinate - BOX_SIZE // 2), 
                                      BOX_SIZE, 
                                      BOX_SIZE, 
                                      linewidth=1, 
                                      edgecolor='lime', 
                                      facecolor='none')
                axes[i].add_patch(rectangle)
        except:
            pass
        
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x_start, y_start, x_end, y_end = box
                rectangle_pred = Rectangle((x_start, y_start), 
                                           x_end - x_start, 
                                           y_end-y_start, 
                                           linewidth=1,
                                           edgecolor='deeppink',
                                           facecolor='none')
                axes[i].add_patch(rectangle_pred)
                axes[i].text(x_start, y_start-5, f'{conf:.2f}', color='cyan')
        
        axes[i].set_title(f"Image: {img_file}\nGround Truth (green) vs Prediction (red)")
    
    plt.tight_layout()
    
    plt.savefig('/kaggle/working/predictions.png')
    plt.show()


import pandas as pd

def plot_distribution_focal_loss(weights_path):
    
    results_csv = os.path.join(weights_path, 'results.csv')
    results_df = pd.read_csv(results_csv)
    
    train_dfl_cols = [col for col in results_df.columns if 'train/dfl_loss' in col]
    val_dfl_cols = [col for col in results_df.columns if 'val/dfl_loss' in col]
    
    train_dfl_cols = train_dfl_cols[0]
    val_dfl_cols = val_dfl_cols[0]

    best_epoch = results_df[val_dfl_cols].idxmin()
    best_val_loss = results_df.loc[best_epoch, val_dfl_cols]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(results_df['epoch'], results_df[train_dfl_cols], label='TRAIN DISTRIBUTION FOCAL LOSS')
    plt.plot(results_df['epoch'], results_df[val_dfl_cols], label='VALIDATION DISTRIBUTION FOCAL LOSS')
    
    plt.axvline(x=results_df.loc[best_epoch, 'epoch'], 
                color='deeppink', 
                label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, ' + \
                      f' Val Loss: {best_val_loss:.4f})')
    
    plt.xlabel('EPOCH')
    plt.ylabel('DISTRIBUTION FOCAL LOSS')
    plt.title('TRAIN & VALIDATION DISTRIBUTION FOCAL LOSS')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = weights_path + '/dfl_plot.png'
    plt.savefig(plot_path)
    
    plt.savefig('/kaggle/working/dfl_plot.png')
    
    print(f"Loss curve saved to {plot_path}")
    plt.close()

    return best_epoch, best_val_loss


from ultralytics import YOLO
import yaml

EPOCHS = 30
BATCH_SIZE = 16
IMG_SIZE = 640
# PATIENCE = 5
PATIENCE = 0
SAVE_PERIOD = 5
VAL = True
VERBOSE = True

def train_yolo_model(yaml_path):

    model = YOLO(yolo_pretrained_weights)
    
    results = model.train(data=yaml_path,
                          epochs=EPOCHS,
                          batch=BATCH_SIZE,
                          imgsz=IMG_SIZE,
                          project=yolo_weights_path,
                          name='flagellar_motor_detector',
                          exist_ok=True,
                          patience=PATIENCE,
                          save_period=SAVE_PERIOD,
                          val=VAL,
                          verbose=VERBOSE)
    
    weights_path = yolo_weights_path + '/flagellar_motor_detector'
    
    best_epoch_info = plot_distribution_focal_loss(weights_path)
    
    if best_epoch_info:
        best_epoch, best_val_loss = best_epoch_info
        print(f"\nBEST EPOCH: {best_epoch} , BEST_VAL_LOSS: {best_val_loss:.4f}")
    
    return model, results


def upload_dataset(yaml_path):
    
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
        
    if 'path' in yaml_data:
        yaml_data['path'] = byu_yolo_dataset_path
        
    upload_yaml_path = "/kaggle/working/upload_dataset.yaml"
    
    with open(upload_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
    
    return upload_yaml_path


yaml_path = upload_dataset(dataset_yaml_path)

with open(yaml_path, 'r') as f:
    yaml_content = f.read()
    
model, results = train_yolo_model(yaml_path)

generalization(model)




