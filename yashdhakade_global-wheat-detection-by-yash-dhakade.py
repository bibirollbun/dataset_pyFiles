|import pandas as pd # Data analysis library

import albumentations as A # data agumentation library
from albumentations.pytorch.transforms import ToTensorV2 # it use to convert images into tensor, requires to train DL models

import torch # it a framework use to build and train deep learning models
import os  # it use to work with local directories
from torch.utils.data import Dataset, DataLoader  # creating custom datasets AND DataLoader is used to load and batch the dataset during training or evaluation.
import cv2 # this is use to work with images and videos in realtime
import ast # module allows you to parse, analyze, and manipulate Python code itself.
import numpy as np # use for neumerical computing in python
import matplotlib.pyplot as plt # to visualize the graph

import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")

DIR_TRAIN = '/kaggle/input/global-wheat-detection/train'
DIR_TEST = '/kaggle/input/global-wheat-detection/test'

data.head(2)


# creating valid and Training set
image_ids = data['image_id'].unique()
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]

valid_df = data[data['image_id'].isin(valid_ids)]
train_df = data[data['image_id'].isin(train_ids)]


class WheatDataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    self.data = csv_file
    self.image_ids = self.data['image_id'].unique()
    self.root_dir = root_dir
    self.transform = transform

  def __getitem__(self, index):

    # grab one sample (image + label)
    image_id = self.image_ids[index]
    records = self.data[self.data['image_id'] == image_id] # get all the records related that image id

    # Construct the image path using os.path.join to ensure correct path handling
    image_path=os.path.join(self.root_dir, f'{image_id}.jpg')

    # Check if file exists!
    if not os.path.exists(image_path):
        print(f"❌ File does not exist: {image_path}")
        return None

    # read the image and covert it into RGB format.
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0  # scale the images

    # store all the bounding boxes
    boxes = []
    for box in records['bbox']:
        box = ast.literal_eval(box) #  safely converts the string
        x_min = box[0]
        y_min = box[1]
        x_max = x_min + box[2]
        y_max = y_min + box[3]
        boxes.append([x_min, y_min, x_max, y_max])

    boxes = torch.as_tensor(boxes, dtype=torch.float32) # convert the list of boxes into tensor
    labels = torch.ones((records.shape[0],), dtype=torch.int64)  # Single class (wheat)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)  # 0 for each box

    # package everything in target dict, This is exactly what PyTorch detection models expect.
    target = {}
    target['boxes'] = boxes
    target['labels'] = labels
    target['area'] = area
    target['iscrowd'] = iscrowd

    target['image_id'] = torch.tensor([index])

    if self.transform:

      sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels.tolist()
            }
      sample = self.transform(**sample)
      image = sample['image']
      target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

    return image, target, image_id


  def __len__(self):
    return len(self.image_ids)


  def show_image(self, index):

    # Get the image using __getitem__
    image, _, _ = self.__getitem__(index)

    # Check if image and target were returned
    if image is None:
        print(f"Error: Could not load image for index {index}")
        return

    # Display the image using matplotlib
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels
    plt.show()

load_data = WheatDataset(csv_file=data, root_dir= DIR_TRAIN)
load_data.show_image(10) # display image on index


# Albumentations
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),   # Randomly flip images horizontally 50% of the time
        ToTensorV2(p=1.0)           # Convert images & bboxes to PyTorch tensors
    ], bbox_params={'format': 'pascal_voc',     # Specifies bbox format
                    'label_fields': ['labels']} # Labels to be transformed along with bboxes
                    )

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)         # Only convert to tensor, no augmentation for validation
    ], bbox_params={'format': 'pascal_voc',      # Specifies bbox format
                    'label_fields': ['labels']}  # Labels to be transformed along with bboxes
                    )


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)                   # fasterrcnn_resnet50_fpn
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)              # fasterrcnn_resnet50_fpn_v2
# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)       # fasterrcnn_mobilenet_v3_large_fpn
# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)   # fasterrcnn_mobilenet_v3_large_320_fpn

# Modify the classifier head for your number of classes (background + wheat)
num_classes = 2  # wheat + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())


# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

# loading the data into dataloader
train_data_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
images, targets, image_ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images) # move the images tensor from CPU to GPU for fater computation
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# take a sample and it's label to plot the image
boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)
sample = images[2].permute(1,2,0).cpu().numpy()

# plot the an image
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)

ax.set_axis_off()
ax.imshow(sample)


# train the model
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# lr_scheduler = None

# run the training loop for 5 epochs
num_epochs = 5


import torch
import os
# Initialize Averager objects to track average loss over iterations
train_loss_hist = Averager()
val_loss_hist = Averager()

itr = 1  # Iteration counter
start_epoch = 0  # Variable to store the epoch to resume training from

# Check if there is a checkpoint to load from
checkpoint_dir = "/kaggle/working//Fast R-CNN"
os.makedirs(name=checkpoint_dir, exist_ok=True)

checkpoint_path = checkpoint_dir+'/checkpoint_fasterrcnn_resnet50_fpn.pth'
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    itr = checkpoint['iteration']
    print(f"Resumed training from epoch {start_epoch}...")
else:
    print("No checkpoint found. Starting from scratch.")
    train_losses = []
    val_losses = []

# Start the training loop for the specified number of epochs
for epoch in range(start_epoch, num_epochs):
    print(f"\nEpoch #{epoch + 1} started\n" + "-"*30)

    # ----------------------------------- TRAINING PHASE ---------------------------------------------
    train_loss_hist.reset()  # Reset the training loss history at the start of each epoch

    model.train()  # Set the model to training mode

    # Iterate through batches in the training data loader
    for images, targets, image_ids in train_data_loader:
        # Move images and targets to the appropriate device (GPU/CPU)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Perform a forward pass through the model and get the loss
        loss_dict = model(images, targets)

        # Sum up the losses (classification loss + bounding box regression loss)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # Update the loss history with the current loss
        train_loss_hist.send(loss_value)

        # Zero the gradients
        optimizer.zero_grad()

        # Calculate the loss on backpropagation
        losses.backward()

        # Update the model parameters
        optimizer.step()

        # Print the current iteration loss every 50 iterations
        if itr % 50 == 0:
            print(f"Iteration #{itr} training loss: {loss_value:.4f}")

        itr += 1  # Increment the iteration counter

    # Step the learning rate scheduler, if provided
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Save the average training loss for the epoch
    train_epoch_loss = train_loss_hist.value
    train_losses.append(train_epoch_loss)
    print(f"Epoch #{epoch + 1} Training loss: {train_epoch_loss:.4f}")

    # ----- VALIDATION PHASE -----
    val_loss_hist.reset()  # Reset validation loss history for the current epoch

    # model.train()  # Stay in train mode to compute validation losses (if required by model)
    # Alternatively, you can use model.eval() if you don't need to compute losses during validation inference.
    with torch.no_grad():
        for images, targets, image_ids in valid_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass through the model to get validation losses
            loss_dict = model(images, targets)

            # Sum all losses
            losses = sum(loss for loss in loss_dict.values())

            # Update the validation loss history
            val_loss_hist.send(losses.item())

    # Save the average validation loss for the epoch
    val_epoch_loss = val_loss_hist.value
    val_losses.append(val_epoch_loss)
    print(f"Epoch #{epoch + 1} Validation loss: {val_epoch_loss:.4f}")

    # Save checkpoint after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'iteration': itr,
    }, checkpoint_path)

print(f"Training Losses: {train_losses}")
print(f"Validation Losses: {val_losses}")

print("\nTraining & validation completed.")


# Plot after training is done
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


images, targets, image_ids = next(iter(valid_data_loader))

images = list(img.to(device) for img in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
sample = images[1].permute(1,2,0).cpu().numpy()
labels = targets[1]['labels'].cpu().numpy()


import matplotlib.pyplot as plt
import numpy as np
# Assuming 'model' and 'device' are defined as in your code snippet
# and you have a DataLoader named 'valid_data_loader'

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    images, targets, image_ids = next(iter(valid_data_loader))
    images = list(img.to(device) for img in images)
    outputs = model(images)  # Get model predictions

    # Example: Plot predictions for the first image in the batch
    boxes = outputs[0]['boxes'].cpu().numpy().astype(np.int32)
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    sample = images[0].permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:  # Only show detections with a confidence score above 0.5
            cv2.rectangle(
                sample,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 255, 0),  # Green color for bounding boxes
                2,
            )
            # Optionally, add labels and scores to the plot
            if label == 1:
                label = 'Wheat'
            else:
                label = 'Background'

            ax.text(box[0], box[1], f"{label}: {score:.2f}", color='white')

    ax.set_axis_off()
    ax.imshow(sample)
    plt.show()


import numpy as np

def compute_iou(box1, box2):
    """Compute IoU between two boxes. Boxes are [xmin, ymin, xmax, ymax]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area



def calculate_ap_at_iou(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Calculate AP at a specific IoU threshold."""
    if len(pred_boxes) == 0:
        return 0.0

    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)  # sort by score

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    matched_gt = set()

    for idx, pred in enumerate(pred_boxes):
        pred_box = pred[:4]
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp[idx] = 1  # True Positive
            matched_gt.add(best_gt_idx)
        else:
            fp[idx] = 1  # False Positive

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recalls = cumulative_tp / (len(gt_boxes) + 1e-6)
    precisions = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-6)

    ap = 0.0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap



def calculate_map(pred_boxes_all, gt_boxes_all, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """Calculate mAP across multiple IoU thresholds."""
    aps = []

    for iou_thresh in iou_thresholds:
        ap_per_image = []
        for preds, gts in zip(pred_boxes_all, gt_boxes_all):
            ap = calculate_ap_at_iou(preds, gts, iou_thresh)
            ap_per_image.append(ap)

        mean_ap = np.mean(ap_per_image)
        print(f"AP @ IoU {iou_thresh:.2f}: {mean_ap:.4f}")
        aps.append(mean_ap)

    map_value = np.mean(aps)
    print(f"mAP@[0.5:0.95]: {map_value:.4f}")

    return map_value



model.eval()  # switch to eval mode

pred_boxes_all = []
gt_boxes_all = []

with torch.no_grad():
    for images, targets, image_ids in valid_data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        # Collect predictions and ground truth for each image
        for i in range(len(images)):
            preds = outputs[i]
            scores = preds['scores'].cpu().numpy()
            boxes = preds['boxes'].cpu().numpy()

            # Combine boxes and scores for prediction list
            pred_boxes = []
            for box, score in zip(boxes, scores):
                pred_boxes.append([box[0], box[1], box[2], box[3], score])

            # Get ground truth boxes
            gt_boxes = targets[i]['boxes'].cpu().numpy().tolist()

            pred_boxes_all.append(pred_boxes)
            gt_boxes_all.append(gt_boxes)

# Calculate mAP
map_score = calculate_map(pred_boxes_all, gt_boxes_all)


!git clone https://github.com/ultralytics/yolov5.git
!cd yolov5
!pip install -r /kaggle/working/yolov5/requirements.txt


import os
from tqdm import tqdm
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define the source directory of images
images_dir = Path('/kaggle/input/global-wheat-detection/train/')

# Collect all image files
image_files = sorted(list(images_dir.glob('*.jpg')))

# Split into 80% train, 20% validation
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)


# Define target folder structure for YOLOv5
base_dir = Path('/kaggle/working/wheat-dataset/')
images_train_dir = base_dir / 'images/train'
images_val_dir = base_dir / 'images/val'
labels_train_dir = base_dir / 'labels/train'
labels_val_dir = base_dir / 'labels/val'

# Create folders
for dirs in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
    dirs.mkdir(parents=True, exist_ok=True)

for train_image_path, valid_image_path in zip(train_images, val_images):
    # print(train_image_path, images_train_dir / train_image_path.name)
    # print(train_image_path.name)   # image.jpg

    # training images
    shutil.copy(src=train_image_path, dst=images_train_dir / train_image_path.name)

    # # validation images
    shutil.copy(src=valid_image_path, dst=images_val_dir / valid_image_path.name)

print(f"Training Set: {len(train_images)}\nValidation set: {len(val_images)}")


from tqdm import tqdm
from PIL import Image
import pandas as pd


csv_file_path = '/kaggle/input/global-wheat-detection/train.csv'  # Contains bounding box coordinates
df = pd.read_csv(csv_file_path)

# Function to create YOLO label files from DataFrame group
def create_labels(image_list, label_output_dir, image_output_dir):
    for img_path in tqdm(image_list):
        image_id = img_path.stem  # image filename without extension
        img = Image.open(img_path)
        w, h = img.size

        # Group annotations for this image_id
        group = df[df['image_id'] == image_id]

        # Skip if there are no annotations
        if group.shape[0] == 0:
            continue

        # Label file path
        label_file = label_output_dir / f"{image_id}.txt"

        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                bbox = eval(row['bbox'])  # format [x_min, y_min, width, height]

                # Convert to YOLO format (normalized)
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                width = bbox[2] / w
                height = bbox[3] / h

                # Only 1 class: wheat (class 0)
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

# Generate labels for train
create_labels(train_images, labels_train_dir, images_train_dir)

# Generate labels for val
create_labels(val_images, labels_val_dir, images_val_dir)

print("Images and label files are ready!")


# Create a YAML file for YOLOv5 (path: data.yaml)
yaml_content = """
path: /kaggle/input/global-wheat-detection  # path to the root of the dataset
train: /kaggle/working/wheat-dataset/images/train  # images for training
val: /kaggle/working/wheat-dataset/images/val  # images for validation
nc: 1  # number of classes
names: ['wheat']  # class names
"""

# Save the YAML configuration file
with open('/kaggle/working/data.yaml', 'w') as f:
    f.write(yaml_content)


# W&B Login
import wandb

# tool used for tracking experiments, logging metrics, visualizations, and collaborating on machine learning projects
# os.environ['WANDB_API_KEY'] = 'wandb_API_Key'
os.environ['WANDB_API_KEY'] = '1233c12c43cf272a78b6d84dc7f19c0546f0e48d'

# start training
!python /kaggle/working/yolov5/train.py --img 640 --batch 16 --epochs 10 --data /kaggle/working/data.yaml --weights yolov5m6.pt --cache


from yolov5 import detect
import torch

model_path = "/kaggle/working/yolov5/runs/train/exp/weights/best.pt"
test_images = "/kaggle/input/global-wheat-detection/test/"

# Run inference
!python /kaggle/working/yolov5/detect.py --weights {model_path} --source {test_images} --img 640 --conf 0.25 --save-txt --save-conf


import os
import matplotlib.pyplot as plt
from PIL import Image

predicted_imgs = "/kaggle/working/yolov5/runs/detect/exp"

# Get all image files in the folder
images = [img for img in os.listdir(predicted_imgs) if img.endswith(('.jpg', '.png', '.jpeg'))]

# Number of images you want to show per row
cols = 3
# Calculate the required number of rows based on the number of images
rows = (len(images) + cols - 1) // cols

# Create a figure with subplots
plt.figure(figsize=(12, rows * 5))

# Loop through the images and create subplots
for idx, img_name in enumerate(images):
    img_path = os.path.join(predicted_imgs, img_name)
    img = Image.open(img_path)

    # Add subplot for each image
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Prediction: {os.path.basename(img_path)}')

# Adjust layout and show the figure
plt.tight_layout()
plt.show()


result = pd.read_csv("/kaggle/working/yolov5/runs/train/exp/results.csv")
display(result.iloc[:, 6:8])

