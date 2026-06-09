import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import cv2


class ObjectDetectionTransforms:
    def __init__(self, resize=(800, 800)):
        self.resize = resize
        self.to_tensor = T.ToTensor()
    
    def __call__(self, image, target):
        print(f"Image type: {type(image)}")

        image = F.resize(image, self.resize)

        image = self.to_tensor(image)

        boxes = target["boxes"]
        boxes = self.resize_boxes(boxes, image)

        target["boxes"] = boxes
        return image, target
    
    def resize_boxes(self, boxes, image):
        if isinstance(image, torch.Tensor):
            _, h, w = image.shape
        else:
            w, h = image.size
        boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
        return boxes


transform = ObjectDetectionTransforms(resize=(800, 800))


class ObjectDetectionDataset(data.Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name.replace('.png', '.txt'))
        
        image = Image.open(image_path).convert("RGB")
        
        boxes = []
        labels = []
        try:
            with open(label_path, "r") as file:
                for line in file:
                    line = line.strip().split()
                    x_center, y_center, width, height = map(float, line[1:])
                    x_min = (x_center - width / 2) * image.width
                    y_min = (y_center - height / 2) * image.height
                    x_max = (x_center + width / 2) * image.width
                    y_max = (y_center + height / 2) * image.height
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(0)  
        except FileNotFoundError:
            print(f"Label file not found for {image_name}")
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target


train_image_dir = "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/train/images"
train_label_dir = "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/train/labels"
val_image_dir = "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/val/images"
val_label_dir = "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/val/labels"

train_dataset = ObjectDetectionDataset(train_image_dir, train_label_dir, transforms=transform)
val_dataset = ObjectDetectionDataset(val_image_dir, val_label_dir, transforms=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

for epoch in range(epochs):
    model.train()
    start_time = time.time()
    
    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {losses.item()}, Time: {time.time() - start_time:.2f} sec")



model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        prediction = model(images)
        
        for idx in range(len(images)):
            img = images[idx].cpu().numpy().transpose((1, 2, 0))
            boxes = prediction[idx]['boxes'].cpu().numpy()
            labels = prediction[idx]['labels'].cpu().numpy()
            
            plt.imshow(img)
            for box in boxes:
                plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, color="red"))
            plt.show()



torch.save(model.state_dict(), "fasterrcnn_soup_model.pth")





