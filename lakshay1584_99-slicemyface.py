import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision import transforms
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torch.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from IPython.display import FileLink
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


submission = []
dice_scores = []
predictions = []


#SegFormer's Defination
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512").to(device)


#Parameters which control training 
num_epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
accumulation_steps = 4
scaler = GradScaler('cuda')


class FaceSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor, transform=None, is_test=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.is_test = is_test
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_test:
            encoded_inputs = self.feature_extractor(images=image, return_tensors="pt", do_rescale=False)
            return encoded_inputs, self.image_filenames[idx]
        
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace('.jpg', '.png'))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        encoded_inputs = self.feature_extractor(images=image, return_tensors="pt", do_rescale=False)
        encoded_inputs['labels'] = torch.tensor(mask, dtype=torch.long)
        return encoded_inputs


train_transform = A.Compose([
    A.Resize(1024,1024),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.GaussianBlur(p=0.1),
    ToTensorV2()
])

dataset = FaceSegmentationDataset("/kaggle/input/slicee-my-face/images/train", "/kaggle/input/slicee-my-face/annotations/train", feature_extractor, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
test_dataset = FaceSegmentationDataset("/kaggle/input/slicee-my-face/images/test", None, feature_extractor, is_test=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def train_class(model, dataloader, criterion, optimizer, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        if batch is None:
            continue
        
        with autocast('cuda'):
            inputs = {k: v.squeeze(0).to(device) if k != 'pixel_values' else v.squeeze(1).to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
    return total_loss / len(dataloader)


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) % 2 != 0:
        runs = np.append(runs, len(pixels) - 1)
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def dice_coefficient(pred_mask, true_mask):
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)
    if union == 0:
        return 1.0
    return 2 * intersection / union


def tta_inference(model, image):
    augments = [
        lambda x: x,
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    ]
    
    for augment in augments:
            augmented_img = augment(image)
            inputs = feature_extractor(images=augmented_img, return_tensors="pt").to(device)
            outputs = model(**inputs)
            pred_mask = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.append(pred_mask)
    
    return np.mean(predictions, axis=0) > 0.5 


model.train()


for epoch in range(num_epochs):
    loss = train_class(model, dataloader, criterion, optimizer, accumulation_steps)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    scheduler.step()


model.eval()


with torch.no_grad():
    for batch, image_name in tqdm(test_dataloader, desc="Inference"):
        inputs = {k: v.squeeze(0).to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        pred_mask = torch.argmax(outputs.logits, dim=1).cpu().numpy().squeeze()
        
        true_mask_path = f"/kaggle/input/slicee-my-face/annotations/test/{image_name[0].replace('.jpg', '.png')}"
        true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
        
        binary_pred_mask = cv2.resize((pred_mask > 0).astype(np.uint8), (true_mask.shape[1], true_mask.shape[0]))
        binary_true_mask = (true_mask > 0).astype(np.uint8)

        dice_score = dice_coefficient(binary_pred_mask, binary_true_mask)
        dice_scores.append(dice_score)
        
        kernel = np.ones((3,3), np.uint8)
        clean_mask = cv2.morphologyEx(binary_pred_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        clean_mask[clean_mask > 0] = 1

        rle_mask = rle_encode(clean_mask)
        submission.append([image_name[0].replace(".jpg", ""), rle_mask])


submission_df = pd.DataFrame(submission, columns=["id", "predicted"])
submission_df = submission_df.sort_values(by="id")
submission_df.to_csv("submission.csv", index=False)
print(f"Average Dice Coefficient: {np.mean(dice_scores):.4f}")
display(FileLink('submission.csv'))

