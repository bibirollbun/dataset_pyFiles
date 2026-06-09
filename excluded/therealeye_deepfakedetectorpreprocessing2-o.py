import os
from PIL import Image
import cv2
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from torchvision import models


# First dataset
train_videos_path = '/kaggle/input/deepfake-detection-challenge/train_sample_videos'
metadata_path = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json'
destination_path = '/kaggle/working/metadata.json'
test_videos_path='/kaggle/input/deepfake-detection-challenge/test_videos'
true_label_test='/kaggle/input/true-label-for-testvideo/true label.csv'
# Second dataset
second_dataset_path = '/kaggle/input/real-vs-fake-img/real-vs-fake'


import pandas as pd

labels_df = pd.read_csv('/kaggle/input/true-label-for-testvideo/metadata_converted.csv')  # Update with your actual CSV path
video_to_label = dict(zip(labels_df['filenames'], labels_df['label(T=0/F=1)']))
# Now video_to_label["aapnvogymq.mp4"] gives 1 (fake) or 0 (real)




import cv2
import os
from tqdm import tqdm  # Add tqdm for progress

video_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos'
frame_interval = 5  # Example: extract every 30th frame

# Wrap the outer video loop with tqdm for overall progress
for video_filename, label in tqdm(video_to_label.items(), desc="Processing videos", total=len(video_to_label)):
    video_path = os.path.join(video_dir, video_filename)
    class_name = 'fake' if label == 1 else 'real'
    save_dir = f'/kaggle/working/training_frames/{class_name}'
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    frame_save_count = 0  # To count how many frames get saved
    # Optional: tqdm in the frame loop only if you want (not strictly necessary if only a few frames will be saved)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = f"{os.path.splitext(video_filename)[0]}_frame{frame_num}.jpg"
            cv2.imwrite(os.path.join(save_dir, frame_filename), frame)
            frame_save_count += 1
        frame_num += 1
    cap.release()
    tqdm.write(f"{video_filename}: saved {frame_save_count} frames to {class_name}")

print("Frame extraction completed.")



import cv2
import os
from tqdm import tqdm

video_dir = '/kaggle/input/deepfake-detection-challenge/test_videos'
save_dir = '/kaggle/working/test_frames'
frame_interval = 5  # Extract every 5th frame

os.makedirs(save_dir, exist_ok=True)

for video_filename in tqdm(os.listdir(video_dir), desc="Processing videos"):
    video_path = os.path.join(video_dir, video_filename)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    frame_save_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = f"{os.path.splitext(video_filename)[0]}_frame{frame_num}.jpg"
            cv2.imwrite(os.path.join(save_dir, frame_filename), frame)
            frame_save_count += 1
        frame_num += 1
    
    cap.release()
    tqdm.write(f"{video_filename}: saved {frame_save_count} frames")

print("Frame extraction completed.")



import os


folder_path1 = '/kaggle/working/training_frames/real'
folder_path2= '/kaggle/working/training_frames/fake'
folder_path3 = '/kaggle/working/test_frames'

jpg_count1 = len([file for file in os.listdir(folder_path1) if file.lower().endswith('.jpg')])
print(f'Total number of jpg files in training_frames/real/ Folder: {jpg_count1}')

jpg_count2 = len([file for file in os.listdir(folder_path2) if file.lower().endswith('.jpg')])
print(f'Total number of jpg files in training_frames/fake/ Folder: {jpg_count2}')

jpg_count3 = len([file for file in os.listdir(folder_path3) if file.lower().endswith('.jpg')])
print(f'Total number of jpg files in test_frames/ Folder: {jpg_count3}')


from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
import os


data_dir_train = '/kaggle/working/training_frames'
data_dir_test = '/kaggle/working/test_frames'
img_size = (224, 224)


train_transform = transforms.Compose([
    transforms.Resize(img_size),           # Resize to 224x224 pixels
    transforms.RandomHorizontalFlip(),     # Randomly flip images left/right
    transforms.ToTensor(),                 # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(img_size),           # Resize for validation (no augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset_aug   = datasets.ImageFolder(data_dir_train, transform=train_transform)
train_dataset_plain = datasets.ImageFolder(data_dir_train, transform=val_transform)
print("Classes:", train_dataset_aug.classes, "Map:", train_dataset_aug.class_to_idx)
print("Done")


print(torch.__version__)


train_full_size = len(train_dataset_aug)
train_size = int(0.70 * train_full_size)
val_size   = train_full_size - train_size

# fixed permutation of indices
g = torch.Generator().manual_seed(42)
indices = torch.randperm(train_full_size, generator=g).tolist()
train_indices = indices[:train_size]
val_indices   = indices[train_size:]

train_subset = Subset(train_dataset_aug,   train_indices)   # augmentation
val_subset   = Subset(train_dataset_plain, val_indices)     # no augmentation
print(f"Train/Val sizes -> {len(train_subset)}/{len(val_subset)}")
print("Done")



batch_size = 32
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=2)
print("Done")


import torchvision.models as models
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(pretrained=True)                 # Use weights trained on millions of images
model.fc = nn.Linear(model.fc.in_features, 2)             # Output: 2 classes
model = model.to(device)                                  # Move model to the GPU


print ("Done")


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
print("Done")


from tqdm import tqdm

num_epochs = 10
best_acc = 0.0
patience = 3
no_improve = 0
best_path = "/kaggle/working/best_model.pt"

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_train_loss = running_loss / max(1, len(train_loader))

    # validation
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        vloop = tqdm(val_loader, total=len(val_loader), desc="Validation", unit="batch")
        for inputs, labels in vloop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = 100.0 * correct / total
    avg_val_loss = val_loss / max(1, len(val_loader))
    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_acc={val_acc:.2f}%")

    scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), best_path)
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

print(f"Best val_acc: {best_acc:.2f}% (saved to {best_path})")
print("Done")



# Load best weights for downstream test/inference
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()
print("Loaded best model.")

# Optional: export to ONNX
dummy = torch.randn(1, 3, 224, 224, device=device)
onnx_path = "/kaggle/working/model.onnx"
torch.onnx.export(model, dummy, onnx_path,
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}},
                  )
print(f"Exported ONNX to {onnx_path}")



from PIL import Image

class UnlabeledImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = [f for f in os.listdir(root) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff'))]
        self.files.sort()
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root, fname)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname

test_dataset_unlabeled = UnlabeledImageFolder(data_dir_test, transform=val_transform)
test_loader = DataLoader(test_dataset_unlabeled, batch_size=32, shuffle=False, num_workers=2)

pred_rows = []
with torch.no_grad():
    for imgs, names in tqdm(test_loader, total=len(test_loader), desc="Test Inference"):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1).cpu().tolist()
        confs = probs.max(1).values.cpu().tolist()
        for n, p, c in zip(names, preds, confs):
            pred_rows.append((n, p, c))

import pandas as pd
pd.DataFrame(pred_rows, columns=["filename","pred_label(0=real,1=fake)","confidence"]).to_csv("/kaggle/working/test_preds.csv", index=False)
print("Saved /kaggle/working/test_preds.csv")



# # Correct way to use f-string with torch.save filename
# torch.save(model.state_dict(), f'deepfake_resnet18ValAcc{val_acc:.2f}.pth')
# print(f'Model saved as deepfake_resnet18ValAcc{val_acc:.2f}.pth')




