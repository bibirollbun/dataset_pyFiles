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
import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch import nn, optim
from PIL import Image
data_path = "/kaggle/input/deepfake-detection-challenge/train_sample_videos"
metadata_path = os.path.join(data_path, "metadata.json")
print("Các file có sẵn:")
print(os.listdir(data_path)[:10])  # Hiển thị 10 file đầu tiên
import json

with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Chuyển metadata thành DataFrame
df = pd.DataFrame(metadata).T
df = df.reset_index().rename(columns={'index': 'filename'})

# Hiển thị dữ liệu
print(df.head())
def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển ảnh sang RGB
            frame = Image.fromarray(frame)
            frames.append(frame)

    cap.release()
    return frames
sample_video = os.path.join(data_path, df.iloc[0]["filename"])
sample_frames = extract_frames(sample_video, num_frames=5)

import matplotlib.pyplot as plt

# Hiển thị ảnh đầu tiên trong danh sách khung hình
plt.imshow(sample_frames[0])
plt.axis("off")  # Ẩn trục
plt.show()
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
class DeepfakeDataset(Dataset):
    def __init__(self, df, data_path, num_frames=5, transform=None):
        self.df = df
        self.data_path = data_path
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_name = self.df.iloc[idx]["filename"]
        label = 1 if self.df.iloc[idx]["label"] == "FAKE" else 0
        video_path = os.path.join(self.data_path, video_name)

        frames = extract_frames(video_path, num_frames=self.num_frames)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        return torch.stack(frames), torch.tensor(label, dtype=torch.long)
dataset = DeepfakeDataset(df, data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Kiểm tra batch đầu tiên
for images, labels in dataloader:
    print("Batch size:", images.shape)  # [batch, num_frames, 3, 128, 128]
    print("Label:", labels)
    break
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0

    for images, labels in dataloader:
        images = images[:, 0, :, :, :].to(device)  # Chỉ lấy frame đầu tiên
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = discriminator(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}") 


import matplotlib.pyplot as plt

def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    output = discriminator(image).item()
    return "FAKE" if output > 0.5 else "REAL"

# Lấy video test
test_video = os.path.join(data_path, df.iloc[5]["filename"])
test_frames = extract_frames(test_video, num_frames=1)

# Kiểm tra nếu có frame thì dự đoán và hiển thị ảnh
if len(test_frames) > 0:
    prediction = predict(test_frames[0])
    print("Kết quả dự đoán:", prediction)

    # Hiển thị ảnh
    plt.imshow(test_frames[0])
    plt.axis("off")  # Ẩn trục
    plt.title(f"Dự đoán: {prediction}")  # Hiển thị nhãn dự đoán trên ảnh
    plt.show()
else:
    print("Lỗi: Không lấy được frame từ video.")



#Lấy nhãn thực tế của video test
actual_label = df[df["filename"] == os.path.basename(test_video)]["label"].values[0]

# So sánh với kết quả dự đoán
print(f"Kết quả dự đoán: {prediction}")
print(f"Nhãn thực tế: {actual_label}")

# Hiển thị ảnh kèm theo nhãn thực tế
plt.imshow(test_frames[0])
plt.axis("off")  
plt.title(f"Dự đoán: {prediction} | Thực tế: {actual_label}")
plt.show()


correct = 0
total = 0

with torch.no_grad():  # Không tính gradient khi kiểm tra
    for images, labels in dataloader:
        images = images[:, 0, :, :, :].to(device)  # Chỉ lấy frame đầu tiên
        labels = labels.to(device)

        outputs = discriminator(images).squeeze()
        predictions = (outputs > 0.5).long()

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"Độ chính xác trên tập validation: {accuracy:.2f}%")


