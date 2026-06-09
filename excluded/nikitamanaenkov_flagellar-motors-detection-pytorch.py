import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam


TRAIN_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train"
TEST_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test"
LABELS_PATH = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv"
SUBMIT_PATH = "/kaggle/working/submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class TomogramDataset(Dataset):
    def __init__(self, root_dir, labels_df=None, transform=None, mode='train', target_shape=(64, 128, 128)):
        self.root_dir = root_dir
        self.tomo_dirs = sorted(os.listdir(root_dir))
        self.transform = transform
        self.mode = mode
        self.labels_df = labels_df
        self.target_shape = target_shape

        self.label_dict = {}
        if labels_df is not None:
            for _, row in labels_df.iterrows():
                self.label_dict.setdefault(row['tomo_id'], []).append(
                    (row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2'])
                )

    def __len__(self):
        return len(self.tomo_dirs)

    def __getitem__(self, idx):
        tomo_id = self.tomo_dirs[idx]
        tomo_path = os.path.join(self.root_dir, tomo_id)

        slice_files = sorted(os.listdir(tomo_path))
        slices = []
        for f in slice_files:
            try:
                img = Image.open(os.path.join(tomo_path, f)).convert("L")
                slices.append(np.array(img, dtype=np.float32))
            except Exception as e:
                print(f"Error reading slice {f} in {tomo_id}: {e}")
                continue

        if len(slices) == 0:
            raise ValueError(f"No valid slices found for {tomo_id}")

        volume = np.stack(slices)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-5)
        volume = torch.tensor(volume).unsqueeze(0)

        volume = F.interpolate(volume.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)

        if self.mode == 'train':
            label = self.label_dict.get(tomo_id, [(-1, -1, -1)])
            orig_shape = (len(slice_files), volume.shape[1], volume.shape[2])
            scale = [self.target_shape[i] / orig_shape[i] for i in range(3)]
            target = torch.tensor([
                label[0][0] * scale[0],
                label[0][1] * scale[1],
                label[0][2] * scale[2]
            ], dtype=torch.float32)
            return volume, target
        else:
            return volume, tomo_id



class MotorNet(nn.Module):
    def __init__(self):
        super(MotorNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)  

        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)  

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool3d((2, 4, 4))  

        self.fc1 = nn.Linear(32 * 2 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)



def main():
    labels_df = pd.read_csv(LABELS_PATH)

    train_dataset = TomogramDataset(TRAIN_DIR, labels_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

    test_dataset = TomogramDataset(TEST_DIR, mode='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = MotorNet().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for volume, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            volume, target = volume.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(volume)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "flagellar_motors_model.pth")

    model.eval()
    predictions = []
    with torch.no_grad():
        for volume, tomo_id in tqdm(test_loader, desc="Predicting"):
            volume = volume.to(device)
            output = model(volume).cpu().numpy()[0]
            output = [int(v) if 0 <= v <= 512 else -1 for v in output]
            predictions.append([tomo_id[0]] + output)

    submission_df = pd.DataFrame(predictions, columns=["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
    submission_df.to_csv(SUBMIT_PATH, index=False)
    print(submission_df.head())

if __name__ == '__main__':
    main()

