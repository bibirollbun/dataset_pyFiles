# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import clear_output



def apply_edge_filter(pil_img, filter_type="sobel"):
    """
    PIL画像を指定したエッジフィルタで処理する
    :param pil_img: PIL.Image (グレースケール前提)
    :param filter_type: "sobel", "laplacian", "canny", None
    :return: PIL.Image
    """
    img = np.array(pil_img)

    if filter_type == "sobel":
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge_img = cv2.magnitude(sobelx, sobely)
    elif filter_type == "laplacian":
        edge_img = cv2.Laplacian(img, cv2.CV_64F)
    elif filter_type == "canny":
        edge_img = cv2.Canny(img, 100, 200)
    elif filter_type is None:
        return pil_img
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    edge_img = np.uint8(np.clip(edge_img, 0, 255))  # 0-255にクリップ
    return Image.fromarray(edge_img)



class ImageDataset(Dataset):
    def __init__(self, data_list, transform=None, filter_type=None):  # ← 修正
        self.data_list = data_list
        self.transform = transform
        self.filter_type = filter_type

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        img = Image.open(img_path).convert("L")  # グレースケール
        img = apply_edge_filter(img, self.filter_type)  # ← フィルタ処理
        if self.transform:
            img = self.transform(img)
        return img, label



class NasConfig:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_dir = os.path.join(root_dir, "train")
        self.test_dir = os.path.join(root_dir, "samples")
        self.cls_map = {"dog": 0, "cat": 1}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_data()
        self._split_data()
        self.train_transform = self.valid_transform = None

    def _load_data(self):
        self.train_datas = [[os.path.join(self.train_dir, f), self.cls_map[f.split(".")[0]]]
                            for f in os.listdir(self.train_dir) if f.endswith(".jpg")]
        self.test_datas = [[os.path.join(self.test_dir, f), int(f.split(".")[0])]
                           for f in os.listdir(self.test_dir) if f.endswith(".jpg")]

    def _split_data(self, ratio=0.1):
        self.train_set, self.valid_set = train_test_split(
            self.train_datas, test_size=ratio, shuffle=True, random_state=42)

    def set_transforms(self, train_tf, valid_tf):
        self.train_transform = train_tf
        self.valid_transform = valid_tf

    def get_dataloaders(self, batch_size=32, filter_type=None):
        train_dataset = ImageDataset(self.train_set, self.train_transform, filter_type)
        valid_dataset = ImageDataset(self.valid_set, self.valid_transform, filter_type)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader




class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # → 64x64
        x = self.pool(F.relu(self.conv2(x)))  # → 32x32
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# パスと設定
data_root = "/kaggle/input/vc-master-24-2-dogs-vs-cats"
batch_size = 64
epochs = 5
learning_rate = 0.001

# データ準備
config = NasConfig(data_root)
train_tf = transforms.Compose([transforms.Resize((128, 128)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()])
valid_tf = transforms.Compose([transforms.Resize((128, 128)),
                               transforms.ToTensor()])
config.set_transforms(train_tf, valid_tf)
train_loader, valid_loader = config.get_dataloaders(batch_size)



device = config.device
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def calculate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


data_root = "/kaggle/input/vc-master-24-2-dogs-vs-cats"
config = NasConfig(data_root)

# 2. Transform を設定
train_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
valid_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
config.set_transforms(train_tf, valid_tf)


# 3. DataLoader の取得
filter_type = "sobel"  # "laplacian", "canny", None 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)




# モデルを複数GPUに並列化
model = SimpleCNN().to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)



train_losses = []
valid_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 精度を評価
    train_acc = calculate_accuracy(model, train_loader)
    valid_acc = calculate_accuracy(model, valid_loader)

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    valid_accuracies.append(valid_acc)

    # 出力とグラフ表示
    clear_output(wait=True)  # 前の出力を消去（Kaggle/Jupyter用）
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}")
    
    # グラフ表示
    plt.figure(figsize=(10,4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label='Valid Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()



# --- 3. モデル推論 ---
model.eval()
all_preds = []
all_ids = []

with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(config.device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_ids.extend(ids.numpy() if torch.is_tensor(ids) else ids)

# --- 4. 整合性チェック ---
assert len(all_ids) == len(all_preds), f"IDと予測数が合いません: {len(all_ids)} vs {len(all_preds)}"

# --- 5. 予測ラベルを文字列に変換 ---
pred_labels = [config.index_map[p] for p in all_preds]



# --- 6. 提出用CSV作成 ---
submission = pd.DataFrame({
    "id": all_ids,
    "label": pred_labels
})

submission.to_csv("submission.csv", index=False)

print("submission.csv を出力しました")

