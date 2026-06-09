import os
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image

# --- ラベルマッピング ---
label_map = {
    'freshapples': 1,
    'freshoranges': 2,
    'freshpotato': 3,
    'freshtomato': 4,
    'rottenapples': 5,
    'rottenoranges': 6,
    'rottenpotato': 7,
    'rottentomato': 8
}

# --- データ変換 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- データセットの読み込み ---
train_dataset = datasets.ImageFolder('/kaggle/input/responsi-komputasi-statistika-ii/Dataset-Responsi/Train', transform=transform)
val_dataset = datasets.ImageFolder('/kaggle/input/responsi-komputasi-statistika-ii/Dataset-Responsi/Validation', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- モデル定義（ResNet18） ---
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 8)  # 出力クラス数8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# --- 学習設定 ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 学習ループ ---
for epoch in range(10):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# --- 推論用画像ロード関数 ---
def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    return transform(image).unsqueeze(0)

# --- インデックスからクラス名へのマッピング ---
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# --- sample_submission.csv の読み込み ---
sample_df = pd.read_csv('/kaggle/input/responsi-komputasi-statistika-ii/Dataset-Responsi/sample_submission.csv')

# --- テスト画像ディレクトリ ---
test_dir = '/kaggle/input/responsi-komputasi-statistika-ii/Dataset-Responsi/Test'

# --- モデル推論 ---
model.eval()
predicted_labels = []

with torch.no_grad():
    for img_name in sample_df['image']:
        img_path = os.path.join(test_dir, img_name)
        input_tensor = load_image(img_path).to(device)
        output = model(input_tensor)
        pred_class_idx = output.argmax(dim=1).item()

        # index → class_name → label_code に変換
        class_name = idx_to_class[pred_class_idx]
        label_code = label_map[class_name]
        predicted_labels.append(label_code)

# --- submission.csv 出力 ---
sample_df['label'] = predicted_labels
sample_df.columns = ['image', 'label']
sample_df.to_csv('submission.csv', index=False)
print("✅ 正しく submission.csv を保存しました。")

