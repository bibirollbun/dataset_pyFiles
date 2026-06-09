import numpy as np 
import pandas as pd
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


base_path = '../input/alaska2-image-steganalysis'

def read_images_path(dir_name, label):
    folder_path = os.path.join(base_path, dir_name)  # Đường dẫn đến thư mục
    return [[os.path.join(folder_path, filename), label] for filename in os.listdir(folder_path)]


sample_size = 15000

cover_img = read_images_path('Cover', 0)[:sample_size]
jmipod_img = read_images_path('JMiPOD', 1)[:(sample_size//3)]
juniward_img = read_images_path('JUNIWARD', 1)[:(sample_size//3)]
uerd_img = read_images_path('UERD', 1)[:(sample_size//3)]

data = cover_img + jmipod_img + juniward_img + uerd_img
df = pd.DataFrame(data=data, columns=['path', 'label'])


def split_dataframe(df, ratio):
    # Xáo trộn dữ liệu để đảm bảo tính ngẫu nhiên
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Chia dữ liệu thành hai phần
    split_point = int(len(df) * ratio)
    return df.iloc[:split_point], df.iloc[split_point:]

train, val = split_dataframe(df, 0.8)


train


from PIL import Image
def preprocess_image(image_path, size=(384, 384), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    
    # Bước 1: Tải hình ảnh bằng PIL
    img = Image.open(image_path)
    
    # Bước 2: Chuyển đổi không gian màu sang YCbCr
    img = img.convert("YCbCr")
    
    # Bước 3: Resize hình ảnh về kích thước cố định
    img = img.resize(size)
    
    # Bước 4: Chuyển đổi thành numpy array và chuẩn hóa giá trị pixel về [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Đổi thứ tự kênh từ (H, W, C) sang (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Chuyển thành tensor PyTorch
    img_tensor = torch.tensor(img_array)
    
    # Chuẩn hóa từng kênh bằng mean và std
    for t, m, s in zip(img_tensor, mean, std):
        t.sub_(m).div_(s)
    
    return img_tensor

def load_image(path):
    img = preprocess_image(path)
    return img


from torchvision.models.efficientnet import efficientnet_b2
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

# Khởi tạo mô hình EfficientNet-B2 với trọng số pre-trained
efficientnet_model = efficientnet_b2(pretrained=True)

# Tùy chỉnh lớp cuối cùng của mô hình
num_classes = 2  # Số lượng lớp trong bài toán (phân loại nhị phân)
efficientnet_model.classifier[1] = torch.nn.Linear(efficientnet_model.classifier[1].in_features, num_classes)

# Đóng băng các tham số trong phần backbone
for name, param in efficientnet_model.named_parameters():
    if "classifier" not in name:  # Chỉ giữ nguyên các lớp không phải classifier
        param.requires_grad = False

# Đưa mô hình lên thiết bị
efficientnet_model.to(device)

# Chuyển sang chế độ đánh giá
efficientnet_model.eval()

print("EfficientNet-B2 model has been customized and is ready!")


from torch.utils.data import Dataset, DataLoader

class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Lấy đường dẫn ảnh và nhãn
        img_path = self.image_paths[index]
        label = self.labels[index]

        # Sử dụng hàm preprocess_image
        image = preprocess_image(img_path)

        return image, label

BATCHSIZE = 256
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 0
WEIGHTS = torch.tensor([4, 1], dtype=torch.float32).to(device)

# Sử dụng resnet_model thay vì model
optimizer = torch.optim.Adam(efficientnet_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Tạo dataset cho train và validation
train_dataset = SimpleImageDataset(train['path'].tolist(), train['label'].tolist())
val_dataset = SimpleImageDataset(val['path'].tolist(), val['label'].tolist())

# Tạo DataLoadera
train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=2, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True)


from sklearn.metrics import accuracy_score, roc_curve
def weighted_auc(y_true, y_score, tpr_thresholds=[0.0, 0.4, 1.0], weights=[2, 1]):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # Tính AUC cho từng vùng
    auc_scores = []
    for i in range(len(tpr_thresholds) - 1):
        start = tpr_thresholds[i]
        end = tpr_thresholds[i+1]
        
        # Lọc các điểm trong vùng hiện tại
        mask = (tpr >= start) & (tpr < end)
        auc_scores.append(np.trapz(tpr[mask], fpr[mask]))
    
    # Tính AUC có trọng số
    weighted_auc = np.sum(np.multiply(auc_scores, weights)) / np.sum(weights)
    
    return weighted_auc

def calculate_metrics(model, dataloader, device='cpu'):
    model.to(device).eval()
    labels, probs, preds = [], [], []

    with torch.no_grad():
        for images, batch_labels in dataloader:
            outputs = model(images.to(device))
            batch_probs = F.softmax(outputs, dim=1)[:, 1]
            batch_preds = torch.argmax(outputs, dim=1)

            labels.extend(batch_labels.cpu().numpy())
            probs.extend(batch_probs.cpu().numpy())
            preds.extend(batch_preds.cpu().numpy())

    labels = np.array(labels)
    probs = np.array(probs)
    preds = np.array(preds)

    return {
        'Weighted AUC': weighted_auc(labels, probs),
        'Accuracy': accuracy_score(labels, preds)
    }


from tqdm import tqdm
train_losses = []  # Store training loss per epoch
accuracies = []  # Store accuracy per epoch
weighted_aucs = []  # Store weighted AUC per epoch
best_auc = 0

for epoch in range(EPOCHS):
    print(f'EPOCH: {epoch+1}')
    
    # Training phase
    efficientnet_model.train()
    train_loss = 0
    for images, labels in tqdm(train_dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        loss = F.cross_entropy(efficientnet_model(images), labels, weight=WEIGHTS)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    print(f'Average Training Loss: {avg_train_loss:.4f}')
    
    # Validation phase
    metrics = calculate_metrics(efficientnet_model, val_dataloader, device)
    acc = metrics['Accuracy']
    current_auc = metrics['Weighted AUC']
    
    accuracies.append(acc)
    weighted_aucs.append(current_auc)
                        
    print(f'Accuracy: {acc:.4f}')
    print(f'Weighted AUC: {current_auc:.4f}')
    
    # Save best model
    if current_auc > best_auc:
        best_auc = current_auc
        torch.save(efficientnet_model.state_dict(), 'best_model.pth')
        print(f"New best model saved with Weighted AUC: {best_auc:.4f}")
    
    print()

# Load best model
efficientnet_model.load_state_dict(torch.load('best_model.pth'))
print("Best model loaded!")


# Plot Training Metrics
plt.figure(figsize=(12, 4))

# Plot Loss
plt.subplot(1, 3, 1)
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

# Plot Accuracy
plt.subplot(1, 3, 2)
plt.plot(range(1, EPOCHS+1), accuracies, marker='o', label='Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()

# Plot Weighted AUC
plt.subplot(1, 3, 3)
plt.plot(range(1, EPOCHS+1), weighted_aucs, marker='o', label='Weighted AUC', color='red')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Weighted AUC per Epoch')
plt.legend()

plt.tight_layout()
plt.show()


test_img = read_images_path('Test', 0.7)
test = pd.DataFrame(data=test_img, columns=['path', 'label'])


def predict(model, test_dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in test_dataloader:
            images = images.to(device)
            outputs = model(images)
            scores = F.softmax(outputs, dim=1)[:, 1]  # Probability of being a stego image
            predictions.extend(scores.cpu().numpy())

    return predictions


test_dataset = SimpleImageDataset(test['path'].tolist(), test['label'].tolist())

test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Predicting...")
test_predictions = predict(efficientnet_model, test_dataloader, device)
print("Finish.")




test_image_ids = [os.path.basename(path) for path in test_dataloader.dataset.image_paths]
submission_df = pd.DataFrame({
    'Id': test_image_ids,
    'Label': test_predictions
})

submission_df['Id'] = submission_df['Id'].str.replace('.jpg', '').astype(int)
submission_df = submission_df.sort_values(by='Id')
submission_df['Id'] = submission_df['Id'].astype(str).str.zfill(4) + '.jpg'

# Save submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
submission_df

