import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset,random_split
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision.models import ResNet34_Weights, ResNet50_Weights
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,ExponentialLR

train_csv = '/kaggle/input/classify-leaves/train.csv'
train_df = pd.read_csv(train_csv)
print(train_df.head())  # 打印前几行数据



# def augment_images(image_folder, train_csv, augmented_csv):
#     transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(30),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
    
#     train_df = pd.read_csv(train_csv)
#     image_paths = train_df['image'].values  # 获取图片路径
#     augmented_images = []  
#     for image_path in image_paths:
#         full_image_path = os.path.join(image_folder, image_path)
#         image = Image.open(full_image_path)
#         # 对每张图片进行多次数据增强
#         for i in range(5):  # 假设每张图片增强5次
#             augmented_image = transform(image)
#             augmented_image_name = f"{os.path.basename(image_path)}_augmented_{i}.jpg"
#             augmented_images.append([augmented_image_name, train_df[train_df['image'] == image_path]['label'].values[0]])
            
#             # 保存增强后的图片
#             augmented_image_path = os.path.join('/kaggle/working', augmented_image_name)
#             torchvision.utils.save_image(augmented_image, augmented_image_path)
    
#     # 将增强后的图片信息保存到新的CSV文件
#     augmented_df = pd.DataFrame(augmented_images, columns=['image', 'label'])
#     augmented_df.to_csv(augmented_csv, index=False)
#     #print(augmented_df.head())  # 打印增强后数据的前几行

train_df = pd.read_csv("/kaggle/input/classify-leaves/train.csv")
unique_labels = sorted(train_df['label'].unique())
label_to_int = {label: idx + 1 for idx, label in enumerate(unique_labels)}
max_label_value = max(label_to_int.values())
print(f"最大标签值: {max_label_value}")

class LeafDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_to_int = label_to_int  # 使用标签映射字典

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(f"Row {idx}: {self.data.iloc[idx]}")  # 打印当前行的内容
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.data.iloc[idx, 1]  # 确保第二列是标签
        label = self.label_to_int[label]
        if self.transform:
           image = self.transform(image)
        return image, label

def predict_test_data(model, test_csv, root_dir, transform, int_to_label, device):
    test_df = pd.read_csv(test_csv)
    test_image_paths = test_df['image'].values  # 假设 test.csv 中的列名为 'image'

    predictions = []
    model.eval()  # 确保 model 在这里已经定义
    with torch.no_grad():
        for image_path in tqdm(test_image_paths, desc="Predicting"):
            full_image_path = os.path.join(root_dir, image_path)
            image = Image.open(full_image_path).convert('RGB')
            if transform:
                image = transform(image)
            image = image.unsqueeze(0).to(device)

            outputs = model(image)  # 使用传递进来的 model
            _, predicted = torch.max(outputs, 1)
            predicted_label = int_to_label[predicted.item()]

            predictions.append(predicted_label)

    test_df['label'] = predictions
    output_csv = os.path.join('/kaggle/working', 'predictions.csv')
    test_df.to_csv(output_csv, index=False)
    print(f"预测结果已写回到 {output_csv} 文件中。")

def main():
    image_folder = '/kaggle/input/classify-leaves'
    train_csv = '/kaggle/input/classify-leaves/train.csv'
    test_csv = '/kaggle/input/classify-leaves/test.csv'
    sample_submission_csv = '/kaggle/input/classify-leaves/sample_submission.csv'
    augmented_csv = '/kaggle/working/augmented_train.csv'

    train_df = pd.read_csv(train_csv)
    unique_labels = sorted(train_df['label'].unique())
    label_to_int = {label: idx + 1 for idx, label in enumerate(unique_labels)}
    
    # 数据增强并保存到新的CSV文件
    # augment_images(image_folder, train_csv, augmented_csv)
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    

    # train_dataset = LeafDataset(csv_file=augmented_csv, root_dir='/home/bhy/test/classify-leaves/working', transform=data_transforms['train'])
    train_dataset = LeafDataset(csv_file=train_csv, root_dir='/kaggle/input/classify-leaves', transform=data_transforms['train'])
    total_size = len(train_dataset)
    test_size = int(0.2 * total_size)  # 20% 作为测试集
    train_size = total_size - test_size  # 剩下的作为训练集
    train_subset, test_subset = random_split(train_dataset, [train_size, test_size])
    
    # test_dataset = LeafDataset(csv_file=test_csv, root_dir=image_folder, transform=data_transforms['val'])
    # test_dataset = LeafDataset(csv_file=sample_submission_csv, root_dir=image_folder, transform=data_transforms['val'])
   
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    train_df = pd.read_csv("/kaggle/input/classify-leaves/train.csv")

    unique_labels = sorted(train_df['label'].unique())
    label_to_int = {label: idx + 1 for idx, label in enumerate(unique_labels)}
    max_label_value = max(label_to_int.values())
    num_classes = max_label_value
    model.fc = nn.Linear(model.fc.in_features, num_classes+1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.9,verbose=True)
    
    num_epochs = 30
    # num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # for inputs, labels in train_loader:
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
    
            if isinstance(labels, str):
               labels = torch.tensor(int(labels))  # 假设标签是字符串形式的整数，将其转换为张量
    
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    root_dir = '/kaggle/input/classify-leaves'
    int_to_label = {v: k for k, v in label_to_int.items()}  # 逆映射字典
    
    # 预测 test.csv 中的图片
    predict_test_data(model=model,
                      test_csv=test_csv,
                      root_dir=root_dir,
                      transform=data_transforms['val'],
                      int_to_label=int_to_label,
                      device=device)
    
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
       for inputs, labels in tqdm(test_loader, desc=f"testing"):
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            if isinstance(labels, tuple):
               labels = labels[0]  # 假设标签是元组的第一个元素
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()

