import pandas as pd
import numpy as np
import os


root_path = '/kaggle/input/paddy-disease-classification/train_images'

image_paths = []
labels = []

for label in os.listdir(root_path):
    label_path = os.path.join(root_path, label)
    if os.path.isdir(label_path):
        for img_file in os.listdir(label_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(label_path, img_file))
                labels.append(label)

df = pd.DataFrame({'image_path': image_paths, 'label': labels})


df.head()


df.tail()


df.shape


df.columns


df.duplicated().sum()


df.isnull().sum()


df.info()


df['label'].unique()


df['label'].value_counts()


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(data=df, x="label", palette="viridis", ax=ax)

ax.set_title("Distribution Types", fontsize=14, fontweight='bold')
ax.set_xlabel("Embryo Type", fontsize=12)
ax.set_ylabel("Count", fontsize=12)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=11, color='black', 
                xytext=(0, 5), textcoords='offset points')

plt.xticks(rotation = 90)
plt.show()

label_counts = df["label"].value_counts()

fig, ax = plt.subplots(figsize=(10, 12))
colors = sns.color_palette("viridis", len(label_counts))

ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', 
       startangle=140, colors=colors, textprops={'fontsize': 12, 'weight': 'bold'},
       wedgeprops={'edgecolor': 'black', 'linewidth': 1})

ax.set_title("Distribution Types - Pie Chart", fontsize=14, fontweight='bold')

plt.show()


from PIL import Image

num_images = 5

unique_labels = df['label'].unique()

plt.figure(figsize=(15, len(unique_labels) * 3))

for row_idx, label in enumerate(unique_labels):
    
    label_images = df[df['label'] == label].head(num_images)['image_path'].tolist()
    
    for col_idx, img_path in enumerate(label_images):
        plt_idx = row_idx * num_images + col_idx + 1
        plt.subplot(len(unique_labels), num_images, plt_idx)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.axis('off')
        if col_idx == 2: 
            plt.title(label, fontsize=10)

plt.tight_layout()
plt.show()


from sklearn.utils import resample

balanced_data = []

max_count = df['label'].value_counts().max()

for label in df['label'].unique():
    df_label = df[df['label'] == label]
    df_resampled = resample(df_label,
                            replace=True,          
                            n_samples=max_count,   
                            random_state=42)
    balanced_data.append(df_resampled)

df_balanced = pd.concat(balanced_data).reset_index(drop=True)

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


df = df_balanced


df


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

class PaddyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(sorted(dataframe['label'].unique()))}
        self.label_map_inv = {idx: label for label, idx in self.label_map.items()}
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.label_map[label]
        return image, label_idx

class Involution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, reduction_ratio=2):
        super(Involution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.reduction_ratio = reduction_ratio
        
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, out_channels * kernel_size * kernel_size, 1)
        )
        
        self.padding = kernel_size // 2 
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=self.padding, stride=1)

    def forward(self, x):
        b, c_in, h, w = x.size()

        inv_kernels = self.kernel_gen(x)
        
        inv_kernels = inv_kernels.view(b, self.out_channels, self.kernel_size * self.kernel_size, h, w)
        inv_kernels = inv_kernels.permute(0, 3, 4, 1, 2)
        inv_kernels = inv_kernels.reshape(-1, self.out_channels, self.kernel_size * self.kernel_size)

        patches = self.unfold(x)
        
        patches = patches.view(b, self.in_channels, self.kernel_size * self.kernel_size, h * w)
        patches = patches.permute(0, 3, 2, 1)
        patches = patches.reshape(-1, self.kernel_size * self.kernel_size, self.in_channels)
        
        out_bmm = torch.bmm(inv_kernels, patches) 

        out = out_bmm.reshape(b, h, w, self.out_channels, self.in_channels)
        out = out.permute(0, 3, 1, 2, 4).sum(dim=-1)
        
        return out

class HybridDenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridDenseNet, self).__init__()
        self.densenet = torchvision.models.densenet169(weights='IMAGENET1K_V1')
        
        self.features_initial = nn.Sequential(
            self.densenet.features.conv0,
            self.densenet.features.norm0,
            self.densenet.features.relu0,
            self.densenet.features.pool0
        )
        
        self.involution1 = Involution(in_channels=128, out_channels=128)
        self.involution2 = Involution(in_channels=256, out_channels=256)
        self.involution3 = Involution(in_channels=640, out_channels=640)
        
        self.final_norm = self.densenet.features.norm5
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(1664, num_classes)
        
    def forward(self, x):
        x = self.features_initial(x)
        
        x = self.densenet.features.denseblock1(x)
        x = self.densenet.features.transition1(x)
        x = self.involution1(x) 
        
        x = self.densenet.features.denseblock2(x)
        x = self.densenet.features.transition2(x)
        x = self.involution2(x) 
        
        x = self.densenet.features.denseblock3(x)
        x = self.densenet.features.transition3(x)
        x = self.involution3(x) 
        
        x = self.densenet.features.denseblock4(x) 
        
        x = self.final_norm(x)
        x = torch.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_outputs, teacher_outputs, labels):
        soft_loss = self.kl_div(
            torch.log_softmax(student_outputs / self.temperature, dim=1),
            torch.softmax(teacher_outputs / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        hard_loss = self.ce_loss(student_outputs, labels)
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss

def prune_model(model, pruning_rate=0.1):
    masks = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'fc' not in name and 'bias' not in name:
            weights = param.data.cpu().numpy()
            threshold = np.percentile(np.abs(weights), pruning_rate * 100)
            mask = np.abs(weights) >= threshold
            param.data = torch.from_numpy(weights * mask).to(param.device)
            masks.append(mask)
    return model, masks

def train_model(model, teacher, train_loader, val_loader, optimizer, criterion, device, epochs=5): # Changed to 5 epochs
    model.train()
    teacher.eval()
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                teacher_outputs = teacher(inputs)
                outputs = model(inputs)
                loss = criterion(outputs, teacher_outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1} finished -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, dataloader, device, label_map_inv):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=[label_map_inv[i] for i in range(len(label_map_inv))])
    return accuracy, cm, report, all_labels, all_preds

def plot_history(train_losses, val_losses, train_accuracies, val_accuracies, title_prefix=''):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{title_prefix} Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = PaddyDataset(df, transform=transform)
    total_samples = len(dataset)
    train_size = 12348
    val_size = 2646
    test_size = 2646

    if (train_size + val_size + test_size) > total_samples:
        print(f"Warning: Dummy dataset size ({total_samples}) is less than sum of desired split sizes ({train_size + val_size + test_size}). Adjusting splits.")
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        test_size = total_samples - train_size - val_size
        print(f"Adjusted Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    num_classes = len(dataset.label_map.keys())
    print(f"Number of classes: {num_classes}")
    
    teacher_model = torchvision.models.densenet169(weights='IMAGENET1K_V1')
    teacher_model.classifier = nn.Linear(teacher_model.classifier.in_features, num_classes)
    teacher_model = teacher_model.to(device)
    print("Teacher model (DenseNet169) initialized.")
    
    student_model = HybridDenseNet(num_classes=num_classes).to(device)
    print("Student model (HybridDenseNet with Involution) initialized.")
    
    print("\n" + "="*30 + "\nTraining Teacher Model (5 Epochs)...\n" + "="*30)
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
    ce_loss = nn.CrossEntropyLoss()
    teacher_model.train()
    for epoch in range(5): 
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            teacher_optimizer.zero_grad()
            outputs = teacher_model(inputs)
            loss = ce_loss(outputs, labels)
            loss.backward()
            teacher_optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        print(f'Teacher Epoch {epoch+1} finished, Average Loss: {running_loss / len(train_loader):.4f}')
    
    print("\n" + "="*30 + "\nStarting Knowledge Distillation (5 Epochs)...\n" + "="*30)
    criterion = DistillationLoss(alpha=0.5, temperature=3.0)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    train_losses_kd, val_losses_kd, train_accuracies_kd, val_accuracies_kd = train_model(
        student_model, teacher_model, train_loader, val_loader, optimizer, criterion, device, epochs=5 # Changed to 5 epochs
    )
    
    plot_history(train_losses_kd, val_losses_kd, train_accuracies_kd, val_accuracies_kd, title_prefix='Knowledge Distillation')
    
    print("\n" + "="*30 + "\nEvaluating Student Model after Knowledge Distillation...\n" + "="*30)
    accuracy_kd, cm_kd, report_kd, _, _ = evaluate_model(student_model, test_loader, device, dataset.label_map_inv)
    print(f'\nAccuracy after Knowledge Distillation: {accuracy_kd:.2f}%')
    print('\nClassification Report after Knowledge Distillation:\n', report_kd)
    plot_confusion_matrix(cm_kd, list(dataset.label_map.keys()), 'Confusion Matrix after Knowledge Distillation')
    
    print("\n" + "="*30 + "\nStarting Weight Pruning (10%)...\n" + "="*30)
    student_model_pruned, masks = prune_model(student_model, pruning_rate=0.1)
    print("Model pruned.")
    
    print("\n" + "="*30 + "\nStarting Fine-tuning Pruned Model (5 Epochs)...\n" + "="*30)
    optimizer_ft = optim.Adam(student_model_pruned.parameters(), lr=0.0001)
    train_losses_ft, val_losses_ft, train_accuracies_ft, val_accuracies_ft = train_model(
        student_model_pruned, teacher_model, train_loader, val_loader, optimizer_ft, criterion, device, epochs=5 # Changed to 5 epochs
    )
    
    plot_history(train_losses_ft, val_losses_ft, train_accuracies_ft, val_accuracies_ft, title_prefix='Post-Pruning Fine-Tuning')
    
    print("\n" + "="*30 + "\nEvaluating Final Model after Pruning and Fine-tuning...\n" + "="*30)
    final_accuracy, final_cm, final_report, _, _ = evaluate_model(student_model_pruned, test_loader, device, dataset.label_map_inv)
    print(f'\nFinal Accuracy after Pruning and Fine-tuning: {final_accuracy:.2f}%')
    print('\nClassification Report after Pruning and Fine-tuning:\n', final_report)
    plot_confusion_matrix(final_cm, list(dataset.label_map.keys()), 'Confusion Matrix after Pruning and Fine-tuning')

if __name__ == '__main__':
    main()

