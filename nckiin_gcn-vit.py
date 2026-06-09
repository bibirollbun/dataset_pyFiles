!pip install torch_geometric


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



class CNN_GNN_ViT(nn.Module):
    def __init__(self, num_classes):
        super(CNN_GNN_ViT, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 256)

        self.conv1 = SAGEConv(256, 128)
        self.conv2 = SAGEConv(128, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.vit(x)  # Feature Extraction with Vision Transformer
        x = self.conv1(x, edge_index)  
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index) 
        return F.log_softmax(x, dim=1)



import os
import cv2
import torch
import random
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

class VideoIterableDataset(IterableDataset):
    def __init__(self, video_dir, transform=None, frame_interval=10, shuffle=True):
        super().__init__()
        self.video_dir = video_dir
        self.transform = transform
        self.frame_interval = frame_interval
        self.shuffle = shuffle

        # Tự động detect các lớp từ thư mục con
        self.classes = sorted(
            [d.name for d in os.scandir(video_dir) if d.is_dir()]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Lấy danh sách video và nhãn
        self.video_paths = []
        for class_name in self.classes:
            class_path = os.path.join(video_dir, class_name)
            for video_file in os.listdir(class_path):
                if video_file.lower().endswith((".mp4", ".avi", ".mov")):
                    self.video_paths.append(
                        (os.path.join(class_path, video_file), 
                        self.class_to_idx[class_name]
                    ))

        # Tính tổng số frame ước lượng (khởi tạo ban đầu)
        self._total_frames = self._estimate_total_frames()

    def _estimate_total_frames(self):
        """Ước lượng tổng số frame dựa trên frame_interval"""
        total = 0
        for video_path, _ in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total += frame_count // self.frame_interval
        return total

    def __len__(self):
        return self._total_frames  # Trả về số frame ước lượng

    def __iter__(self):
        # Phân chia video cho các worker (multi-processing)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            video_subset = self.video_paths
        else:
            per_worker = len(self.video_paths) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.video_paths)
            video_subset = self.video_paths[start:end]

        # Xáo trộn video subset
        if self.shuffle:
            random.shuffle(video_subset)

        # Generator để yield từng frame
        for video_path, label in video_subset:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_interval == 0:
                    # Chuyển đổi và augment frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    if self.transform:
                        frame = self.transform(frame)
                    yield frame, label
                
                frame_count += 1
            cap.release()


# Transform (giữ nguyên)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Khởi tạo dataset (không cần truyền tham số classes)
dataset = VideoIterableDataset(
    video_dir="/kaggle/input/datadoan/data/data",  # Thư mục cha chứa các thư mục lớp (class0, class1, ...)
    transform=transform,
    frame_interval=10,
    shuffle=True
)

# DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True
)
class_names = dataset.classes


# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# data_dir = "data"
# dataset = ImageFolder(root=data_dir, transform=transform)
# data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
# class_names = dataset.classes



def create_graph_data(features, labels=None):
    num_nodes = features.size(0)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    x = features.clone().detach()
    y = labels.clone().detach() if labels is not None else None
    return Data(x=x, edge_index=edge_index, y=y)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
# Building a Model Using Vision Transformer and GNN Layers
model = CNN_GNN_ViT(num_classes=len(class_names)).to(device)
# Loss Function and Optimization Settings with Focal Loss
criterion = FocalLoss(gamma=2, alpha=[0.25] * len(class_names))
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)



device.type


def train(epochs=20):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, batch in enumerate(data_loader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            data = create_graph_data(images, labels)
            data = data.to(device)

            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        scheduler.step()
        avg_loss = running_loss / len(data_loader)
        train_losses.append(avg_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Accuracy: {train_accuracy:.2f}%")

        # Validation Accuracy and Loss Calculation
        val_loss, val_accuracy = validate()
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    # Plotting graphs after training
    plot_training_graphs(train_losses, val_losses, train_accuracies, val_accuracies)

def validate():
    model.eval()
    correct_val = 0
    total_val = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            data = create_graph_data(images, labels)
            data = data.to(device)

            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, labels)
            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct_val / total_val
    return avg_loss, accuracy

def plot_roc_curve():
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            data = create_graph_data(images, labels)
            data = data.to(device)

            output = model(data.x, data.edge_index, data.batch)
            probs = F.softmax(output, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'Class {class_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix():
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            data = create_graph_data(images, labels)
            data = data.to(device)

            output = model(data.x, data.edge_index, data.batch)
            _, predicted = torch.max(output.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_training_graphs(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_accuracies, 'r', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

def calculate_metrics():
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            data = create_graph_data(images, labels)
            data = data.to(device)

            output = model(data.x, data.edge_index, data.batch)
            _, predicted = torch.max(output.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')

    print(f'F1 Score: {f1:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')

def calculate_accuracy():
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            data = create_graph_data(images)
            data = data.to(device)

            output = model(data.x, data.edge_index, data.batch)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')



train(epochs=20)


calculate_accuracy()
calculate_metrics()
plot_roc_curve()
plot_confusion_matrix()


model_path = "/content/drive/MyDrive/MPox/cnn_gnn_vit_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")























































