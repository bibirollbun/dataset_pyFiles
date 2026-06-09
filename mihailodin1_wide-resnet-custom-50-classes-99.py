import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, models
import torchvision.transforms.functional as F
import torch.nn.functional as Func

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

import os
import gc
import shutil
import random
from tqdm import tqdm

from PIL import Image

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({
    'figure.figsize': (16, 8),    
    'figure.facecolor': 'white',    
    'figure.autolayout': True,     
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()
batch_size = 32


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if device:
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sorted_classes(path):
    folders = os.listdir(path)
    sorted_folders = sorted(folders, key=lambda x: int(x.split('_')[-1]))
    return sorted_folders


data_dir = "/kaggle/input/butterflies-classification/train_butterflies/train_split"
class_names = sorted_classes(data_dir)


class_counts = {
    class_name: len(os.listdir(os.path.join(data_dir, class_name)))
    for class_name in class_names
}

df = pd.DataFrame(
    list(class_counts.items()),
    columns=["class", "count"]
)

sns.barplot(data=df, x="class", y="count", palette="viridis")
plt.xticks(rotation=45)
plt.title("Распределение изображений по классам", pad=20)
plt.xlabel("class")
plt.ylabel("count")

mean_val = df["count"].mean()
plt.axhline(mean_val, color="red", linestyle="--")

plt.text(
    len(df)+0.5,
    mean_val,
    f"Mean: {mean_val:.0f}", 
    va="center",
    ha="left",
    color="red",
    backgroundcolor="white"
)

plt.show()


def show_images(data_dir, classes, images_per_row=5):
    num_classes = len(classes)
    num_rows = (num_classes + images_per_row - 1) // images_per_row
    
    for i, cls in enumerate(classes):
        ax = plt.subplot(num_rows, images_per_row, i + 1)
        cls_dir = os.path.join(data_dir, cls)
        image = random.choice(os.listdir(cls_dir))
            
        img_path = os.path.join(cls_dir, image)
        img = Image.open(img_path)
        
        plt.imshow(img)
        plt.axis('off')
        plt.annotate(
            cls,
            xy=(0.05, 0.9),             
            xycoords='axes fraction',    
            fontsize=12,
            fontweight='bold',
            color='black',            
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )  
    plt.show()


show_images(
    data_dir=data_dir,
    classes=class_names,
    images_per_row=5
)


widths, heights, channels = [], [], []

for cls in os.listdir(data_dir):
    cls_path = os.path.join(data_dir, cls)
    for img_name in os.listdir(cls_path)[:30]:
        img = Image.open(os.path.join(cls_path, img_name))
        w, h = img.size
        widths.append(w)
        heights.append(h)
        channels.append(len(img.getbands()))

fig, (ax1, ax2) = plt.subplots(2, 1)

sns.histplot(widths, bins=20, ax=ax1, kde=True)
ax1.set_title(f"Распределение ширины")

sns.histplot(heights, bins=20, ax=ax2, kde=True)
ax2.set_title(f"Распределение высоты")

plt.show()

print(f"Уникальные каналы: {set(channels)}")


def load_dataset(class_names, data_dir, train_dir, val_dir, test_dir, train_number=0.7, val_number=0.15, shuffle=False):
    for dir_name in [train_dir, val_dir, test_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)

    for class_name in tqdm(class_names):
        source_dir = os.path.join(data_dir, class_name)
        class_files = os.listdir(source_dir)
        
        train_size = int(np.ceil(len(class_files) * train_number))
        val_size = int(np.ceil(len(class_files) * val_number))
        
        if shuffle:
            random.shuffle(class_files)

        train_files = class_files[:train_size]
        val_files = class_files[train_size:train_size + val_size]
        test_files = class_files[train_size + val_size:]
        
        for sample, dir_path in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
            for file in sample:
                original_image_path = os.path.join(source_dir, file)
                save_path = os.path.join(dir_path, class_name, file)
                shutil.copy(original_image_path, save_path)


train_dir = "train"
val_dir = "val"
test_dir = "test"

load_dataset(class_names, data_dir, train_dir, val_dir, test_dir, shuffle=True)


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

normalize = transforms.Normalize(
    mean=mean,
    std=std
)

affine = transforms.RandomAffine(
    degrees=(-30, 30), 
    translate=(0.1, 0.1),
)

flip = transforms.RandomOrder([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

brightness = transforms.ColorJitter(brightness=0.2)

contrast = transforms.ColorJitter(contrast=0.1)

add_gaussian_noise = transforms.Lambda(
    lambda img: torch.randn_like(img)*0.1 + img
)

shift = transforms.RandomAffine(0, translate=(0.3, 0.3))

grayscale = transforms.RandomGrayscale(p=1)

to_tensor =  transforms.ToTensor()

def add_random_shadows(img):
    if not isinstance(img, torch.Tensor):
        img = F.to_tensor(img)
    
    h, w = img.shape[1], img.shape[2]
    shadow_mask = torch.ones((h, w))
    
    for _ in range(random.randint(1, 2)):
        x_center = random.randint(0, w)
        y_center = random.randint(0, h)
        radius = random.randint(50, min(h, w) // 2)
        
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist = ((x - x_center) ** 2 + (y - y_center) ** 2).float().sqrt()
        shadow = torch.clamp(1 - (dist / radius), 0, 1)
        shadow_mask = torch.min(shadow_mask, shadow)
    
    img = img * shadow_mask.unsqueeze(0)
    return F.to_pil_image(img)


train_transforms = [
    # Аффинные преобразования + нормализация
    transforms.Compose([
        affine,
        to_tensor,
        normalize,
    ]),
    
    # Только базовая нормализация
    transforms.Compose([
        to_tensor,
        normalize,
    ]),
    
    # Коррекция контраста + шум
    transforms.Compose([
        contrast,
        to_tensor,
        add_gaussian_noise,
        normalize,
    ]),

    # Поворот + искажение
    transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        to_tensor,
        add_gaussian_noise,
        normalize,
    ]),
    
    # Отражения + шум
    transforms.Compose([
        flip,
        to_tensor,
        add_gaussian_noise,
        normalize,
    ]),
    
    # Сдвиг + шум
    transforms.Compose([
        shift,
        to_tensor,
        add_gaussian_noise,
        normalize,
    ]),

    # Размытие
    transforms.Compose([
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  
        add_random_shadows,
        to_tensor,
        normalize,
    ]),
    
    # Яркость + контраст
    transforms.Compose([
        brightness,
        contrast,
        to_tensor,
        normalize,
    ]),
]

val_transforms = [
    transforms.Compose([
        to_tensor,
        normalize,
    ])
]


class ImageFolderWithPaths(ImageFolder):
    def __init__(self, root, transform=None):
        classes = sorted_classes(root)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        super().__init__(root, transform=transform)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = self.make_dataset(
            self.root, class_to_idx, extensions=self.extensions, is_valid_file=None
        )
        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


train_dataset = ConcatDataset([
    ImageFolderWithPaths(train_dir, train_transform)
    for train_transform in train_transforms
])

val_dataset = ConcatDataset([
    ImageFolderWithPaths(val_dir, val_transform)
    for val_transform in val_transforms
])

test_dataset = ConcatDataset([
    ImageFolderWithPaths(test_dir,transform)
    for transform in val_transforms
])

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True
)

test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
)


def show_batch_with_paths(dataloader, class_names, n_images=4):
    batch = next(iter(dataloader))
    images, labels, paths = batch[:3]
    
    for i in range(min(n_images, len(images))):
        plt.subplot(1, n_images, i+1)
        
        img = images[i].permute(1, 2, 0).numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        true_class = os.path.basename(os.path.dirname(paths[i]))
        
        plt.imshow(img)
        plt.title(f"True: {true_class}\nLabel: {class_names[labels[i]]}\n{os.path.basename(paths[i])}")
        plt.axis('off')
    plt.show()


show_batch_with_paths(train_dataloader, class_names)


print(f"Train size : {len(train_dataloader)} \nVal size : {len(val_dataloader)}")


X_batch, y_batch, _ = next(iter(train_dataloader))
plt.imshow(X_batch[0].permute(1, 2, 0).numpy() * std + mean);


class BasicBlock(nn.Module):
    def __init__(self, inp, out, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        
        if downsample or inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out)
            )
        
        self.bn1 = nn.BatchNorm2d(out)
        self.bn2 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class Bottleneck(nn.Module):
    def __init__(self, inp, out, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(inp, out//4, 1, bias=False)
        self.conv2 = nn.Conv2d(out//4, out//4, 3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out//4, out, 1, bias=False)
        self.shortcut = nn.Sequential()
        
        if downsample or inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out)
            )
        
        self.bn1 = nn.BatchNorm2d(out//4)
        self.bn2 = nn.BatchNorm2d(out//4)
        self.bn3 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)


class NN(nn.Module):
    def __init__(self, outputs_number=50):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),  # (batch_size, 32, 224, 224)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)  # (batch_size, 32, 112, 112)
        )
        
        self.layers = nn.Sequential(
            BasicBlock(32, 64, downsample=True),   # (batch_size, 64, 56, 56)
            BasicBlock(64, 64),                    # (batch_size, 64, 56, 56)
            
            Bottleneck(64, 128, downsample=True),  # (batch_size, 128, 28, 28) 
            Bottleneck(128, 128),                  # )batch_size, 128, 28, 28)
            
            BasicBlock(128, 256, downsample=True), # (batch_size, 256, 14, 14)
            BasicBlock(256, 256),                  # (batch_size, 256, 14, 14)
            
            Bottleneck(256, 512, downsample=True), # (batch_size, 512, 7, 7)
            Bottleneck(512, 512)                   # (batch_size, 512, 7, 7)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)     # (batch_size, 512, 1, 1)
        self.fc = nn.Linear(512, outputs_number)      # (batch_size, 50)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class WRN(torch.nn.Module):
    def __init__(self, outputs_number):
        super(WRN, self).__init__()
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        
        for param in self.net.parameters():
            param.requires_grad = False
        
        for param in self.net.layer4.parameters():
            param.requires_grad = True

        TransferModelOutputs = self.net.fc.in_features
        self.net.fc = torch.nn.Sequential(
            torch.nn.Linear(TransferModelOutputs, 256, bias=False),
            torch.nn.BatchNorm1d(256), 
            torch.nn.ReLU(),  
            torch.nn.Dropout(0.5),  
            torch.nn.Linear(256, outputs_number) 
        )

    def forward(self, x):
        return self.net(x)


model = WRN(50)
# model = NN(50)
model = model.to(device)

if gpu_count > 1:
    model = nn.DataParallel(model)

loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


def train_model(model, train_dataloader, val_dataloader, loss, optimizer, scheduler, num_epochs, save_path='best_model.pth'):
    best_val_loss = float('inf')
    best_val_acc = 0
    best_model_weights = None
    
    train_loss = torch.zeros(num_epochs)
    val_loss = torch.zeros(num_epochs)
    train_acc = torch.zeros(num_epochs)
    val_acc = torch.zeros(num_epochs)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}:')

        for phase in ['train', 'val']:
            dataloader = train_dataloader if phase == 'train' else val_dataloader
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels, _ in tqdm(dataloader, desc=phase, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    preds_class = preds.argmax(dim=1)

                    loss_value = loss(preds, labels)
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                running_loss += loss_value.item() * inputs.size(0)
                running_corrects += (preds_class == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total

            if phase == 'train':
                train_loss[epoch] = epoch_loss
                train_acc[epoch] = epoch_acc
            else:
                val_loss[epoch] = epoch_loss
                val_acc[epoch] = epoch_acc
            
                scheduler.step(epoch_loss)

                if best_val_acc < epoch_acc:
                    best_val_loss = epoch_loss
                    best_val_acc = epoch_acc
                    if isinstance(model, nn.DataParallel):
                        best_model_weights = model.module.state_dict()
                    else:
                        best_model_weights = model.state_dict()

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    if best_model_weights is not None:
        torch.save(best_model_weights, save_path)
        print(f'Best model saved to {save_path} with loss: {best_val_loss:.4f}, acc: {best_val_acc:.4f}')

    return train_loss, val_loss, train_acc, val_acc



train_loss, val_loss, train_acc, val_acc = train_model(
    model, train_dataloader, val_dataloader, loss, optimizer, scheduler, num_epochs=1, save_path="wrn_35.pth"
)


def plot_training_curves(train_loss, val_loss, train_acc, val_acc, save_path='training_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(train_loss, label='Train Loss', linewidth=2)
    axes[0].plot(val_loss, label='Val Loss', linewidth=2)
    axes[0].set_title('Loss per Epoch', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10)

    axes[1].plot(train_acc, label='Train Accuracy', linewidth=2)
    axes[1].plot(val_acc, label='Val Accuracy', linewidth=2)
    axes[1].set_title('Accuracy per Epoch', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(loc='lower right', fontsize=10)

    plt.savefig(save_path, dpi=150)
    plt.show()


plot_training_curves(train_loss.numpy(), val_loss.numpy(), train_acc.numpy(), val_acc.numpy())


def evaluate_model(model, dataloader):
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='micro')
    recall = recall_score(all_labels, all_preds, average='micro')
    f1 = f1_score(all_labels, all_preds, average='micro')

    print(f'Accuracy : {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall   : {recall:.4f}')
    print(f'F1-score : {f1:.4f}')

    return all_labels, all_preds

def plot_confusion_matrix(labels, preds, class_names, save_path='confmatrix.png'):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(16, 14))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)

    plt.title('Confusion Matrix', fontsize=18)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)

    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)

    plt.savefig(save_path, dpi=150)
    plt.show()

    return cm


all_labels, all_preds = evaluate_model(model, val_dataloader)


def plot_roc_curves(model, dataloader, class_names):
    model.eval()
    
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels, _ in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    y_true = label_binarize(all_labels, classes=range(len(class_names)))
    n_classes = y_true.shape[1]

    plt.figure(figsize=(12, 10))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves')
    plt.legend(loc='lower right', fontsize='small', ncol=2)
    plt.savefig('roc_curves.png', dpi=150)
    plt.show()


plot_roc_curves(model, val_dataloader, class_names)


cm = plot_confusion_matrix(all_labels, all_preds, class_names)


def confused_classes(cm, top_n=6):
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    confused_pairs = []

    for _ in range(top_n):
        idx = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
        confused_pairs.append(idx)
        cm_no_diag[idx] = 0
        cm_no_diag[idx[1], idx[0]] = 0 
    
    return sorted(set(i for pair in confused_pairs for i in pair))

def filter_dataset_by_classes(dataset, target_classes):
    indices = [i for i, (_, label, _) in enumerate(dataset)
               if label in target_classes]
    return Subset(dataset, indices)


def finetune(model, val_dataset, confused_classes, num_epochs=1, lr=1e-5):
    subset = filter_dataset_by_classes(val_dataset, confused_classes)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels, _ in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch {epoch} Loss: {running_loss/total:.4f}, Acc: {correct/total:.4f}")



top_confused_classes = confused_classes(cm, top_n=6)
finetune(model, val_dataset, top_confused_classes, num_epochs=6)


all_labels, all_preds = evaluate_model(model, test_dataloader)


def inference(model, folder_path, output_file='predictions.csv'):
    model.eval()
    image_files = [f for f in os.listdir(folder_path)]
    image_files.sort()
    results = []
    
    with torch.no_grad():
        for filename in tqdm(image_files):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            img_tensor = val_transforms[0](img).unsqueeze(0).to(device)
            
            preds = model(img_tensor)
            probs = torch.softmax(preds, dim=1)
            pred_class = probs.argmax(dim=1).item()
            
            index = os.path.splitext(filename)[0]
            results.append({'index': index, 'label': pred_class})
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, header=True)


# inference(model, "/kaggle/input/butterflies-classification/test_butterflies/valid")


class EnsembleModel(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        
        probs1 = Func.softmax(logits1, dim=1)
        probs2 = Func.softmax(logits2, dim=1)
        
        avg_probs = (probs1 + probs2) / 2
        
        return avg_probs


model1 = NN(50)
model2 = WRN(50)

model2.load_state_dict(torch.load('/kaggle/working/wrn_35.pth'))
model1.load_state_dict(torch.load('/kaggle/input/models/best_model.pth'))

ensemble_model = EnsembleModel(model1, model2).to(device)


# inference(ensemble_model, "/kaggle/input/butterflies-classification/test_butterflies/valid", output_file='ensemble_predictions.csv')


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
clear_gpu_memory()


