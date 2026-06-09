import pandas as pd
import os
import zipfile
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn.functional as F


from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import copy  # For deep copy in early stopping


index_to_labels_mapping = {
    0: "No Cancer",
    1: "Cancer"
}


train_images_root_dir = '/kaggle/input/histopathologic-cancer-detection/train'


df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')


df


img_path = os.path.join(train_images_root_dir, df["id"].values[0]+".tif")
img = Image.open(img_path)
print("Mode:", img.mode)  # 'RGB' for color, 'L' for grayscale

to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img)  # shape: [C, H, W]
img_tensor.shape


df_class_0 = df[df['label'] == 0].sample(n=5000, random_state=42)
df_class_1 = df[df['label'] == 1].sample(n=5000, random_state=42)
df_sampled = pd.concat([df_class_0, df_class_1]).reset_index(drop=True)


filenames = [f"{id}.tif" for id in df_sampled['id']]


len(filenames), filenames[0], filenames[-1]


# 80% data train; 40% explicitly test + validation
train_df, temp_df = train_test_split(df_sampled, test_size=0.4, stratify=df_sampled['label'], random_state=42)

# 20% for test and 20% for validation
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)


train_df.reset_index(drop=True)
val_df.reset_index(drop=True)
test_df.reset_index(drop=True)


print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


def plot_a_random_image_from_trainDf(label=0):
    df_random = train_df[train_df.label == label].sample(1)
    filename = df_random["id"].values[0]
    file_path = os.path.join(train_images_root_dir, filename + ".tif")

    img = Image.open(file_path)

    plt.figure(figsize=(6, 6)) 
    plt.imshow(img)
    plt.axis('off')  
    plt.title(f"Class: {index_to_labels_mapping[label]}", fontsize=16, fontweight='bold', pad=15)
    plt.show()


plot_a_random_image_from_trainDf(0)


plot_a_random_image_from_trainDf(1)


def plot_batch_of_images(number_of_images = 16, label = 0):
    df_random = train_df[train_df.label == label].sample(number_of_images, random_state = 42)["id"].values
    sixteen_random_samples = [os.path.join(train_images_root_dir, i + ".tif") for i in df_random]
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f"16 Random '{index_to_labels_mapping[label]}' Samples", fontsize=18, fontweight='bold', y=1)
    
    for i, img_path in tqdm(enumerate(sixteen_random_samples)):
        ax = axes[i // 4, i % 4]  
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


plot_batch_of_images(label=0)


plot_batch_of_images(label=1)


class CancerDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{self.df.iloc[idx]['id']}.tif")
        image = Image.open(img_name)
        label = self.df.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# we only apply augmentation to training data
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flipping image horizontally
    transforms.RandomVerticalFlip(), # randomly flipping image vertically
    transforms.RandomRotation(degrees=90),  # radomly rotating image up to 90 degrees
    transforms.ToTensor(),
    # transforms.Pad(padding=64, fill=0),           # 96 + 128 = 224; as resnet requires input at 3 channel * 224 * 224 W H
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # these are imagenet dataset RGB mean and std for normalizing our input
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Pad(padding=64, fill=0),           # 96 + 128 = 224; as resnet requires input at 3 channel * 224 * 224 W H
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


%%time

train_dataset = CancerDataset(train_df, train_images_root_dir, transform=train_transform)
val_dataset = CancerDataset(val_df, train_images_root_dir, transform=val_transform)
test_dataset = CancerDataset(test_df, train_images_root_dir, transform=val_transform)


# first image in train dataset
image, label = train_dataset[0]
print("Image shape:", image.shape)
print("Label:", label)


batch_size = 256  # we have 2x T4 gpu so this should be good; each t4 will get 128 batch size in parallel in each epoch
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# THIS IS COPIED FROM JOVIAN.AI
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()
device


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)


# pretrained imagenet ResNet50
model = models.resnet50(pretrained=True)


model


model.fc 


# only 2 predictions (one for each class; will go through softmax activation to get predicted probaility)
# cross entropy loss will use softmax activation by default; can replace with just one neuron to be passed through sigmoid but will require using
# binary cross entropy loss instead of CE loss
model.fc = nn.Linear(model.fc.in_features , 2) 


model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model = model.to(device)


# these will be same for both model
patience = 8  # number of epochs to wait for improvement for early stopping
num_epochs = 50


criterion = nn.CrossEntropyLoss()

# very small learning rate; we don't want rapid updates, otherwise no point in using ImageNet weights
optimizer = optim.Adam(model.parameters(), lr=0.0001) 

# reduce LR by factor of 0.9 every 5 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
early_stop_counter = 0

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for inputs, labels in train_dl:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss /= len(train_dataset)
    train_acc = 100 * train_correct / train_total

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_dl:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_dataset)
    val_acc = 100 * val_correct / val_total

    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    scheduler.step()
    
    # early stopping check to prevent overfitting
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# will load best weights after training is done
model.load_state_dict(best_model_wts)


model.eval();


training_stats = pd.DataFrame({"Training_Accuracy": train_accs,
                               "Training_Loss": train_losses,
                               "Val_Loss": val_losses,
                               "Val_Accuracy": val_accs})


training_stats


fig, axes = plt.subplots(1, 2, figsize=(20, 8))

n_epochs = len(training_stats)
ax1 = plt.subplot(1,2, 1)

plot_1 = ax1.plot(range(n_epochs), training_stats['Training_Loss'], color = 'blue', label = 'Train Loss',\
             marker = 's', linewidth=2.0, markersize = 10)

plot_2 = ax1.plot(range(n_epochs), training_stats['Val_Loss'], color = 'blue', label = 'Val Loss',\
             marker = 'o', linewidth=2.0, markersize = 10)

ax1.tick_params(axis ='y', labelcolor = 'blue',labelsize=20, width=3) 
ax1.tick_params(axis ='x', labelcolor = 'black',labelsize=20, width=3)
ax1.legend(fontsize = 30)
plt.xticks(range(0,n_epochs,1))
ax1.set_ylabel("Cross Entropy Loss", fontsize = 30, labelpad = 10, color = 'blue')


ax1a = plt.subplot(1,2, 2)
plot_11 = ax1a.plot(range(n_epochs), training_stats['Training_Accuracy'], color = 'red', label = 'Train Acc',\
             marker = 's', linewidth=2.0, markersize = 10)

plot_22 = ax1a.plot(range(n_epochs), training_stats['Val_Accuracy'], color = 'red', label = 'Val Acc',\
             marker = 'o', linewidth=2.0, markersize = 10)
ax1a.legend(fontsize = 30)
plt.xticks(range(0,n_epochs,1))
ax1a.tick_params(axis ='y', labelcolor = 'red',labelsize=20, width=3) 
ax1a.tick_params(axis ='x', labelcolor = 'black',labelsize=20, width=3)
ax1a.set_ylabel("Accuracy", fontsize = 30, labelpad = 10, color = 'red')


ax1.tick_params(which='both', width=2.5)
ax1.tick_params(which='major', length=15)
ax1.tick_params(which='minor', length=5)
ax1.tick_params(which = 'both', direction = 'in')

ax1a.tick_params(which='both', width=2.5)
ax1a.tick_params(which='major', length=15)
ax1a.tick_params(which='minor', length=5)
ax1a.tick_params(which = 'both', direction = 'in') 

ax1a.spines['bottom'].set_color('black')
ax1a.spines['top'].set_color('black') 
ax1a.spines['right'].set_color('black')
ax1a.spines['right'].set_linewidth(2)
ax1a.spines['top'].set_linewidth(2)
ax1a.spines['bottom'].set_linewidth(2)
ax1a.spines['left'].set_color('black')
ax1a.spines['left'].set_lw(2)

ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['right'].set_linewidth(2)
ax1.spines['top'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_color('black')
ax1.spines['left'].set_lw(2)


ax1.grid(True, which = 'major', alpha = 1, linestyle='--', linewidth = 1)
ax1a.grid(True, which = 'major', alpha = 1, linestyle='--', linewidth = 1)


plt.subplots_adjust(wspace=0.25,hspace=0.)
fig.text(0.5, 0.01, 'Epochs', ha='center', va='center', fontsize = 30)

fig.text(0.5, 0.95, 'ResNet-Finetuned Performance on Training and Validation Datasets', ha='center', va='center', fontsize = 30)


def get_test_predictions_with_probs(best_model, dataloader, device="cuda"):
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            outputs = best_model(inputs)  
            probs = F.softmax(outputs, dim=1)  # convert logits to probabilities with softmax
            _, predicted = torch.max(outputs, 1)  

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_labels, all_preds, all_probs


%%time

train_actuals, train_predictions, train_predicted_probabilities = get_test_predictions_with_probs(model, train_dl)


val_actuals, val_predictions, val_predicted_probabilities = get_test_predictions_with_probs(model, val_dl)
test_actuals, test_predictions, test_predicted_probabilities = get_test_predictions_with_probs(model, test_dl)


from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score


print("The testing accuracy is: {}".format(accuracy_score(train_actuals, train_predictions)*100))


print("Classification report for training set")
print(classification_report(train_actuals, train_predictions))


print("Classification report for val set")
print(classification_report(val_actuals, val_predictions))


print("Classification report for test set")
print(classification_report(test_actuals, test_predictions))


cf_matrix = confusion_matrix(test_actuals, test_predictions)

classes = list(index_to_labels_mapping.values())
dataframe = pd.DataFrame(cf_matrix, index = classes, columns = classes)


fig, axes = plt.subplots(1, 1, figsize=(8, 6))

ax1 = plt.subplot(1, 1, 1)

sns.heatmap(dataframe, cmap="Blues", annot = True, fmt="d", cbar =False)
fig.text(0.5, 0.00, 'Predicted', ha='center', va='center', fontsize = 20)
fig.text(0.0, 0.5, 'Actual', ha='center', va='center', rotation='vertical', fontsize = 20)
ax1.text(0.5, 1.08, 'ResNet50-Finetuned',
    horizontalalignment='center',
    fontsize=20,
    transform = ax1.transAxes);


import numpy as np

test_predicted_probabilities = np.array(test_predicted_probabilities)
test_actuals = np.array(test_actuals)


y_true = test_actuals
y_score = test_predicted_probabilities[:, 1]  # column 1 = cancer probabilities

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc_score_value = roc_auc_score(y_true, y_score)
print("AUC (Test Set):", round(roc_auc_score_value, 3))


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='red', lw=3, linestyle='--',
         label=f'Cancer vs No Cancer, AUC = {roc_auc_score_value:.3f}')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve - ResNet Finetuned', fontsize=22, fontweight='bold')
plt.legend(loc='lower right', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()



image_size = (96, 96)
batch_size = 256
CHANNELS_IMG = 3


class RawDataset(Dataset):
    def __init__(self, df, root_dir):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transforms.ToTensor()  # [0,1]

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, f"{self.df.iloc[idx]['id']}.tif")
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

# Compute stats
raw_dataset = RawDataset(train_df, '/kaggle/input/histopathologic-cancer-detection/train')
loader = DataLoader(raw_dataset, batch_size=64, shuffle=False, num_workers=4)


mean = 0.0
std = 0.0
n_pixels = 0

for images in tqdm(loader, desc="Computing stats"):
    batch_size, c, h, w = images.shape
    n_pixels += batch_size * h * w
    mean += images.sum([0, 2, 3])
    std += (images ** 2).sum([0, 2, 3])

mean /= n_pixels
std = torch.sqrt(std / n_pixels - mean ** 2)

print(f"Mean: {mean.tolist()}")
print(f"Std : {std.tolist()}")


mean = mean.tolist()
std = std.tolist()


# we only apply augmentation to training data
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flipping image horizontally
    transforms.RandomVerticalFlip(), # randomly flipping image vertically
    transforms.RandomRotation(degrees=90),  # radomly rotating image up to 90 degrees
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std) 
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std) 
])


%%time

train_dataset = CancerDataset(train_df, train_images_root_dir, transform=train_transform)
val_dataset = CancerDataset(val_df, train_images_root_dir, transform=val_transform)
test_dataset = CancerDataset(test_df, train_images_root_dir, transform=val_transform)


# first image in train dataset
image, label = train_dataset[0]
print("Image shape:", image.shape)
print("Label:", label)


batch_size = 256  # we have 2x T4 gpu so this should be good; each t4 will get 128 batch size in parallel in each epoch
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # if input and output shapes differ, we will adjust skip connection to make original input and output same to add them
        self.skip = None
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.skip: # if not the same shape, 
            identity = self.skip(identity)
            
        out += identity # adding original and new output 
        return F.relu(out)

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.res1 = ResidualBlock(32, 64, stride=2) # first residual block
        self.res2 = ResidualBlock(64, 128, stride=2) # second one
        self.res3 = ResidualBlock(128, 256, stride=2) # third one

        # average pooling followed by linear layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = SimpleResNet(num_classes=2)
x = torch.randn(1, 3, 96, 96) # batch, channel, widdth, hight
out = model(x)
print(out.shape)  # torch.Size([1, 2])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model = model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,          
    momentum=0.9,
    weight_decay=1e-4 # L2 regularization
)
# reduce LR by 5% every epoch
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
early_stop_counter = 0

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for inputs, labels in train_dl:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss /= len(train_dataset)
    train_acc = 100 * train_correct / train_total

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_dl:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_dataset)
    val_acc = 100 * val_correct / val_total

    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    scheduler.step()
    
    # early stopping check to prevent overfitting
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# will load best weights after training is done
model.load_state_dict(best_model_wts)


model.eval();


training_stats = pd.DataFrame({"Training_Accuracy": train_accs,
                               "Training_Loss": train_losses,
                               "Val_Loss": val_losses,
                               "Val_Accuracy": val_accs})


fig, axes = plt.subplots(1, 2, figsize=(20, 8))

n_epochs = len(training_stats)
ax1 = plt.subplot(1,2, 1)

plot_1 = ax1.plot(range(n_epochs), training_stats['Training_Loss'], color = 'blue', label = 'Train Loss',\
             marker = 's', linewidth=2.0, markersize = 10)

plot_2 = ax1.plot(range(n_epochs), training_stats['Val_Loss'], color = 'blue', label = 'Val Loss',\
             marker = 'o', linewidth=2.0, markersize = 10)

ax1.tick_params(axis ='y', labelcolor = 'blue',labelsize=20, width=3) 
ax1.tick_params(axis ='x', labelcolor = 'black',labelsize=20, width=3)
ax1.legend(fontsize = 30)
# plt.xticks(range(0,n_epochs,1))
ax1.set_ylabel("Cross Entropy Loss", fontsize = 30, labelpad = 10, color = 'blue')


ax1a = plt.subplot(1,2, 2)
plot_11 = ax1a.plot(range(n_epochs), training_stats['Training_Accuracy'], color = 'red', label = 'Train Acc',\
             marker = 's', linewidth=2.0, markersize = 10)

plot_22 = ax1a.plot(range(n_epochs), training_stats['Val_Accuracy'], color = 'red', label = 'Val Acc',\
             marker = 'o', linewidth=2.0, markersize = 10)
ax1a.legend(fontsize = 30)
# plt.xticks(range(0,n_epochs,1), )
ax1a.tick_params(axis ='y', labelcolor = 'red',labelsize=20, width=3) 
ax1a.tick_params(axis ='x', labelcolor = 'black',labelsize=20, width=3)
ax1a.set_ylabel("Accuracy", fontsize = 30, labelpad = 10, color = 'red')


ax1.tick_params(which='both', width=2.5)
ax1.tick_params(which='major', length=15)
ax1.tick_params(which='minor', length=5)
ax1.tick_params(which = 'both', direction = 'in')

ax1a.tick_params(which='both', width=2.5)
ax1a.tick_params(which='major', length=15)
ax1a.tick_params(which='minor', length=5)
ax1a.tick_params(which = 'both', direction = 'in') 

ax1a.spines['bottom'].set_color('black')
ax1a.spines['top'].set_color('black') 
ax1a.spines['right'].set_color('black')
ax1a.spines['right'].set_linewidth(2)
ax1a.spines['top'].set_linewidth(2)
ax1a.spines['bottom'].set_linewidth(2)
ax1a.spines['left'].set_color('black')
ax1a.spines['left'].set_lw(2)

ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['right'].set_linewidth(2)
ax1.spines['top'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_color('black')
ax1.spines['left'].set_lw(2)


ax1.grid(True, which = 'major', alpha = 1, linestyle='--', linewidth = 1)
ax1a.grid(True, which = 'major', alpha = 1, linestyle='--', linewidth = 1)


plt.subplots_adjust(wspace=0.25,hspace=0.)
fig.text(0.5, 0.01, 'Epochs', ha='center', va='center', fontsize = 30)

fig.text(0.5, 0.95, 'ResNet-Finetuned Performance on Training and Validation Datasets', ha='center', va='center', fontsize = 30)


%%time

train_actuals, train_predictions, train_predicted_probabilities = get_test_predictions_with_probs(model, train_dl)


val_actuals, val_predictions, val_predicted_probabilities = get_test_predictions_with_probs(model, val_dl)
test_actuals, test_predictions, test_predicted_probabilities = get_test_predictions_with_probs(model, test_dl)


print("Classification report for training set")
print(classification_report(train_actuals, train_predictions))


print("Classification report for val set")
print(classification_report(val_actuals, val_predictions))


print("Classification report for test set")
print(classification_report(test_actuals, test_predictions))


cf_matrix = confusion_matrix(test_actuals, test_predictions)

classes = list(index_to_labels_mapping.values())
dataframe = pd.DataFrame(cf_matrix, index = classes, columns = classes)

fig, axes = plt.subplots(1, 1, figsize=(8, 6))

ax1 = plt.subplot(1, 1, 1)

sns.heatmap(dataframe, cmap="Blues", annot = True, fmt="d", cbar =False)
fig.text(0.5, 0.00, 'Predicted', ha='center', va='center', fontsize = 20)
fig.text(0.0, 0.5, 'Actual', ha='center', va='center', rotation='vertical', fontsize = 20)
ax1.text(0.5, 1.08, 'Custom-ResNet',
    horizontalalignment='center',
    fontsize=20,
    transform = ax1.transAxes);


test_predicted_probabilities = np.array(test_predicted_probabilities)
test_actuals = np.array(test_actuals)


y_true = test_actuals
y_score = test_predicted_probabilities[:, 1]  # column 1 = cancer probabilities

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc_score_value = roc_auc_score(y_true, y_score)
print("AUC (Test Set):", round(roc_auc_score_value, 3))


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='red', lw=3, linestyle='--',
         label=f'Cancer vs No Cancer, AUC = {roc_auc_score_value:.3f}')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve - Custom ResNet', fontsize=22, fontweight='bold')
plt.legend(loc='lower right', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()





