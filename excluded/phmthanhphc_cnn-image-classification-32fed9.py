import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import copy
import os
import torch
from PIL import Image
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torchvision import utils
import pandas as pd


# library which allows us to view model summary like keras/tf
!pip install torchsummary


labels_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
print(labels_df.head().to_markdown())


# No duplicate ids found
labels_df[labels_df.duplicated(keep=False)]


labels_df['label'].value_counts()


# define transformation that converts a PIL image into PyTorch tensors
import torchvision.transforms as transforms
data_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((46,46))])


torch.manual_seed(0) # fix random seed

class ImageDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None, max_get=4000):
        data = pd.read_csv(labels_file)
        print(f"-> Data pd read_csv: {data.head(5)}")
        
        self.data = data.head(max(4000, max_get))
        
        print(f"-> Img dir: {img_dir}")
        self.img_dir = img_dir

        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])+".tif"
        image = Image.open(img_path)
        label = self.data.iloc[idx, 1] 
        # print(f"-> getitem: {str(img_path)} : {label}")
        if self.transform:
            image = self.transform(image)
        return image, label

# Define an object of the custom dataset for the train folder.
data_dir = '/kaggle/input/histopathologic-cancer-detection/'
train_path = str(data_dir + "train")
label_train_path = str(data_dir + "train_labels.csv")
img_dataset = ImageDataset(data_dir + "train_labels.csv", train_path, data_transformer, 5000)


len_img=len(img_dataset)
len_train=int(0.8*len_img)
len_val=len_img-len_train

# Split Pytorch tensor
train_ts,val_ts=random_split(img_dataset,
                             [len_train,len_val]) # random split 80/20

print("train dataset size:", len(train_ts))
print("validation dataset size:", len(val_ts))


# Define the following transformations for the training dataset
tr_transf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5),  
    transforms.RandomRotation(45),         
    transforms.ToTensor()])
tr_transf


# For the validation dataset, we don't need any augmentation; simply convert images into tensors
val_transf = transforms.Compose([
    transforms.ToTensor()])

# After defining the transformations, overwrite the transform functions of train_ts, val_ts
train_ts.transform=tr_transf
val_ts.transform=val_transf

train_ts, val_ts


from torch.utils.data import DataLoader

# Training DataLoader
train_dl = DataLoader(train_ts, 
                      batch_size=32, 
                      shuffle=True)

# Validation DataLoader
val_dl = DataLoader(val_ts,
                    batch_size=32,
                    shuffle=False)

train_dl, val_dl


# CNNModel.py
import torch.nn.functional as F

class Network(nn.Module):
    
    # Network Initialisation
    def __init__(self, num_fc1 = 256, num_classes = 2, dropout_rate = 0.3):
        
        super(Network, self).__init__()
    
        self.dropout_rate= dropout_rate
        
        # transform resize 46 46, ảnh có chiều dài và chiều rộng là 46 x 46
        
        # Convolution Layers - Tầng tích chập
        # 3, 8, 3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        # (46 - 3 + 2*0) / 1 + 1 = 44
        
        # 8, 2*8, 3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        
        # 2*8, 4*8, 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        
        # 4*8, 8*8, 3
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        
        # Pooling layer
        # 22 -> 10 -> 4 -> 1 => 1x1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # Tầng fully connected
        # VD: từ 64 kênh ở conv cuối * 1x1 conv cuối sau pooling (conv4 => pool) (64)
        self.fc1 = nn.Linear(128 * 1 * 1, num_fc1)
        # => 0 hoặc 1 => num_classes = 2
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self,X):
        
        # Convolution => ReLu => Pool Layers
        # Convolution và Relu 1: (46 - 3 + 2*0) / 1 + 1 = 44 => Pool: 44/2 = 22
        X = self.pool(F.relu(self.conv1(X)))
        # Convolution và Relu 2: (22 - 3 + 2*0) / 1 + 1 = 20 => Pool: 20/2 = 10
        X = self.pool(F.relu(self.conv2(X))) 
        # Convolution và Relu 3: (10 - 3 + 2*0) / 1 + 1 = 8 => Pool: 44/2 = 4
        X = self.pool(F.relu(self.conv3(X)))
        # Convolution và Relu 4: (4 - 3 + 2*0) / 1 + 1 = 2 => Pool: 2/2 = 1 => 1x1
        X = self.pool(F.relu(self.conv4(X)))

        X = X.view(-1, 128 * 1 * 1)
        
        X = F.relu(self.fc1(X))
        X=F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return X


# Create instantiation of Network class
cnn_model = Network(512, 2, 0.3)

# define computation hardware approach (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cnn_model.to(device)


from tqdm.notebook import trange, tqdm
from torch import optim

def train_model(model, train_loader, criterion, optimizer, epochs=50):
    loss_history, accuracy_history = [], []
    
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        
        loss_history.append(total_loss / len(train_loader))
        accuracy_history.append(correct / total)
        print(f"Epoch {epoch+1}, Loss: {loss_history[-1]:.4f}, Accuracy: {accuracy_history[-1]:.4f}")
    
    return model, loss_history, accuracy_history


from torchsummary import summary
summary(cnn_model, input_size=(3, 46, 46),device=device.type)


criterion = nn.CrossEntropyLoss()
epochs = 50
# cnn_model,loss_hist,metric_hist=train_model(cnn_model,train_dl,criterion,optim.Adam(cnn_model.parameters(),lr=0.005),epochs)
# cnn_model,loss_hist,metric_hist=train_model(cnn_model,train_dl,criterion,optim.Adam(cnn_model.parameters(),lr=0.0025),epochs)
cnn_model,loss_hist,metric_hist=train_model(cnn_model,train_dl,criterion,optim.Adam(cnn_model.parameters(),lr=0.001),epochs)


import seaborn as sns; sns.set(style='whitegrid')

fig,ax = plt.subplots(1,2,figsize=(12,5))

sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist,ax=ax[0],label='loss_hist_train')
sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist,ax=ax[1],label='metric_hist_train')

ax[0].set_title("Loss Curve")
ax[1].set_title("Accuracy Curve")

plt.show()


def evaluate_model(model, test_loader, criterion, epochs=50):
    model.eval()
    loss_history, accuracy_history = [], []
    
    for i in tqdm(range(epochs)):
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        loss_history.append(total_loss / len(test_loader))
        accuracy_history.append(correct / total)
    
    print(f"Test Loss: {sum(loss_history)/len(loss_history):.4f}, Accuracy: {sum(accuracy_history)/len(accuracy_history):.4f}")
    return loss_history, accuracy_history


loss_val, accuracy_val = evaluate_model(cnn_model,val_dl,criterion)


fig,ax = plt.subplots(1,2,figsize=(12,5))

sns.lineplot(x=[*range(1,epochs+1)],y=loss_val,ax=ax[0],label='loss_val')
sns.lineplot(x=[*range(1,epochs+1)],y=accuracy_val,ax=ax[1],label='accuracy_val')
ax[0].set_title("Loss Val")
ax[1].set_title("Accuracy Val")
plt.show()


loss_val, accuracy_val = evaluate_model(cnn_model,train_dl,criterion, epochs)
fig,ax = plt.subplots(1,2,figsize=(12,5))

sns.lineplot(x=[*range(1,epochs+1)],y=loss_val,ax=ax[0],label='loss_train_test')
sns.lineplot(x=[*range(1,epochs+1)],y=accuracy_val,ax=ax[1],label='accuracy_train_test')
ax[0].set_title("Loss Train Test")
ax[1].set_title("Accuracy Train Test")
plt.show()


from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(model, dataloader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:   
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

class_labels = ["Không Ung Thư", "Ung Thư"]
plot_confusion_matrix(model, val_dl, device, class_labels)


# Load ảnh và xử lý
path = '/kaggle/input/histopathologic-cancer-detection/train/'
image_name = '5622f473549868709943855f3ee4ce5fe8a0bb4e'
image_path = path + image_name + ".tif"  # Đường dẫn ảnh test
image = Image.open(image_path).convert("RGB")  # Mở ảnh và chuyển sang RGB
image = data_transformer(image).unsqueeze(0)  # Chuyển thành batch 1 ảnh

print(image_path)


# Dự đoán
with torch.no_grad():
    output = model(image)

# Chuyển output thành lớp dự đoán
predicted_class = torch.argmax(output, dim=1).item()

print(f"Ảnh này thuộc lớp: {predicted_class}")


def visualize_prediction(image_path):  
    image = Image.open(image_path).convert("RGB")  # Mở ảnh và chuyển sang RGB
    transformed_image = data_transformer(image).unsqueeze(0).to(device)  # Chuyển thành batch 1 ảnh và đưa vào GPU/CPU
    
    with torch.no_grad():
        output = model(transformed_image)
        predicted_class = torch.argmax(output, dim=1).item()
    
    # Vẽ ảnh và hiển thị dự đoán
    plt.figure(figsize=(4, 4))
    plt.imshow(image)  # Hiển thị ảnh gốc
    plt.title(f"Dự đoán: {predicted_class}")
    plt.axis('off')
    plt.show()

# Gọi hàm để hiển thị dự đoán cho ảnh hiện tại
visualize_prediction(image_path)



# weight_path = "weights.pt"
# torch.save(model.state_dict(), weight_path)

