import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pickle
import os
import torch.nn.functional as F
import requests
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
torch.manual_seed(1) # Set manual seed
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, Subset


IMAGE_SIZE = 32

mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
# These values are mostly used by researchers as found to very useful in fast convergence


# https://pytorch.org/vision/stable/transforms.html
# We can try various transformation for good generalization of model
composed_train = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize the image in a 32X32 shape
                                     transforms.RandomRotation(20), # Randomly rotate some images by 20 degrees
                                     transforms.RandomHorizontalFlip(0.1), # Randomly horizontal flip the images
                                     transforms.ColorJitter(brightness = 0.1, # Randomly adjust color jitter of the images
                                                            contrast = 0.1, 
                                                            saturation = 0.1), 
                                     transforms.RandomAdjustSharpness(sharpness_factor = 2,
                                                                      p = 0.1), # Randomly adjust sharpness
                                     transforms.ToTensor(),   # Converting image to tensor
                                     transforms.Normalize(mean, std), # Normalizing with standard mean and standard deviation
                                     transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)])


composed_test = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])



# Load the data and transform the dataset
train_dataset =  dsets.CIFAR10(root='./data', train=True, download=True, transform = composed_train)
validation_dataset = dsets.CIFAR10(root='./data', train=False, download=True, transform = composed_test)

# Create train and validation batch for training
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100)


def show_data(img):
    try:
        plt.imshow(img[0])
    except Exception as e:
        print(e)
    print(img[0].shape, img[0].permute(1,2,0).shape)
    plt.imshow(img[0].permute(1,2,0))
    plt.title('y = '+ str(img[1]))
    plt.show()
    
# We need to convert the images to numpy arrays as tensors are not compatible with matplotlib.
def im_convert(tensor):
    #Lets
    img = tensor.cpu().clone().detach().numpy() #
    img = img.transpose(1, 2, 0)
    img = img * np.array(tuple(mean)) + np.array(tuple(std))
    img = img.clip(0, 1) # Clipping the size to print the images later
    return img


show_data(train_dataset[8])


# Different classes in CIPHAR 10 dataset. 
classes = ('airplane', 
           'automobile', 
           'bird',
           'cat',
           'deer',
           'dog', 
           'frog', 
           'horse', 
           'ship',
           'truck')

# Define an iterable on the data
data_iterable = iter(train_loader) # converting our train_dataloader to iterable so that we can iter through it. 
images, labels = next(data_iterable) #going from 1st batch of 100 images to the next batch
fig = plt.figure(figsize=(20, 10)) 

# Lets plot 50 images from our train_dataset
for idx in np.arange(10):
    ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
    
    # Note: imshow cant print tensor !
    # Lets convert tensor image to numpy using im_convert function for imshow to print the image
    plt.imshow(im_convert(images[idx]))
    ax.set_title(classes[labels[idx].item()])


def train_model(model, train_loader, validation_loader, optimizer, n_epochs = 20):
    
    # Global variable
    N_test = len(validation_dataset)
    accuracy_list = []
    train_loss_list = []
    train_cost_list = []
    val_cost_list = []
    
    for epoch in range(n_epochs):
        train_COST = 0
        for x,y in train_loader:
            model.train()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z,y)
            loss.backward()
            optimizer.step()
            train_COST+=loss.item()
            
        train_COST = train_COST/len(train_loader)
        train_cost_list.append(train_COST)
        correct = 0
        
        # Perform the prediction on the validation data
        val_COST = 0
        for x_test, y_test in validation_loader:
            model.eval()
            z = model(x_test)
            val_loss = criterion(z, y_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat==y_test).sum().item()
            val_COST+=val_loss.item()
        
        val_COST = val_COST/ len(validation_loader)
        val_cost_list.append(val_COST)
            
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
        
        print("--> Epoch Number : {}".format(epoch + 1),
              " | Training Loss : {}".format(round(train_COST,4)),
              " | Validation Loss : {}".format(round(val_COST,4)),
              " | Validation Accuracy : {}%".format(round(accuracy * 100, 2)))
        
    return accuracy_list, train_cost_list, val_cost_list


class CNN(nn.Module):
    '''
    CNN Model V1: 
    1. 2 convolution + max pool layers 
    2. 1 fully connected layers
    3. Default runtime using 0 momentum and 0 dropout value
    '''
    
    # Constructor
    def __init__(self, out_1 = 32, out_2 = 64, number_of_classes = 10):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = 3, out_channels = out_1, kernel_size = 5, padding = 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        
        
        self.cnn2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 5, padding = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(out_2 * 8 * 8, number_of_classes)
        # Calculation of how we got 8*8 is mentioned in the below comment
        
    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return(x)


# Define the model
model_v1 = CNN(out_1=32, out_2=64, number_of_classes = 10)

# Define model training hyperparameters
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model_v1.parameters(), lr = learning_rate)

# Train the model
accuracy_list_normal_v1, train_cost_list_v1, val_cost_list_v1 = train_model(model=model_v1,n_epochs=20,train_loader=train_loader,validation_loader=validation_loader,optimizer=optimizer)


# Define the model
model_mmt_v1 = CNN(out_1=32, out_2=64, number_of_classes = 10)

# Define the model learning hyperparameters
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model_mmt_v1.parameters(), lr = learning_rate, momentum = 0.2)

# Train the model
accuracy_list_normal_mmt_v1, train_cost_list_mmt_v1, val_cost_list_mmt_v1=train_model(model=model_mmt_v1,n_epochs=20,train_loader=train_loader,validation_loader=validation_loader,optimizer=optimizer)


class CNN_V2(nn.Module):    
    '''
    CNN Model V2: 
    1. 2 convolution & max pool layers 
    2. 2 fully connected layers
    3. Default runtime using 0.2 momentum and dropout value p = 0.5
    '''
    # Constructor
    def __init__(self, out_1 = 32, out_2 = 64, number_of_classes = 10, p = 0):
        super(CNN_V2, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = 3, out_channels = out_1, kernel_size = 5, padding = 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        
        
        self.cnn2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 5, padding = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(out_2 * 8 * 8, 1000) # Roughly taken seein the input and the output
        self.drop = nn.Dropout(p=p)
        self.fc2 = nn.Linear(1000, number_of_classes)
        
    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(self.drop(x))
        x = self.fc2(x)
        return(x)


# Define the model
model_mmt_v2 = CNN_V2(out_1=32, out_2=64, number_of_classes = 10, p=0.5)

# Define model learning hyperparamters
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model_mmt_v2.parameters(), lr = learning_rate, momentum = 0.2)

# Train the model
accuracy_list_normal_v2, train_cost_list_v2, val_cost_list_v2=train_model(model=model_mmt_v2,n_epochs=20,train_loader=train_loader,validation_loader=validation_loader,optimizer=optimizer)


class CNN_V3(nn.Module):
    '''
    CNN Model V3: 
    1. 2 convolution & max pool layers 
    2. 3 fully connected layers
    3. Default runtime using 0.2 momentum and dropout value p = 0.5
    '''
    # Constructor
    def __init__(self, out_1 = 32, out_2 = 64, number_of_classes = 10, p = 0):
        super(CNN_V3, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = 3, out_channels = out_1, kernel_size = 5, padding = 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.cnn2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 5, padding = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        
        # Hidden layer 1
        self.fc1 = nn.Linear(out_2 * 8 * 8, 1000) # Roughly taken seein the input and the output
        self.drop = nn.Dropout(p=p)
        
        # Hidden layer 2
        self.fc2 = nn.Linear(1000, 1000)
        
        # Final layer
        self.fc3 = nn.Linear(1000, 10)
        
    # Predictiona
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(self.drop(x))
        x = self.fc2(x)
        x = F.relu(self.drop(x))
        x = self.fc3(x)
        return(x)


# Define the model
model_mmt_v31 = CNN_V3(out_1=32, out_2=64, number_of_classes = 10, p=0.5)

# Define the model training hyperparameters
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model_mmt_v31.parameters(), lr = learning_rate, momentum = 0.2)

# Train the model
accuracy_list_normal_v31, loss_list_normal_v31=train_model(model=model_mmt_v31,n_epochs=20,train_loader=train_loader,validation_loader=validation_loader,optimizer=optimizer)


class CNN_V3_V2(nn.Module):
    
    '''
    CNN Model V3-V2: 
    1. 3 convolution & max pool layers 
    2. 3 fully connected layers
    3. Default runtime using 0.2 momentum and dropout value p = 0.5
    '''
    # Constructor
    def __init__(self, out_1 = 32, out_2 = 64, out_3 = 128, number_of_classes = 10, p = 0):
        super(CNN_V3_V2, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = 3, out_channels = out_1, kernel_size = 5, padding = 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.cnn2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 5, padding = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.cnn3 = nn.Conv2d(in_channels = out_2, out_channels = out_3, kernel_size = 5, padding = 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)
        
        # Hidden layer 1
        self.fc1 = nn.Linear(out_3 * 4 * 4, 1000) 
        # 8x8 will change to 4x4 as we added a convolution & max pool layer refer calculation comment above
        self.drop = nn.Dropout(p=p)
        
        # Hidden layer 2
        self.fc2 = nn.Linear(1000, 1000)
        
        # Final layer
        self.fc3 = nn.Linear(1000, 10)
        
    # Predictiona
    def forward(self, x):
        
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        
        x = self.cnn3(x)
        x = torch.relu(x)
        x = self.maxpool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        x = F.relu(self.drop(x))
        x = self.fc2(x)
        
        x = F.relu(self.drop(x))
        x = self.fc3(x)
        
        return(x)


# Define model
model_mmt_v32 = CNN_V3_V2(out_1=32, out_2=64, out_3 =128, number_of_classes = 10, p=0.5)

# Define the model hyperparameters
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model_mmt_v32.parameters(), lr = learning_rate, momentum = 0.2)

# Train the model
accuracy_list_normal_v32, loss_list_normal_v32=train_model(model=model_mmt_v32, n_epochs=20, train_loader=train_loader, validation_loader=validation_loader, optimizer=optimizer)


class CNN_V3_V3(nn.Module):
    '''
    CNN Model V3-V2: 
    1. 3 convolution & max pool layers 
    2. 4 fully connected layers
    3. Default runtime using 0.2 momentum and dropout value p = 0.5
    4. Weights initialized using He weight initialization
    '''
    # Constructor
    def __init__(self, out_1 = 32, out_2 = 64, out_3 = 128, number_of_classes = 10, p = 0):
        super(CNN_V3_V3, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = 3, out_channels = out_1, kernel_size = 5, padding = 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.cnn2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 5, padding = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.cnn3 = nn.Conv2d(in_channels = out_2, out_channels = out_3, kernel_size = 5, padding = 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)
        
        # Hidden layer 1
        self.fc1 = nn.Linear(out_3 * 4 * 4, 1000) 
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.drop = nn.Dropout(p=p)
        
        # Hidden layer 2
        self.fc2 = nn.Linear(1000, 1000)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu') # He weight initialization

        # Hidden layer 3
        self.fc3 = nn.Linear(1000, 1000)
        torch.nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu') # He weight initialization
        
        # Final layer
        self.fc4 = nn.Linear(1000, 10)
        torch.nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu') # He weight initialization
        
    # Predictiona
    def forward(self, x):
        
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        
        x = self.cnn3(x)
        x = torch.relu(x)
        x = self.maxpool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        x = F.relu(self.drop(x))
        x = self.fc2(x)
        
        x = F.relu(self.drop(x))
        x = self.fc3(x)

        x = F.relu(self.drop(x))
        x = self.fc4(x)
        
        return(x)


# Define the model
model_mmt_v33 = CNN_V3_V3(out_1=32, out_2=64, out_3 =128, number_of_classes = 10, p=0.5)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model_mmt_v33.parameters(), lr = learning_rate, momentum = 0.2)

# Train the model
accuracy_list_normal_v33, loss_list_normal_v33=train_model(model=model_mmt_v33, n_epochs=20, train_loader=train_loader, validation_loader=validation_loader, optimizer=optimizer)


class CNN_V3_V4(nn.Module):
    """
    Adding one more hidden layer & dropout value & one more convolution layer
    Total 3 hidden layers, 3 convolution layer & Batch Normalization
    """
    # Constructor
    def __init__(self, out_1 = 32, out_2 = 64, out_3 = 128, number_of_classes = 10, p = 0):
        super(CNN_V3_V4, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = 3, out_channels = out_1, kernel_size = 5, padding = 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.drop_conv = nn.Dropout(p=0.2)
        
        self.cnn2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 5, padding = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        
        self.cnn3 = nn.Conv2d(in_channels = out_2, out_channels = out_3, kernel_size = 5, padding = 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)
        self.conv3_bn = nn.BatchNorm2d(out_3)
        
        # Hidden layer 1
        self.fc1 = nn.Linear(out_3 * 4 * 4, 1000) 
        self.drop = nn.Dropout(p=p)
        self.fc1_bn = nn.BatchNorm1d(1000)
        
        # Hidden layer 2
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)

        # Hidden layer 3
        self.fc3 = nn.Linear(1000, 1000)
        self.fc3_bn = nn.BatchNorm1d(1000)
        
        # Hidden layer 4
        self.fc4 = nn.Linear(1000, 1000)
        self.fc4_bn = nn.BatchNorm1d(1000)
        
        # Final layer
        self.fc5 = nn.Linear(1000, 10)
        self.fc5_bn = nn.BatchNorm1d(10)
        
    # Predictiona
    def forward(self, x):
        
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = self.maxpool1(x)
        x = self.drop_conv(x)
        
        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.drop_conv(x)
        
        x = self.cnn3(x)
        x = self.conv3_bn(x)
        x = torch.relu(x)
        x = self.maxpool3(x)
        x = self.drop_conv(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        
        x = F.relu(self.drop(x))
        x = self.fc2(x)
        x = self.fc2_bn(x)
        
        x = F.relu(self.drop(x))
        x = self.fc3(x)
        x = self.fc3_bn(x)
        
        x = F.relu(self.drop(x))
        x = self.fc4(x)
        x = self.fc4_bn(x)

        x = F.relu(self.drop(x))
        x = self.fc5(x)
        x = self.fc5_bn(x)
        
        return(x)


model_mmt_v34 = CNN_V3_V4(out_1=32, out_2=64, out_3 =128, number_of_classes = 10, p=0.5)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model_mmt_v34.parameters(), lr = learning_rate, momentum = 0.2)

# Train the model
accuracy_list_normal_v34, train_cost_list_v34, val_cost_list_v34=train_model(model=model_mmt_v34, n_epochs=200, train_loader=train_loader, validation_loader=validation_loader, optimizer=optimizer)


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(train_cost_list_v34,color=color)
ax1.set_xlabel('epoch',color=color)
ax1.set_ylabel('total loss',color=color)
ax1.tick_params(axis='y', color=color)
    
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)  
ax2.plot( accuracy_list_normal_v34, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()


plt.plot(train_cost_list_v34, 'r', label='Training Loss')
plt.plot(val_cost_list_v34,  'g',  label='Validation Loss')
plt.xlabel("Iteration")
plt.title("Loss")
plt.legend()


y_pred = []
y_true = []

# iterate over test data
for x, y in torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1):
    z = model_mmt_v34(x)
    _, yhat = torch.max(z, 1)
    pred = yhat.data.cpu().numpy()
    y_pred.extend(pred) # Save Prediction

    labels = y.data.cpu().numpy()
    y_true.extend(labels) # Save Truth

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')


print(classification_report(y_true, y_pred))


# Plot the mis-classified samples
count = 0
i = 0
for x, y in torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1):
    z = model_mmt_v34(x)
    _, yhat = torch.max(z, 1)
    if yhat != y:
        show_data(validation_dataset[i])
        plt.show()
        print("yhat: ",yhat)
        count += 1
    if count >= 3:
        break
    i+=1


data_iterable = iter(validation_loader)
images, labels = next(data_iterable)
output = model_mmt_v34(images)
_, preds = torch.max(output, 1)

fig = plt.figure(figsize=(25, 5))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(str(classes[preds[idx].item()]), str(classes[labels[idx].item()])), color=("green" if preds[idx]==labels[idx] else "red"))


class TestImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Define the path where images are stored by yourself.
        # Remember, the first image is ./train/1.png
        img_path = self.img_dir + '/' + str(self.img_labels.iloc[idx, 0]) +'.png'

        # Read image. Recall how you converted an image to numpy array in the previous step.
        image = np.asarray(Image.open(img_path).convert('RGB'))

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Creating transform pipeline for test data
composed_test_check = transforms.Compose([transforms.ToPILImage(), 
                                          transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

test_dataset =  dsets.CIFAR10(root='./data/test', download=True, transform = composed_test_check)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=500, shuffle=False)


y_test_pred = []
y_test_true = []

# iterate over test data
for x, y in torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1):
    z = model_mmt_v34(x)
    _, yhat = torch.max(z, 1)
    pred = yhat.data.cpu().numpy()
    y_pred.extend(pred) # Save Prediction

    labels = y.data.cpu().numpy()
    y_true.extend(labels) # Save Truth

# Build confusion matrix
cf_matrix = confusion_matrix(y_test_true, y_test_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')


print(classification_report(y_test_true, y_test_pred))

