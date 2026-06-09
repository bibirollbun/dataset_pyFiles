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
import cv2
import glob
import torch
import torchvision
import pandas as pd
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm





class CustomImageDataset(Dataset):
    
    def __init__(self, img_dir , transform = None, target_transform = None):
        self.img = glob.glob(img_dir)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.target_transform = target_transform
        self.labels = ['Black-grass', 'Charlock' , 'Cleavers' , 'Common Chickweed' , 'Common wheat' , 'Fat Hen' , 'Loose Silky-bent' , 'Maize' , 'Scentless Mayweed' , 'Shepherds Purse', 'Small-flowered Cranesbill' , 'Sugar beet']

    def __len__(self):
        return len(self.img)


    def __getitem__(self, idx):
        img_path = self.img[idx]
        image = cv2.imread(img_path)
        image = self.to_tensor(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = img_path.split('/')[-2]
        idx = self.labels.index(label)
        
        # target = F.one_hot(torch.tensor(idx) , num_classes = len(self.labels))
        # target = torch.tensor(target , dtype=torch.float32)
        return image, idx

class TestDataset(Dataset):
    def __init__(self, img_dir , transform = None, target_transform = None):
        self.img = glob.glob(img_dir)
        # self.name = os.path.basename(img_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = self.img[idx]
        image_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        image = self.to_tensor(image)
        
        if self.transform:
            image = self.transform(image)

        
            
        return image_name , image

    def __len__(self):
        return len(self.img)

# if __name__ == '__main__': # int main()
#    
#     path = "/kaggle/input/plant-seedlings-classification/train/*/*.png"
#     transform = transforms.Compose([transforms.Resize((256, 256))])
#     plant_data = CustomImageDataset(path , transform = transform)

#     dataloader = DataLoader(
#     plant_data, 
#     batch_size=32,
#     shuffle=True,   # set to True in training stage and False in others
#     num_workers=8,  # for multi-processing
#     pin_memory=True,
#     )
    
#     for img, label in dataloader:
#         print(img, label)
    
#     # print("here")


class resnet_50(nn.Module):
    def __init__(self, num_classes = 12):
        super(resnet_50, self).__init__()
        self.resnet50 = models.resnet50(pretrained = True) 
        self.resnet50.fc = nn.Sequential(
                nn.Linear(2048, num_classes),
                nn.ReLU(True),
                # nn.Dropout(),
                # nn.Linear(1024, 512),
                # nn.ReLU(True),
                # nn.Dropout(),
                # nn.Linear(512, 128),
                # nn.ReLU(True),
                # nn.Dropout(),
                # nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.resnet50(x)
        return x


# if __name__ == "__main__":
#     model = resnet_50(12)
#     print(model)



device_num = 0
train_loss = []
train_acc = []
val_loss = []
val_acc = []
train_epoch = []

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total_loss = 0
    total_samples = len(train_loader.dataset)
    
    pbar = tqdm(enumerate(train_loader), total = len(train_loader), desc = f"Epoch {epoch}", unit = "batch")
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim = 1, keepdim=True) #predicted answer
        correct += pred.eq(target.view_as(pred)).sum().item() #currect answer

        total_loss += loss.item()
        pbar.set_postfix({"Loss": total_loss / (batch_idx + 1), "Acc": correct / total_samples})

    acc = round(correct / total_samples, 4)
    train_acc.append(acc)
    train_loss.append(total_loss / len(train_loader))

    print('\nTrain acc: ', acc, 'Train loss: ', total_loss / len(train_loader))
   

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)           
            test_loss += F.cross_entropy(output, target, reduction = 'sum').item()       
            pred = output.argmax(dim = 1, keepdim=True) #predicted answer
            correct += pred.eq(target.view_as(pred)).sum().item() #currect answer

    test_loss /= len(test_loader.dataset)
    acc = round(correct / len(test_loader.dataset), 4)
    val_loss.append(test_loss)
    val_acc.append(acc)
    print('\r\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * acc), end='')

def draw(title, xlabel, ylabel, x, y1, y2, filename):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(filename)

def main():
    # train_dict, train_info_dict = load_data.read_image_folder((224, 224), "./train")
    # whole_dataset = load_data.ImageDataset(train_dict["data"], train_dict["labels"])
    path = "/kaggle/input/plant-seedlings-classification/train/*/*.png"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.RandomRotation(180),
        transforms.RandomAffine(degrees = 45 , translate = (0.25 , 0.25) , scale = (0.5 , 1.5)),
        # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    plant_data = CustomImageDataset(path , transform = transform)
    
    # Random split
    train_set_size = int(len(plant_data) * 0.8)
    valid_set_size = len(plant_data) - train_set_size
    train_set, valid_set = data.random_split(plant_data, [train_set_size, valid_set_size])

    train_loader = DataLoader(train_set, batch_size = 64, shuffle = True)
    valid_loader = DataLoader(valid_set, batch_size = 64)

    with torch.cuda.device(device_num):
        model = resnet_50(num_classes = 12).to(device_num)
        adam = torch.optim.Adam(model.parameters(), lr = 1e-5 , weight_decay = 0.001)

        for epoch in range(1, 31):
            train_epoch.append(epoch)
            train(model, device_num, train_loader, adam, epoch)
            test(model, device_num, valid_loader)

        #print loss graph
        draw('train loss', 'epoch', 'loss', train_epoch, train_loss, val_loss,'resnet50_train_loss.jpg')
        draw('train accuracy', 'epoch', 'accuracy', train_epoch, train_acc, val_acc,'resnet50_train_acc.jpg')
        torch.save(model.state_dict(), 'resnet50.pt')

if __name__ == "__main__":
    main()
    


# from tqdm import tqdm
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import pandas as pd
# import numpy as np
# import torch
# import torchvision

image_names_list = []
ans = []
device_num = 0
label = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
        'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 
        'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

path = '/kaggle/input/plant-seedlings-classification/test/*.png'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
])

test_data = TestDataset(path , transform = transform)
loader = DataLoader(test_data, batch_size = 1)

with torch.cuda.device(device_num):
    model = resnet_50(num_classes = 12).cuda()
    model.load_state_dict(torch.load('/kaggle/working/resnet50.pt'))
    model.eval()
    with torch.no_grad():
        for image_names , data in tqdm(loader):
            # image , idx = data
            data = data.to(device_num)
            output = model(data)
            res = output.argmax(dim = 1, keepdim = True)
            ans.append(label[res])
            image_names_list.extend(list(image_names))


dict = {'file' : image_names_list , 'species' : ans}
df = pd.DataFrame(dict)
df.to_csv('/kaggle/working/submission.csv', index = False)
print("Your submission was successfully saved!")




