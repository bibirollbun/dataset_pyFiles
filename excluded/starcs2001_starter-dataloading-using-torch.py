import os 
import pandas 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from PIL import Image 
import torch
import torchvision.transforms as transforms

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
print('Done')





### baseline IDEA we need to create data and map each image to its 
### z slice and y x axis this will be our target 


labels_data=pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
print('Done')


labels_data.columns=[iterable.replace(' ','_') for iterable in labels_data.columns]


labels_data





class ImageDataset(Dataset):
    def __init__(self,dir_path,transform=None):
        self.root_dir=dir_path
        self.image_folders=[os.path.join(dir_path,image_folder_path) for image_folder_path in os.listdir(dir_path)]

        self.transform=transform

    def __len__(self):
        return len(self.image_folders)


    def __getitem__(self,idx):
        image_paths = [os.path.join(self.image_folders[idx], actual_image) for actual_image in os.listdir(self.image_folders[idx])]
        slices=[]
        target_image=0
        filtered_values = labels_data[labels_data['tomo_id'] == image_paths[0].split('/')[-2]]['Motor_axis_0']
        if not filtered_values.empty:
            image_target = filtered_values.values[0] 
        if (image_target!=-1.0):
            image_target=labels_data[labels_data['tomo_id']==image_paths[0].split('/')[-2]]['Motor_axis_0']
        target_image_path='slice_0'+str(image_target)+'.jpg'                  
        for image_path in image_paths:
            if image_path.split('/')[-1]==target_image_path:
                target_image=Image.open(target_image_path)
                if self.transform:
                    target_image=self.transform(target_image)
                

            else:
                image=Image.open(image_path)
                if self.transform: 
                    image=self.transform(image)
                slices.append(image)
        three_d_image=torch.stack(slices,dim=-1)
        

        return three_d_image,target_image
        
        


transform = transforms.Compose([
    transforms.ToTensor(),
])


Data_set_of_targets=ImageDataset('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train',transform)



len(Data_set_of_slices) ## now we can call the folders 


first_image,first_target=Data_set_of_slices[0]


first_image


first_image[0, 1, :]



import torch
import torchvision.transforms as transforms
from PIL import Image



# Convert tensor to PIL Image
trans = transforms.ToPILImage()
out = trans(first_image[0, 1, :])

# Show the image (optional)
out.show()




















