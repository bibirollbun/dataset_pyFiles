import numpy as np
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
import torchvision
import random
from torchvision import transforms,datasets,models
import math
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time
from tempfile import TemporaryDirectory
import torch
cudnn.benchmark = True


brightness=(0.98, 1.02)
gamma=(0.5, 1.1)
gaussian_std=(0,0.02)
resolution = (256 , 256)


def delete_np(y,indice):
  y = np.concatenate((y[:indice],y[indice+1:]))
  return y


def data_augmentation(img, brightness=(0.98, 1.02), gamma=(0.5, 1.1), gaussian_std=(0,0.02)):
  img = (img + 1) / 2  # Map from [-1, 1] to [0, 1] if normalized

  gamma_val = random.uniform(gamma[0], gamma[1])
  img = transforms.functional.adjust_gamma(img, gamma_val)

  brightness_val = random.uniform(brightness[0], brightness[1])
  img = transforms.functional.adjust_brightness(img, brightness_val)
  # Gaussian noise (if needed)
  noise = torch.randn_like(img) * random.uniform(gaussian_std[0],gaussian_std[1])
  img = img + noise

  # Clamp and re-normalize to [-1, 1]
  img = torch.clamp(img, 0, 1)  # First clamp to [0, 1]
  img = img * 2 - 1  # Map back to [-1, 1]

  return img


import os 
os.listdir('/kaggle/input/image-matching-challenge-2025/train')


import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


transform = transforms.Compose([
    transforms.Resize(resolution),
    transforms.ToTensor(),  # [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5]),  # [-1, 1]
    #transforms.Grayscale(),
    transforms.RandomApply(
        [transforms.Lambda(lambda x: data_augmentation(x))],
        p=0.5  # 50% chance to apply
    ),
])

kaggle_data = torchvision.datasets.ImageFolder(
    '/kaggle/input/image-matching-challenge-2025/train',
    transform=transform)

data_loader = torch.utils.data.DataLoader(kaggle_data,batch_size=64,shuffle=True, num_workers=4)


class_names = kaggle_data.classes
classes_map = []
for i in enumerate(kaggle_data.classes):
  classes_map.append(i)
classes_map = dict(classes_map)


device = "cuda"
print(f"Using {device} device")


dif_classes ={}
for label in range(13):
    dif_classes[label] =0
for _,label in kaggle_data:
    dif_classes[label] +=1


y = list(dif_classes.values())
x = list(dif_classes.keys())


import seaborn as sns
import pandas as pd
sns.barplot(x=x,y=y)


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(kaggle_data))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out,title =classes)


from tqdm import tqdm
def train_model(model, criterion, optimizer, scheduler, train_loader,test_loader, num_epochs=25):
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                data_loader_phase = train_loader
            else :
                data_loader_phase =test_loader
            dataset_size = len(data_loader_phase.dataset)
            
            
            for inputs, labels in data_loader_phase:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

            if phase == 'train':
                scheduler.step()

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'best_model.pt')
    return model, history



def visualize_model(model,num_images = 6):
  was_training = model.training
  model.eval()
  images_so_far = 0
  fig = plt.figure()
  with torch.no_grad():
    for i, (inputs,labels) in enumerate(kaggle_data_test):
      inputs = inputs.to(device)
      labels = inputs.to(device)

      outputs = model(inputs)
      _, preds = torch.max(outputs,1)

      for j in range(inputs.size()[0]):
        images_so_far+=1
        ax = plt.subplot(num_images//2,2,images_so_far)
        ax.axis('off')
        ax.set_title(f'predicted:{class_names[preds[j]]}')
        imshow(inputs.cpu.data[j])
        if images_so_far == num_images:
          model.train(mode=was_training)
          return

    model.train(mode=was_training)


from torch.utils.data import DataLoader, Subset
def k_fold_cross_validation(kaggle_data=kaggle_data,batch_size =3,k_folds =2,num_workers=8):
    results ={}
    kf = KFold(n_splits=k_folds, shuffle=True)
    
    model_ft = models.resnet18(weights = 'IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc =nn.Linear(num_ftrs,len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(),lr = 0.001, momentum =0.9)
    exp_lr_scheduler=lr_scheduler.StepLR(optimizer_ft,step_size =7, gamma =0.1)

    
    for fold,(train_idx,val_idx) in tqdm(enumerate(kf.split(kaggle_data))):
        print(f'FOLD {fold + 1}')
        print('--------------------------------')
        
        # Create subsets
        train_subsampler = Subset(kaggle_data, train_idx)
        val_subsampler = Subset(kaggle_data, val_idx)
        
        # Create data loaders
        train_loader = DataLoader(
            train_subsampler,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True  # Shuffle training data each epoch
        )
        
        val_loader = DataLoader(
            val_subsampler,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False  # No need to shuffle validation data
        )
        
    
        model_ft = model_ft.to(device)
    
        model_ft , val_results  = train_model(model_ft,
                               criterion,
                               optimizer_ft, 
                               exp_lr_scheduler,
                               train_loader,
                               val_loader,
                               num_epochs=10)
    return val_results


model_ft = models.resnet18(weights = 'IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc =nn.Linear(num_ftrs,len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(),lr = 0.001, momentum =0.9)
exp_lr_scheduler=lr_scheduler.StepLR(optimizer_ft,step_size =7, gamma =0.1)
model_ft = model_ft.to(device)


import optuna
import torch.optim as optim

def objective(trial):
    # Hyperparameters
    batch = trial.suggest_int('batch', 3, 16)
    lr = trial.suggest_float('lr', 0.0001, 0.1, log=True)  # Added log scale for better exploration
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'Adadelta'])
    
    # Model setup
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Optimizer selection
    if optimizer_name == 'SGD':
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)
    elif optimizer_name == 'Adadelta':
        optimizer_ft = optim.Adadelta(model_ft.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = model_ft.to(device)
    
    # Validation
    val_results = k_fold_cross_validation(kaggle_data=kaggle_data, batch_size=batch, k_folds=3)
    
    # Return the final validation accuracy
    return float(val_results['val_acc'][-1])


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

trial = study.best_trial


optuna.visualization.plot_optimization_history(study)

#Plotting the accuracies for each hyperparameter for each trial.

optuna.visualization.plot_slice(study)

#Plotting the accuracy surface for the hyperparameters involved in the random forest model.

optuna.visualization.plot_contour(study, params=["batch", "lr","optimizer"])




