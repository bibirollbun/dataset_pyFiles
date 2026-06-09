# this cell downloads zip archive with data
# ! wget "https://www.dropbox.com/s/r11z0ugf2mezxvi/dogs.zip?dl=0" -O dogs.zip


# this cell extract the archive. You'll now have "dogs" folder in colab
# ! unzip -qq dogs.zip


# model = models.vgg11(pretrained=True)
# model


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy




train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),          
    transforms.RandomHorizontalFlip(),       
    transforms.RandomRotation(15),         
    transforms.ColorJitter(brightness=0.2, 
                           contrast=0.2, 
                           saturation=0.2, 
                           hue=0.1),        
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]), 
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))  
])

val_transforms = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])




# train_dataset = datasets.ImageFolder(root='/kaggle/working/dogs/train', transform=train_transforms)
# val_dataset = datasets.ImageFolder(root='/kaggle/working/dogs/valid', transform=val_transforms)


# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)




# device = 'cuda'

# import torch.nn as nn
# from torchvision import models
# model = models.vgg11(pretrained=True)


# model.classifier[2] = nn.Dropout(p=0.2)
# model.classifier[5] = nn.Dropout(p=0.2)

# classifier_layers = list(model.classifier.children())


# classifier_layers = classifier_layers[:-1]

# num_classes = len(train_dataset.classes)  


# classifier_layers.append(nn.Linear(4096, num_classes))

# model.classifier = nn.Sequential(*classifier_layers)


# model = model.to(device)




# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)




# def train_model(model, criterion, optimizer, scheduler, num_epochs=12):
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         model.train()
#         running_loss = 0.0
#         running_corrects = 0

#         for inputs, labels in train_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = running_corrects.double() / len(train_loader.dataset)

#         print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#         model.eval()
#         running_loss = 0.0
#         running_corrects = 0

#         for inputs, labels in val_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             with torch.no_grad():
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / len(val_loader.dataset)
#         epoch_acc = running_corrects.double() / len(val_loader.dataset)

#         print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#         if epoch_acc > best_acc:
#             best_acc = epoch_acc
#             best_model_wts = copy.deepcopy(model.state_dict())


#         scheduler.step()

#     time_elapsed = time.time() - since

#     model.load_state_dict(best_model_wts)
#     return model

# model = train_model(model, criterion, optimizer, scheduler, num_epochs=15)



# import torch
# torch.save(model.state_dict(), 'model.pth')



# model = train_model(model, criterion, optimizer, scheduler, num_epochs=7)



# torch.save(model.state_dict(), 'model.pth')
# 


# model = train_model(model, criterion, optimizer, scheduler, num_epochs=5)



# torch.save(model.state_dict(), 'model.pth')



# model = train_model(model, criterion, optimizer, scheduler, num_epochs=5)



# torch.save(model.state_dict(), 'model.pth')



# model.load_state_dict(torch.load('/kaggle/input/imbbaaa/model.pth', map_location=device))


# test_transforms = transforms.Compose([
#     transforms.Resize(256),                   
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])  
# ])


# test_dataset = datasets.ImageFolder(root='/kaggle/working/dogs/test', transform=test_transforms)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# def test_model(model, test_loader):
#     model.eval() 
#     correct = 0
#     total = 0
#     with torch.no_grad():  
#         for inputs, labels in test_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             correct += torch.sum(preds == labels).item()
#             total += labels.size(0)
#     accuracy = correct / total
#     print('Test Accuracy: {:.4f}'.format(accuracy))
#     return accuracy

# accuracy = test_model(model, test_loader)




