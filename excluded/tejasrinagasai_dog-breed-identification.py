import torch
import os
from PIL import Image
from pathlib import Path
import pandas as pd
import torchvision
from sklearn.model_selection import train_test_split


labels=pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
train,valid=train_test_split(labels,train_size=0.8,shuffle=True,stratify=labels['breed'],random_state=42)


train,train_labels=train['id'].reset_index(drop=True),train['breed'].reset_index(drop=True)
val,val_labels=valid['id'].reset_index(drop=True),valid['breed'].reset_index(drop=True)


'''Encoding labels'''

breeds=dict()
breed_count=1

for breed in labels['breed'].value_counts().index:

    breeds[breed]=breed_count-1
    breed_count+=1

val_labels_torch=torch.zeros(len(val_labels),1)
train_labels_torch=torch.zeros(len(train_labels),1)

for index in val_labels.index:
    val_labels_torch[index]=breeds[val_labels.iloc[index]]

for index in train_labels.index:
    train_labels_torch[index]=breeds[train_labels.iloc[index]]

val_labels_torch = val_labels_torch.long()
train_labels_torch = train_labels_torch.long()


val_labels_torch[91]
# val_labels=torch.tensor(val_labels.values)
# train_labels=torch.tensor(train_labels.values)


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    torchvision.transforms.ToTensor()
])


def create_dataset(series, directory_path, transform):
    tensors = []
    for i in range(len(series)):
        if(series[i].endswith('.jpg')):
            img_file=series[i]
        else:
            img_file = series[i] + '.jpg'
        img_path = os.path.join(directory_path, img_file)
        img = Image.open(img_path).convert('RGB')  # Ensure RGB
        img_t = transform(img)
        tensors.append(img_t)
    return torch.stack(tensors, dim=0)



directory_path=Path('/kaggle/input/dog-breed-identification/train')
train_dataset=create_dataset(train,directory_path,transform)




# For validation: no augmentation
val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])
val_dataset = create_dataset(val, directory_path, transform=val_transform)



from torch.utils.data import TensorDataset

train_dataset_torch=TensorDataset(train_dataset,train_labels_torch)
val_dataset_torch=TensorDataset(val_dataset,val_labels_torch)


train_dataset.shape,train_labels_torch.shape


val_dataset_torch[0]


from torchvision.models import resnet101, ResNet101_Weights
import torch.nn as nn
import torch.optim as optim

resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
for param in resnet.parameters():
    param.requires_grad = False


resnet.fc = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 256),  # From ResNet's output to 256 units
    nn.ReLU(),                              # Activation function
    nn.BatchNorm1d(256),                    # Normalization for stability
    nn.Linear(256, 256),                    # Additional dense layer
    nn.ReLU(),                              # Activation again
    nn.Dropout(0.3),                        # Dropout to reduce overfitting
    nn.BatchNorm1d(256),                    # Another batch norm layer
    nn.Linear(256, 120)                     # Final layer: 120 output classes
)
model_torch = resnet

optimizier=optim.Adam(model_torch.parameters(),lr=1e-4)
loss_fn=nn.CrossEntropyLoss()


from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset_torch, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset_torch, batch_size=64, shuffle=False)

n_epochs=20
model_torch=model_torch.to('cuda')
for i in range(1,n_epochs+1):
    model_torch.train()
    training_loss=0.0
    for img,label in train_loader:
        img=img.to('cuda')
        label=label.to('cuda')
        label=label.squeeze(1)
        outputs=model_torch(img)
        optimizier.zero_grad()
        losses=loss_fn(outputs,label)
        training_loss+=losses.item()
        losses.backward()
        optimizier.step()

    model_torch.eval()
    val_loss=0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for img,label in val_loader:
            img=img.to('cuda')
            label=label.to('cuda')
            label=label.squeeze(1)
            outputs=model_torch(img)
            loss=loss_fn(outputs,label)
            val_loss+=loss.item()

            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print(f"Epoch [{i}/{n_epochs}], Training Loss: {training_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")



torch.save(resnet.state_dict(), 'resnet_custom_head.pth')



os.listdir('/kaggle/input/dog-breed-identification/test')[1:10]



test=os.listdir('/kaggle/input/dog-breed-identification/test')
test_dataset=create_dataset(test,'/kaggle/input/dog-breed-identification/test',val_transform)
test_dataset=test_dataset.to(device='cuda')


import pandas as pd
output=pd.DataFrame(columns=['file',*breeds.keys()])


output


breeds


import torch.nn.functional as F
i=0
for file,sample in zip(os.listdir('/kaggle/input/dog-breed-identification/test'),test_dataset):
    sample=sample.cuda()
    output_values=model_torch(sample.unsqueeze(dim=0))
    output_values=output_values.cpu()
    probabilities = F.softmax(output_values, dim=1)
    output.loc[i]=[file,*probabilities[0].detach().numpy()]
    i+=1


output.head()


output.to_csv('test_submission.csv',index=False)




