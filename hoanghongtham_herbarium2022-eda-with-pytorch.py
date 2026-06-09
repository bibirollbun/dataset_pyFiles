import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

train_dir = '../input/herbarium-2022-fgvc9/train_images/'
test_dir = '../input/herbarium-2022-fgvc9/test_images/'

with open("../input/herbarium-2022-fgvc9/train_metadata.json") as json_file:
    train_meta = json.load(json_file)
with open("../input/herbarium-2022-fgvc9/test_metadata.json") as json_file:
    test_meta = json.load(json_file)


image_ids = [image["image_id"] for image in train_meta["images"]]
image_dirs = [train_dir + image['file_name'] for image in train_meta["images"]]
category_ids = [annotation['category_id'] for annotation in train_meta['annotations']]
genus_ids = [annotation['genus_id'] for annotation in train_meta['annotations']]

test_ids = [image['image_id'] for image in test_meta]
test_dirs = [test_dir + image['file_name'] for image in test_meta]

train_df = pd.DataFrame({
    "image_id" : image_ids,
    "image_dir" : image_dirs,
    "category" : category_ids,
    "genus" : genus_ids})

test_df = pd.DataFrame({
    "test_id" : test_ids,
    "test_dir" : test_dirs
})


train_df.head()


genus_map = {genus['genus_id'] : genus['genus'] for genus in train_meta['genera']}
train_df['genus'] = train_df['genus'].map(genus_map)
train_df


print('Top 15 Genus ')
print(train_df['genus'].value_counts().head(15))
print()


data = train_df['genus'].value_counts().head(15)
data = pd.DataFrame({'Genus' : data.index,
                     'values' : data.values})
plt.figure(figsize = (20, 10))
sns.barplot(x='values', y = 'Genus', data = data , palette='summer_r')
plt.show()


def show_images(speices):
    images = train_df.loc[train_df['genus'] == speices]['image_dir'][:6]
    i = 1
    fig = plt.figure(figsize = (18, 18))
    plt.suptitle(speices, fontsize = '30')
    for image in images:
        img = cv2.imread(image)
        ax = fig.add_subplot(2, 3, i)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
        i += 1
    plt.show()


show_images('Carex')


show_images('Astragalus')


show_images('Penstemon')


show_images('Eriogonum')


show_images('Erigeron')


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


BATCH = 128
EPOCHS = 1

LR = 0.01
IM_SIZE = 224

X_Train, Y_Train = train_df['image_dir'].values, train_df['category'].values

Transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class GetData(Dataset):
    def __init__(self, FNames, Labels, Transform):
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels         
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        x = Image.open(self.fnames[index])
    
        if "train" in self.fnames[index]:             
            return self.transform(x), self.labels[index]
        elif "test" in self.fnames[index]:            
            return self.transform(x), self.fnames[index]
                
trainset = GetData(X_Train, Y_Train, Transform)
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)

N_Classes = train_df['category'].nunique()
next(iter(trainloader))[0].shape

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = torchvision.models.densenet169(pretrained=True)


train_df['category'].nunique()


print(model.classifier.in_features) 
print(model.classifier.out_features)

for param in model.parameters():
    param.requires_grad = False
    
n_inputs = model.classifier.in_features
last_layer = nn.Linear(n_inputs, N_Classes)
model.classifier = last_layer
if torch.cuda.is_available():
    model.cuda()
print(model.classifier.out_features)    

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters())


training_history = {'accuracy':[],'loss':[]}
validation_history = {'accuracy':[],'loss':[]}

from tqdm import tqdm

def train(trainloader, model, criterion, optimizer, scaler, device=torch.device("cpu")):
    train_acc = 0.0
    train_loss = 0.0
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=True):
        output = model(images)
        loss = criterion(output, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc = ((output.argmax(dim=1) == labels).float().mean())
        train_acc += acc
        train_loss += loss
    return train_acc/len(trainloader), train_loss/len(trainloader)


## Normal Evaluation
def evaluate(testloader, model, criterion, device=torch.device("cpu")):
    eval_acc = 0.0
    eval_loss = 0.0
    for images, labels in tqdm(testloader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, labels)

        acc = ((output.argmax(dim=1) == labels).float().mean())
        eval_acc += acc
        eval_loss += loss
  
    return eval_acc/len(testloader), eval_loss/len(testloader)


%%time
#%dirsrmal Training
scaler = torch.cuda.amp.GradScaler(enabled=True)
for epoch in range(EPOCHS):
    train_acc, train_loss = train(trainloader, model, criterion, optimizer, scaler, device=device)
    eval_acc, eval_loss = evaluate(val_loader, model, criterion, device=torch.device("cuda"))
    print("")
    print(f"Epoch {epoch + 1} | Train Acc: {train_acc*100} | Train Loss: {train_loss}")
    print(f"\t Val Acc: {eval_acc*100} | Val Loss: {eval_loss}")
    print("===="*8)


X_Test = test_df['test_dir'].values
testset = GetData(X_Test, None, Transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

s_ls = []

with torch.no_grad():
    model.eval()
    for image, fname in testloader: 
        image = image.to(device)
        
logits = model(image)        
ps = torch.exp(logits)        
_, top_class = ps.topk(1, dim=1)
        
for pred in top_class:
    s_ls.append([fname[0].split('/')[-1][:-4], pred.item()])
            
sub = pd.DataFrame.from_records(s_ls, columns=['Id', 'Predicted'])
sub.head()
sub.to_csv("submission.csv", index=False)

