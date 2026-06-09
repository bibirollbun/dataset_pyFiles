TEST_MODE=True


import numpy as np
import os
import torch
import random

seed=42

os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)



device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from torch.utils.data import Dataset, DataLoader
from torch import nn 
import timm
from torchvision import transforms as v2
import PIL
from PIL import Image
import tqdm as tqdm 
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split





train=pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')
#test=pd.read_csv('')
sample=pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv')

train_img_dir='/kaggle/input/cassava-leaf-disease-classification/train_images'
test_img_dir='/kaggle/input/cassava-leaf-disease-classification/test_images'


train.info()


if TEST_MODE:
    train=train[:10000]
else:
    train=train


class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transforms, is_train):
        self.df=df
        self.img_dir=img_dir
        self.transforms=transforms
        self.is_train=is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        img_df=row['image_id']
        img_all=os.path.join(self.img_dir, f'{img_df}')
        image=Image.open(img_all).convert('RGB')

        if self.transforms is not None:
            image=self.transforms(image)
        else:
            image=image


        label=torch.tensor(int(row['label']), dtype=torch.long)

        if self.is_train:
            return {
                'image': image,
                'label': label
            }
        else:
            return{
                'image': image, 
                'id': img_df
            }


IMG_SIZE=512


train_transforms=v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms=v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_transforms=v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_transforms = v2.Compose([
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    #v2.VerticalFlip(p=0.5),
    #v2.RandomRotation(15),
    v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    #v2.RandomErasing(p=0.25)
])


eval_transforms = v2.Compose([
    v2.Resize(IMG_SIZE),
    
    v2.ToTensor(),
    v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
test_transforms = eval_transforms


train_data, eval_data=train_test_split(train, random_state=seed, stratify=train['label'])


train_dataset=CassavaDataset(train_data, train_img_dir, train_transforms, is_train=True)
eval_dataset=CassavaDataset(eval_data, train_img_dir, eval_transforms, is_train=True)
test_dataset=CassavaDataset(sample, test_img_dir, test_transforms, is_train=False)


train_dataloader=DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
eval_dataloader=DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=8)
test_dataloader=DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)


model=timm.create_model('resnext50_32x4d', pretrained=True, num_classes=5).to(device)


EPOCHS=5


#criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.1)
class_counts = train['label'].value_counts().sort_index().values.astype(float)
w = (1.0 / np.maximum(class_counts, 1))
w = w / w.mean()
cls_weights = torch.tensor(w, dtype=torch.float, device=device)

criterion = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.05)



optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)


#optimizer=torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)



steps_per_epoch = len(train_dataloader)
total_steps = steps_per_epoch * EPOCHS


scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS,)


'''
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss, running_correct, n=0,0.0, 0.0
    optimizer.zero_grad()
    pbar=tqdm.tqdm(train_dataloader, desc='Training', leave=False)

    for step, batch in enumerate(pbar):
        X=batch['image'].to(device)
        y=batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits=model(X)

        loss=criterion(logits, y)
        loss.backward()
        
        running_loss+=loss.item()*X.size(0)
        
        optimizer.step()
        
        preds=logits.argmax(dim=1)
        running_correct+=(preds==y).sum().item()
        n+=X.size(0)
        pbar.set_postfix(loss=running_loss/ max(n, 1), acc=running_correct / max(n, 1))
        

    scheduler.step()
    print(f'train {epoch} epoch : loss={running_loss/n:.4f}, acc={running_correct/n:.4f}')
    model.eval()
    loss_sum, correct, n=0.0,0,0
    with torch.no_grad():
        
        pbar_eval=tqdm.tqdm(eval_dataloader, desc='Evaluation: ', leave=False)
        for batch in pbar_eval:
            X=batch['image'].to(device)
            y=batch['label'].to(device)

            logits=model(X)
            loss=criterion(logits, y)

            loss_sum+=loss.item()* X.size(0)

            

            n+=X.size(0)
            preds=logits.argmax(dim=1)

            correct+=(preds==y).sum().item()
            
            #acc=accuracy_score(preds, y)

        
            #pbar_eval.set_prefix()
        val_loss = loss_sum / n
        val_acc  = correct / n
        print(f"valid {epoch} epoch : loss={val_loss:.4f}, acc={val_acc:.4f}")

        
    
  '''  


scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
best_acc = 0.0
BEST_PATH = "/kaggle/working/best_resnext50.pth"

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    running_correct = 0
    n = 0

    for batch in tqdm.tqdm(train_dataloader, desc=f'Training {epoch}', leave=False):
        X = batch['image'].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            logits = model(X)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * X.size(0)
        running_correct += (logits.argmax(1) == y).sum().item()
        n += X.size(0)

    scheduler.step()
    print(f'train {epoch}: loss={running_loss/n:.4f}, acc={running_correct/n:.4f}')

    # ------ eval ------
    model.eval()
    loss_sum = 0.0; correct = 0; nval = 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
        for batch in tqdm.tqdm(eval_dataloader, desc='Evaluation', leave=False):
            X = batch['image'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss_sum += loss.item() * X.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            nval += X.size(0)

    val_loss = loss_sum / nval
    val_acc  = correct / nval
    print(f"valid {epoch}: loss={val_loss:.4f}, acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "state_dict": model.state_dict(),
            "meta": {
                "model_name": "resnext50_32x4d",
                "num_classes": 5,
                "img_size": IMG_SIZE,
                "mean": [0.485,0.456,0.406],
                "std":  [0.229,0.224,0.225],
            }
        }, BEST_PATH)
        print(">> saved best to", BEST_PATH)









SAVE_PATH = "/kaggle/working/resnext50_32x4d.pth"

meta = {
    "model_name": "resnext50_32x4d",   # важно для recreate
    "num_classes": 5,
    "img_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
}
torch.save({"state_dict": model.state_dict(), "meta": meta}, SAVE_PATH)
print("saved to", SAVE_PATH)


