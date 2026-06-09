seed=42


import os
import numpy as np
import pandas as pd
import torch
import tqdm


np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED']=str(seed)

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from sklearn.model_selection import train_test_split

from PIL import Image
import timm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score
from torch import nn


TEST_MODE=True


trainn_csv=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test_csv=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
sample=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

train_img_root='/kaggle/input/siim-isic-melanoma-classification/jpeg/train'
test_img_root='/kaggle/input/siim-isic-melanoma-classification/jpeg/test'


if TEST_MODE:
    trainn_csv = trainn_csv.sample(n=11000, random_state=seed).reset_index(drop=True)
else:
    trainn_csv=trainn_csv


train_csv, eval_csv=train_test_split(trainn_csv,  test_size=0.2,
    
    stratify=trainn_csv["target"])


#train_tfms=T.Compose(
#    T.ResizedCrop
#)


IMG_SIZE = 224
train_tfms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_tfms = T.Compose([
    T.Resize(IMG_SIZE + 32),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MelanomaDataset(Dataset):
    def __init__(self, df, img_root, transform, with_labels):
        self.df=df
        self.with_labels = with_labels
        self.img_root=img_root
        self.transform=transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        image_name=row['image_name']
        image_path=os.path.join(self.img_root, f'{image_name}.jpg')
        image=Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image=self.transform(image)
        if self.with_labels and 'target' in row.index:
            label = torch.tensor(int(row['target']), dtype=torch.long)
            return {'image': image, 'label': label}
        else:
            return {'image': image, 'image_name': row['image_name']}




train_dataset = MelanomaDataset(train_csv, img_root=train_img_root, transform=train_tfms, with_labels=True)
eval_dataset = MelanomaDataset(eval_csv, img_root=train_img_root, transform=eval_tfms, with_labels=True)
test_dataset  = MelanomaDataset(test_csv,  img_root=test_img_root,  transform=eval_tfms, with_labels=False)



train_dataloader=DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8,)
eval_dataloader=DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=8,)
test_dataloader=DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8,)


model=timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=2)


model.to(device)


criterion=nn.CrossEntropyLoss()


BATCH_SIZE = 32  # matches train_loader above
base_lr = 0.1 * (BATCH_SIZE / 256.0) 


optim=torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)


scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=5)


EPOCHS=5


best_auc = -1.0
for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS} • lr={optim.param_groups[0]['lr']:.6f}")
    model.train()
    running_loss, running_correct, n= 0.0,0,0

    optim.zero_grad()
    pbar=tqdm.tqdm(train_dataloader, desc='train', leave=False)

    for step, batch in enumerate(pbar):
        X=batch['image'].to(device)
        y=batch['label'].to(device)
        optim.zero_grad()
        logits=model(X)
        
        loss=criterion(logits, y)
        loss.backward()
        optim.step()
        

        running_loss+=loss.item()*X.size(0)

        preds=logits.argmax(dim=1)

        running_correct+=(preds==y).sum().item()

        n+=X.size(0)
        pbar.set_postfix(loss=running_loss / max(n, 1), acc=running_correct / max(n, 1))

    scheduler.step()
    print(f"train: loss={running_loss / n:.4f}, acc={running_correct / n:.4f}")
    #@torch.no_grad()
    model.eval()
    loss_sum, correct, n=0.0,0,0
    all_probs, all_targets = [], []
    with torch.no_grad():
        pbar=tqdm.tqdm(eval_dataloader, desc='validation', leave=False)
    
        for batch in pbar:
            X=batch['image'].to(device)
            y=batch['label'].to(device)

            logits=model(X)
            loss=criterion(logits, y)

            loss_sum+=loss.item()* X.size(0)

            preds=logits.argmax(dim=1)

            correct+=(preds==y).sum().item()

            n+=X.size(0)
            probs=torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            all_probs.append(probs)
            all_targets.append(y.detach().cpu().numpy())
        all_probs = np.concatenate(all_probs) if len(all_probs) else np.array([])
        all_targets = np.concatenate(all_targets) if len(all_targets) else np.array([])
        auc = roc_auc_score(all_targets, all_probs)
        val_loss = loss_sum / n
        val_acc  = correct / n
        print(f"valid: loss={val_loss:.4f}, acc={val_acc:.4f}, AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save({"model": model.state_dict()}, "best_resnet50.pt")
            print(f"✓ Saved new best (AUC {best_auc:.4f})")


ckpt_path="best_resnet50.pt"
state=torch.load(ckpt_path, map_location=device)
model.load_state_dict(state["model"])

model.eval()

test_names, test_probs=[], []

with torch.no_grad():
    pbar = tqdm.tqdm(test_dataloader, desc="test", leave=False)
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)                       # [B, 2]
        probs  = torch.softmax(logits, 1)[:, 1] # positive-class probability
        test_probs.append(probs.cpu().numpy())
        test_names.extend(batch["image_name"])
test_probs = np.concatenate(test_probs, axis=0)
pred_df = pd.DataFrame({"image_name": test_names, "target": test_probs})

# 3) Align to sample order & save
submission = sample[["image_name"]].merge(pred_df, on="image_name", how="left")
#submission["target"] = submission["target"].fillna(0.0)  # just in case
submission.to_csv("submission.csv", index=False)
print(submission.head(), "\nSaved -> submission.csv")
    

