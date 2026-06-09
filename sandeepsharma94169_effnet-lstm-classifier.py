import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch 
from  torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import Resize
import pydicom
import glob
import os
import cv2
from sklearn.metrics import accuracy_score, roc_auc_score
import torchvision.transforms as transforms


BASE_PATH = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train'
test_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test'

required_slices = 32


label_df = pd.read_csv('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv')
label_df['MGMT_value'].value_counts()


patient_ids = [x for x in label_df['BraTS21ID'].tolist() if x not in ['00109','00123','00709']]
len(patient_ids)


label_df['BraTS21ID'] = label_df['BraTS21ID'].apply(lambda x:f"{int(x):05d}")
label_df[label_df['MGMT_value']==0]


label_dict = label_df.set_index('BraTS21ID')['MGMT_value'].to_dict()


sub_df = pd.read_csv('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv')
test_ids = sub_df['BraTS21ID'].tolist()
test_ids = [str(x).zfill(5) for x in test_ids]


from torchvision import transforms
import torch

def get_training_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3),
    ])


class BrDataset(Dataset):
    def __init__(self,base_path,patient_ids,label_dict,resize = (128,128),transform = None):
        self.base_path = base_path 
        self.patient_ids = patient_ids 
        self.label_dict = label_dict 
        self.modalities = ['FLAIR','T1w']
        self.resize = Resize(resize) 
        self.transform = transform 

    def load_slices(self, case_id, modality):
        paths = sorted(glob.glob(os.path.join(self.base_path, case_id, modality, '*.dcm')),
                       key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
        slices = []
        for path in paths:
            dcm = pydicom.dcmread(path)
            image = dcm.pixel_array.astype('float32')
            slices.append(image)
    
        num_slices = len(slices)
    
        if num_slices < required_slices:
            pad_needed = required_slices - num_slices
            start = pad_needed // 2
            end = pad_needed - start
            h, w = slices[0].shape
            zero_slice = torch.zeros(h, w)
            slices = [zero_slice] * start + slices + [zero_slice] * end
        else:
            mid = num_slices // 2
            start = mid - required_slices // 2
            slices = slices[start:start + required_slices]
    
        volume = torch.stack([torch.tensor(s, dtype=torch.float32) for s in slices])  # [D, H, W]
    
        # ✅ Resize to uniform shape
        volume = self.resize(volume.unsqueeze(0)).squeeze(0)  # [D, H, W] -> [1, D, H, W] -> back
    
        return volume
        # num_cols = 4 
        # num_rows = 8
        # fig, axis = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 3, num_rows * 3))

        # for i in range(num_rows * num_cols):
        #     row = i // num_cols
        #     col = i % num_cols
        #     ax = axis[row][col]
        #     ax.imshow(slices[i], cmap='gray')
        #     ax.axis('off')  # remove axis ticks
        
        # plt.subplots_adjust(wspace=0.05, hspace=0.05)
        # plt.show()

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self,idx):
        case_id = str(self.patient_ids[idx]).zfill(5)
        vol = []
        for modality in self.modalities:
            # print(f"{'='*10} {modality} {'='*10}\n\n")
            volume = self.load_slices(case_id,modality)
            vol.append(volume)
        vol = torch.stack(vol)

        if self.transform:
            transformed_vol = []
            for modality_vol in vol:  # shape: [D, H, W]
                transformed_slices = []
                for slice_img in modality_vol:
                    img = transforms.ToPILImage()(slice_img.unsqueeze(0))
                    img = self.transform(img)
                    img_tensor = transforms.ToTensor()(img).squeeze(0)
                    transformed_slices.append(img_tensor)
                transformed_vol.append(torch.stack(transformed_slices))
            vol = torch.stack(transformed_vol)  # [2, D, H, W]

        if self.label_dict is not None:
            label = torch.tensor(self.label_dict[case_id], dtype=torch.long)
            return vol, label
        else:
            return vol, case_id


train_transform = get_training_transform()

train_dataset = BrDataset(
    base_path=BASE_PATH,
    patient_ids=patient_ids,
    label_dict=label_dict,
    resize=(128,128),
    transform=train_transform
)
x,y = train_dataset[2]


x.shape


import matplotlib.pyplot as plt

def plot_modalities(x, title_prefix="Modality"):
    num_modalities, num_slices, height, width = x.shape
    num_cols = 8
    num_rows = num_slices // num_cols

    for m in range(num_modalities):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        fig.suptitle(f"{title_prefix} {m+1}", fontsize=16)

        for i in range(num_slices):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row][col]
            ax.imshow(x[m, i].cpu(), cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

# Usage
plot_modalities(x)



# import timm
# model = timm.create_model('efficientnet_b0', pretrained=True)
# torch.save(model.state_dict(), 'efficientnet_b0_offline.pth')



import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # You need to have timm installed

class brclassifier(nn.Module):
    def __init__(self, hidden_size=256, num_classes=2):
        super().__init__()
        # Load efficientnet_b0 from timm (pretrained=True will NOT try to download if weights are already cached)
        # self.backbone = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=False,  # Disable download
            features_only=True
        )
        # Load saved weights
        state_dict = torch.load('/kaggle/input/effnet0-weights/efficientnet_b0_offline.pth', map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict=False)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.backbone_out_dim = self.backbone.feature_info[-1]['num_chs']  # usually 1280

        self.lstm = nn.LSTM(
            input_size=self.backbone_out_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        B, C, D, H, W = x.shape
        features = []

        for d in range(D):
            flair_slice = x[:, 0, d]
            t1w_slice = x[:, 1, d]
            avg_slice = (flair_slice + t1w_slice) / 2.0
            img = torch.stack([flair_slice, t1w_slice, avg_slice], dim=1)  # [B, 3, H, W]
            img = F.interpolate(img, size=(224, 224), mode='bilinear')    # Resize

            feats = self.backbone(img)[-1]  # Get last feature map
            pooled = self.pool(feats).view(B, -1)  # [B, 1280]
            features.append(pooled)

        features = torch.stack(features, dim=1)  # [B, D, 1280]
        lstm_out, _ = self.lstm(features)
        final_feat = lstm_out[:, -1]  # [B, 2*hidden]

        return self.classifier(self.dropout(final_feat))



def train_one_fold(model, train_loader, val_loader, optimizer, criterion, device, fold=0, num_epochs=5, patience=2):
    best_auc = 0
    best_model_wts = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss = 0
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Train Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)[:, 1]
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        val_cls = (np.array(val_preds) > 0.5).astype(int)
        val_acc = accuracy_score(val_labels, val_cls)

        print(f"Val Accuracy: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        scheduler.step(val_auc)
        
        if val_auc > best_auc:
            print("Saving best model...")
            best_auc = val_auc
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_model_wts, f"best_model_fold{fold}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break

    return best_auc


import warnings 
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
all_test_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(label_df['BraTS21ID'], label_df['MGMT_value'])):
    train_ids = [patient_ids[i] for i in train_idx]
    val_ids = [patient_ids[i] for i in val_idx]

    train_dataset = BrDataset(BASE_PATH, train_ids, label_dict, resize=(128, 128), transform=train_transform)
    val_dataset = BrDataset(BASE_PATH, val_ids, label_dict, resize=(128, 128), transform=None)
    test_dataset = BrDataset(test_path, test_ids, label_dict=None, resize=(128, 128), transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = brclassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    best_auc = train_one_fold(model, train_loader, val_loader, optimizer, criterion, device)
    print(f"\nFold {fold+1} AUC: {best_auc:.4f}")

    model.eval()
    fold_preds = []
    with torch.no_grad():
        for inputs,_ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            fold_preds.extend(probs)

    all_test_preds.append(fold_preds)


final_results = np.mean(all_test_preds,axis=0)
sub_df['MGMT_value'] = final_results

sub_df.to_csv('/kaggle/working/submission.csv',index=False)

