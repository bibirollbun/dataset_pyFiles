##---------------------- IMPORTING ALL LIBRARIES -------------------------
import pandas as pd
import pydicom #for dicom files
import numpy as np
from pathlib import Path
import os
import cv2
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics.classification import AUROC
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.nn import BCEWithLogitsLoss


###----------------------- COMPETITION CONSTANTS ------------------------------

ID_COL = 'SeriesInstanceUID'

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]


##---------------- LOAD DATA PATHS ---------------------

train_csv = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
images_path = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"
dicom_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647"
localizer_csv = "/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv"


##---------------- Let's understand the data by actually loading and visualizing --------------------


slices = [pydicom.dcmread(os.path.join(dicom_dir, file)) for file in os.listdir(dicom_dir)]

#sort the 2d slices by z position. (ImagePositionPatient is metadata in dicom)
slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

three_d_volume = np.stack([s.pixel_array for s in slices])

plt.imshow(three_d_volume[len(three_d_volume) // 2], cmap = "gray")
plt.title("Middle slice of 3d Volume (DICOM series)")
plt.show()



####------------------- Explore Tabular data provided ----------------------------

train_df = pd.read_csv(train_csv)

localizer_df = pd.read_csv(localizer_csv)

print("Train CSV shape", train_df.shape)
print("Localizer CSV shape", localizer_df.shape)

train_df.head()


total_cases = train_df["Aneurysm Present"].count()
print("Number of total cases", total_cases)
positive_cases = train_df["Aneurysm Present"].sum()
print("Number of Positive cases", positive_cases)
neg_cases = total_cases - positive_cases
print("Number of Negative Cases:", neg_cases)


#---------------------- VISUALIZING DICOM FILES IN FIRST DIRECTORY --------------------
import os
def visualize(series_path):
    series_path = Path(series_path)
    all_filepaths = []
    count = 0 
    for sub_dir in os.listdir(series_path):
        count += 1
        if count > 3:
            break
        dir_path = os.path.join(series_path, sub_dir)
        
        for file in os.listdir(dir_path):  #get all files of sub director
            if file.endswith(".dcm"):
                all_filepaths.append(os.path.join(dir_path, file))
                
    if len(all_filepaths) == 0: 
        #no files extracted
        print("Ops! NO files extracted")
        volume = np.zeros((num_slices, image_size, image_size), dtype = np.uint8)
        metadeta = {'age': 40, 'sex': 0, 'modality': 'CT'}
        return volume, 
    
    
    rows = int(np.ceil(50 / 5))
    plt.figure(figsize=(20, 50))
    ### Only 50 files from the first directory
    for i, filepath in enumerate(all_filepaths):
        if i == 50:
            break

        ds = pydicom.dcmread(filepath, force = True)
        img = ds.pixel_array 
        plt.subplot(rows, 5 , i + 1)
        plt.imshow(img, cmap="gray")
    plt.show()
        
                    
visualize(images_path)        


#---------------------------------- IMAGE PROCESSING ---------------------------------

image_size = 512
num_slices = 32

def adaptive_windowing(image, modality):
    percentile_range = (5, 95)

    img_flat = image.flatten()
    img_flat = img_flat[img_flat > 0]
    if len(img_flat) == 0:
        return np.zeros_like(image, dtype=np.uint8)
    low_val = np.percentile(img_flat, percentile_range[0])
    
    high_val = np.percentile(img_flat, percentile_range[1])

    if modality in ['CTA']:
        window_width = (high_val - low_val) * 1.5
        window_center = (high_val + low_val) / 2
    elif modality in ['MRA']:
        window_width = (high_val - low_val) * 1.2
        window_center = high_val * 0.7
    else:
        window_width = high_val - low_val
        window_center = (high_val + low_val) / 2    

    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    img_windowed = np.clip(image, img_min, img_max)

## normalize 0-255
    if img_max > img_min:
        img_normalized = ((img_windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else: 
        img_normalized = np.zeros_like(image, dtype = np.uint8)
    return img_normalized


def process_dicom_series(series_path):

    all_filepaths = []
    for file in os.listdir(series_path):  #get all files of sub director
        if file.endswith(".dcm"):
            all_filepaths.append(os.path.join(series_path, file))
                
    if len(all_filepaths) == 0: 
        #no files extracted
        print("Ops! NO files extracted")
        volume = np.zeros((num_slices, image_size, image_size), dtype = np.uint8)
        metadeta = {'age': 40, 'sex': 0, 'modality': 'CT'}
        return volume, metadata

    metadata = {}
    dicom_data = []
    for i, filepath in enumerate(all_filepaths):
        ds = pydicom.dcmread(filepath, force = True)
        img = ds.pixel_array 
        if img.ndim == 3:
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                img = img[:, :, 0]
        instance_num = getattr(ds, 'InstanceNumber', i) # get instance number for proper slice sorting
        if i == 0: # extract the metadata
            metadata["modality"] = getattr(ds, 'Modality', 'CT')
            try: 
                age_str = getattr(ds, 'PatientAge', '50Y')
                age = int(''.join(filter(str.isdigit, age_str[:3]))) or '50'
                metadata['age'] = min(age, 100)
            except:
                metadata['age'] = 50
            try:
                sex = getattr(ds, 'PatientSex', 'M')
                metadata['sex'] = 1 if sex == 'M' else 0
            except:
                metadata['sex'] = 0
        
        #### ------ APPLY RESCALING------------
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept
            
        #input arrays shape not matching so resizing image should be done before stacking
        img_resized = cv2.resize(img, (image_size, image_size))
        dicom_data.append((instance_num, img_resized))

    if len(dicom_data) == 0:
        volume = np.zeros((num_slices, image_size, image_size), dtype = np.uint8)
        return volume, metadata
            
    dicom_data.sort(key = lambda x: x[0])
    raw_slices = [d[1] for d in dicom_data]
    volume_3d = np.stack(raw_slices, axis = 0) #stacking on top of each other (row-wise)

    ## apply modality specific intensity windowing
    volume_windowed = adaptive_windowing(volume_3d, metadata['modality'])

    ## Resize slices 
    processed_slices = []
    for img in volume_windowed:
        resized = cv2.resize(img, (image_size, image_size))
        processed_slices.append(resized)
    volume = np.array(processed_slices)

    if len(processed_slices) > num_slices:
        indices = np.linspace(0, len(processed_slices) - 1, num_slices).astype(int)
        volume = volume[indices]
    elif len(processed_slices) < num_slices:
        pad_size = num_slices - len(processed_slices)
        volume = np.pad(volume, ((0,pad_size), (0,0) , (0,0)), mode = 'edge')
    return volume, metadata

img_folder = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647"
volume, metadata = process_dicom_series(img_folder)        


print(metadata.keys())
print(volume.shape)
metadata


##----------- LET'S VISUALIZE SOME PROCESSED IMAGES--------------

import matplotlib.pyplot as plt
plt.figure(figsize = (10, 10))
for i in range(12):
    plt.subplot(4, 3, i+1)
    plt.imshow(volume[i], cmap='gray')  
plt.show()


#####----------------- CREATE RICH MULTI-CHANNEL REPRESENTATON OF 3D IMAGE------------------
"""
Since "volumes" obtained are 3D, in order to pass it in 2D network we need a smart way fro converting 3D to 2D without losing finer details 


"""
def create_multichannel_img(volume):
    depth, height, width = volume.shape
    #--------Channel 1: Adaptive maximum intensity projection---------
    start = int(depth * 0.15)
    end = int(depth * 0.85)
    core_vol = volume[start: end]
    mip = np.max(core_vol, axis = 0)

    #------ Channel 2: weighted avg of high intensity slices------
    #------------ Focusing on brightness-----------
    slices_means = np.mean(volume, axis = (1,2))
    top_percentile = np.percentile(slices_means, 75)
    high_intensity = slices_means >= top_percentile
    if np.any(high_intensity):
        weighted_avg = np.mean(volume[high_intensity], axis = 0)
    else:
        weighted_avg = np.mean(volume, axis = 0)

    ##-------- Channel 3: Std projection ----------
    std_proj = np.zeros_like(volume[0])
    window_size = min(5, depth // 4)
    for i in range(depth - window_size + 1):
        window_std = np.std(volume[i : i + window_size], axis = 0)
        std_proj = np.maximum(std_proj, window_std)
    #------- normalize all channels 0-255-----------
    channels = []
    for channel in [mip, weighted_avg, std_proj]:
        if channel.max() > channel.min():
            channel_norm = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)
        else:  
            channel_norm = np.zeros_like(channel, dtype = np.uint8)
        channels.append(channel_norm)
    return np.stack(channels, axis = -1)



vol = create_multichannel_img(volume)

plt.figure(figsize=(12, 4))
## display channel wise
# Channel 1 (MIP)
plt.subplot(1, 3, 1)
plt.imshow(vol[:, :, 0], cmap="gray")
plt.title("Adaptive MIP")

# Channel 2 (Weighted Avg High-Intensity)
plt.subplot(1, 3, 2)
plt.imshow(vol[:, :, 1], cmap="gray")
plt.title("Weighted Avg")

# Channel 3 (Std Projection")
plt.subplot(1, 3, 3)
plt.imshow(vol[:, :, 2], cmap="gray")
plt.title("Std Projection")

plt.show()

#3 channels
plt.figure(figsize=(6,6))
plt.imshow(vol)  
plt.title("Multi-channel projection image")
plt.axis("off")
plt.show()


avail_pretrained_models = timm.list_models(pretrained=True)
print(len(avail_pretrained_models)) # timm currently supports 1599 models


vol.shape ### height * width * channels 


###-------------- EXPLORING CSV FILE---------------
df = pd.DataFrame(train_df)

row = df.iloc[1]
print(row)
print("Series Instance ID for 1st row:",row[ID_COL])
df


# WE KNOW THAT SeriesINstanceUID IS UNIQUE REPRESENTATION OF CASES 
# SeriesINstanceUID FROM THE CSV DIRECTLY CONNECTS TO SERIES/SERIESINSTANCEID AND THIS IS HOW WE ARE GOING TO CONNECT LABELS (TARGETS) TO EACH BRAIN SCAN IMAGES (CASES)


###------------------- Model Architecture --------------------

class ClassificationModel(nn.Module):
    def __init__(self, model_name, num_classes = 14, pretrained = True, drop_rate = 0.3, drop_path_rate = 0):
        super().__init__()

        
        self.backbone = timm.create_model(
            model_name,
            pretrained = pretrained,
            in_chans = 3, # 3 for RGB Images
            num_classes = 0, # removes the classifier head entirely
            exportable = True, #model made exportable for ONNX
            drop_rate = 0, 
            drop_path_rate = 0,
            global_pool = '' # skip global pooling. 
        )

        with torch.no_grad():
            dummy_input = torch.zeros(5, 3, image_size, image_size)
            features = self.backbone(dummy_input)
            
            if len(features.shape) == 4:
                num_features = features.shape[1]
                self.needs_pool = True
            elif len(features.shape) == 3:
                num_features = features.shape[1]
                self.needs_pool = False 
            else: 
                num_features = features.shape[1] ## features.shape torch.Size([1, 1280]), backbone is already doing global pooling internally
                self.needs_pool = False
        if self.needs_pool == True:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes) # 14 classes
            
        )

    def forward(self, image):
        #pass image to model
        img_features = self.backbone(image)
        
        if hasattr(self, 'needs_pool') and self.needs_pool:
            img_features = self.global_pool(img_features)
            img_features = img_features.flatten(1)
        elif len(img_features.shape) == 4:
            img_features = F.adaptive_avg_pool2d(img_features, 1).flatten(1)
        elif len(img_features.shape) == 3:
            img_features = img_features.mean(dim=1)

        output = self.classifier(img_features)
        return output

inputs = torch.rand((5, 3, 512, 512))
model = ClassificationModel("tf_efficientnetv2_s")
out = model(inputs)
print(out.shape)


## Since now we have Generated 3 channels 2D image, we need to pass this to model along with the target (y) values. So, next step is preparing data reading for training model

###---------------IMAGE AUGMENTATIONS----------------------
image_size = 512
## Note: transformation on train and validation data would be different
def apply_train_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p = 0.5),
        A.VerticalFlip(p = 0.5),
        A.RandomRotate90(p = 0.5),
        A.Affine(
            translate_percent={"x": 0.0625, "y": 0.0625},
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def apply_val_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

#------------------------------PREPARING DATA READING FOR TRAINING-------------------------------
import torch
import pandas as pd
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, df, img_series, is_train, transform):
        self.df = df
        self.img_series = img_series
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        instance_id = row[ID_COL] 

        ## you have series_id now, you can take that particular folder from series/ and preprocess it!!
        volume, metadata = process_dicom_series(os.path.join(self.img_series, instance_id))
        #you got 3d volume, now its time to transform it to 2D 3channels imgs
        img = create_multichannel_img(volume)

        transformed_img = self.transform(image = img)['image']
        labels = torch.tensor(row[LABEL_COLS].values.astype(float), dtype=torch.float32)
        return transformed_img, labels


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_df = pd.read_csv(train_csv)
train_df, val_df = train_test_split(
    train_df,
    test_size = 0.2, #20% for validation
    random_state = 42,
    stratify = train_df["Aneurysm Present"]   # Since Aneurysm present is rare case in the dataset, number of neg cases > pos cases so this is class imbalance case. 
)

train_transform = apply_train_transform(image_size)
val_transform = apply_val_transform(image_size)


train_data = CustomDataset(train_df, images_path, is_train = True, transform = train_transform)
val_data = CustomDataset(val_df, images_path, is_train = False, transform = val_transform)

train_loader = DataLoader(train_data, batch_size = 5, shuffle = True, num_workers = os.cpu_count())
val_loader = DataLoader(val_data, batch_size = 5, shuffle = False, num_workers = os.cpu_count())


model = ClassificationModel(
    model_name = "tf_efficientnetv2_s",
    num_classes = 14,
    pretrained = True,
    drop_rate = 0.3,
    drop_path_rate = 0.2
).to(device)

## in the evaluation section, it's mentioned that - 13 weight is assigned to Aneurysm Present and for the rest (locations) only 1. total there are 14 labels to predict. 

weights = torch.ones(14).to(device)
aneurysm_idx = LABEL_COLS.index('Aneurysm Present') #these index weight to be 13
weights[aneurysm_idx] = 13

##loss function
criterion = BCEWithLogitsLoss(pos_weight = weights)
## optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.0003)

## scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',       # we want to maximize the metric (e.g. accuracy, AUC)
    factor=0.5,       # multiply LR by 0.5 (reduce by half)
    patience=3,       # wait 3 epochs of no improvement before reducing LR
    verbose=True      # print when LR is reduced
)

scaler = GradScaler(enabled = True)
metric_auroc = AUROC(task = "multilabel", num_labels = 14, average = None).to(device)

best_score = -1.0

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    train_pred = []
    train_labels = []

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled = True):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        train_pred.append(torch.sigmoid(outputs).detach().cpu())
        train_labels.append(labels.detach().cpu())

    epoch_train_loss = running_loss / len(train_data)
    train_pred = torch.cat(train_pred)
    train_labels = torch.cat(train_labels)

    #----------Calculating AUROC per class and separate main target--------------
    train_auroc_per_class = metric_auroc(train_pred, train_labels.long())
    train_auroc_present = train_auroc_per_class[aneurysm_idx].item()
    sum_other_aurocs = torch.sum(train_auroc_per_class[[i for i in range(14) if i != aneurysm_idx]]).item()
    final_score = 0.5 * (train_auroc_present + (1/13) * sum_other_aurocs)
    
    # auroc = torch.mean(train_auroc_per_classp[[i for i in range(14)if i != aneurysm_idx]]).item()
    # final_score = 0.5 * (train_auroc_per_class + auroc) ALSO WORKS!!
    print(f"Epoch: {epoch + 1}, Train Loss: {epoch_train_loss}, Train AUROC for Present: {train_auroc_present}, Final Score: {final_score}")

    ## Now validation part!!!
    model.eval()
    val_preds = []
    val_labels = []
    val_loss = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            with autocast(enabled = True):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_preds.append(torch.sigmoid(outputs).cpu())
            val_labels.append(labels.cpu())  
        
        epoch_val_loss = val_loss / len(val_data)
        val_pred = torch.cat(val_preds)
        val_labels = torch.cat(val_labels) ## for passing to metric auroc

    
        val_auroc_per_class = metric_auroc(val_pred, val_labels.long())
        val_auroc_present = val_auroc_per_class[aneurysm_idx].item()
        sum_other_aurocs = torch.sum(val_auroc_per_class[[i for i in range(14) if i != aneurysm_idx]]).item()
        final_score = 0.5 * (val_auroc_present + (1/13) * sum_other_aurocs)
        
        print(f"Epoch: {epoch + 1}, Val Loss: {epoch_val_loss}, Val AUROC for Present: {val_auroc_present}, Final Score: {final_score}")

        scheduler.step(final_score)
        if final_score > best_score:
            best_score = final_score
            print(f"Best score found! Saving Model with score {best_score}")

            torch.save({
                'model_state_dict' : model.state_dict(),
                'best_score' : best_score
            }, 'best_model.pth')
        del images, labels, outputs, loss
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("Training Complete!")


