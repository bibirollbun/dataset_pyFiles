!pip install -q segmentation-models-pytorch


# === 1. Imports ===
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet101, densenet121
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp


# === 2. Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # === 3. Load & Prepare CSV ===
# df = pd.read_csv('/kaggle/input/our-pacs-data-yolo-640/filtered_data_640.csv')
# df_grouped = df.groupby('Scan ID')['Disease'].apply(lambda x: '|'.join(sorted(set(x)))).reset_index()
# df_v1 = df_grouped.copy()
# df_v1['label'] = df_v1['Disease'].apply(lambda x: 0 if x == 'No Finding' else 1)


# def downsample_df(df, label_col='label'):
#     df_normal = df[df[label_col] == 0]
#     df_abnormal = df[df[label_col] == 1]
#     df_normal_downsampled = resample(df_normal, replace=False, n_samples=len(df_abnormal), random_state=42)
#     df_balanced = pd.concat([df_normal_downsampled, df_abnormal])
#     return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


# def split_and_report(df_balanced):
#     train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label'])
#     return train_df, test_df


# # === 4. Split data ===
# df_v1_balanced = downsample_df(df_v1)
v1_train = pd.read_csv("/kaggle/input/our-data-ensemble/v1_train_total_balanced.csv")
v1_test  = pd.read_csv("/kaggle/input/our-data-ensemble/v1_test_total_balanced.csv")


# === 5. Transforms ===
IMG_SIZE = 256
val_transform_chex = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
])

val_transform_seg = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
])


# === 6. Load segmentation model ===
def load_segmentation_model(model_path):
    model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

SEG_MODEL_PATH = "/kaggle/input/x-ray-segmention-model/xray_Segmention_model.pth"
unet_model = load_segmentation_model(SEG_MODEL_PATH)


# === 7. Custom Dataset for Ensemble ===
class EnsembleDataset(Dataset):
    def __init__(self, df, image_dir, transform_chexnet, transform_segmentation, seg_model):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform_chexnet = transform_chexnet
        self.transform_seg = transform_segmentation
        self.seg_model = seg_model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["Scan ID"]
        label = row["label"]
        image_path = os.path.join(self.image_dir, f"{image_id}.png")

        image = Image.open(image_path).convert("RGB")
        img_chexnet = self.transform_chexnet(image)

        # 1. Segment image
        img_for_seg = self.transform_seg(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.seg_model(img_for_seg)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.float32)
        
        # 2. Resize image to match mask
        image_resized = image.resize((512, 512))
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        
        # 3. Apply mask
        masked_image = image_np * np.expand_dims(mask, axis=-1)
        
        # 4. Back to PIL for transform
        masked_image_pil = Image.fromarray((masked_image * 255).astype(np.uint8))
        img_segmented = self.transform_chexnet(masked_image_pil)

        return {
            "chexnet_img": img_chexnet,
            "chexnet2_img": img_chexnet,
            "seg_img": img_segmented,
            "label": torch.tensor(label, dtype=torch.float32)
        }


# === 8. Load Models ===
def get_resnet101_model():
    model = resnet101(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))
    return model.to(DEVICE)

def get_chexnet_model(num_classes=2):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(1024, num_classes)
    return model.to(DEVICE)

# Load models
model_loaded_seg = get_resnet101_model()
model_loaded_seg.load_state_dict(torch.load("/kaggle/input/models_normal_abnormal/pytorch/default/1/ResNet_Segmentation.pth", map_location=DEVICE))

model_loaded_chexnet_all = get_chexnet_model()
model_loaded_chexnet_all.load_state_dict(torch.load("/kaggle/input/models_normal_abnormal/pytorch/default/1/ChexNet_Full.pth", map_location=DEVICE))

model_loaded_chexnet_2_classes = get_chexnet_model()
model_loaded_chexnet_2_classes.load_state_dict(torch.load("/kaggle/input/models_normal_abnormal/pytorch/default/1/ChexNet_2classes.pth", map_location=DEVICE))


# === 9. Ensemble Dataloader ===
IMAGE_DIR = "/kaggle/input/our-pacs-data-yolo-640/Images/Images"
ensemble_dataset = EnsembleDataset(
    df=v1_test,
    image_dir=IMAGE_DIR,
    transform_chexnet=val_transform_chex,
    transform_segmentation=val_transform_seg,
    seg_model=unet_model
)
ensemble_loader = DataLoader(ensemble_dataset, batch_size=32, shuffle=False, num_workers=0)



def soft_voting_ensemble(models_info, dataloader, device, weights=None):
    """
    models_info: Ù‚Ø§Ø¦Ù…Ø© ØªØ­ØªÙˆÙŠ Ø¹Ù„Ù‰ tuples Ø¨Ø§Ù„Ø´ÙƒÙ„ Ø§Ù„ØªØ§Ù„ÙŠ:
        [(model_object, model_name_string), ...]
    
    dataloader: ÙŠØ¬Ø¨ Ø£Ù† ÙŠÙ�Ø±Ø¬Ø¹ (image, label, model_name)
    """
    all_probs = []

    for model, model_name in models_info:
        model.eval()
        model.to(device)
        probs = []

        with torch.no_grad():
            for batch in dataloader:
                image = batch["image"]
 # Ù�Ù‚Ø· Ø§Ù„ØµÙˆØ±Ø©
                image = image.to(device)
            
                outputs = model(image)
                prob = torch.softmax(outputs, dim=1)[:, 1]  # Ø§Ø­ØªÙ…Ø§Ù„ abnormal
                probs.extend(prob.cpu().numpy())

        all_probs.append(np.array(probs))

    all_probs = np.stack(all_probs, axis=0)  # Ø§Ù„Ø´ÙƒÙ„: (n_models, n_samples)

    if weights is None:
        weights = np.ones(len(models_info)) / len(models_info)

    final_probs = np.average(all_probs, axis=0, weights=weights)
    return final_probs



# Dataset Ø¨Ø¯ÙˆÙ† transform (Ø³ÙŠØªÙ… ØªØ·Ø¨ÙŠÙ‚Ù‡Ø§ Ø¯Ø§Ø®Ù„ Ø§Ù„Ù€ ensemble function)
class RawLungDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["Scan ID"]
        label = row["label"]
        image_path = os.path.join(ORIGINAL_IMAGE_DIR, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")
        return image, label



import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet101, densenet121
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === 1. Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Load & Prepare CSV ===
df = pd.read_csv('/kaggle/input/our-pacs-data-yolo-640/filtered_data_640.csv')
df_grouped = df.groupby('Scan ID')['Disease'].apply(lambda x: '|'.join(sorted(set(x)))).reset_index()
df_grouped['label'] = df_grouped['Disease'].apply(lambda x: 0 if x == 'No Finding' else 1)

def downsample_df(df, label_col='label'):
    df_normal = df[df[label_col] == 0]
    df_abnormal = df[df[label_col] == 1]
    df_normal_downsampled = resample(df_normal, replace=False, n_samples=len(df_abnormal), random_state=42)
    df_balanced = pd.concat([df_normal_downsampled, df_abnormal])
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

def split_and_report(df_balanced):
    return train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label'])

df_balanced = downsample_df(df_grouped)
v1_train, v1_test = split_and_report(df_balanced)

# === 3. Transforms ===
IMG_SIZE = 256
val_transform_chex = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
])

val_transform_seg = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
])

# === 4. Load segmentation model ===
def load_segmentation_model(model_path):
    model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

unet_model = load_segmentation_model("/kaggle/input/x-ray-segmention-model/xray_Segmention_model.pth")

# === 5. Custom Dataset ===
class EnsembleDataset(Dataset):
    def __init__(self, df, image_dir, transform_chexnet, transform_segmentation, seg_model):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform_chexnet = transform_chexnet
        self.transform_seg = transform_segmentation
        self.seg_model = seg_model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["Scan ID"]
        label = row["label"]
        image_path = os.path.join(self.image_dir, f"{image_id}.png")

        image = Image.open(image_path).convert("RGB")
        img_chexnet = self.transform_chexnet(image)

        # segmentation
        img_for_seg = self.transform_seg(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.seg_model(img_for_seg)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.float32)

        image_resized = image.resize((512, 512))
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        masked_image = image_np * np.expand_dims(mask, axis=-1)
        masked_image_pil = Image.fromarray((masked_image * 255).astype(np.uint8))
        img_segmented = self.transform_chexnet(masked_image_pil)

        return {
            "chexnet_img": img_chexnet,
            "chexnet2_img": img_chexnet,
            "seg_img": img_segmented,
            "label": torch.tensor(label, dtype=torch.float32)
        }

# === 6. Load Classifier Models ===
def get_resnet101_model():
    model = resnet101(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))
    return model.to(DEVICE)

def get_chexnet_model(num_classes=2):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(1024, num_classes)
    return model.to(DEVICE)

model_loaded_seg = get_resnet101_model()
model_loaded_seg.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/best_fc_only_resnet101_epoch10_Segmentation.pth", map_location=DEVICE))

model_loaded_chexnet_all = get_chexnet_model()
model_loaded_chexnet_all.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/chexnet_fc_only_epoch7.pth", map_location=DEVICE))

model_loaded_chexnet_2_classes = get_chexnet_model()
model_loaded_chexnet_2_classes.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/chexnet_fc_only_epoch9_2classes.pth", map_location=DEVICE))

# === 7. Ensemble Loader ===
IMAGE_DIR = "/kaggle/input/our-pacs-data-yolo-640/Images/Images"
ensemble_dataset = EnsembleDataset(
    df=v1_test,
    image_dir=IMAGE_DIR,
    transform_chexnet=val_transform_chex,
    transform_segmentation=val_transform_seg,
    seg_model=unet_model
)
ensemble_loader = DataLoader(ensemble_dataset, batch_size=1, shuffle=False, num_workers=0)

# === 8. Ensemble Prediction ===
def soft_voting_ensemble(models_info, dataloader, device, weights=None):
    all_probs = []

    for model, input_key in models_info:
        model.eval()
        model.to(device)
        probs = []

        with torch.no_grad():
            for batch in dataloader:
                image = batch[input_key].to(device)
                output = model(image)
                prob = torch.softmax(output, dim=1)[:, 1]
                probs.append(prob.cpu().numpy())

        all_probs.append(np.concatenate(probs))

    all_probs = np.stack(all_probs, axis=0)
    if weights is None:
        weights = np.ones(len(models_info)) / len(models_info)

    final_probs = np.average(all_probs, axis=0, weights=weights)
    return final_probs

models_info = [
    (model_loaded_seg, "seg_img"),
    (model_loaded_chexnet_all, "chexnet_img"),
    (model_loaded_chexnet_2_classes, "chexnet2_img")
]

# Run ensemble
ensemble_probs = soft_voting_ensemble(models_info, ensemble_loader, DEVICE)
final_preds = [1 if p >= 0.5 else 0 for p in ensemble_probs]
true_labels = [sample["label"].item() for sample in ensemble_dataset]

# === 9. Evaluation ===
f1 = f1_score(true_labels, final_preds)
precision = precision_score(true_labels, final_preds)
recall = recall_score(true_labels, final_preds)
cm = confusion_matrix(true_labels, final_preds)

print(f"âœ… Ensemble F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Ensemble")
plt.grid(False)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# === Step 1: Get model predictions (probabilities) and true labels ===
# Assumes: ensemble_probs â†’ list of predicted probs (floats)
#          v1_test["label"].tolist() â†’ list of true labels
y_scores = ensemble_probs  # output from soft_voting_ensemble
y_true = v1_test["label"].tolist()

# === Step 2: Try thresholds from 0.0 to 1.0 ===
thresholds = np.linspace(0, 1, 101)
precisions, recalls, f1s = [], [], []

for t in thresholds:
    preds = [1 if p >= t else 0 for p in y_scores]
    precisions.append(precision_score(y_true, preds, zero_division=0))
    recalls.append(recall_score(y_true, preds, zero_division=0))
    f1s.append(f1_score(y_true, preds, zero_division=0))

# === Step 3: Find best threshold (by F1 score) ===
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]
print(f"\nâœ… Best Threshold = {best_threshold:.2f} with F1 = {f1s[best_idx]:.4f}")

# === Step 4: Final predictions using best threshold ===
final_preds = [1 if p >= best_threshold else 0 for p in y_scores]

# === Step 5: Classification Report and Confusion Matrix ===
print("\nğŸ“Š Classification Report:")
print(classification_report(y_true, final_preds, target_names=["Normal", "Abnormal"]))

print("\nğŸ“‰ Confusion Matrix:")
cm = confusion_matrix(y_true, final_preds)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (Threshold = {best_threshold:.2f})")
plt.grid(False)
plt.show()

# === Optional: Plot Precision, Recall, F1 vs Threshold ===
plt.figure(figsize=(10, 5))
plt.plot(thresholds, precisions, label="Precision", color='blue')
plt.plot(thresholds, recalls, label="Recall", color='green')
plt.plot(thresholds, f1s, label="F1 Score", color='red')
plt.axvline(best_threshold, color='gray', linestyle='--', label=f"Best Threshold = {best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Metrics vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



def predict_image_class(image_path):
    import torch
    import torch.nn as nn
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import resnet101, densenet121
    import segmentation_models_pytorch as smp

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load transforms
    transform_chexnet = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
    ])
    transform_seg = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
    ])

    # === Load segmentation model
    def load_seg_model(path):
        model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=1)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model.to(DEVICE)

    seg_model = load_seg_model("/kaggle/input/x-ray-segmention-model/xray_Segmention_model.pth")

    # === Load classification models
    def get_resnet101_model():
        model = resnet101(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))
        return model.to(DEVICE)

    def get_chexnet_model():
        model = densenet121(weights=None)
        model.classifier = nn.Linear(1024, 2)
        return model.to(DEVICE)

    model_seg = get_resnet101_model()
    model_seg.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/best_fc_only_resnet101_epoch10_Segmentation.pth", map_location=DEVICE))
    model_seg.eval()

    model_chex_all = get_chexnet_model()
    model_chex_all.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/chexnet_fc_only_epoch7.pth", map_location=DEVICE))
    model_chex_all.eval()

    model_chex_2 = get_chexnet_model()
    model_chex_2.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/chexnet_fc_only_epoch9_2classes.pth", map_location=DEVICE))
    model_chex_2.eval()

    # === Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # â†’ CheXNet input
    img_chexnet = transform_chexnet(image).unsqueeze(0).to(DEVICE)

    # â†’ Segmentation preprocessing
    img_seg_input = transform_seg(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = torch.sigmoid(seg_model(img_seg_input)).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.float32)

    # Apply mask
    image_resized = image.resize((512, 512))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    masked_image = image_np * np.expand_dims(mask, axis=-1)
    masked_pil = Image.fromarray((masked_image * 255).astype(np.uint8))
    img_segmented = transform_chexnet(masked_pil).unsqueeze(0).to(DEVICE)

    # === Inference from each model
    def get_prob(model, img_tensor):
        with torch.no_grad():
            out = model(img_tensor)
            prob = torch.softmax(out, dim=1)[0, 1].item()
        return prob

    p1 = get_prob(model_seg, img_segmented)
    p2 = get_prob(model_chex_all, img_chexnet)
    p3 = get_prob(model_chex_2, img_chexnet)

    # === Soft voting
    probs = [p1, p2, p3]
    final_prob = np.mean(probs)
    pred_class = 1 if final_prob >= 0.71  else 0

    label = "Abnormal" if pred_class == 1 else "Normal"

    # === Print result
    print(f"\nğŸ–¼ï¸� Image: {image_path}")
    print(f"ğŸ“Š Ensemble Probability: {final_prob:.4f}")
    print(f"ğŸ§  Prediction: {label}")
    return label



predict_image_class("/kaggle/input/our-pacs-data-yolo-640/Images/Images/01010078100.png")



def predict_image_class(dicom_path):
    import torch
    import torch.nn as nn
    import numpy as np
    import pydicom
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import resnet101, densenet121
    import segmentation_models_pytorch as smp

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load transforms
    transform_chexnet = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
    ])
    transform_seg = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
    ])

    # === Load segmentation model
    def load_seg_model(path):
        model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=1)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model.to(DEVICE)

    seg_model = load_seg_model("/kaggle/input/x-ray-segmention-model/xray_Segmention_model.pth")

    # === Load classification models
    def get_resnet101_model():
        model = resnet101(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))
        return model.to(DEVICE)

    def get_chexnet_model():
        model = densenet121(weights=None)
        model.classifier = nn.Linear(1024, 2)
        return model.to(DEVICE)

    model_seg = get_resnet101_model()
    model_seg.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/best_fc_only_resnet101_epoch10_Segmentation.pth", map_location=DEVICE))
    model_seg.eval()

    model_chex_all = get_chexnet_model()
    model_chex_all.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/chexnet_fc_only_epoch7.pth", map_location=DEVICE))
    model_chex_all.eval()

    model_chex_2 = get_chexnet_model()
    model_chex_2.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/chexnet_fc_only_epoch9_2classes.pth", map_location=DEVICE))
    model_chex_2.eval()

    # === Load DICOM and convert to PIL image
    dicom = pydicom.dcmread(dicom_path)
    image_array = dicom.pixel_array.astype(np.float32)

    # Normalize to [0,255] then convert to uint8
    image_array -= image_array.min()
    image_array /= image_array.max()
    image_array *= 255
    image_array = image_array.astype(np.uint8)

    # If grayscale, convert to RGB
    if len(image_array.shape) == 2:
        image_pil = Image.fromarray(image_array).convert("RGB")
    else:
        image_pil = Image.fromarray(image_array)

    # â†’ CheXNet input
    img_chexnet = transform_chexnet(image_pil).unsqueeze(0).to(DEVICE)

    # â†’ Segmentation preprocessing
    img_seg_input = transform_seg(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = torch.sigmoid(seg_model(img_seg_input)).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.float32)

    # Apply mask
    image_resized = image_pil.resize((512, 512))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    masked_image = image_np * np.expand_dims(mask, axis=-1)
    masked_pil = Image.fromarray((masked_image * 255).astype(np.uint8))
    img_segmented = transform_chexnet(masked_pil).unsqueeze(0).to(DEVICE)

    # === Inference from each model
    def get_prob(model, img_tensor):
        with torch.no_grad():
            out = model(img_tensor)
            prob = torch.softmax(out, dim=1)[0, 1].item()
        return prob

    p1 = get_prob(model_seg, img_segmented)
    p2 = get_prob(model_chex_all, img_chexnet)
    p3 = get_prob(model_chex_2, img_chexnet)

    # === Soft voting
    probs = [p1, p2, p3]
    final_prob = np.mean(probs)
    pred_class = 1 if final_prob >= 0.71 else 0
    label = "Abnormal" if pred_class == 1 else "Normal"

    # === Print result
    print(f"\nğŸ“� DICOM File: {dicom_path}")
    print(f"ğŸ“Š Ensemble Probability: {final_prob:.4f}")
    print(f"ğŸ§  Prediction: {label}")
    return label




predict_image_class("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/000434271f63a053c4128a0ba6352c7f.dicom")


