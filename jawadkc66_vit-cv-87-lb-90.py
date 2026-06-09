import requests
import math
import matplotlib.pyplot as plt
import shutil
from getpass import getpass
from PIL import Image, UnidentifiedImageError
from requests.exceptions import HTTPError
from io import BytesIO
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTForImageClassification
import pandas as pd



HOME = "/kaggle/input/sheep-classification-challenge-2025"

original_train_dir = Path(f'{HOME}/Sheep Classification Images/train')
labels_file = Path(f'{HOME}/Sheep Classification Images/train_labels.csv')

data_dir = Path('sheap_imagefolder_dataset')

# Clean up and create the root directory for ImageFolder
if data_dir.exists():
    print(f"Removing existing organized dataset directory: {data_dir}")
    shutil.rmtree(data_dir)
data_dir.mkdir(parents=True, exist_ok=True)
print(f"Created new directory for organized dataset: {data_dir}")

print(f"Reading labels from {labels_file}...")
df_labels = pd.read_csv(labels_file)

print(f"Organizing images from {original_train_dir} into {data_dir}...")
copied_count = 0
missing_count = 0
for index, row in df_labels.iterrows():
    filename = row['filename']
    label = str(row['label']) # Ensure label is a string for directory naming

    # Create label directory if it doesn't exist
    label_dir = data_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    # Source and destination paths
    src_image_path = original_train_dir / filename
    dst_image_path = label_dir / filename

    if src_image_path.exists():
        shutil.copy(src_image_path, dst_image_path)
        copied_count +=1
    else:
        print(f"Warning: Image '{filename}' not found in '{original_train_dir}'")
        missing_count += 1

print(f"Data organization complete. Copied {copied_count} images. Missing {missing_count} images")



ds = ImageFolder(data_dir)
indices = torch.randperm(len(ds)).tolist()
n_val = math.floor(len(indices) * .15)
train_ds = torch.utils.data.Subset(ds, indices[:-n_val])
val_ds = torch.utils.data.Subset(ds, indices[-n_val:])


label2id = {}
id2label = {}

for i, class_name in enumerate(ds.classes):
    label2id[class_name] = str(i)
    id2label[str(i)] = class_name


class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)
collator = ImageClassificationCollator(feature_extractor)
train_loader = DataLoader(train_ds, batch_size=8, collate_fn=collator, num_workers=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collator, num_workers=2)


class Classifier(pl.LightningModule):

    def __init__(self, model, lr: float = 2e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.model = model
        self.forward = self.model.forward
        self.val_acc = Accuracy(
            task='multiclass' if model.config.num_labels > 2 else 'binary',
            num_classes=model.config.num_labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"val_loss", outputs.loss)
        acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
        self.log(f"val_acc", acc, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


pl.seed_everything(42)
classifier = Classifier(model, lr=2e-5)
trainer = pl.Trainer(accelerator='gpu', devices=1, precision="16-mixed", max_epochs=4)
trainer.fit(classifier, train_loader, val_loader)


model_save_path = Path('sheap_vit_model')
model_save_path.mkdir(parents=True, exist_ok=True)

model.save_pretrained(model_save_path)
feature_extractor.save_pretrained(model_save_path)

print(f"Model and feature extractor saved to {model_save_path}")


from sklearn.metrics import f1_score
import numpy as np
import torch

# Ensure the model is in evaluation mode
model.eval()

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        labels = batch['labels'].numpy() # Get labels
        
        # Move inputs to the correct device
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}

        outputs = model(**inputs)
        logits = outputs.logits
        predictions = logits.argmax(-1).cpu().numpy()
        
        all_preds.extend(predictions)
        all_labels.extend(labels)

# Calculate Macro F1 score
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
print(f"Validation Macro F1 Score: {f1:.4f}")


import os
test_dir = Path(f'{HOME}/Sheep Classification Images/test')
output_dir = Path('submission_output')
output_dir.mkdir(parents=True, exist_ok=True) 
submission_file = output_dir / 'submission.csv'

test_image_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

predictions = []

model.eval()
if torch.cuda.is_available():
    model.to('cuda')

for image_file in test_image_files:
    image_path = test_dir / image_file
    try:
        image = Image.open(image_path).convert('RGB') 
        # Apply the same transformations as training/validation
        inputs = feature_extractor(images=image, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[str(predicted_class_idx)] # Convert index to string
        predictions.append({'filename': image_file, 'label': predicted_label})
    except UnidentifiedImageError:
        print(f"Skipping file {image_file} as it could not be identified as an image.")
    except Exception as e:
        print(f"Error processing file {image_file}: {e}")

# Create a DataFrame and save to CSV
submission_df = pd.DataFrame(predictions)
submission_df.to_csv(submission_file, index=False)

print(f"Submission file created at {submission_file}")
print(submission_df.head())


# --- Workaround for Kaggle UI ---
# Copy submission.csv into the lightning_logs/version_X directory
# This is to help if the Kaggle submission UI dropdown only shows files from there.
import shutil
if 'trainer' in globals() and hasattr(trainer, 'logger') and hasattr(trainer.logger, 'log_dir'):
    try:
        log_dir_path = Path(trainer.logger.log_dir)
        target_submission_path = log_dir_path / "submission.csv"
        shutil.copy(submission_file, target_submission_path)
        print(f"Copied submission.csv to {target_submission_path} as a workaround for Kaggle UI.")
    except Exception as e:
        print(f"Could not copy submission.csv to log_dir: {e}")
else:
    print("Could not copy submission.csv to log_dir: Trainer object or logger not found as expected.")
# --- End Workaround ---

