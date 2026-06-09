import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
import torch
import pandas as pd
import torchvision.transforms as T
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


from kaggle_secrets import UserSecretsClient
import wandb

user_secrets = UserSecretsClient()

wandb.login(key=user_secrets.get_secret("wandb"))


def get_transforms():
    """Get data transformations for training and validation"""
    train_transform = T.Compose([
        T.ToTensor(),
        T.Resize((224,224)),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Resize((224,224)),
    ])

    return train_transform, val_transform


# Custom dataset class to extract and process patches from HDF5 slides
def CustomDataset(data_path, slide_names, is_train=True,transform=None):
        data = []
        targets = []
        patch_size = 224
        slide_names = slide_names
        is_train = is_train
        transform = transform

        with h5py.File(data_path, "r") as f:
            images = f["images/Train"] if is_train else f["images/Test"]
            coords = f["spots/Train"] if is_train else f["spots/Test"]
            for slide_name in slide_names:
                slide = np.array(images[slide_name])
                spots = np.array(coords[slide_name])
                df = pd.DataFrame(spots)

                # Apply specific x/y shifts for alignment if needed
                x_shift, y_shift = 0, 0
                if slide_name == 'S_1':
                    x_shift, y_shift = 50, 60
                elif slide_name == 'S_2':
                    x_shift, y_shift = 95, 55
                df['x'] -= x_shift
                df['y'] -= y_shift

                # Extract patches and corresponding targets
                for _, row in df.iterrows():
                    x_center, y_center = int(row['x']), int(row['y'])
                    x0 = x_center - patch_size // 2
                    y0 = y_center - patch_size // 2
                    patch = slide[y0:y0 + patch_size, x0:x0 + patch_size, :]
                    if transform:
                        patch = transform(patch)
                    data.append(patch)
                    targets.append(row[2:].values)
        return data, targets


# Dataset paths and slide setup
data_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
train_slides = ['S_1', 'S_2', 'S_3', 'S_4', 'S_5','S_6']
# Load full dataset and split into train/val subsets
dataset = CustomDataset(data_path, slide_names=train_slides, is_train=True,transform=get_transforms()[0])


data_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
test_slide = ['S_7']
test_data = CustomDataset(data_path, slide_names=test_slide, is_train=False,transform=get_transforms()[1])


train_images, val_images, train_targets, val_targets = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=42)

train_data = {'image': train_images, 'targets': train_targets}
val_data = {'image': val_images, 'targets': val_targets}


from datasets import Dataset, concatenate_datasets

def load_dataset_in_batches(data_dict, batch_size=500):
    datasets = []
    for i in range(0, len(data_dict['image']), batch_size):
        batch = {
            key: val[i:i+batch_size]
            for key, val in data_dict.items()
        }
        datasets.append(Dataset.from_dict(batch))
    return concatenate_datasets(datasets)


train_data = load_dataset_in_batches(train_data)
val_data = load_dataset_in_batches(val_data)
dataset = {'train': train_data, 'val': val_data}


from datasets import load_dataset, DatasetDict
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTModel, TrainingArguments, Trainer
from torch import nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file as safetensors_load_file
import logging
import os
import json
import shutil
from datasets import Dataset

def get_device():
    """Get the appropriate device"""
    if torch.cuda.is_available():
        return torch.device("cuda")  
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device is: {device}")


class ViTRegressionModel(nn.Module):
    def __init__(self,num_targets):
        super(ViTRegressionModel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224',device_map=0, ignore_mismatched_sizes=True)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_targets)

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token
        values = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            # print(f"label: {labels}")
            # print(f"pixel_values: {values}")
            loss = loss_fct(values, labels)
        return (loss, values) if loss is not None else values



def train_model(dataset, value_column_name, test_split, output_dir, num_train_epochs, learning_rate):
    # Load the dataset
    dataset = dataset

    # Split the dataset into train and test
    dataset = DatasetDict({
        'train': dataset['train'],
        'val': dataset['val']
    })
    
    def collate_fn(batch):
        pixel_values = torch.stack([
            item['image'] if isinstance(item['image'], torch.Tensor) else torch.tensor(item['image']) 
            for item in batch])
        labels = torch.tensor([item[value_column_name] for item in batch], dtype=torch.float)
        return {'pixel_values': pixel_values, 'labels': labels}


    model = ViTRegressionModel(num_targets=35)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_steps=10,
        save_total_limit=5,
        logging_steps=10,
        remove_unused_columns=False,
        resume_from_checkpoint=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        data_collator=collate_fn,
    )

    # Add logging to inspect the model outputs and labels
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        mse = ((preds - labels) ** 2).mean().item()
        return {"mse": mse}

    trainer.compute_metrics = compute_metrics

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")


def predict_batch(test_images, checkpoint_path="/kaggle/working/results/checkpoint-2090/model.safetensors"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViTRegressionModel(num_targets=35)
    state_dict = safetensors_load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if isinstance(test_images, Image.Image):
        test_images = [test_images]

    batch = torch.stack([
        img if isinstance(img, torch.Tensor) else transform(img)
        for img in test_images
    ]).to(device)

    with torch.no_grad():
        output = model(pixel_values=batch)
        preds = output[1] if isinstance(output, tuple) else output

    return preds.cpu().numpy().tolist()


train_model(dataset=dataset,
            value_column_name='targets',
            test_split=0.2,
            output_dir='/kaggle/working/results',
            num_train_epochs=5,
            learning_rate=1e-4)


def predict_all(images, batch_size=30):
    """
    All images will be divided into batches and input into the model.
    """
    all_preds = []

    # all image len
    total = len(images)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_images = images[start:end]

        preds = predict_batch(batch_images)
        all_preds.extend(preds)

    return all_preds
preds = predict_all(test_data[0])


len(preds)


with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as f:
    train_spots = f["spots/Train"]
    
    # Dictionary to store DataFrames for each slide
    train_spot_tables = {}
    
    for slide_name in train_spots.keys():
        # Load dataset as NumPy structured array
        spot_array = np.array(train_spots[slide_name])
        
        # Convert to DataFrame
        df = pd.DataFrame(spot_array)
        
        # Store in dictionary
        train_spot_tables[slide_name] = df

# Example: Display the spots table for slide 'S_1'
train_spot_tables['S_1']


# Display spot table for Test slide (only the spot coordinates on 2D array)
with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as f:
    test_spots = f["spots/Test"]
    spot_array = np.array(test_spots['S_7'])
    test_spot_table = pd.DataFrame(spot_array)
    
# Show the test spots coordinates for slide 'S_7'
test_spot_table


# Use the cell type columns from the train spots table; assuming first two columns are (x, y)
cell_type_columns = train_spot_tables['S_1'].columns[2:].values  # Expecting 35 cell types here
indices = test_spot_table.index.values  # All spots on the Test slide

# Create a 2D array of random floats between 0 and 2 for each spot and cell type
predicted_labels = pd.DataFrame(preds, columns=cell_type_columns, index=indices)

predicted_labels.head()


# Prepare submission DataFrame: spot_id column and then predictions for each cell type
submission_df = predicted_labels.copy()
submission_df.insert(0, 'ID', submission_df.index)

# Save the submission file as submission.csv
submission_df.to_csv("./submission.csv", index=False)
print("Submission file 'submission.csv' created!")

