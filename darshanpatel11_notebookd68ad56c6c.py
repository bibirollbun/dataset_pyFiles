# import zipfile
# from glob import glob
# all_zip_files = glob("/kaggle/input/kuzushiji-recognition/*.zip")
# for f in all_zip_files:
#     with zipfile.ZipFile(f, "r") as zip_file:
#         zip_file.extractall(path='/kaggle/working/')

import zipfile
with zipfile.ZipFile("/kaggle/input/kuzushiji-recognition/test_images.zip", "r") as zip_file:
    zip_file.extractall(path='/kaggle/working/test_images/')

with zipfile.ZipFile("/kaggle/input/kuzushiji-recognition/train_images.zip", "r") as zip_file:
    zip_file.extractall(path='/kaggle/working/train_images/')


import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import DetrForObjectDetection, DetrFeatureExtractor
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

# Enable Hugging Face hub to work in offline mode
import os

os.environ["HF_HUB_OFFLINE"] = "0"

# Load data
train_df = pd.read_csv("/kaggle/input/kuzushiji-recognition/train.csv")#train_df = pd.read_csv("./data/train.csv")
unicode_translation_df = pd.read_csv("/kaggle/input/kuzushiji-recognition/unicode_translation.csv")#unicode_translation_df = pd.read_csv("./data/unicode_translation.csv")

# Create a dictionary mapping Unicode to character
unicode_to_char = unicode_translation_df.set_index("Unicode")["char"].to_dict()
unique_labels = sorted(set(unicode_to_char.values()))
label_to_class_idx = {label: idx for idx, label in enumerate(unique_labels)}


class KuzushijiDataset(Dataset):
    def __init__(self, df, image_dir, feature_extractor):
        self.df = df
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["image_id"]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        labels = self.df.iloc[idx]["labels"]
        boxes = []
        labels_list = []
        for i in range(0, len(labels.split()), 5):
            label, x, y, w, h = labels.split()[i : i + 5]
            boxes.append([int(x), int(y), int(x) + int(w), int(y) + int(h)])
            # labels_list.append(ord(unicode_to_char[label]))
            label_char = unicode_to_char[label]
            class_idx = label_to_class_idx[label_char]
            labels_list.append(class_idx)

        encoding = self.feature_extractor(images=image, return_tensors="pt", size={"height":800, "width":800})
        encoding["labels"] = torch.tensor(labels_list, dtype=torch.int64)
        encoding["boxes"] = torch.tensor(boxes, dtype=torch.float32)

        return encoding


def collate_fn(batch):
    return {
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        "labels": [b["labels"] for b in batch],
        "boxes": [b["boxes"] for b in batch],
    }


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Define model and feature extractor
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.config.num_labels = len(unique_labels)
    model.class_labels_classifier = torch.nn.Linear(model.config.d_model, model.config.num_labels)
    model.to(device)
except EnvironmentError as e:
    print(f"Error loading model: {e}")
    print(
        "Please ensure the model is cached locally or adjust the code to download it manually."
    )


# Train function
def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = [
            {"class_labels": labels, "boxes": boxes}
            for labels, boxes in zip(batch["labels"], batch["boxes"])
        ]
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(loader)


# Evaluate function
def evaluate(model, device, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = [
                {"class_labels": labels, "boxes": boxes}
                for labels, boxes in zip(batch["labels"], batch["boxes"])
            ]
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(loader)


# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    # train_dataset = KuzushijiDataset(
    #     train_df.iloc[train_idx], "./data/train_images", feature_extractor
    # )
    # val_dataset = KuzushijiDataset(
    #     train_df.iloc[val_idx], "./data/train_images", feature_extractor
    # )
    train_dataset = KuzushijiDataset(
        train_df.iloc[train_idx], "/kaggle/working/train_images", feature_extractor
    )
    val_dataset = KuzushijiDataset(
        train_df.iloc[val_idx], "/kaggle/working/train_images", feature_extractor
    )
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(5):
        train(model, device, train_loader, optimizer)
        evaluate(model, device, val_loader)

    scores.append(evaluate(model, device, val_loader))

print(f"5-fold cross-validation score: {np.mean(scores)}")

# Make predictions on test set
# test_dataset = KuzushijiDataset(
#     pd.read_csv("./data/sample_submission.csv"),
#     "./data/test_images",
#     feature_extractor,
# )
test_dataset = KuzushijiDataset(
    pd.read_csv("/kaggle/input/kuzushiji-recognition/sample_submission.csv"),
    "/kaggle/working/test_images",
    feature_extractor,
)
test_loader = DataLoader(
    test_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn
)

model.eval()
test_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        pixel_values = batch["pixel_values"].to(device)
        outputs = model(pixel_values=pixel_values)
        for i, output in enumerate(outputs):
            scores = output["scores"].cpu().numpy()
            boxes = output["pred_boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            for score, box, label in zip(scores, boxes, labels):
                if score > 0.5:
                    test_preds.append(
                        {
                            "image_id": test_dataset.df.iloc[i]["image_id"],
                            "label": chr(label),
                            "x": (box[0] * 512 + box[2] * 512) / 2,
                            "y": (box[1] * 512 + box[3] * 512) / 2,
                        }
                    )

# Save predictions to submission.csv
submission_df = pd.DataFrame(test_preds)
submission_df = (
    submission_df.groupby("image_id")
    .apply(
        lambda x: " ".join(
            f"{label} {x:.2f} {y:.2f}"
            for label, x, y in zip(x["label"], x["x"], x["y"])
        )
    )
    .reset_index()
)
submission_df.columns = ["image_id", "labels"]
submission_df.to_csv("submission.csv", index=False)


# len(unique_labels)




