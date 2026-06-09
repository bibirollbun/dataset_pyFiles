# Import Libraries
import os
import cv2
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Print PyTorch version and set device
print(f"PyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_to_tfrecord(df, label_columns, image_dir, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting to TFRecord"):
            img_path = os.path.join(image_dir, row['Image_name'])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((224, 224), dtype=np.uint8)
            else:
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
            labels = row[label_columns].values.astype(np.float32)

            feature = {
                'image': _bytes_feature(img_bytes),
                'label': _float_list_feature(labels),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

convert_to_tfrecord(train_data, label_columns, '/kaggle/input/grand-xray-slam-division-a/train1/', 'train.tfrecord')
convert_to_tfrecord(val_data, label_columns, '/kaggle/input/grand-xray-slam-division-a/train1/', 'val.tfrecord')


from torch.utils.data import Dataset

class TFRecordDataset(Dataset):
    def __init__(self, tfrecord_path, transforms=None):
        self.tfrecord_path = tfrecord_path
        self.transforms = transforms
        # Load the entire dataset into memory as a list of (image, label) pairs
        self.data = []
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        for example_proto in dataset:
            example = self._parse_tfrecord(example_proto)
            self.data.append(example)

    def _parse_tfrecord(self, example_proto):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([14], tf.float32),
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_jpeg(example['image'], channels=3).numpy()
        label = example['label'].numpy()
        return image, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(label)


train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = TFRecordDataset('train.tfrecord', transforms=train_transforms)
val_dataset = TFRecordDataset('val.tfrecord', transforms=val_transforms)


batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Disable multiprocessing
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,  # Disable multiprocessing
    pin_memory=True,
)

print("TFRecord DataLoaders created successfully.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(num_classes=14):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    for param in model.layer4.parameters():
        param.requires_grad = True
    return model

model = create_model().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    return model

model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)


!ls /kaggle/input/grand-xray-slam-division-a/


!ls /kaggle/working/


def convert_test_to_tfrecord(df, image_dir, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting TEST to TFRecord"):
            img_path = os.path.join(image_dir, row['Image_name'])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((224, 224), dtype=np.uint8)
            else:
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_bytes = cv2.imencode('.jpg', img)[1].tobytes()

            feature = {
                'image': _bytes_feature(img_bytes),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

# Load sample_submission (for image names)
sample_submission = pd.read_csv("/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv")
convert_test_to_tfrecord(sample_submission, "/kaggle/input/grand-xray-slam-division-a/test1", "test.tfrecord")


# Create the same model structure
model = create_model(num_classes=14).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))


class TFRecordTestDataset(Dataset):
    def __init__(self, tfrecord_path, transforms=None):
        self.transforms = transforms
        self.data = []
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        for example_proto in dataset:
            example = self._parse_tfrecord(example_proto)
            self.data.append(example)

    def _parse_tfrecord(self, example_proto):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_jpeg(example["image"], channels=3).numpy()
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transforms:
            image = self.transforms(image)
        return image


# Transforms (same as val)
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# DataLoader
test_dataset = TFRecordTestDataset("test.tfrecord", transforms=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Load model
model = create_model(num_classes=14).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Predictions
predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        batch_preds = torch.sigmoid(outputs).cpu().numpy()
        predictions.append(batch_preds)

predictions = np.vstack(predictions)
predictions = predictions[:len(sample_submission)]


label_columns


label_columns = sample_submission.columns[1:]
print("Final label columns:", label_columns)


submission_df


submission_df = sample_submission.copy()
submission_df[label_columns] = predictions
submission_df.to_csv("submission.csv", index=False)
print("✅ Submission file created: submission.csv")


submission_df

