import pandas as pd
import numpy as np
import os
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from skimage.transform import resize
import warnings
warnings.filterwarnings('ignore')

class SimpleMRIDataset(Dataset):
    """Simplified dataset for MRI data with basic preprocessing."""

    def __init__(self, df, data_dir, target_size=(32, 32, 32)):
        self.df = df
        self.data_dir = data_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row['BraTS21ID']).zfill(5)
        patient_path = os.path.join(self.data_dir, patient_id)

        # Load all 4 modalities
        modalities = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
        volume_data = []

        for modality in modalities:
            modality_path = os.path.join(patient_path, modality)
            dicom_files = sorted([f for f in os.listdir(modality_path) if f.endswith('.dcm')])

            if len(dicom_files) == 0:
                volume = np.zeros(self.target_size, dtype=np.float32)
            else:
                try:
                    middle_idx = len(dicom_files) // 2
                    dicom_file = os.path.join(modality_path, dicom_files[middle_idx])
                    ds = pydicom.dcmread(dicom_file)
                    slice_data = ds.pixel_array.astype(np.float32)
                    slice_resized = resize(slice_data, self.target_size[:2], preserve_range=True)
                    volume = np.repeat(slice_resized[:, :, np.newaxis], self.target_size[2], axis=2)
                    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
                except:
                    volume = np.zeros(self.target_size, dtype=np.float32)

            volume_data.append(volume)

        multi_modal_volume = np.stack(volume_data, axis=0)
        return torch.FloatTensor(multi_modal_volume), torch.FloatTensor([row['MGMT_value']])

class Simple3DCNN(nn.Module):
    """Simplified 3D CNN for better performance."""

    def __init__(self, in_channels=4, num_classes=1):
        super(Simple3DCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(4)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

def build_and_train_model(train_df, val_df, train_data_dir):
    """Build and train the 3D CNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = SimpleMRIDataset(train_df, train_data_dir)
    val_dataset = SimpleMRIDataset(val_df, train_data_dir)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = Simple3DCNN(in_channels=4, num_classes=1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    best_val_auc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                preds = torch.sigmoid(outputs).cpu().numpy()
                train_preds.extend(preds.flatten())
                train_labels.extend(labels.cpu().numpy().flatten())

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(preds.flatten())
                val_labels.extend(labels.cpu().numpy().flatten())

        # Calculate metrics
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_auc = roc_auc_score(val_labels, val_preds)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_auc




if __name__ == "__main__":
    # Load data
    train_labels_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv'
    train_data_dir = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train'
    test_data_dir = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test'

    train_labels = pd.read_csv(train_labels_path)

    # Exclude problematic cases
    exclude_ids = ['00109', '00123', '00709']
    train_labels = train_labels[~train_labels['BraTS21ID'].astype(str).isin(exclude_ids)]

    # Split data
    train_df, val_df = train_test_split(
        train_labels, test_size=0.2, random_state=42, stratify=train_labels['MGMT_value']
    )

    # Train model
    model, best_val_auc = build_and_train_model(train_df, val_df, train_data_dir)

    # Generate validation predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    val_dataset = SimpleMRIDataset(val_df, train_data_dir)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    val_predictions = []
    patient_ids = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            start_idx = i * val_loader.batch_size
            end_idx = min(start_idx + len(probs), len(val_df))
            batch_patient_ids = val_df.iloc[start_idx:end_idx]['BraTS21ID'].values

            for patient_id, prob in zip(batch_patient_ids, probs):
                patient_ids.append(str(patient_id).zfill(5))
                val_predictions.append(float(prob))

    val_predictions_df = pd.DataFrame({
        'BraTS21ID': patient_ids,
        'MGMT_value': val_predictions
    })
    val_predictions_df.to_csv('validation_predictions.csv', index=False)

    # Generate test predictions
    test_patient_ids = sorted([f for f in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, f))])

    test_predictions = []
    for patient_id in test_patient_ids:
        patient_path = os.path.join(test_data_dir, patient_id)

        try:
            modalities = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
            volume_data = []

            for modality in modalities:
                modality_path = os.path.join(patient_path, modality)
                dicom_files = sorted([f for f in os.listdir(modality_path) if f.endswith('.dcm')])

                if len(dicom_files) == 0:
                    volume = np.zeros((32, 32, 32), dtype=np.float32)
                else:
                    try:
                        middle_idx = len(dicom_files) // 2
                        dicom_file = os.path.join(modality_path, dicom_files[middle_idx])
                        ds = pydicom.dcmread(dicom_file)
                        slice_data = ds.pixel_array.astype(np.float32)
                        slice_resized = resize(slice_data, (32, 32), preserve_range=True)
                        volume = np.repeat(slice_resized[:, :, np.newaxis], 32, axis=2)
                        volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
                    except:
                        volume = np.zeros((32, 32, 32), dtype=np.float32)

                volume_data.append(volume)

            multi_modal_volume = np.stack(volume_data, axis=0)
            image_tensor = torch.FloatTensor(multi_modal_volume).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                prob = torch.sigmoid(output).cpu().numpy()[0, 0]

            test_predictions.append({'BraTS21ID': patient_id, 'MGMT_value': float(prob)})

        except:
            test_predictions.append({'BraTS21ID': patient_id, 'MGMT_value': 0.5})

    test_predictions_df = pd.DataFrame(test_predictions)
    test_predictions_df.to_csv('submission.csv', index=False)

    print(f"Best validation ROC-AUC: {best_val_auc:.4f}")
    print(f"Validation predictions saved: {len(val_predictions_df)} patients")
    print(f"Test predictions saved: {len(test_predictions_df)} patients")




