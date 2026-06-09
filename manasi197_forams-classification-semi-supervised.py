import pandas as pd
import numpy as np
from glob import glob
import tifffile
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix, log_loss, accuracy_score
import xgboost as xgb
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import feature, measure
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
import os
import re

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)



def compute_3d_features(volume):
    """Extract advanced 3D features from a volume."""
    # Basic statistical features
    features = [
        volume.mean(),  # Mean intensity
        volume[volume > 0].mean() if np.any(volume > 0) else 0,  # Mean of non-zero voxels
        volume.max(),  # Maximum intensity
        volume.std(),  # Standard deviation
    ]

    # Shape features (Hu moments on central slice)
    slice_z = volume[:, :, 64].astype(np.uint8)
    moments = cv2.HuMoments(cv2.moments(slice_z)).flatten()
    features.extend(moments)

    # Texture features (3D local binary patterns - simplified)
    lbp = np.zeros_like(volume)
    for i in range(1, volume.shape[0]-1):
        for j in range(1, volume.shape[1]-1):
            for k in range(1, volume.shape[2]-1):
                center = volume[i, j, k]
                neighbors = volume[i-1:i+2, j-1:j+2, k-1:k+2].flatten()
                lbp[i, j, k] = np.sum(neighbors > center)
    features.append(lbp.mean())

    # Gradient features
    grad_x = np.abs(np.diff(volume, axis=0)).mean()
    grad_y = np.abs(np.diff(volume, axis=1)).mean()
    grad_z = np.abs(np.diff(volume, axis=2)).mean()
    features.extend([grad_x, grad_y, grad_z])

    return np.array(features)

def extract_cnn_features(image_path):
    """Extract features from JPG visualizations using pre-trained ResNet18."""
    # Load pre-trained ResNet18
    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    # Remove the final classification layer
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)

    # Extract features
    with torch.no_grad():
        features = feature_extractor(img_tensor).flatten().numpy()
    return features

def make_feature_array(vol_list, vis_list=None):
    """Compute features from volumes and optionally visualizations."""
    n_features = 14 + 512  # 14 from 3D + 512 from ResNet18
    X = np.full((len(vol_list), n_features), np.nan)
    for i, vol_path in enumerate(tqdm(vol_list)):
        # Load volume
        volume = tifffile.imread(vol_path)
        # Extract 3D features
        X[i, :14] = compute_3d_features(volume)
        # Extract CNN features from visualization if provided
        if vis_list:
            vis_path = vol_path.replace('volumes/volumes', 'visualizations/visualizations').replace('.tif', '.jpg')
            X[i, 14:] = extract_cnn_features(vis_path)
    return X



def build_autoencoder():
    """Build a 3D convolutional autoencoder for anomaly detection."""
    input_shape = (64, 64, 64, 1)  # Downsample to save memory
    input_vol = Input(shape=input_shape)
    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(input_vol)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_vol, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def detect_anomalies(volumes, autoencoder, threshold=0.05):
    """Detect anomalies (unknown class) using autoencoder reconstruction error."""
    anomalies = []
    for vol in tqdm(volumes):
        # Resize 3D volume to 64x64x64
        vol_resized = np.zeros((64, 64, 64))
        for z in range(64):
            vol_resized[:, :, z] = cv2.resize(vol[:, :, z], (64, 64))
        vol_resized = vol_resized.reshape(1, 64, 64, 64, 1) / 255.0
        recon = autoencoder.predict(vol_resized, verbose=0)
        mse = np.mean((recon - vol_resized) ** 2)
        anomalies.append(mse > threshold)
    return np.array(anomalies)




# Read labeled data
labels = pd.read_csv('/kaggle/input/forams-classification-2025/labelled.csv', index_col='id')['label']
vol_list_labeled = sorted(glob('/kaggle/input/forams-classification-2025/volumes/volumes/labelled/*.tif'))
vis_list_labeled = sorted(glob('/kaggle/input/forams-classification-2025/visualizations/visualizations/labelled/*.jpg'))

# Read unlabeled data
vol_list_unlabeled = sorted(glob('/kaggle/input/forams-classification-2025/volumes/volumes/unlabelled/*.tif'))
vis_list_unlabeled = sorted(glob('/kaggle/input/forams-classification-2025/visualizations/visualizations/unlabelled/*.jpg'))

# Extract features
print("Extracting features for labeled data...")
X_labeled = make_feature_array(vol_list_labeled, vis_list_labeled)
scaler = StandardScaler()
X_labeled = scaler.fit_transform(X_labeled)



print("Training autoencoder for anomaly detection...")
volumes_labeled = [tifffile.imread(vol) for vol in tqdm(vol_list_labeled)]
volumes_labeled_resized = []
for vol in volumes_labeled:
    vol_resized = np.zeros((64, 64, 64))
    for z in range(64):
        # Resize each 2D slice along the z-axis
        slice_idx = int(z * vol.shape[2] / 64)  # Sample slices evenly
        vol_resized[:, :, z] = cv2.resize(vol[:, :, slice_idx], (64, 64))
    vol_resized = vol_resized.reshape(64, 64, 64, 1) / 255.0
    volumes_labeled_resized.append(vol_resized)
autoencoder = build_autoencoder()
autoencoder.fit(np.array(volumes_labeled_resized), np.array(volumes_labeled_resized), epochs=10, batch_size=8, verbose=1)


# Initialize model (XGBoost)
model = xgb.XGBClassifier(objective='multi:softprob', num_class=14, eval_metric='mlogloss', random_state=42)

# Initial training on labeled data
print("Training initial model on labeled data...")
model.fit(X_labeled, labels)

# Cross-validation
oof = cross_val_predict(model, X_labeled, labels, cv=5)
print(f"Initial F1: {f1_score(labels, oof, average='macro'):.3f}")
print(f"Initial Accuracy: {accuracy_score(labels, oof):.3f}")

# Pseudo-labeling
print("Extracting features for unlabeled data...")
X_unlabeled = make_feature_array(vol_list_unlabeled, vis_list_unlabeled)
X_unlabeled = scaler.transform(X_unlabeled)

# Detect anomalies (unknown class)
print("Detecting anomalies...")
volumes_unlabeled = [tifffile.imread(vol) for vol in tqdm(vol_list_unlabeled[:2000])]  # Subset for speed
anomalies = detect_anomalies(volumes_unlabeled, autoencoder)

# Pseudo-labeling loop
n_iterations = 3
confidence_threshold = 0.9
for iter in range(n_iterations):
    print(f"Pseudo-labeling iteration {iter+1}/{n_iterations}...")
    probs = model.predict_proba(X_unlabeled)
    max_probs = np.max(probs, axis=1)
    pseudo_labels = np.argmax(probs, axis=1)
    
    # Assign unknown class to anomalies or low-confidence predictions
    pseudo_labels[max_probs < confidence_threshold] = 14
    if iter == 0:  # Apply anomalies in first iteration
        pseudo_labels[:len(anomalies)][anomalies] = 14
    
    # Select high-confidence pseudo-labels
    confident = max_probs >= confidence_threshold
    X_combined = np.vstack([X_labeled, X_unlabeled[confident]])
    y_combined = np.concatenate([labels, pseudo_labels[confident]])
    
    # Retrain model
    model.fit(X_combined, y_combined)
    
    # Evaluate on labeled data
    oof = cross_val_predict(model, X_labeled, labels, cv=5)
    print(f"Iter {iter+1} F1: {f1_score(labels, oof, average='macro'):.3f}")



print("Generating submission...")
y_pred = model.predict(X_unlabeled)
# Re-apply anomaly detection for final predictions
y_pred[:len(anomalies)][anomalies] = 14

unlabelled_index = pd.read_csv('/kaggle/input/forams-classification-2025/unlabelled.csv', index_col='id').index
submission = pd.Series(y_pred, index=unlabelled_index[:len(y_pred)], name='label')
submission.to_csv('submission.csv')

print("Submission saved. First few rows:")
print(submission.head())

# Visualize confusion matrix
sns.heatmap(confusion_matrix(labels, cross_val_predict(model, X_labeled, labels, cv=5)), annot=True)
plt.title("Confusion Matrix (Labeled Data)")
plt.show()

