#!pip install umap-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap.umap_ as umap
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway, kruskal
from scipy.stats import levene, bartlett
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import shapiro
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')


DATA_DIR = '/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot'
CSV_PATH = '/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/train.csv'

df = pd.read_csv(CSV_PATH)

data_list = []
labels = []

expected_shape = (128,128,125)

for _, row in df.iterrows():
    file_path = os.path.join(DATA_DIR, row['id'])
    
    try:
        cube = np.load(file_path)

        if cube.shape != expected_shape:
            continue
        mean_spectrum = cube.reshape(-1, cube.shape[2]).mean(axis=0)
        data_list.append(mean_spectrum)
        labels.append(row['label'])

    except Exception as e:
        print(f'Error with {file_path} {e}')
        

X = np.array(data_list)
y = np.array(labels)


reducer = umap.UMAP(random_state=42)
X_embedded = reducer.fit_transform(X)

plt.figure(figsize=(10, 8), constrained_layout=True)
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette='tab10')
plt.title('UMAP Projection of Hyperspectral Data by Label')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("umap_plot.png")
plt.show()



df_spectra = pd.DataFrame(X)
df_spectra['id'] = y

plt.figure(figsize=(12, 6))
for label in df_spectra['id'].unique():
    mean_spectrum = df_spectra[df_spectra['id'] == label].drop('id', axis=1).mean()
    plt.plot(mean_spectrum, label=label)

plt.title('Mean Reflectance Spectra per Class')
plt.xlabel('Bands (Spectral Channels)')
plt.ylabel('Reflectance')
plt.legend(title='Class Label')
plt.tight_layout()
plt.show()


df_plot = df_spectra.copy()
selected_bands = [10, 50, 100] 

for band in selected_bands:
    plt.figure(figsize=(14, 5))
    sns.boxplot(data=df_plot, x='id', y=band)
    plt.title(f'Boxplot for Band {band}')
    plt.xlabel('Class')
    plt.ylabel(f'Reflectance at Band {band}')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    sns.violinplot(data=df_plot, x='id', y=band)
    plt.title(f'Violinplot for Band {band}')
    plt.xlabel('Class')
    plt.ylabel(f'Reflectance at Band {band}')
    plt.tight_layout()
    plt.show


correlation_matrix = df_spectra.drop('id', axis=1).corr()

corr_unstacked = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
corr_pairs = corr_unstacked.unstack().dropna()
top_corr = corr_pairs.abs().sort_values(ascending=False).head(10)

print("Top 10 most correlated band pairs (by absolute correlation):")
for (band1, band2), corr_val in top_corr.items():
    print(f"Bands {band1} & {band2}: correlation = {corr_val:.3f}")

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True)
plt.title('Spectral Band Correlation Heatmap')
plt.xlabel('Band')
plt.ylabel('Band')
plt.tight_layout()
plt.show()


mean_spectra_by_class = df_spectra.groupby('id').mean()
linked = linkage(mean_spectra_by_class, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked, labels=mean_spectra_by_class.index.tolist(), leaf_rotation=90)
plt.title('Dendrogram of Class Mean Spectra')
plt.xlabel('Class')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()


R_band, G_band, B_band = 90, 60, 30

def create_rgb_image_for_class(class_label):
    sample_path = os.path.join(DATA_DIR, df[df['label'] == class_label]['id'].iloc[0])
    cube = np.load(sample_path)

    rgb_image = np.stack([
        cube[:, :, R_band],
        cube[:, :, G_band],
        cube[:, :, B_band]
    ], axis=-1)

    
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    return rgb_image


class_labels = df['label'].unique()[:4]  
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, class_label in zip(axes.flatten(), class_labels):
    rgb_image = create_rgb_image_for_class(class_label)
    ax.imshow(rgb_image)
    ax.set_title(f'Class: {class_label}')
    ax.axis('off')

plt.tight_layout()
plt.show()


anova_results = {}
for band in range(X.shape[1]): 
    groups = [X[y == label, band] for label in np.unique(y)]  
    f_stat, p_value = f_oneway(*groups)
    anova_results[band] = p_value


significant_bands_anova = {band: p for band, p in anova_results.items() if p < 0.05}
print("ANOVA significant bands:", significant_bands_anova)

kruskal_results = {}
for band in range(X.shape[1]):
    groups = [X[y == label, band] for label in np.unique(y)] 
    h_stat, p_value = kruskal(*groups)
    kruskal_results[band] = p_value

significant_bands_kruskal = {band: p for band, p in kruskal_results.items() if p < 0.05}
print("Kruskal-Wallis significant bands:", significant_bands_kruskal)

plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

for band in range(X.shape[1]):
    _, p_value = shapiro(X[:, band])
    print(f"Shapiro-Wilk test for Band {band}: p-value = {p_value}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10')
plt.title('PCA Projection')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.tight_layout()
plt.show()


class1_data = X[y == df['label'].unique()[0]]  
class2_data = X[y == df['label'].unique()[1]] 


t_stat, p_value_t = ttest_ind(class1_data, class2_data, axis=0)

u_stat, p_value_u = mannwhitneyu(class1_data.flatten(), class2_data.flatten())

print("t-test p-values:", p_value_t)
print("Mann-Whitney U test p-value:", p_value_u)


levene_results = {}
for band in range(X.shape[1]):
    groups = [X[y == label, band] for label in np.unique(y)]
    stat, p_value = levene(*groups)
    levene_results[band] = p_value

bartlett_results = {}
for band in range(X.shape[1]):
    groups = [X[y == label, band] for label in np.unique(y)]
    stat, p_value = bartlett(*groups)
    bartlett_results[band] = p_value

significant_bands_levene = {band: p for band, p in levene_results.items() if p < 0.05}
significant_bands_bartlett = {band: p for band, p in bartlett_results.items() if p < 0.05}

print("Levene Test significant bands:", significant_bands_levene)
print("Bartlett Test significant bands:", significant_bands_bartlett)


def calculate_ndvi(cube, red_band=30, nir_band=90):
    red = cube[:, :, red_band]
    nir = cube[:, :, nir_band]
    return (nir - red) / (nir + red)

sample_path = os.path.join(DATA_DIR, df[df['label'] == df['label'].unique()[0]]['id'].iloc[0])
cube = np.load(sample_path)
ndvi_image = calculate_ndvi(cube)

plt.imshow(ndvi_image, cmap='RdYlGn')
plt.title("NDVI (NIR-Red)/(NIR+Red)")
plt.colorbar()
plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=25, random_state=42)
kmeans.fit(X)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("KMeans Clustering")
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar(label='Cluster')
plt.show()


def calculate_snr(X, y, class_label):
    class_data = X[y == class_label]
    mean_signal = class_data.mean(axis=0)
    noise = class_data.std(axis=0)
    snr = mean_signal / noise
    return snr

snr_values = {}
for label in np.unique(y):
    snr_values[label] = calculate_snr(X, y, label)

for label, snr in snr_values.items():
    print(f"SNR for class {label}: {snr[:5]}")  


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import kornia.augmentation as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


 train_df = pd.read_csv('/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/train.csv')
base_path = '/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot'


BANDS = 100
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_BANDS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HyperspectralDataset(Dataset):
    def __init__(self, df, base_path, patch_size=64, augment=False, num_bands=100):
        self.df = df
        self.base_path = base_path
        self.patch_size = patch_size
        self.augment = augment
        self.num_bands = num_bands
        
        self.transform = nn.Sequential(
            K.RandomHorizontalFlip(p=0.3),     
            K.RandomVerticalFlip(p=0.3),
            K.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), p=0.5),
            K.RandomCrop((patch_size, patch_size), padding=4, p=0.5)
        )
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.base_path}/{row['id']}"

        try:
            img = np.load(img_path)

            if len(img.shape) == 2:
                img = np.repeat(img[:, :, np.newaxis], self.num_bands, axis=2)
            elif len(img.shape) == 3:
                if img.shape[2] > self.num_bands:
                    img = img[:, :, :self.num_bands]
                elif img.shape[2] < self.num_bands:
                    pad_width = ((0, 0), (0, 0), (0, self.num_bands - img.shape[2]))
                    img = np.pad(img, pad_width, mode='constant')

            img = img.astype(np.float32) / 65535.0  # Normalize image

            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Convert to [C, H, W]

            if self.augment:
                img = self.transform(img.unsqueeze(0)).squeeze(0)

            if img.shape[1] != self.patch_size or img.shape[2] != self.patch_size:
                img = F.interpolate(img.unsqueeze(0), size=(self.patch_size, self.patch_size), mode='bilinear').squeeze(0)

            label = torch.tensor(row['label'], dtype=torch.long)  

            if label > 0:
                label = label - 1

            return img, label

        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            dummy_img = torch.zeros(self.num_bands, self.patch_size, self.patch_size)
            dummy_label = torch.tensor(0, dtype=torch.long)  
            return dummy_img, dummy_label



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return out.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class HyperspectralCNN(nn.Module):
    def __init__(self, in_channels=NUM_BANDS, num_classes=100):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes) 
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        
        x = self.conv2(x)
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


sample = np.load('/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot/sample1024.npy')  # (128, 128, 125)
plt.imshow(sample[:, :, 0])
plt.title('First channel')
plt.colorbar()
plt.show()


def evaluate_model(model, loader, criterion, device = DEVICE):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probabilities, dim=1)
            
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item() * inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_labels)




def train_model(model, train_loader, val_loader, epochs, criterion, optimizer):
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        valid_samples = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                continue
                
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if torch.isnan(outputs).any():
                continue
                
            loss = criterion(outputs.squeeze(), labels) 
            
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                valid_samples += inputs.size(0)
        
        if valid_samples > 0:
            train_loss /= valid_samples
            val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if len(val_preds.shape) == 2: 
                val_preds = np.argmax(val_preds, axis=1)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Sample predictions: {val_preds[:5]}, True labels: {val_labels[:5]}")
            
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'Spectrum_CNN.pth')
        else:
            print(f"Epoch {epoch+1}: No valid training samples")
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return model


train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
train_dataset = HyperspectralDataset(train_df, base_path, augment=True)
val_dataset = HyperspectralDataset(val_df, base_path, augment=False)
    
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    



model = HyperspectralCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)


model = train_model(model, train_loader, val_loader, EPOCHS, criterion, optimizer)
    
model.load_state_dict(torch.load('Spectrum_CNN.pth'))


model = HyperspectralCNN(in_channels=100).to(DEVICE)
model.load_state_dict(torch.load('Spectrum_CNN.pth'))
model.eval()
print("Model weights:", list(model.parameters())[0][0, 0, :5])
test_input = torch.randn(1, 100, 64, 64).to(DEVICE)

class TestHyperspectralDataset(Dataset):
    def __init__(self, test_csv, base_path, patch_size=64, num_bands=100):
        self.df = pd.read_csv(test_csv)
        self.base_path = base_path
        self.patch_size = patch_size
        self.num_bands = num_bands
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_path, row['id'])
        
        try:
            img = np.load(img_path)
            
            if len(img.shape) == 2:
                img = np.repeat(img[:, :, np.newaxis], self.num_bands, axis=2)
            elif len(img.shape) == 3:
                if img.shape[2] > self.num_bands:
                    img = img[:, :, :self.num_bands] 
                elif img.shape[2] < self.num_bands:
                    pad_width = ((0, 0), (0, 0), (0, self.num_bands - img.shape[2]))
                    img = np.pad(img, pad_width, mode='constant')
            
            normalized_img = np.zeros_like(img)
            for band in range(img.shape[2]):
                band_data = img[:, :, band]
                if np.max(band_data) > 0:  
                    normalized_img[:, :, band] = (band_data - np.min(band_data)) / (np.max(band_data) - np.min(band_data))
            
            img_tensor = torch.tensor(normalized_img, dtype=torch.float32).permute(2, 0, 1)
            
            if img_tensor.shape[1] != self.patch_size or img_tensor.shape[2] != self.patch_size:
                img_tensor = F.interpolate(img_tensor.unsqueeze(0), 
                                         size=(self.patch_size, self.patch_size),
                                         mode='bilinear').squeeze(0)
            
            return img_tensor, row['id']
        
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            dummy_img = torch.zeros(self.num_bands, self.patch_size, self.patch_size)
            return dummy_img, row['id']






test_csv_path = '/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/test.csv'
base_path = '/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot'

test_dataset = TestHyperspectralDataset(test_csv_path, base_path, num_bands=100)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

predictions = []
ids = []

with torch.no_grad():
    for inputs, img_ids in test_loader:
        inputs = inputs.to(DEVICE)
        
        if torch.isnan(inputs).any():
            print(f"Skipping batch with NaN values")
            predictions.extend([50] * len(img_ids))  
            ids.extend(img_ids)
            continue
            
        outputs = model(inputs)
        preds = outputs.squeeze().cpu().numpy()
        print(preds)
        preds = np.clip(preds, 1, 100).round().astype(int)
        print(preds)
        if isinstance(preds, np.ndarray) and preds.ndim > 1:
            preds = np.max(preds, axis=1)  

        if len(preds) != len(img_ids):
            preds = preds[:len(img_ids)]  

        predictions.extend(preds.tolist())  
        ids.extend(img_ids)



submission_df = pd.DataFrame({'ID': ids, 'TARGET': predictions})
submission_df.to_csv('submission.csv', index=False)
print("Submission created successfully")
print("\nSubmission preview:")
print(submission_df.head())

