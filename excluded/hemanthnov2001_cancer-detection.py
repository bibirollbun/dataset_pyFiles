import os



#! pip install kaggle


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image




train_path = '/kaggle/input/isic-2024-challenge/train-metadata.csv'
test_path = '/kaggle/input/isic-2024-challenge/test-metadata.csv'
image_folder = '/kaggle/input/isic-2024-challenge/train-image/image'
training_validation_hdf5 = h5py.File(f"/kaggle/input/isic-2024-challenge/train-image.hdf5", 'r')
testing_hdf5 = h5py.File(f"/kaggle/input/isic-2024-challenge/test-image.hdf5", 'r')
data=pd.read_csv(train_path)


data.head(20)


num_images = 20

for i in range(num_images):
  target = data.target.iloc[i]

  label = f"Target: {target}"
  isic_id = data.isic_id.iloc[i]
  byte_string = training_validation_hdf5[isic_id][()]

  nparr = np.frombuffer(byte_string, np.uint8)
  image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[...,::-1] # reverse last axis for bgr -> rgb

  plt.subplot(4, 5, i + 1)
  plt.imshow(image)
  plt.axis('off')

  plt.title(label, fontsize=8)

# Display the grid of images
plt.tight_layout()
plt.show()


data.info()


data.shape


data.columns


data['target'].value_counts().plot(kind='bar', figsize=(20,10))


from collections import Counter
print("Original class distribution:", Counter(data['target']))


print(data['target'].unique().tolist())


print(data['iddx_1'].unique().tolist())


print(data['iddx_2'].unique().tolist())


print(data['iddx_3'].unique().tolist())


print(data['iddx_4'].unique().tolist())


print(data['anatom_site_general'].unique().tolist())


missing_data_report = pd.DataFrame({
    'Total Missing': data.isnull().sum(),
    'Percentage Missing': (data.isnull().sum() / len(data)) * 100
})
print("Detailed Missing Data Report:")
print(missing_data_report[missing_data_report['Total Missing'] > 0])




sns.heatmap(data.isnull()) #show  missing values


missing_values = data.isnull().sum()
plt.figure(figsize=(20, 6))
missing_values.plot(kind='bar')
plt.title('Number of Missing Values per Column', fontsize=16)
plt.xlabel('Columns', fontsize=14)
plt.ylabel('Number of Missing Values', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()





co_matrix = data.select_dtypes(include=[float, int])


corr_matrix = co_matrix .corr()

fig, ax = plt.subplots(figsize=(20, 10))
colormap = sns.color_palette("Reds")
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt='.2f', cmap=colormap)

plt.show()


crosstab = pd.crosstab(data['anatom_site_general'], data['target'])
print(crosstab)

from scipy.stats import chi2_contingency
chi2, p, _, _ = chi2_contingency(crosstab)
print(f"Chi-square statistic: {chi2}, p-value: {p}")


crosstab = pd.crosstab(data['tbp_lv_location'], data['target'])
print(crosstab)

from scipy.stats import chi2_contingency
chi2, p, _, _ = chi2_contingency(crosstab)
print(f"Chi-square statistic: {chi2}, p-value: {p}")


crosstab = pd.crosstab(data['tbp_lv_location_simple'], data['target'])
print(crosstab)

from scipy.stats import chi2_contingency
chi2, p, _, _ = chi2_contingency(crosstab)
print(f"Chi-square statistic: {chi2}, p-value: {p}")


crosstab = pd.crosstab(data['image_type'], data['target'])
print(crosstab)

from scipy.stats import chi2_contingency
chi2, p, _, _ = chi2_contingency(crosstab)
print(f"Chi-square statistic: {chi2}, p-value: {p}")


data_cleaned = data.drop(["lesion_id", "anatom_site_general", "attribution","iddx_1", "tbp_tile_type","copyright_license", "image_type", "patient_id", "iddx_full","iddx_2", "iddx_3", "iddx_4", "iddx_5", "mel_mitotic_index", "mel_thick_mm"], axis=1)
data_cleaned.columns


data_cleaned_imputed = data_cleaned.copy()

data_cleaned_imputed['age_approx'] = data_cleaned_imputed['age_approx'].fillna(data_cleaned['age_approx'].mean())

data_cleaned_imputed['sex'] = data_cleaned_imputed['sex'].fillna(data_cleaned['sex'].mode()[0])



data_report = pd.DataFrame({
    'Total Missing':data_cleaned_imputed.isnull().sum(),
    'Percentage Missing': (data_cleaned_imputed.isnull().sum() / len(data_cleaned_imputed)) * 100
})
print("Detailed Missing Data Report:")
print(data_report[data_report['Total Missing'] > 0])


#save the clean data to df
df = data_cleaned_imputed


df.columns


#One= encode
cols = ['target']
for col in cols:
    df= pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    df.drop([col], axis=1, inplace=True)

df.rename(columns={'target_0': 'target_absent', 'target_1': 'target_present'}, inplace=True)
df.drop(["target_absent"], axis=1, inplace=True)


cols = ['sex']
for col in cols:
    df= pd.concat([df, pd.get_dummies(df[col], prefix=col).astype(int)], axis=1)
    df.drop([col], axis=1, inplace=True)

df.drop(["sex_female"], axis=1, inplace=True)


#one hot encode tbp_lv_location_simple
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)

cols = ['tbp_lv_location_simple']
encoded_features = encoder.fit_transform(df[cols])
df= pd.concat([df, pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cols))], axis=1)
df.drop(cols, axis=1, inplace=True)



#one hot encode anatom_site_general
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)

cols = ['tbp_lv_location']
encoded_features = encoder.fit_transform(df[cols])
df= pd.concat([df, pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cols))], axis=1)
df.drop(cols, axis=1, inplace=True)


df.columns


pd.set_option('display.max_columns', None)


df.head(40)


import pandas as pd


control_cancer_images = data[data.target == 1].sample(n=80, random_state=42)


train_cancer_images = data[data.target == 1].drop(control_cancer_images.index)


num_zeros_in_target = data[data.target == 0].shape[0]


control_healthy_images = data[data.target == 0].sample(n=len(control_cancer_images), random_state=42)
train_healthy_images = data[data.target == 0].sample(n=len(train_cancer_images), random_state=42)


control_images = pd.concat([control_cancer_images, control_healthy_images])
train_images = pd.concat([train_cancer_images, train_healthy_images])


print(f'Control array length: {len(control_images)}')
print(f'Train array length: {len(train_images)}')



image_size = 100

# Data Augmentation Pipeline
transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.9, 1.1)),  # Random zoom from 90% to 110%
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),  # Random rotation by up to 15 degrees
    transforms.ColorJitter(contrast=0.2),  # Random contrast adjustment
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Custom Dataset Class with Augmentation
class CustomImageDataset(Dataset):
    def __init__(self, data, training_validation_hdf5, transform=None):
        self.data = data
        self.training_validation_hdf5 = training_validation_hdf5
        self.transform = transform

    def __len__(self):
        return len(self.data) * 3  # Increase the dataset size by 3x for augmentation

    def __getitem__(self, idx):
        # Apply modulo operation to get index within original dataset length
        original_idx = idx % len(self.data)
        target = self.data.target.iloc[original_idx]
        isic_id = self.data['isic_id'].iloc[original_idx]

        byte_string = self.training_validation_hdf5[isic_id][()]
        nparr = np.frombuffer(byte_string, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[..., ::-1]  # BGR to RGB
        image = cv2.resize(image, (image_size, image_size))  # Resize to square

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, target  # Return image and target

# Create the Dataset and DataLoader for training and validation
dataset = CustomImageDataset(train_images, training_validation_hdf5, transform=transform)

# Create DataLoader
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Residual Block Definition
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout after first convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(dropout_prob)  # Dropout after second convolution

        # For matching dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)  # Apply dropout
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)  # Apply dropout
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# ResNet Architecture Definition
class SimpleResNet(nn.Module):
    def __init__(self, block, layers):
        super(SimpleResNet, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)

        # Calculate the correct input size for the fully connected layer
        dummy_input = torch.randn(1, 3, image_size, image_size)
        dummy_output = self.conv_forward(dummy_input)
        fc_input_size = dummy_output.view(dummy_output.size(0), -1).shape[1]

        self.fc = nn.Linear(fc_input_size, 1)  # Adjusted for binary classification

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def conv_forward(self, x):  # New method to encapsulate the convolutional part
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x):
        x = self.conv_forward(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)

# Initialize the model
model = SimpleResNet(ResidualBlock, [2, 2, 2])  # Example: 2 blocks per layer group
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# History dictionary to store metrics
history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': []
}


from sklearn.metrics import roc_curve, auc

def calculate_partial_auc(y_true, y_scores, min_fpr=0, max_fpr=0.3):
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Create a mask for the desired FPR range
    mask = (fpr >= min_fpr) & (fpr <= max_fpr)

    # If no points in the specified range, return 0
    if not np.any(mask):
        return 0.0

    # Interpolate TPR and FPR
    fpr_subset = fpr[mask]
    tpr_subset = tpr[mask]

    # Ensure the first point is at min_fpr and last point is at max_fpr
    if fpr_subset[0] > min_fpr:
        fpr_subset = np.insert(fpr_subset, 0, min_fpr)
        tpr_subset = np.insert(tpr_subset, 0, 0)

    if fpr_subset[-1] < max_fpr:
        fpr_subset = np.append(fpr_subset, max_fpr)
        tpr_subset = np.append(tpr_subset, tpr_subset[-1])

    # Calculate pAUC
    pauc = auc(fpr_subset, tpr_subset)

    # Normalize pAUC by the FPR range
    pauc_normalized = pauc / (max_fpr - min_fpr)

    return pauc_normalized


num_epochs =3
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.view(-1), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Collect predictions and labels for pAUC calculation
        predicted = outputs.view(-1).detach().numpy()
        all_preds.extend(predicted)
        all_labels.extend(labels.numpy())

        # Calculate training accuracy
        predicted_binary = (outputs.view(-1) > 0.5).float()
        total += labels.size(0)
        correct += (predicted_binary == labels.float()).sum().item()

    # Calculate average loss and training accuracy for the epoch
    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # Calculate pAUC for training data
    train_pauc = calculate_partial_auc(np.array(all_labels), np.array(all_preds))

    # Validation phase (similar modification)
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss += criterion(outputs.view(-1), labels.float()).item()
            predicted = outputs.view(-1).numpy()
            val_preds.extend(predicted)
            val_labels.extend(labels.numpy())

            predicted_binary = (outputs.view(-1) > 0.5).float()
            val_total += labels.size(0)
            val_correct += (predicted_binary == labels.float()).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    # Calculate pAUC for validation data
    val_pauc = calculate_partial_auc(np.array(val_labels), np.array(val_preds))

    # Print epoch details including pAUC
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {avg_loss:.4f}, '
          f'Validation Loss: {avg_val_loss:.4f}, '
          f'Train Accuracy: {train_accuracy:.2f}%, '
          f'Validation Accuracy: {val_accuracy:.2f}%, '
          f'Train pAUC: {train_pauc:.4f}, '
          f'Validation pAUC: {val_pauc:.4f}')

# Optional: Add pAUC to history tracking
history['train_pauc'] = []
history['val_pauc'] = []


from sklearn.metrics import roc_curve, precision_recall_curve, auc



def plot_roc_and_pauc(val_labels, val_preds, min_fpr=0, max_fpr=0.3):
  
 
    # Compute full ROC curve
    fpr, tpr, _ = roc_curve(val_labels, val_preds)
    roc_auc = auc(fpr, tpr)

    # Compute partial AUC
    pauc = calculate_partial_auc(val_labels, val_preds, min_fpr, max_fpr)

    # Highlight the partial ROC range
    mask = (fpr >= min_fpr) & (fpr <= max_fpr)
    fpr_partial = fpr[mask]
    tpr_partial = tpr[mask]

    # ROC Curve Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
    plt.plot(fpr_partial, tpr_partial, label=f'Partial AUC (pAUC = {pauc:.2f})', color='orange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.title('ROC Curve with Partial AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(val_labels, val_preds)
    pr_auc = auc(recall, precision)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})', color='green')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'roc_auc': roc_auc,
        'partial_auc': pauc,
        'pr_auc': pr_auc
    }

# Usage in training loop or after training
metrics = plot_roc_and_pauc(np.array(val_labels), np.array(val_preds))


pauc = metrics['partial_auc']
print(f"The pAUC value is: {pauc:.4f}")


df.head()



X= df.drop(columns=['isic_id', 'target_present'], axis=1)
y= df["target_present"]


X.head()


y


#treating the imbalanced data
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


print("New class distribution in training set after SMOTE:", Counter(y_train_res))

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
pd.Series(y_train).value_counts().plot(kind='bar', ax=axs[0], color='skyblue')
y_train.value_counts().plot(kind='bar', ax=axs[0])
axs[0].set_title('Original Class Distribution in Training Set')
axs[0].set_xlabel('Class')
axs[0].set_ylabel('Count')

y_train_res.value_counts().plot(kind='bar', ax=axs[1])
axs[1].set_title('Balanced Class Distribution in Training Set (After SMOTE)')
axs[1].set_xlabel('Class')
axs[1].set_ylabel('Count')

plt.show()


balanced_counts = y_train_res.value_counts()
print("Class distribution after SMOTE:")
print(balanced_counts)


from sklearn.preprocessing import StandardScaler
#  Instantiate the StandardScaler
scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

dt_model = DecisionTreeClassifier( max_depth = 5,  random_state=42)

#  Train the model using the scaled training data
dt_model.fit(X_train_scaled, y_train_res)

#  Make predictions on the scaled test data
y_pred_dt = dt_model.predict(X_test_scaled)


# Evaluate the model's performance on the test data
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt,average="weighted")
print(f"Accuracy: {accuracy_dt:.2f} Precision: {precision_dt:.2f}, Recall: {recall_dt:.2f}, F1 score:{f1_dt:.2f} ")


print(f'{classification_report(y_test, y_pred_dt)}')

cm = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square =True, cmap = 'Blues');
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Confusion Matrix'
plt.title(all_sample_title, size = 15)
plt.show()


#pAUC
y_scores_dt = dt_model.predict_proba(X_test_scaled)[:, 1]

# Calculate full ROC curve
fpr, tpr, _ = roc_curve(y_test, y_scores_dt)

# Calculate full AUC
full_auc = auc(fpr, tpr)

min_fpr, max_fpr = 0, 0.3
pauc_dt = calculate_partial_auc(y_test, y_scores_dt, min_fpr, max_fpr)

print(f"Full AUC: {full_auc:.2f}")
print(f"Partial AUC (0 to 0.3 FPR): {pauc_dt:.2f}")

import matplotlib.pyplot as plt

# Mask for partial ROC range
mask = (fpr >= min_fpr) & (fpr <= max_fpr)
fpr_partial = fpr[mask]
tpr_partial = tpr[mask]

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {full_auc:.2f})', color='blue')
plt.plot(fpr_partial, tpr_partial, label=f'Partial AUC (pAUC = {pauc_dt:.2f})', color='orange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.title('Decision Tree ROC Curve with Partial AUC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()





from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train_res)
# Make predictions on the scaled test data
y_pred_nb = nb_model.predict(X_test_scaled)


# Evaluate the model's performance on the test data
accuracy_nb = accuracy_score(y_test, y_pred_nb )
precision_nb = precision_score(y_test, y_pred_nb )
recall_nb= recall_score(y_test, y_pred_nb )
f1_nb = f1_score(y_test, y_pred_nb ,average="weighted")



print(f"Accuracy: {accuracy_nb:.2f} Precision: {precision_nb:.2f}, Recall: {recall_nb:.2f}, F1 score:{f1_nb:.2f} ")

print(" ")
print(f'{classification_report(y_test, y_pred_nb )}')
print(" ")
cm = confusion_matrix(y_test, y_pred_nb )
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square =True, cmap = 'Blues');
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Confusion Matrix'
plt.title(all_sample_title, size = 15)
plt.show()


#pAUC
y_scores_nb = nb_model.predict_proba(X_test_scaled)[:, 1]

# Calculate full ROC curve
fpr, tpr, _ = roc_curve(y_test, y_scores_nb)

# Calculate full AUC
full_auc = auc(fpr, tpr)

def calculate_partial_auc(fpr, tpr, min_fpr, max_fpr):
    mask = (fpr >= min_fpr) & (fpr <= max_fpr)
    partial_fpr = fpr[mask]
    partial_tpr = tpr[mask]
    return auc(partial_fpr, partial_tpr)

# Calculate partial AUC in a specific FPR range (e.g., 0 to 0.3)
min_fpr, max_fpr = 0, 0.3
pauc_nb = calculate_partial_auc(y_test, y_scores_nb, min_fpr, max_fpr)

# Print AUC and pAUC
print(f"Full AUC: {full_auc:.2f}")
print(f"Partial AUC (0 to 0.3 FPR): {pauc_nb:.2f}")

# Plot ROC curve with partial AUC
import matplotlib.pyplot as plt

# Mask for partial ROC range
mask = (fpr >= min_fpr) & (fpr <= max_fpr)
fpr_partial = fpr[mask]
tpr_partial = tpr[mask]

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {full_auc:.2f})', color='blue')
plt.plot(fpr_partial, tpr_partial, label=f'Partial AUC (pAUC = {pauc_nb:.2f})', color='orange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.title('Naive Bayes ROC Curve with Partial AUC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()


from sklearn.ensemble import RandomForestClassifier
rf_model =RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train_res)
#  Make predictions on the scaled test data
y_pred_rf = rf_model.predict(X_test_scaled)




# Evaluate the model's performance on the test data
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf,average="weighted")



print(f"Accuracy: {accuracy_rf:.3f} Precision: {precision_rf:.2f}, Recall: {recall_rf:.2f}, F1 score:{f1_rf:.2f} ")

print(" ")
print(f'{classification_report(y_test, y_pred_rf)}')
print(" ")

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square =True, cmap = 'Blues');
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Confusion Matrix'
plt.title(all_sample_title, size = 15)
plt.show()


y_scores_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Step 2: Define the function to calculate partial AUC
def calculate_partial_auc(y_true, y_scores, min_fpr, max_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    mask = (fpr >= min_fpr) & (fpr <= max_fpr)
    partial_fpr = fpr[mask]
    partial_tpr = tpr[mask]
    return auc(partial_fpr, partial_tpr)

# Step 3: Calculate partial AUC (pAUC) for Random Forest
min_fpr = 0
max_fpr = 0.3
pauc_rf = calculate_partial_auc(y_test, y_scores_rf, min_fpr, max_fpr)

print(f"Partial AUC (pAUC) for Random Forest: {pauc_rf:.4f}")

# Step 4: Calculate and plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_scores_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.4f})'.format(auc(fpr, tpr)))

# Step 5: Highlight the partial ROC curve
mask = (fpr >= min_fpr) & (fpr <= max_fpr)
plt.fill_between(fpr[mask], tpr[mask], step='post', alpha=0.3, color='orange', label=f'Partial ROC (pAUC = {pauc_rf:.4f})')

# Adding labels and legend
plt.title("ROC Curve with Partial AUC")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train_res)
#  Make predictions on the scaled test data
y_pred_lr= lr_model.predict(X_test_scaled)




# Evaluate the model's pelrormance on the test data
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr,average="weighted")



print(f"Accuracy: {accuracy_lr:.2f} Precision: {precision_lr:.2f}, Recall: {recall_lr:.2f}, F1 score:{f1_lr:.2f} ")

print(" ")
print(f'{classification_report(y_test, y_pred_lr)}')
print(" ")

cm_rf = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square =True, cmap = 'Blues');
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Confusion Matrix'
plt.title(all_sample_title, size = 15)
plt.show()


#pAUC
# Step 1: Get predicted probabilities for the positive class (class 1)
y_scores_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
min_fpr = 0
max_fpr = 0.3
pauc_lr = calculate_partial_auc(y_test, y_scores_lr, min_fpr, max_fpr)

print(f"Partial AUC (pAUC) for Logistic Regression: {pauc_lr:.4f}")

# Step 3: Plot the ROC curve and highlight the partial segment
fpr, tpr, _ = roc_curve(y_test, y_scores_lr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.4f})'.format(auc(fpr, tpr)))

# Highlight the partial ROC curve
mask = (fpr >= min_fpr) & (fpr <= max_fpr)
plt.fill_between(fpr[mask], tpr[mask], step='post', alpha=0.3, color='orange', label=f'Partial ROC (pAUC = {pauc_lr:.4f})')

plt.title("ROC Curve with Partial AUC")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


import xgboost as xgb
scale_pos_weight = len(y_train_res[y_train_res == 0]) / len(y_train_res[y_train_res == 1])
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
xgb_model.fit(X_train_scaled, y_train_res)


y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, y_pred_xgb))



# Evaluate the model's pelrormance on the test data
accuracy_xgb = accuracy_score(y_test,y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb,average="weighted")



print(f"Accuracy: {accuracy_xgb:.2f} Precision: {precision_xgb:.2f}, Recall: {recall_xgb:.2f}, F1 score:{f1_xgb:.2f} ")

print(" ")
print(f'{classification_report(y_test, y_pred_xgb)}')
print(" ")

cm_rf = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square =True, cmap = 'Blues');
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Confusion Matrix'
plt.title(all_sample_title, size = 15)
plt.show()


#pAUC

y_scores_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]


min_fpr = 0
max_fpr = 0.3
pauc_xgb = calculate_partial_auc(y_test, y_scores_xgb, min_fpr, max_fpr)

print(f"Partial AUC (pAUC) for XGBoost: {pauc_xgb:.4f}")


fpr, tpr, _ = roc_curve(y_test, y_scores_xgb)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.4f})'.format(auc(fpr, tpr)))

# Highlight the partial ROC curve
mask = (fpr >= min_fpr) & (fpr <= max_fpr)
plt.fill_between(fpr[mask], tpr[mask], step='post', alpha=0.3, color='orange', label=f'Partial ROC (pAUC = {pauc_xgb:.4f})')

plt.title("ROC Curve with Partial AUC")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()





# Define weights for each model
y_scores_xgb = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
y_scores_nb = nb_model.predict_proba(X_test)[:, 1]
y_scores_dt = dt_model.predict_proba(X_test)[:, 1]


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Assuming X_test, y_test, min_fpr, and max_fpr are defined

# Define the weights for the fusion
weights = [0.3, 0.3, 0.3]  # Example weights summing to 1
weights_pauc = 0.1

# Generate predictions from models
y_scores_xgb = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
y_scores_nb = nb_model.predict_proba(X_test)[:, 1]
y_scores_dt = dt_model.predict_proba(X_test)[:, 1]

# Calculate the fused scores using weighted average
y_scores_fusion = (
    weights[0] * y_scores_xgb +
    weights[1] * y_scores_nb +
    weights[2] * y_scores_dt
) + weights_pauc * pauc

# Generate binary predictions from fused scores
y_preds_fusion = (y_scores_fusion > 0.5).astype(int)

# Calculate the partial AUC for the fused model
fused_pauc = calculate_partial_auc(y_test, y_scores_fusion, min_fpr, max_fpr)
print(f"Partial AUC (pAUC) for Fused Model: {fused_pauc:.4f}")

# Plot the ROC curve with the partial AUC
fpr, tpr, _ = roc_curve(y_test, y_scores_fusion)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Fused ROC Curve (AUC = {:.4f})'.format(auc(fpr, tpr)))

# Highlight the partial ROC curve
mask = (fpr >= min_fpr) & (fpr <= max_fpr)
plt.fill_between(fpr[mask], tpr[mask], step='post', alpha=0.3, color='orange', label=f'Partial ROC (pAUC = {fused_pauc:.4f})')

plt.title("Fused ROC Curve with Partial AUC")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()



import pandas as pd

test_df = pd.read_csv('/kaggle/input/isic-2024-challenge/test-metadata.csv')

num_test_samples = len(test_df)
print(f"Number of test samples: {num_test_samples}")

if len(y_preds_fusion) < num_test_samples:
    print("Filling missing predictions with default values (0).")
    y_preds_fusion.extend([0] * (num_test_samples - len(y_preds_fusion)))
elif len(y_preds_fusion) > num_test_samples:
    print("Truncating extra predictions.")
    y_preds_fusion = y_preds_fusion[:num_test_samples]

# Verify the length matches
assert len(y_preds_fusion) == num_test_samples, "Mismatch between predictions and test samples!"

sub_df = pd.DataFrame({'isic_id': test_df['isic_id'], 'target': y_preds_fusion})

sub_df.to_csv("submission.csv", index=False)
print("Submission file saved successfully.")



y_preds_fusion = y_preds_fusion[:len(test_df)]  # Truncate to match the number of test samples

sub_df = pd.DataFrame({'isic_id': test_df['isic_id'], 'target': y_preds_fusion})
sub_df.to_csv("submission.csv", index=False)
print("Submission file created successfully.")


sub_df.head()

