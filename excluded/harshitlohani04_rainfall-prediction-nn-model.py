# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

train_data_rt = "/kaggle/input/playground-series-s5e3/train.csv"
test_data_rt = "/kaggle/input/playground-series-s5e3/test.csv"


import torch.nn as nn
import torch

# Initializing The model

class RainfallClassifierModel(nn.Module):
    def __init__(self):
        super(RainfallClassifierModel, self).__init__()
        self.base_layers = nn.Sequential(
            nn.Linear(54, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1)
)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.base_layers(x)

        return x


# Definfing the feature engineering process

def feature_engineering(df):
    """
    Create new features based on meteorological understanding and data analysis,
    with 'day' representing day of the year (1-365).
    Ensures no data leakage by avoiding use of the target variable (rainfall).
    """
    # Make a copy to avoid modifying the original dataframe
    enhanced_df = df.copy()
    
    # 1. temparature range (difference between max and min temparatures)
    enhanced_df['temp_range'] = enhanced_df['maxtemp'] - enhanced_df['mintemp']
    
    # 2. Dew point depression (difference between temparature and dew point)
    enhanced_df['dewpoint_depression'] = enhanced_df['temparature'] - enhanced_df['dewpoint']
    
    # 3. Pressure change from previous day
    enhanced_df['pressure_change'] = enhanced_df['pressure'].diff().fillna(0)
    
    # 4. Humidity to dew point ratio
    enhanced_df['humidity_dewpoint_ratio'] = enhanced_df['humidity'] / enhanced_df['dewpoint'].clip(lower=0.1)
    
    # 5. Cloud coverage to sunshine ratio (inverse relationship)
    enhanced_df['cloud_sunshine_ratio'] = enhanced_df['cloud'] / enhanced_df['sunshine'].clip(lower=0.1)
    
    # 6. Wind intensity factor (combination of speed and humidity)
    enhanced_df['wind_humidity_factor'] = enhanced_df['windspeed'] * (enhanced_df['humidity'] / 100)
    
    # 7. temparature-humidity index (simple version of heat index)
    enhanced_df['temp_humidity_index'] = (0.8 * enhanced_df['temparature']) + \
                                        ((enhanced_df['humidity'] / 100) * \
                                        (enhanced_df['temparature'] - 14.3)) + 46.4
    
    # 8. Pressure change rate (acceleration)
    enhanced_df['pressure_acceleration'] = enhanced_df['pressure_change'].diff().fillna(0)
    
    # 9. Seasonal features (based on day of year)
    # Convert day to month (1-365 to 1-12)
    enhanced_df['month'] = ((enhanced_df['day'] - 1) // 30) + 1
    enhanced_df['month'] = enhanced_df['month'].clip(upper=12)  # Ensure month doesn't exceed 12
    
    # 10. Convert day to season (1-365 to 1-4)
    enhanced_df['season'] = ((enhanced_df['month'] - 1) // 3) + 1
    
    # 11. Sine and cosine transformations to capture cyclical nature of days in a year
    enhanced_df['day_of_year_sin'] = np.sin(2 * np.pi * enhanced_df['day'] / 365)
    enhanced_df['day_of_year_cos'] = np.cos(2 * np.pi * enhanced_df['day'] / 365)
    
    # 12. Rolling averages for key meteorological variables
    for window in [3, 7, 14]:
        enhanced_df[f'temparature_rolling_{window}d'] = enhanced_df['temparature'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'pressure_rolling_{window}d'] = enhanced_df['pressure'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'humidity_rolling_{window}d'] = enhanced_df['humidity'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'cloud_rolling_{window}d'] = enhanced_df['cloud'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'windspeed_rolling_{window}d'] = enhanced_df['windspeed'].rolling(window=window, min_periods=1).mean()
    
    # 13. Weather pattern change features
    # temparature trend
    enhanced_df['temp_trend_3d'] = enhanced_df['temparature'].diff(3).fillna(0)
    # Pressure trend
    enhanced_df['pressure_trend_3d'] = enhanced_df['pressure'].diff(3).fillna(0)
    # Humidity trend
    enhanced_df['humidity_trend_3d'] = enhanced_df['humidity'].diff(3).fillna(0)
    
    # 14. Extreme weather indicators
    enhanced_df['extreme_temp'] = (enhanced_df['temparature'] > enhanced_df['temparature'].quantile(0.95)) | \
                                 (enhanced_df['temparature'] < enhanced_df['temparature'].quantile(0.05))
    enhanced_df['extreme_temp'] = enhanced_df['extreme_temp'].astype(int)
    
    enhanced_df['extreme_humidity'] = (enhanced_df['humidity'] > enhanced_df['humidity'].quantile(0.95)) | \
                                     (enhanced_df['humidity'] < enhanced_df['humidity'].quantile(0.05))
    enhanced_df['extreme_humidity'] = enhanced_df['extreme_humidity'].astype(int)
    
    enhanced_df['extreme_pressure'] = (enhanced_df['pressure'] > enhanced_df['pressure'].quantile(0.95)) | \
                                     (enhanced_df['pressure'] < enhanced_df['pressure'].quantile(0.05))
    enhanced_df['extreme_pressure'] = enhanced_df['extreme_pressure'].astype(int)
    
    # 15. Interaction terms between key variables
    enhanced_df['temp_humidity_interaction'] = enhanced_df['temparature'] * enhanced_df['humidity']
    enhanced_df['pressure_wind_interaction'] = enhanced_df['pressure'] * enhanced_df['windspeed']
    enhanced_df['cloud_sunshine_interaction'] = enhanced_df['cloud'] * enhanced_df['sunshine']
    enhanced_df['dewpoint_humidity_interaction'] = enhanced_df['dewpoint'] * enhanced_df['humidity']
    
    # 16. Moving standard deviations for measuring variability
    for window in [7, 14]:
        enhanced_df[f'temp_std_{window}d'] = enhanced_df['temparature'].rolling(window=window, min_periods=4).std().fillna(0)
        enhanced_df[f'pressure_std_{window}d'] = enhanced_df['pressure'].rolling(window=window, min_periods=4).std().fillna(0)
        enhanced_df[f'humidity_std_{window}d'] = enhanced_df['humidity'].rolling(window=window, min_periods=4).std().fillna(0)
    
    return enhanced_df


train_data = pd.read_csv(train_data_rt)
test_data = pd.read_csv(test_data_rt)
indices = test_data['id']
test_data = test_data.drop(columns = ["id"])

train_data_fe = feature_engineering(train_data)
test_data_fe = feature_engineering(test_data).to_numpy()


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x = train_data["rainfall"])
plt.title("Class Distribution")
plt.show()


from imblearn.over_sampling import RandomOverSampler

train_data_fe_copy = train_data_fe.copy()
# Extract features and labels
X = train_data_fe_copy.drop(columns=["id", "rainfall"]).values  # Features
y = train_data_fe_copy["rainfall"].values  # Labels

# Apply oversampling
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print(X_resampled.shape)
print(y_resampled.shape)

X_dataframe = pd.DataFrame(X_resampled, columns = train_data_fe.drop(columns = ["id", "rainfall"]).columns)
y_dataframe = pd.DataFrame(y_resampled, columns = ["rainfall"])
train_data_fe_new = pd.concat([X_dataframe, y_dataframe], axis = 1)

print(train_data_fe.head())


sns.countplot(x = train_data_fe_new["rainfall"])
plt.title("New Class Distribution")
plt.show()


# Creating the custom dataloader for loading the data

from torch.utils.data import Dataset, DataLoader
# from sklearn.utils.compute_weight import calculate

class CSVDataset(Dataset):
    def __init__(self, df):
        self.data = df.drop(columns = ["rainfall"]).values # In order to load them as a numpy array
        self.target = df["rainfall"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        features = torch.tensor(self.data[idx, :], dtype = torch.float32)
        labels = torch.tensor(self.target[idx], dtype = torch.float32)

        return features, labels


# Initializing the model and other functionalities
import torch.optim as optim

# model = RainfallClassifierModel()
# optimizer = optim.Adam(model.parameters(), lr = 0.0001)
# criterion = nn.BCEWithLogitsLoss()

dataset = CSVDataset(train_data_fe_new)
# dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)


# Function to initialize weights

import torch.nn.init as init

def initialize_weights(model, init_type="xavier"):
    for m in model.modules():
        if isinstance(m, nn.Linear):  # Apply to linear layers
            if init_type == "xavier":
                init.xavier_uniform_(m.weight)  # Xavier initialization
            elif init_type == "he":
                init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He initialization
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight)  # Orthogonal initialization
            else:
                raise ValueError("Unknown initialization type")
            if m.bias is not None:
                init.zeros_(m.bias)  # Initialize bias to zero



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# Hyperparameters
epochs = 2000
learning_rate = 0.000035
k_folds = 5
batch_size = 32
patience = 50

# Cross-validation
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
roc_auc_scores = []
models = []

# Extract features and target
X = train_data_fe.drop(columns = ["id", "rainfall"]).values  # Feature matrix
y = train_data_fe["rainfall"].values  # Target labels (binary: 0/1)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nðŸ”¹ Fold {fold + 1}/{k_folds}")

    # Extract training and validation subsets
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Apply SMOTE only on training data (to balance classes)
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights for loss function
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y[train_idx])
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Define model, optimizer, and loss function
    model = RainfallClassifierModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])  # Handling class imbalance

    # Early stopping variables
    best_roc_auc = 0
    no_improvement_count = 0

    # Training loop
    for epoch in tqdm(range(epochs), desc=f"Training Fold {fold + 1}"):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_outputs = []
        val_targets = []
        
        with torch.no_grad():
            for val_x, val_y in val_loader:
                outputs = model(val_x).squeeze(1)
                val_outputs.append(outputs.cpu().numpy())  # Ensure CPU conversion
                val_targets.append(val_y.cpu().numpy())    # Ensure CPU conversion

        # Compute ROC-AUC score
        val_outputs = np.concatenate(val_outputs)
        val_targets = np.concatenate(val_targets)

        roc_auc = roc_auc_score(val_targets, val_outputs)
        print(f"Epoch {epoch + 1}: ROC AUC = {roc_auc:.4f}")

        # Early stopping
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"âš  Early stopping at epoch {epoch + 1}, Best ROC AUC = {best_roc_auc:.4f}")
                break

    # Store results
    roc_auc_scores.append(best_roc_auc)
    if best_roc_auc > 0.85:
        models.append(model)

# Final results
print(f"\nðŸ”¥ Average ROC AUC across {k_folds} folds: {np.mean(roc_auc_scores):.4f}")



# Training and Validation Code

# from sklearn.metrics import roc_auc_score
# from torch.utils.data import Subset
# from sklearn.model_selection import KFold

# num_of_epochs = 50

# for i in range(num_of_epochs):
#     model.train()
#     for batch_x, batch_y in dataloader:
#         optimizer.zero_grad()
#         y_pred = model(batch_x).squeeze(1)
        
#         loss = criterion(y_pred, batch_y.float())  # BCE loss
        
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch : {i} --------- Loss : {loss}")


# Prediction code

X_test_tensor = torch.tensor(test_data_fe, dtype=torch.float32)
test_predictions = np.zeros ((X_test_tensor.shape [0]))

for model in models:
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_predictions = np.add(predictions.numpy().flatten(), test_predictions)
        
test_predictions = test_predictions / len(models)
clean_test_predictions = np.nan_to_num(test_predictions, nan=0.8)

# model.eval()

# with torch.no_grad():  # No gradients needed for inference
#     logits = model(X_test_tensor)  # Forward pass
#     probabilities = torch.sigmoid(logits)  # Apply Sigmoid (for binary classification)

# probabilities = torch.nan_to_num(probabilities, nan=0.5)

# print(torch.isnan(probabilities).sum())  # Count NaN values
# print(torch.isinf(probabilities).sum())


# Concatenation

final_dataframe = np.concatenate((np.array(indices).reshape(-1, 1), clean_test_predictions.reshape(-1, 1)), axis = 1)
submission = pd.DataFrame(final_dataframe, columns = ["id", "rainfall"])
submission["id"] = submission["id"].astype(int)
submission.to_csv("submission.csv", index = False)

submission.isnull().sum()

