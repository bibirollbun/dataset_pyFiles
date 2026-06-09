import os
import gc

import sys

import pandas as pd
import numpy as np

import re

import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno
import scipy.stats as stats
from patsy import dmatrices
import statsmodels as sm



plt.style.use('seaborn-v0_8-darkgrid')

if sys.platform == 'win32':
    print('Win')
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
elif sys.platform == 'darwin':
    print('Mac')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rcParams['axes.unicode_minus']=False

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

# pd.set_option('display.max_columns', 20)  # or 1000
# pd.set_option('display.max_rows', 80)  # or 1000
# pd.set_option('display.max_colwidth', 20)  # or 199


import pandas as pd
# Form https://www.kaggle.com/code/valliansayoga/how-to-load-train-and-sub-data
Is_local = True  # Set to True if running locally, False for Kaggle
if Is_local:
    data_path = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/'
else:
    data_path = './'

def load_train():
    df = pd.read_csv(f"{data_path}transfer_plate.csv")
    # df = pd.read_csv("./transfer_plate.csv")

    X = df.iloc[:, :-4]
    y = df.iloc[:, -4:].dropna()

    X.columns = ["sample_id"] + [i for i in range(X.shape[1]-1)]
    X.sample_id = X.sample_id.ffill().str.strip()

    col_to_cast = X.select_dtypes(object).columns[1:]
    
    for col in col_to_cast:
        X[col] = X[col].str.replace("[\[\]]", "", regex=True).astype("int64")
        
    return X, y

X, y = load_train()
display(X.head())
display(y.head())

y= y.rename(columns={"Analyte concentration": "sample_id"})
print(X.shape, y.shape)
train_df = pd.merge(X, y, on="sample_id", how="left")
display(train_df.head())
train_df.to_csv("./clear_transfer_plate.csv", index=False, encoding="utf-8-sig")




# read testing set 
# Clear testing set

df_test = pd.read_csv(f"{data_path}96_samples.csv", header=None)
df_test.columns = ["sample_id"] + [i for i in range(df_test.shape[1]-1)]

df_test = df_test.ffill()

col_to_cast = df_test.select_dtypes(object).columns[1:]

for col in col_to_cast:
    df_test[col] = df_test[col].str.replace("[\[\]]", "", regex=True).astype("int64")
        
display(df_test.head())
df_test.to_csv("./clear_96_samples.csv", index=False, encoding="utf-8-sig")





from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np


# Create spectrum feature columns
spectrum_cols = [str(i) for i in range(2048)]
transfer_df = pd.read_csv("./clear_transfer_plate.csv")
sample_df = pd.read_csv("./clear_96_samples.csv")

# Extract features and target variables
X_train = transfer_df[spectrum_cols]
y_train = transfer_df[["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]]
X_test = sample_df[spectrum_cols]
test_sample_ids = sample_df["sample_id"].values

# Create a Ridge regression model with cross-validation
alphas = np.logspace(-3, 3, 50)
model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=5))

# Fit the model for each target variable and make predictions
predictions = {}
for col in y_train.columns:
    model.fit(X_train, y_train[col])
    preds = model.predict(X_test)
    predictions[col] = preds

# Create a DataFrame for predictions
pred_df = pd.DataFrame(predictions)
pred_df["sample_id"] = test_sample_ids

# Reorder columns to match the original format
final_pred = pred_df.groupby("sample_id", as_index=False).mean()

# Rename columns to match the submission format
final_pred.rename(columns={'sample_id': 'ID','Glucose (g/L)': 'Glucose','Sodium Acetate (g/L)': 'Sodium Acetate','Magnesium Acetate (g/L)': 'Magnesium Sulfate'}, inplace=True)


# Extract numeric part from ID column
final_pred['ID'] = final_pred['ID'].str.extract('(\d+)').astype(int)
final_pred = final_pred.sort_values(by='ID')
display(final_pred.head())
final_pred.to_csv("./submission.csv", index=False, encoding="utf-8-sig")


# PCA Testing 
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd

# Read the data
transfer_df = pd.read_csv("./clear_transfer_plate.csv")
sample_df = pd.read_csv("./clear_96_samples.csv")

# Create spectrum feature columns
spectrum_cols = [str(i) for i in range(2048)]

# Extract features and target variables
X_train = transfer_df[spectrum_cols]
y_train = transfer_df[["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]]
X_test = sample_df[spectrum_cols]
test_sample_ids = sample_df["sample_id"].values


# Feature scaling and PCA
scaler = StandardScaler()
pca = PCA(n_components=100)  

X_train_scaled = scaler.fit_transform(X_train)
X_train_pca = pca.fit_transform(X_train_scaled)

X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Create a Ridge regression model with cross-validation
alphas = np.logspace(-3, 3, 50)
model = RidgeCV(alphas=alphas, cv=5)

# Fit the model for each target variable and make predictions
predictions = {}
for col in y_train.columns:
    model.fit(X_train_pca, y_train[col])
    preds = model.predict(X_test_pca)
    predictions[col] = preds

pred_df = pd.DataFrame(predictions)
pred_df["sample_id"] = test_sample_ids

final_pred = pred_df.groupby("sample_id", as_index=False).mean()

# Rename columns to match the submission format
final_pred.rename(columns={
    'sample_id': 'ID',
    'Glucose (g/L)': 'Glucose',
    'Sodium Acetate (g/L)': 'Sodium Acetate',
    'Magnesium Acetate (g/L)': 'Magnesium Sulfate'
}, inplace=True)


final_pred['ID'] = final_pred['ID'].str.extract('(\d+)').astype(int)
final_pred = final_pred.sort_values(by='ID')

display(final_pred.head())
# final_pred.to_csv("./submission.csv", index=False, encoding="utf-8-sig")



from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

# Define models to be used
models = {
    "RidgeCV": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)),
    "LassoCV": make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3, 3, 50), cv=5, max_iter=5000)),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    # "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}


all_preds = {}

# Iterate through each model and make predictions
for model_name, model in models.items():
    print(f"\nğŸ”§ Model ï¼š{model_name}")
    preds_dict = {}
    for target in y_train.columns:
        model.fit(X_train, y_train[target])
        preds = model.predict(X_test)
        preds_dict[target] = preds
    pred_df = pd.DataFrame(preds_dict)
    pred_df["sample_id"] = test_sample_ids
    pred_df = pred_df.groupby("sample_id", as_index=False).mean()
    pred_df.rename(columns={
        'sample_id': 'ID',
        'Glucose (g/L)': 'Glucose',
        'Sodium Acetate (g/L)': 'Sodium Acetate',
        'Magnesium Acetate (g/L)': 'Magnesium Sulfate'
    }, inplace=True)
    pred_df['ID'] = pred_df['ID'].str.extract('(\d+)').astype(int)
    pred_df = pred_df.sort_values(by='ID')
    all_preds[model_name] = pred_df
    pred_df.to_csv(f"./submission_{model_name}.csv", index=False, encoding="utf-8-sig")
    print(f"submission_{model_name}.csv")

# 
final_avg_pred = pd.concat(all_preds.values()).groupby("ID").mean().reset_index()
final_avg_pred.rename(columns={
    'ID': 'ID',
    'Glucose': 'Glucose',
    'Sodium Acetate': 'Sodium Acetate',
    'Magnesium Sulfate': 'Magnesium Sulfate'
}, inplace=True)

# final_avg_pred.to_csv("./submission_avg.csv", index=False, encoding="utf-8-sig")





from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd


transfer_df = pd.read_csv("./clear_transfer_plate.csv")
sample_df = pd.read_csv("./clear_96_samples.csv")


spectrum_cols = [str(i) for i in range(2048)]


X_train = transfer_df[spectrum_cols]
y_train = transfer_df[["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]]
X_test = sample_df[spectrum_cols]
test_sample_ids = sample_df["sample_id"].values


scaler = StandardScaler()
pca = PCA(n_components=100)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Define models to be used
models = {
    "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5),
    "LassoCV": LassoCV(alphas=np.logspace(-3, 3, 50), cv=5, max_iter=5000),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
}


all_preds = {}

for model_name, model in models.items():
    print(f"\n Model{model_name}")
    preds_dict = {}
    for target in y_train.columns:
        model.fit(X_train_pca, y_train[target])
        preds = model.predict(X_test_pca)
        preds_dict[target] = preds

    pred_df = pd.DataFrame(preds_dict)
    pred_df["sample_id"] = test_sample_ids
    pred_df = pred_df.groupby("sample_id", as_index=False).mean()
    pred_df.rename(columns={
        'sample_id': 'ID',
        'Glucose (g/L)': 'Glucose',
        'Sodium Acetate (g/L)': 'Sodium Acetate',
        'Magnesium Acetate (g/L)': 'Magnesium Sulfate'
    }, inplace=True)
    pred_df['ID'] = pred_df['ID'].str.extract('(\d+)').astype(int)
    pred_df = pred_df.sort_values(by='ID')
    all_preds[model_name] = pred_df
    pred_df.to_csv(f"./submission_{model_name}.csv", index=False, encoding="utf-8-sig")
    print(f"submission_{model_name}.csv")

# Calculate the average of all predictions
final_avg_pred = pd.concat(all_preds.values()).groupby("ID").mean().reset_index()
final_avg_pred.rename(columns={
    'ID': 'ID',
    'Glucose': 'Glucose',
    'Sodium Acetate': 'Sodium Acetate',
    'Magnesium Sulfate': 'Magnesium Sulfate'
}, inplace=True)

final_avg_pred.to_csv("./submission_avg.csv", index=False, encoding="utf-8-sig")
print("submission_avg.csv")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# file list and mechanical devices
file_paths = {
    "anton_532": f"{data_path}anton_532.csv",
    "anton_785": f"{data_path}anton_785.csv",
    "kaiser": f"{data_path}kaiser.csv",
    "metrohm": f"{data_path}metrohm.csv",
    "mettler_toledo": f"{data_path}mettler_toledo.csv",
    "tec5": f"{data_path}tec5.csv",
    "timegate": f"{data_path}timegate.csv",
    "tornado": f"{data_path}tornado.csv"
}

# wave numbers 
wavenumbers = np.linspace(65, 3350, 2048)

plt.figure(figsize=(16, 8))

for label, path in file_paths.items():
    df = pd.read_csv(path)

    # kepp only numeric columns (usually the spectrum columns)
    spectrum_cols = df.select_dtypes(include=[np.number]).columns
    spectrum = df[spectrum_cols].iloc[0].values  # ç¬¬ä¸€ç­†

    # Define wavenumbers for the spectrum
    wavenumbers = np.linspace(65, 3350, len(spectrum_cols))

    plt.plot(wavenumbers, spectrum, label=label)



plt.title("FIRST Wave From device", fontsize=16, fontweight='bold')
plt.xlabel("Wave (cmâ�»Â¹)", fontsize=14)
plt.ylabel("Intensity", fontsize=14)
plt.legend(title="Device Name")
plt.grid(True)
plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# file list and mechanical devices
file_paths = {
    "anton_532": f"{data_path}anton_532.csv",
    "anton_785": f"{data_path}anton_785.csv",
    "kaiser": f"{data_path}kaiser.csv",
    "metrohm": f"{data_path}metrohm.csv",
    "mettler_toledo": f"{data_path}mettler_toledo.csv",
    "tec5": f"{data_path}tec5.csv",
    "timegate": f"{data_path}timegate.csv",
    "tornado": f"{data_path}tornado.csv"
}

# wave numbers
plt.figure(figsize=(16, 8))
glucose_record = []
for label, path in file_paths.items():
    df = pd.read_csv(path)

    # Extract numeric columns as spectrum data
    spectrum_cols = list(df.select_dtypes(include=[np.number]).columns)
    for col in ['glucose', 'Na_acetate', 'Mg_SO4', 'MSM_present', 'fold_idx']:
        if col in spectrum_cols:
            spectrum_cols.remove(col)  # Remove non-spectrum columns
    
    
    # IF it is "timegate.csv", scale all values by 10000000
    if label == "timegate":
        df.loc[:, spectrum_cols] *= 10000000

    if label == "anton_532":
        df.loc[:, spectrum_cols] *= 5

    # If no numeric columns or data is empty, skip this file
    if len(spectrum_cols) == 0 or df.empty:
        print(f"âš ï¸� File {label} No valid spectral data, skipped.")
        continue

    
    # Visualize glucose values in a specific range
    # cond = [1.4357,2,'Na_acetate']
    cond = [2,5,'glucose']
    
    condition = (df[cond[2]] > cond[0]) & (df[cond[2]] < cond[1])
    spectrum = df[condition][spectrum_cols].iloc[0].values

    # Normalize the spectrum
    spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
    
    # display(spectrum)
    glucose_record.append([label,df[condition][[cond[2]]].iloc[0].values,spectrum_cols[0],spectrum_cols[-1],len(spectrum_cols)])  # è¨˜éŒ„ glucose å€¼
    # Transform columns to wavenumbers
    try:
        wavenumbers = np.array([float(col) for col in spectrum_cols])
    except ValueError:
        
        print(f"Failed to parse wavenumbers from {label} columns, please check the column format.")
        continue

    # Order to avoid wavenumber order confusion
    sorted_idx = np.argsort(wavenumbers)
    wavenumbers = wavenumbers[sorted_idx]
    spectrum = spectrum[sorted_idx]

    plt.plot(wavenumbers, spectrum, label=label)


display(glucose_record)
plt.title(f"First Raman device's first spectrum data {cond[2]} [{cond[0]},{cond[1]}]", fontsize=16, fontweight='bold')

plt.xlabel("Wave (cmâ�»Â¹)", fontsize=14)
plt.ylabel("Intensity", fontsize=14)
plt.legend(title="Device Name")

# Draw vertical lines at specific wavenumbers
plt.axvline(x=250, color='red', linestyle='--', label='250 cmâ�»Â¹')
plt.axvline(x=3000, color='blue', linestyle='--', label='3000 cmâ�»Â¹')


plt.grid(True)
plt.tight_layout()
plt.show()



# Read the transfer plate data

transfer_df = pd.read_csv("./clear_transfer_plate.csv")

# 
retain_cols_transfer = [
    'Analyte concentration',
    'Glucose (g/L)',
    'Sodium Acetate (g/L)',
    'Magnesium Acetate (g/L)'
]


retain_part = transfer_df[[col for col in retain_cols_transfer if col in transfer_df.columns]].reset_index(drop=True)

# Extract numeric columns, excluding retained columns
numeric_cols_transfer = transfer_df.select_dtypes(include=[np.number]).columns


spectrum_cols = [str(i) for i in range(2048)]

plt.figure(figsize=(16, 8))

# Extract the first spectrum data that meets the condition
# Visualize glucose values in a specific range
# cond = [1.4357, 2, 'Sodium Acetate (g/L)']
cond = [2,5,'Glucose (g/L)']
condition = ( (transfer_df[cond[2]] > cond[0]) & (transfer_df[cond[2]] > cond[0]))
spectrum = transfer_df[condition][spectrum_cols].iloc[0].values

# Normalize the spectrum
spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))    

# display(spectrum)
glucose_record.append([cond[2], transfer_df[condition][[cond[2]]].iloc[0].values, spectrum_cols[0], spectrum_cols[-1], len(spectrum_cols)])  # è¨˜éŒ„ glucose å€¼


try:
    wavenumbers = np.array([float(col) for col in spectrum_cols])
except ValueError:      
    print("Failed to parse wavenumbers, using default range.")
    wavenumbers = np.arange(len(spectrum_cols))

# Order to avoid wavenumber order confusion
sorted_idx = np.argsort(wavenumbers)
wavenumbers = wavenumbers[sorted_idx]
spectrum = spectrum[sorted_idx]


plt.plot(wavenumbers, spectrum, label=cond[2])

plt.xlabel("Wave (cmâ�»Â¹)", fontsize=14)
plt.ylabel("Intensity", fontsize=14)
plt.legend(title="Device Name")
plt.grid(True)
plt.tight_layout()
plt.show()

display(retain_part.head())
display(transfer_df.head())




import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def median_baseline(y, window=101):
    baseline = pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    return baseline


class RamanDataset(Dataset):
    def __init__(self, X, y=None, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.augment = augment

    def __getitem__(self, idx):
        x = self.X[idx].numpy()


        baseline = median_baseline(x)
        x = x - baseline

        # smoothing: Savitzkyâ€“Golay
        x = savgol_filter(x, window_length=17, polyorder=3, deriv=1)

        # augmentation
        if self.augment:
            x *= np.random.uniform(0.95, 1.05)
            x = np.roll(x, np.random.randint(-3, 4))
            x += np.random.normal(0, 0.005, x.shape)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        return (x, self.y[idx]) if self.y is not None else x

    def __len__(self):
        return len(self.X)

# Define a simple CNN for Raman spectra classification
class RamanCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, 5)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, 3)
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.conv3(x))).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Data resampling and normalization

uniform_grid = np.linspace(65, 3350, 2048)
device_files = [
    f"{data_path}anton_532.csv", f"{data_path}anton_785.csv", f"{data_path}kaiser.csv", f"{data_path}metrohm.csv",
    f"{data_path}mettler_toledo.csv", f"{data_path}tec5.csv", f"{data_path}timegate.csv", f"{data_path}tornado.csv"
]
all_spectra = []

for file in device_files:
    df = pd.read_csv(file)
    numeric_cols = df.select_dtypes(include=np.number).columns
    spec_cols = [c for c in numeric_cols if c not in ['glucose', 'Na_acetate', 'Mg_SO4', 'MSM_present', 'fold_idx']]
    
    try:
        orig_wavenumbers = np.array([float(c) for c in spec_cols])
    except:
        print(f"âš ï¸� Failed to parse wavenumbers from {file} columns, skipping.")
        continue

    df = df.dropna(subset=spec_cols)
    for _, row in df.iterrows():
        f = interp1d(orig_wavenumbers, row[spec_cols], bounds_error=False, fill_value='extrapolate')
        spectrum = f(uniform_grid)

        if "timegate" in file:
            spectrum *= 1e7
        elif "anton_532" in file:
            spectrum *= 5

        spectrum /= np.max(spectrum)  # Normalize to 0~1
        all_spectra.append(spectrum)

X_pretrain = np.vstack(all_spectra)
scaler = MinMaxScaler()
X_pretrain = scaler.fit_transform(X_pretrain)

# ===== 4. Pretraining =====
model = RamanCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.SmoothL1Loss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pretrain_loader = DataLoader(RamanDataset(X_pretrain), batch_size=64, shuffle=True)

# Pretraining loop
for epoch in range(3): 
    model.train()
    total_loss = 0
    for inputs in pretrain_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.zeros_like(outputs))  # dummy label
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    print(f"Pretrain Epoch {epoch+1}, Loss: {total_loss / len(pretrain_loader.dataset):.6f}")

# ===== 5. Fine-tuning with Transfer Plate =====
df = pd.read_csv("./clear_transfer_plate.csv")
spectrum_cols = [str(i) for i in range(2048)]
X = scaler.transform(df[spectrum_cols])
y = df[["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]].values

dataset = RamanDataset(X, y, augment=True)
train_len = int(0.9 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
best_loss = float('inf')
wait = 0
patience = 10

for epoch in range(100):
    model.train()
    tr_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(Xb)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            val_loss += criterion(model(Xb), yb).item() * len(Xb)

    tr_loss /= len(train_ds)
    val_loss /= len(val_ds)
    scheduler.step(epoch)

    print(f"Epoch {epoch+1}: Train Loss = {tr_loss:.4f}, Val Loss = {val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        wait = 0
        torch.save(model.state_dict(), "best_transfer_model.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break
        
# ===== 6. Inference on Test Set =====
model.load_state_dict(torch.load("best_transfer_model.pt"))
model.eval()

test_df = pd.read_csv("./clear_96_samples.csv")
X_test = scaler.transform(test_df[spectrum_cols].values)
sample_ids = test_df["sample_id"]

test_loader = DataLoader(RamanDataset(X_test), batch_size=64)
all_preds = []

with torch.no_grad():
    for xb in test_loader:
        preds = model(xb.to(device)).cpu().numpy()
        all_preds.append(preds)

pred_df = pd.DataFrame(np.vstack(all_preds), columns=["Glucose", "Sodium Acetate", "Magnesium Sulfate"])
pred_df["ID"] = sample_ids.str.extract(r"(\d+)").astype(int)
final = pred_df.groupby("ID", as_index=False).mean().sort_values("ID")
final.to_csv("submission_cnn_transfer.csv", index=False, encoding="utf-8-sig")




import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import torch.nn.functional as F

def median_baseline(y, window=101):
    baseline = pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    return baseline

# Define the Raman dataset class
class RamanDataset(Dataset):
    def __init__(self, X, y=None, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.augment = augment

    def __getitem__(self, idx):
        x = self.X[idx].numpy()

        # âœ… baseline correction
        baseline = median_baseline(x)
        x = x - baseline

        # âœ… smoothing
        x = savgol_filter(x, window_length=17, polyorder=3, deriv=1)

        # âœ… augmentation
        if self.augment:
            x *= np.random.uniform(0.98, 1.02)
            x = np.roll(x, np.random.randint(-2, 3))
            x += np.random.normal(0, 0.002, x.shape)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        return (x, self.y[idx]) if self.y is not None else x

    def __len__(self):
        return len(self.X)

# Create a CNN model with Squeeze-and-Excitation (SE) block 
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, l = x.size()
        z = F.adaptive_avg_pool1d(x, 1).view(b, c)
        z = torch.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z)).view(b, c, 1)
        return x * z.expand_as(x)

class RamanMultiHeadCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.se = SEBlock(128)
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        self.fc_shared = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)

        self.head_glucose = nn.Linear(64, 1)
        self.head_sodium = nn.Linear(64, 1)
        self.head_magnesium = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.se(x)
        x = self.pool3(x).squeeze(-1)

        x = torch.relu(self.fc_shared(x))
        x = self.dropout(x)

        out1 = self.head_glucose(x)
        out2 = self.head_sodium(x)
        out3 = self.head_magnesium(x)
        return torch.cat([out1, out2, out3], dim=1)

# Define multitask loss function
def multitask_loss(output, target):
    return nn.SmoothL1Loss()(output, target)

# Data resampling and normalization
uniform_grid = np.linspace(65, 3350, 2048)
device_files = [
    f"{data_path}anton_532.csv", f"{data_path}anton_785.csv", f"{data_path}kaiser.csv", f"{data_path}metrohm.csv",
    f"{data_path}mettler_toledo.csv", f"{data_path}tec5.csv", f"{data_path}timegate.csv", f"{data_path}tornado.csv"
]


all_spectra = []

for file in device_files:
    df = pd.read_csv(file)
    numeric_cols = df.select_dtypes(include=np.number).columns
    spec_cols = [c for c in numeric_cols if c not in ['glucose', 'Na_acetate', 'Mg_SO4', 'MSM_present', 'fold_idx']]

    try:
        orig_wavenumbers = np.array([float(c) for c in spec_cols])
    except:
        print(f"âš ï¸� Failed to parse wavenumbers from {file} columns, skipping.")
        continue

    df = df.dropna(subset=spec_cols)
    for _, row in df.iterrows():
        f = interp1d(orig_wavenumbers, row[spec_cols], bounds_error=False, fill_value='extrapolate')
        spectrum = f(uniform_grid)

        if "timegate" in file:
            spectrum *= 1e7
        elif "anton_532" in file:
            spectrum *= 5

        spectrum /= np.max(spectrum)
        all_spectra.append(spectrum)

X_pretrain = np.vstack(all_spectra)
scaler = MinMaxScaler()
X_pretrain = scaler.fit_transform(X_pretrain)


# Initialize model, optimizer, and device
model = RamanMultiHeadCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pretrain_loader = DataLoader(RamanDataset(X_pretrain), batch_size=64, shuffle=True)

# Pretraining loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for inputs in pretrain_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = multitask_loss(outputs, torch.zeros_like(outputs))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    print(f"[Pretrain] Epoch {epoch+1}, Loss: {total_loss / len(pretrain_loader.dataset):.6f}")


# Fine-tuning with transfer plate data
df = pd.read_csv("./clear_transfer_plate.csv")
spectrum_cols = [str(i) for i in range(2048)]
X = scaler.transform(df[spectrum_cols])
y = df[["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]].values

dataset = RamanDataset(X, y, augment=True)
train_len = int(0.9 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
best_loss = float('inf')
wait = 0
patience = 10

for epoch in range(100):
    model.train()
    tr_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = multitask_loss(out, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(Xb)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            val_loss += multitask_loss(model(Xb), yb).item() * len(Xb)

    tr_loss /= len(train_ds)
    val_loss /= len(val_ds)
    scheduler.step(epoch)

    print(f"[Fine-tune] Epoch {epoch+1}: Train Loss = {tr_loss:.4f}, Val Loss = {val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        wait = 0
        torch.save(model.state_dict(), "best_transfer_model.pt")
    else:
        wait += 1
        if wait >= patience:
            print("ğŸ›‘ Early stopping.")
            break

# ===== Inference on Test Set =====
model.load_state_dict(torch.load("best_transfer_model.pt"))
model.eval()

test_df = pd.read_csv("./clear_96_samples.csv")
X_test = scaler.transform(test_df[spectrum_cols].values)
sample_ids = test_df["sample_id"]

test_loader = DataLoader(RamanDataset(X_test), batch_size=64)
all_preds = []

with torch.no_grad():
    for xb in test_loader:
        preds = model(xb.to(device)).cpu().numpy()
        all_preds.append(preds)

pred_df = pd.DataFrame(np.vstack(all_preds), columns=["Glucose", "Sodium Acetate", "Magnesium Sulfate"])
pred_df["ID"] = sample_ids.str.extract(r"(\d+)").astype(int)
final = pred_df.groupby("ID", as_index=False).mean().sort_values("ID")
final.to_csv("submission_cnn_transfer.csv", index=False, encoding="utf-8-sig")


