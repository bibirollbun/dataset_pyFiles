import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import pydicom
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import OrderedDict
from tqdm import tqdm
import pickle
from skimage.transform import resize
from types import SimpleNamespace
import torch.nn.functional as F
import random
import math
import torch.optim as optim



base_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/"

train = pd.read_csv(base_path + "train.csv")
test = pd.read_csv(base_path + "test.csv")
sample_submission = pd.read_csv(base_path + "sample_submission.csv")

print("training")
print(train.head())
print("-" * 90)
print("testing")
print(test.head())
print("-" * 90)
print("sample submission")
print(sample_submission.head())


sns.histplot(train['FVC'], kde=True)
plt.title("Distribution of FVC")
plt.show()

sns.histplot(train['Age'], kde=True)
plt.title("Age distribution")
plt.show()

sns.countplot(data=train, x='Sex')
plt.title("Sex distribution")
plt.show()

sns.countplot(data=train, x='SmokingStatus')
plt.title("SmokingStatus")
plt.show()



sample_patient = train[train["Patient"] == train["Patient"].iloc[0]]
print(sample_patient)
plt.plot(sample_patient["Weeks"], sample_patient["FVC"])
plt.xlabel("Weeks")
plt.ylabel("FVC")
plt.show()


sample_patient = train[train["Patient"] == train["Patient"].iloc[9]]
print(sample_patient)
plt.plot(sample_patient["Weeks"], sample_patient["FVC"])
plt.xlabel("Weeks")
plt.ylabel("FVC")
plt.show()



example_patient = train["Patient"].iloc[0]
path = os.path.join(base_path + "train", example_patient)
files = sorted(os.listdir(path))

dcm = pydicom.dcmread(os.path.join(path, files[len(files)//2]))
plt.imshow(dcm.pixel_array, cmap="gray")
plt.title(f"CT Scan of {example_patient}")



# Set a clean style
sns.set(style="whitegrid", palette="muted")

# Set the figure size
plt.figure(figsize=(18, 5))

# 1. FVC vs Sex
plt.subplot(1, 3, 1)
sns.boxplot(data=train, x='Sex', y='FVC')
plt.title('FVC by Sex')

# 2. FVC vs Age (we'll use scatterplot here)
plt.subplot(1, 3, 2)
sns.scatterplot(data=train, x='Age', y='FVC', alpha=0.4)
plt.title('FVC vs Age')

# 3. FVC vs SmokingStatus
plt.subplot(1, 3, 3)
sns.boxplot(data=train, x='SmokingStatus', y='FVC')
plt.xticks(rotation=20)
plt.title('FVC by Smoking Status')

plt.tight_layout()
plt.show()



# Set a clean style
sns.set(style="whitegrid", palette="muted")

# Set the figure size
plt.figure(figsize=(18, 5))

# 1. Percent by Sex
plt.subplot(1, 3, 1)
sns.boxplot(data=train, x='Sex', y='Percent')
plt.title('Percent by Sex')

# 2. Percent vs Age
plt.subplot(1, 3, 2)
sns.scatterplot(data=train, x='Age', y='Percent', alpha=0.4)
plt.title('Percent vs Age')

# 3. Percent by SmokingStatus
plt.subplot(1, 3, 3)
sns.boxplot(data=train, x='SmokingStatus', y='Percent')
plt.xticks(rotation=20)
plt.title('Percent by Smoking Status')

plt.tight_layout()
plt.show()



plt.figure(figsize=(6, 5))
sns.scatterplot(data=train, x='FVC', y='Percent', alpha=0.5)
sns.regplot(data=train, x='FVC', y='Percent', scatter=False, color='red', label='Trend')
plt.title('Relation between FVC and Percent')
plt.xlabel('FVC (ml)')
plt.ylabel('Percent (%)')
plt.legend()
plt.tight_layout()
plt.show()



def generate_pairwise_fvc_dataset(df):
    df = df.dropna()
    df = df.sort_values(['Patient', 'Weeks'])

    all_rows = []

    for patient_id, group in df.groupby("Patient"):
        group = group.reset_index(drop=True)
        n = len(group)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                row_i = group.iloc[i]
                row_j = group.iloc[j]
                all_rows.append({
                    'Patient_ID': row_i['Patient'],
                    'Reading_week': row_i['Weeks'],
                    'target_week': row_j['Weeks'],
                    'Weeks_diff': row_j['Weeks'] - row_i['Weeks'],
                    'Curr_FVC': row_i['FVC'],
                    'Target_FVC': row_j['FVC'],
                    'Percent': row_i['Percent'],
                    'Age': row_i['Age'],
                    'Sex': row_i['Sex'],
                    'SmokingStatus': row_i['SmokingStatus'],
                })

    return pd.DataFrame(all_rows)



# Load original data and generate pairs
df = pd.read_csv(base_path + 'train.csv')
pairwise_df = generate_pairwise_fvc_dataset(df)
print(pairwise_df.head())


# Encode categorical variables
le_sex = LabelEncoder().fit(pairwise_df['Sex'])
le_smoke = LabelEncoder().fit(pairwise_df['SmokingStatus'])

pairwise_df['Sex'] = le_sex.transform(pairwise_df['Sex'])
pairwise_df['SmokingStatus'] = le_smoke.transform(pairwise_df['SmokingStatus'])

# Select features and target
features = ['Reading_week', 'target_week', 'Weeks_diff', 'Curr_FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']
target = 'Target_FVC'

X = pairwise_df[features]
y_processed = pairwise_df[target]

# Normalize continuous features
scaler = StandardScaler()
x_processed = scaler.fit_transform(X)


def get_model(x, y):
    seed = random.randint(1, 1000)
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.5,
        random_state=seed

    )
    model.fit(x, y)
    return model, seed


def adaBoost(k, x, y, thre):
    m = 0
    models = []
    weights = []
    maes = []
    seeds = []
    while m < k:
        split_seed = random.randint(1, 1000)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=split_seed)
        model, model_seed = get_model(x_train, y_train)
        y_pred = model.predict(x_val)
        mae = mean_absolute_error(y_val, y_pred)
        if mae < thre:
            m += 1
            w = math.log((thre * 2 - mae) / mae)
            models.append(model)
            weights.append(w)
            maes.append(mae)
            seeds.append([split_seed, model_seed])

    return models, weights, maes, seeds


# x_data = x_processed
# y_data = y_processed

# models, weights, maes, seeds = adaBoost(10, x_data, y_data, 75)


# print(maes)


# print(seeds)


# print(weights)


def normalize_to_sum_one(values):
    total = sum(values)
    if total == 0:
        raise ValueError("Cannot normalize list with sum = 0.")
    return [x / total for x in values]


# weights = normalize_to_sum_one(weights)


def weighted_model_prediction_with_confidence(x_values, models, model_weights):
    if len(models) != len(model_weights):
        raise ValueError("Number of models and weights must be the same.")
    if not abs(sum(model_weights) - 1.0) < 1e-6:
        raise ValueError("Model weights must sum to 1.")

    all_preds = [np.array(model.predict(x_values)) for model in models]  # shape: [num_models, num_samples]
    all_preds = np.array(all_preds)  # shape: (n_models, n_samples)

    # Weighted average prediction
    weights = np.array(model_weights).reshape(-1, 1)  # shape: (n_models, 1)
    weighted_pred = np.sum(weights * all_preds, axis=0)  # shape: (n_samples,)

    # Weighted standard deviation (a proxy for uncertainty or confidence)
    weighted_mean = weighted_pred
    variance = np.sum(weights * (all_preds - weighted_mean) ** 2, axis=0)
    std_dev = np.sqrt(variance)  # shape: (n_samples,)
    confidence = 1 / (1 + std_dev) * 100 # confidence ∈ (0, 1], higher = more confident
    confidence = np.clip(confidence, 420, 1000)

    return weighted_pred.tolist(), confidence.tolist()



# x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=17)


# ada_boost_preds, conf = weighted_model_prediction_with_confidence(x_val, models, weights)
# mae = mean_absolute_error(y_val, ada_boost_preds)
# print(mae)

# print(conf[:10])


# test = pd.read_csv(base_path + 'test.csv')
# sub_sample = pd.read_csv(base_path + 'sample_submission.csv')


# print(test.head())
# print(sub_sample.head())


# # Extract Patient ID and Target Week from Patient_Week in df2
# sub_sample[['Patient_ID', 'Target_Week']] = sub_sample['Patient_Week'].str.rsplit('_', n=1, expand=True)
# sub_sample['Target_Week'] = sub_sample['Target_Week'].astype(int)

# # Rename 'Patient' in df1 to match the new column name
# test = test.rename(columns={'Patient': 'Patient_ID', 'Weeks': 'Reading_Week'})

# # Merge the two DataFrames on Patient_ID (one-to-one mapping assumed)
# merged = test.merge(sub_sample[['Patient_ID', 'Target_Week']], on='Patient_ID')

# # Compute Weeks_diff
# merged['Weeks_Diff'] = merged['Target_Week'] - merged['Reading_Week']

# # Rename/Select columns for final format
# final_df = merged[[
#     'Patient_ID',
#     'Reading_Week',
#     'Target_Week',
#     'Weeks_Diff',
#     'FVC',          # This is Curr_FVC
#     'Percent',
#     'Age',
#     'Sex',
#     'SmokingStatus'
# ]].rename(columns={'FVC': 'Curr_FVC'})

# print(final_df.head())


# # Encode categorical variables
# le_sex = LabelEncoder().fit(final_df['Sex'])
# le_smoke = LabelEncoder().fit(final_df['SmokingStatus'])

# final_df['Sex'] = le_sex.transform(final_df['Sex'])
# final_df['SmokingStatus'] = le_smoke.transform(final_df['SmokingStatus'])

# # Select features and target
# features = ['Reading_Week', 'Target_Week', 'Weeks_Diff', 'Curr_FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']

# X = final_df[features]

# # Normalize continuous features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


# ada_boost_preds, conf = weighted_model_prediction_with_confidence(X_scaled, models, weights)
# print(ada_boost_preds[:10])
# print(conf[:10])


# print(len(ada_boost_preds))
# print(len(X_scaled))


# # Attach predictions and confidence to the DataFrame
# final_df['FVC'] = ada_boost_preds
# final_df['Confidence'] = conf

# # Create the Patient_Week column
# final_df['Patient_Week'] = final_df['Patient_ID'] + '_' + final_df['Target_Week'].astype(str)

# # Select the final submission columns
# submission = final_df[['Patient_Week', 'FVC', 'Confidence']].copy()

# # Optional: round to reasonable precision
# submission['FVC'] = submission['FVC'].round(1)
# submission['Confidence'] = submission['Confidence'].round(1)

# # Save or print
# print(submission.head())
# print(submission.shape)
# submission.to_csv("submission.csv", index=False)



def compute_loss(y_pred, y_true, sigma):
    f = torch.sqrt(torch.tensor(2.0, device=y_pred.device))
    # sigma = torch.clamp(sigma, min=70.0)  # element-wise clamp

    # Compute delta = min(1000, abs(y_pred - y_true))
    delta = torch.abs(y_pred - y_true)
    delta = torch.clamp(delta, max=1000.0)

    # Compute loss
    loss = f * delta / sigma + torch.log(f * sigma)

    return loss.mean()  # return scalar loss



class ImprovedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            
            nn.Linear(64, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)


def train_MLP(model, device, optimizer, train_loader, val_loader, num_epochs=250):
    best_model = model
    min_loss = 1000
    no_imprv = 0
    
    # Training loop
    for epoch in range(num_epochs):  # number of epochs
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
    
            pred = model(xb)              # shape: [batch_size, 2]
            y_pred = pred[:, 0]           # shape: [batch_size]
            sigma = pred[:, 1]            # shape: [batch_size]
            
            loss = compute_loss(y_pred, yb, sigma)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                y_pred, sigma = pred[:, 0], pred[:, 1]
                loss = compute_loss(y_pred, yb, sigma)
                val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            if val_loss - min_loss < 0:
                min_loss = val_loss
                best_model = model
                print(f"new best model with loss {val_loss}")
            else:
                no_imprv += 1
    
        if no_imprv == 20:
            print("Early stopping...")
            break
    
    return best_model, min_loss


x = x_processed
y = y_processed


def ensemble_MLPs(n, X, Y):
    models = []
    weights = []
    for _ in range(n):
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=random.randint(1, 1000))
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleMLP().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3,  weight_decay=1e-5)
        
        mlp, loss = train_MLP(model=model,
                               device=device,
                               optimizer=optimizer,
                               train_loader=train_loader,
                               val_loader=val_loader)

        if loss < 10:
            models.append(mlp)
            weights.append(math.log((20 - loss) / loss))

    return models, weights
        


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ensemble_models, ensemble_weights = ensemble_MLPs(10, x, y)


print(len(ensemble_models))


ensemble_weights = normalize_to_sum_one(ensemble_weights)


test = pd.read_csv(base_path + 'test.csv')
sub_sample = pd.read_csv(base_path + 'sample_submission.csv')


# Extract Patient ID and Target Week from Patient_Week in df2
sub_sample[['Patient_ID', 'Target_Week']] = sub_sample['Patient_Week'].str.rsplit('_', n=1, expand=True)
sub_sample['Target_Week'] = sub_sample['Target_Week'].astype(int)

# Rename 'Patient' in df1 to match the new column name
test = test.rename(columns={'Patient': 'Patient_ID', 'Weeks': 'Reading_Week'})

# Merge the two DataFrames on Patient_ID (one-to-one mapping assumed)
merged = test.merge(sub_sample[['Patient_ID', 'Target_Week']], on='Patient_ID')

# Compute Weeks_diff
merged['Weeks_Diff'] = merged['Target_Week'] - merged['Reading_Week']

# Rename/Select columns for final format
final_df = merged[[
    'Patient_ID',
    'Reading_Week',
    'Target_Week',
    'Weeks_Diff',
    'FVC',          # This is Curr_FVC
    'Percent',
    'Age',
    'Sex',
    'SmokingStatus'
]].rename(columns={'FVC': 'Curr_FVC'})

print(final_df.head())


# Encode categorical variables
le_sex = LabelEncoder().fit(final_df['Sex'])
le_smoke = LabelEncoder().fit(final_df['SmokingStatus'])

final_df['Sex'] = le_sex.transform(final_df['Sex'])
final_df['SmokingStatus'] = le_smoke.transform(final_df['SmokingStatus'])

# Select features and target
features = ['Reading_Week', 'Target_Week', 'Weeks_Diff', 'Curr_FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']

X = final_df[features]

# Normalize continuous features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)  # shape: [12158, 8]


preds = []
confs = []
for model in ensemble_models:
    model.to(device)
    model.eval()
    with torch.no_grad():
        p = model(x_tensor)
        p = p.cpu().detach().numpy()
        preds.append(p[:, 0])
        confs.append(p[:, 1])

# preds.append(ada_boost_preds)
# confs.append(conf)


# mean_preds = np.mean(preds, axis=0)
# mean_confs = np.mean(confs, axis=0)

weights = np.array(ensemble_weights).reshape(-1, 1)  # shape: (n_models, 1)
weighted_pred = np.sum(weights * preds, axis=0)  # shape: (n_samples,)
weighted_conf = np.sum(weights * confs, axis=0)


print(weighted_pred[:10])
print(weighted_conf[:10])


# Attach predictions and confidence to the DataFrame
final_df['FVC'] = weighted_pred
final_df['Confidence'] = weighted_conf

# Create the Patient_Week column
final_df['Patient_Week'] = final_df['Patient_ID'] + '_' + final_df['Target_Week'].astype(str)

# Select the final submission columns
submission = final_df[['Patient_Week', 'FVC', 'Confidence']].copy()

# Optional: round to reasonable precision
submission['FVC'] = submission['FVC'].round(1)
submission['Confidence'] = submission['Confidence'].round(1)

# Save or print
print(submission.head())
print(submission.shape)
submission.to_csv("submission.csv", index=False)

