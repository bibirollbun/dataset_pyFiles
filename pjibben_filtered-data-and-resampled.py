# Install required packages (run this only if needed)
!pip install -q kaggle
!pip install -q pandas numpy scikit-learn torch matplotlib
!pip install -q openpyxl
!pip install -q imblearn

# Standard libraries
import kagglehub
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Utility
import matplotlib.pyplot as plt

# Warnings
import warnings
warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


importances_07_path = kagglehub.dataset_download('pjibben/importances-07')
columns_keep = pd.read_csv(f"{importances_07_path}/importances_07.csv")
columns_keep.head(3)


# === Load TRAIN data ===
# wid_path = kagglehub.competition_download('widsdatathon2025')
train_path = "/kaggle/input/widsdatathon2025/TRAIN_NEW"

connectome_train = pd.read_csv(f"{train_path}/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
# find the intersection of columns between connectome_train and columns_keep
connectome_keep_colums = list(set(connectome_train.columns) & set(columns_keep["Feature"]))
connectome_keep_colums.append('participant_id')
connectome_train = connectome_train[connectome_keep_colums]

print("Connectome:", connectome_train.shape)
connectome_train.info()


quant_meta_train = pd.read_excel(f"{train_path}/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
quant_meta_train_keep_columns = list(set(quant_meta_train.columns) & set(columns_keep["Feature"]))
quant_meta_train_keep_columns.append('participant_id')
quant_meta_train = quant_meta_train[quant_meta_train_keep_columns]

print("Quantitative metadata:", quant_meta_train.shape)
quant_meta_train.info()


cat_meta_train = pd.read_excel(f"{train_path}/TRAIN_CATEGORICAL_METADATA_new.xlsx")
cat_meta_train_keep_columns = list(set(cat_meta_train.columns) & set(columns_keep["Feature"]))
cat_meta_train_keep_columns.append('participant_id')
cat_meta_train = cat_meta_train[cat_meta_train_keep_columns]

print("Categorical metadata:", cat_meta_train.shape)
cat_meta_train.info()


targets_train = pd.read_excel(f"{train_path}/TRAINING_SOLUTIONS.xlsx")


print("Targets:", targets_train.shape)
targets_train.info()


# === Load TEST data ===
test_path = "/kaggle/input/widsdatathon2025/TEST"
connectome_test = pd.read_csv(f"{test_path}/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
connectome_test_keep_colums = list(set(connectome_test.columns) & set(columns_keep["Feature"]))
connectome_test_keep_colums.append('participant_id')
connectome_test = connectome_test[connectome_test_keep_colums]


print("Connectome:", connectome_test.shape)
connectome_test.info()


quant_meta_test = pd.read_excel(f"{test_path}/TEST_QUANTITATIVE_METADATA.xlsx")
quant_meta_test_keep_columns = list(set(quant_meta_test.columns) & set(columns_keep["Feature"]))
quant_meta_test_keep_columns.append('participant_id')
quant_meta_test = quant_meta_test[quant_meta_test_keep_columns]


print("Quantitative metadata:", quant_meta_test.shape)
quant_meta_test.info()


cat_meta_test = pd.read_excel(f"{test_path}/TEST_CATEGORICAL.xlsx")
cat_meta_test_keep_columns = list(set(cat_meta_test.columns) & set(columns_keep["Feature"]))
cat_meta_test_keep_columns.append('participant_id')
cat_meta_test = cat_meta_test[cat_meta_test_keep_columns]

print("Categorical metadata:", cat_meta_test.shape)
cat_meta_test.info()


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

def preprocess_data(connectome, quant_meta, cat_meta, targets=None, is_train=True, scaler=None, feature_columns=None):
    # Merge all inputs
    df = connectome.merge(quant_meta, on="participant_id").merge(cat_meta, on="participant_id")
    
    if is_train:
        df = df.merge(targets, on="participant_id")

   # random undersample
    # random undersample each target separately
    # if is_train:
    #     X = df.drop(columns=["participant_id", "ADHD_Outcome", "Sex_F"])
        
    #     # Undersample for ADHD
    #     rus_adhd = RandomUnderSampler(random_state=42)
    #     X_resampled_adhd, y_resampled_adhd = rus_adhd.fit_resample(X, df["ADHD_Outcome"])
        
    #     # Undersample for Sex
    #     rus_sex = RandomUnderSampler(random_state=42)
    #     X_resampled_sex, y_resampled_sex = rus_sex.fit_resample(X, df["Sex_F"])
        
    #     # Use the intersection of both resampled datasets
    #     common_indices = np.intersect1d(X_resampled_adhd.index, X_resampled_sex.index)
    #     X = X_resampled_adhd.loc[common_indices]
    #     df = pd.DataFrame(X)
    #     df["ADHD_Outcome"] = y_resampled_adhd.loc[common_indices]
    #     df["Sex_F"] = y_resampled_sex.loc[common_indices]
    #     df["participant_id"] = common_indices

    # random undersample Sex
    if is_train:
        X = df.drop(columns=["participant_id", "ADHD_Outcome", "Sex_F"])
        rus_sex = RandomUnderSampler(sampling_strategy="majority", random_state=42)
        X_resampled_sex, y_resampled_sex = rus_sex.fit_resample(X, df["Sex_F"])
        common_indices = np.intersect1d(X_resampled_sex.index, df.index)
        df = df.loc[common_indices]
        df["Sex_F"] = y_resampled_sex.loc[common_indices]
        df["participant_id"] = common_indices 

    # Store IDs for debugging or tracking
    ids = df["participant_id"]
    df = df.drop(columns=["participant_id"])

    # Separate targets if available
    if is_train:
        y_adhd = df["ADHD_Outcome"]
        y_sex = df["Sex_F"]
        X = df.drop(columns=["ADHD_Outcome", "Sex_F"])
    else:
        y_adhd = y_sex = None
        X = df

    # Replace infs with NaNs
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns with >10% missing values (only in train)
    # Play around with this threshold to see what works best and other techniques for cleaning the data
    if is_train:
        missing_ratio = X.isnull().median()
        to_drop = missing_ratio[missing_ratio > 0.10].index
        print(f"Dropping {len(to_drop)} columns with >10% missing values")
        X = X.drop(columns=to_drop)
    else:
        # Drop same columns as train
        X = X[feature_columns]

    # Separate types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute
    # should look at each var to see what is the best way to impute
    if numeric_cols:
        imp_num = SimpleImputer(strategy='mean')
        X[numeric_cols] = imp_num.fit_transform(X[numeric_cols]) if is_train else imp_num.fit(X[numeric_cols]).transform(X[numeric_cols])

    if categorical_cols:
        imp_cat = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = imp_cat.fit_transform(X[categorical_cols]) if is_train else imp_cat.fit(X[categorical_cols]).transform(X[categorical_cols])
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Align test with train
    if not is_train:
        X = X.reindex(columns=feature_columns, fill_value=0)

    # Scale
    # normalize the data
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    

    # Final safety check
    assert not np.isnan(X_scaled).any(), "NaNs in processed data!"
    assert not np.isinf(X_scaled).any(), "Infs in processed data!"

    return X_scaled, y_adhd, y_sex, scaler, X.columns.tolist()



# === Preprocess TRAIN ===
X_train_full, y_adhd, y_sex, scaler, feature_cols = preprocess_data(
    connectome=connectome_train,
    quant_meta=quant_meta_train,
    cat_meta=cat_meta_train,
    targets=targets_train,
    is_train=True
)

X_train, X_val, y_train_adhd, y_val_adhd, y_train_sex, y_val_sex = train_test_split(
    X_train_full, y_adhd, y_sex, test_size=0.1, random_state=42
)

# === Preprocess TEST ===
X_test, _, _, _, _ = preprocess_data(
    connectome=connectome_test,
    quant_meta=quant_meta_test,
    cat_meta=cat_meta_test,
    targets=None,
    is_train=False,
    scaler=scaler,
    feature_columns=feature_cols
)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)


# Convert data to a tensor so that it can be used in PyTorch
# y1 = ADHD, y2 = Sex
class BrainDataset(Dataset):
    def __init__(self, X, y1, y2):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y1 = torch.tensor(y1.values, dtype=torch.float32)
        self.y2 = torch.tensor(y2.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y1[idx], self.y2[idx]


# This is a typical feedforward neural network in pytorch
class BrainNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # batchnum which normalizes the data but not used here.
        self.out_adhd = nn.Linear(128, 1)
        self.out_sex = nn.Linear(128, 1)

    def forward(self, x):
        feat = self.net(x)
        return self.out_adhd(feat).squeeze(1), self.out_sex(feat).squeeze(1)


def evaluate(true, pred, label=""):
    acc = accuracy_score(true, pred)
    prec = precision_score(true, pred, zero_division=0)
    rec = recall_score(true, pred)
    f1 = f1_score(true, pred)
    return acc, prec, rec, f1

def log_loss(y_true, y_pred, epsilon=1e-15):
    y_pred_new = np.maximum(np.minimum(y_pred, 1-epsilon), epsilon)
    return -np.mean(y_true * np.log(y_pred_new) + (1-y_true) * np.log(1-y_pred_new))

def train_model(model, loader, val_loader, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define optimizer
    # play around with learning rate choose a value that is lower than the one used here 1e-3
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Use pos_weight to handle ADHD imbalance
    adhd_pos_weight = torch.tensor([y_train_adhd.value_counts()[0] / y_train_adhd.value_counts()[1]]).to(device)
    print(adhd_pos_weight)
    sex_pos_weight = torch.tensor([y_train_sex.value_counts()[0] / y_train_sex.value_counts()[1]]).to(device)
    print(sex_pos_weight)
    # Define separate losses
    loss_adhd = nn.BCEWithLogitsLoss(adhd_pos_weight)
    # original empty()
    loss_sex = nn.BCEWithLogitsLoss() # Think about handling class imbalance in sex variable as well.
    
    # Start training
    adhd_f1 = []
    sex_f1 = []
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for xb, yb1, yb2 in loader:
            xb, yb1, yb2 = xb.to(device), yb1.to(device), yb2.to(device)

            opt.zero_grad()
            
            pred1, pred2 = model(xb) # gets the data out of the model
            # look at the data and determine how to calculate the loss. Should they be added together?
            # weighted loss.
            loss = loss_adhd(pred1, yb1) + loss_sex(pred2, yb2)
            total_loss += loss.item()
            # clear out previous gradients
            # opt.zero_grad()
            # compute gradients
            loss.backward()
            # step in the direction of the gradient
            opt.step()

        model.eval() # set model to evaluation mode to use all nodes
        with torch.no_grad():
            all_preds, all_trues = { "ADHD": [], "Sex": [] }, { "ADHD": [], "Sex": [] }
            for xb, yb1, yb2 in val_loader:
                xb = xb.to(device)
                p1, p2 = model(xb)
                all_preds["ADHD"].extend((torch.sigmoid(p1) > 0.5).cpu().numpy())
                all_preds["Sex"].extend((torch.sigmoid(p2) > 0.5).cpu().numpy())
                all_trues["ADHD"].extend(yb1.numpy())
                all_trues["Sex"].extend(yb2.numpy())

        print(f"\nEpoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")
        acc, prec, rec, f1_adhd_new = evaluate(np.array(all_trues["ADHD"]), np.array(all_preds["ADHD"]), "ADHD")
        adhd_f1.append(f1_adhd_new)
        acc, prec, rec, f1_sex_new = evaluate(np.array(all_trues["Sex"]), np.array(all_preds["Sex"]), "Sex")
        sex_f1.append(f1_sex_new)

        for label in ["ADHD", "Sex"]:
            acc, prec, rec, f1 = evaluate(np.array(all_trues[label]), np.array(all_preds[label]), label)
            print(f"{label:<6} → Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    return np.array(adhd_f1), np.array(sex_f1)


epochs = 100
train_ds = BrainDataset(X_train, y_train_adhd, y_train_sex)
val_ds = BrainDataset(X_val, y_val_adhd, y_val_sex)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
model = BrainNet(X_train.shape[1])
adhd_f1, sex_f1 = train_model(model, train_loader, val_loader, epochs=epochs)


plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, epochs+1), adhd_f1, label="ADHD")
plt.plot(np.arange(1, epochs+1), sex_f1, label="Sex")
plt.xticks(np.arange(1, epochs+1,10))
plt.title("F1 Scores")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.show()


# Create test loader
test_loader = DataLoader(torch.tensor(X_test, dtype=torch.float32), batch_size=32)

# Predict on test set
model.eval()
all_preds_adhd, all_preds_sex = [], []

with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(next(model.parameters()).device)
        out_adhd, out_sex = model(xb)
        pred_adhd = (torch.sigmoid(out_adhd) > 0.5).cpu().numpy().astype(int)
        pred_sex = (torch.sigmoid(out_sex) > 0.5).cpu().numpy().astype(int)
        all_preds_adhd.extend(pred_adhd)
        all_preds_sex.extend(pred_sex)

# Combine predictions
predictions_df = pd.DataFrame({
    "ADHD_Outcome": all_preds_adhd,
    "Sex_F": all_preds_sex
})
sample_sub = pd.read_excel("/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")
submission = sample_sub.copy()
submission["ADHD_Outcome"] = predictions_df["ADHD_Outcome"]
submission["Sex_F"] = predictions_df["Sex_F"]

# Save submission
submission_path = "/kaggle/working/submission_07.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path}")
submission.head()





