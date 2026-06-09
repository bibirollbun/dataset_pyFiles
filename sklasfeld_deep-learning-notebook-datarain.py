# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotting libraries
import matplotlib as plt 
import matplotlib.pyplot as plt
import seaborn as sns

# math libraries
from scipy import stats


# Model building
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LassoCV


# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Warnings
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Set up Kaggle API
# os.environ['KAGGLE_CONFIG_DIR'] = "/root/.kaggle"
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
# !kaggle competitions download -c widsdatathon2025

# Unzip
# !unzip -q widsdatathon2025.zip -d wids_data



# === Load TRAIN data ===
train_path = "/kaggle/input/widsdatathon2025/TRAIN_NEW"
connectome_train = pd.read_csv(f"{train_path}/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
quant_meta_train = pd.read_excel(f"{train_path}/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
cat_meta_train = pd.read_excel(f"{train_path}/TRAIN_CATEGORICAL_METADATA_new.xlsx")
targets_train = pd.read_excel(f"{train_path}/TRAINING_SOLUTIONS.xlsx")

# Check shapes
print("Train Connectome:", connectome_train.shape)
print("Train Quantitative metadata:", quant_meta_train.shape)
print("Train Categorical metadata:", cat_meta_train.shape)
print("Train Targets:", targets_train.shape)


# === Load TEST data ===
test_path = "/kaggle/input/widsdatathon2025/TEST"
connectome_test = pd.read_csv(f"{test_path}/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
quant_meta_test = pd.read_excel(f"{test_path}/TEST_QUANTITATIVE_METADATA.xlsx")
cat_meta_test = pd.read_excel(f"{test_path}/TEST_CATEGORICAL.xlsx")

# Check shapes
print("Test Connectome:", connectome_test.shape)
print("Test Quantitative metadata:", quant_meta_test.shape)
print("Test Categorical metadata:", cat_meta_test.shape)


# function to engineer the categorical variables
def categorical_featEngineering(cat_meta):
    # add column for single parents (1 means has single parent)
    cat_meta['single_parent'] = (
        (
            (cat_meta['Barratt_Barratt_P1_Edu'].notnull()) | 
            (cat_meta['Barratt_Barratt_P1_Occ'].notnull())
        ) &
        (cat_meta['Barratt_Barratt_P2_Edu'].isnull()) & 
        (cat_meta['Barratt_Barratt_P2_Occ'].isnull())
    ).astype(int)
    # get the parent with the highest education
    cat_meta['highest_edu_parent'] = (
        cat_meta[['Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P2_Edu']].max(axis=1)
    )
    # get the parent with the highest occupation
    cat_meta['highest_occ_parent'] = (
        cat_meta[['Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Occ']].max(axis=1)
    )
    # add column if one of the parents is a homegiver
    cat_meta['Barratt_Barratt_HomeGiver'] = (
        (cat_meta['Barratt_Barratt_P1_Occ'] == 0) |
        (cat_meta['Barratt_Barratt_P2_Occ'] == 0)
    ).astype(int)

    # drop the columns with individual parent data
    cat_meta = cat_meta.drop(columns=[
        'Barratt_Barratt_P1_Edu', 
        'Barratt_Barratt_P1_Occ', 
        'Barratt_Barratt_P2_Edu', 
        'Barratt_Barratt_P2_Occ'])

    # MRI_Track_Scan_Location: set NA to 2 since all NA individuals were at Study Site 1 
    # and most people who went to study site 1 did their MRI Track Scan at Location 2
    cat_meta['MRI_Track_Scan_Location'] = cat_meta['MRI_Track_Scan_Location'].fillna(2.0)

    # set Ethnicity values 9-11 to NA. set null value to 2.0 (Hispanic) if Ethnicity is 1.0 (Hispanic or Latino). 
    cat_meta.loc[(
        (cat_meta["PreInt_Demos_Fam_Child_Ethnicity"]==1.0) &
        (cat_meta["PreInt_Demos_Fam_Child_Race"].isnull())),"PreInt_Demos_Fam_Child_Race"] = 2.0
    cat_meta.loc[cat_meta["PreInt_Demos_Fam_Child_Race"] > 8,"PreInt_Demos_Fam_Child_Race"] = np.nan

    # set Race values 2-3 to NA. set null value to 1.0 if Ethnicity is 2.0 
    cat_meta.loc[(
        (cat_meta["PreInt_Demos_Fam_Child_Race"]==2.0) &
        (cat_meta["PreInt_Demos_Fam_Child_Ethnicity"].isnull())),"PreInt_Demos_Fam_Child_Ethnicity"] = 1.0
    cat_meta.loc[cat_meta["PreInt_Demos_Fam_Child_Ethnicity"] > 1,"PreInt_Demos_Fam_Child_Ethnicity"] = np.nan

    # Creating a list of all of the columns except the first
    columns_to_encode = cat_meta.columns[1:].tolist()
    categorical_ordinal_cols=[
        'single_parent',
        'highest_edu_parent',
        'highest_occ_parent',
        'Barratt_Barratt_HomeGiver',
    ]
    columns_to_encode = [x for x in columns_to_encode if not x in categorical_ordinal_cols]

    # encoding categorical data
    for col in columns_to_encode:
        cat_meta[col] = cat_meta[col].astype('category')
    drop_first_cols = [x for x in columns_to_encode if cat_meta[x].isnull().sum()==0]
    keep_allval_cols = [x for x in columns_to_encode if cat_meta[x].isnull().sum()>0]

    train_encoded1 = pd.get_dummies(cat_meta[drop_first_cols], drop_first=True, prefix_sep=':')
    train_encoded1 = train_encoded1.map(lambda x: 1 if x is True else (0 if x is False else x))
    train_encoded2 = pd.get_dummies(cat_meta[keep_allval_cols], prefix_sep=':')
    train_encoded2 = train_encoded2.map(lambda x: 1 if x is True else (0 if x is False else x))
    cat_meta_final = pd.concat([cat_meta.drop(columns=columns_to_encode), train_encoded1, train_encoded2], axis=1)
    return(columns_to_encode, cat_meta_final)



quant_meta_train


# principal components of the connectome
def getConnectomePC(in_connectome, is_train, pca_connectome=None, n_components=0.99):
    connectome_matrix = in_connectome.drop(columns=["participant_id"]).to_numpy()

    # normalize the matrix
    scaler = StandardScaler()
    connectome_matrix = scaler.fit_transform(connectome_matrix)

    # perform PCA
    if is_train:
        pca_connectome = PCA(n_components=n_components)
        pca_connectome.fit(connectome_matrix)
    transform_connectome = pca_connectome.transform(connectome_matrix)

    # get the components into a pandas dataframe
    transform_connectome_df = pd.DataFrame(
        data = transform_connectome,
        columns = ["PC"+str(x+1) for x in range(0,len(pca_connectome.explained_variance_ratio_))]
    )
    transform_connectome_df = pd.concat([in_connectome['participant_id'], transform_connectome_df],axis=1)

    return pca_connectome, transform_connectome_df


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(connectome, quant_meta, cat_meta, targets=None, is_train=True, 
    scaler=None, feature_columns=None, pca_connectome=None, imp_num=None,
    n_components=0.99, random_state=42):
    # edit the categorical features
    encoded_cols, cat_meta = categorical_featEngineering(cat_meta)
    
    # add any encoded cols found in the training data to the test data
    if not is_train:
        train_encoded_cols = [x for x in feature_columns if ":" in x]
        missing_cols = [x for x in train_encoded_cols if x not in cat_meta.columns]
        if len(missing_cols) > 0:
            print(f"Missing encoded columns in test data: {missing_cols}")
        cat_meta.loc[:,missing_cols] = 0

    # Get principal components
    pca_connectome, connectome_pc = getConnectomePC(
        connectome, is_train, 
        pca_connectome=pca_connectome, 
        n_components=n_components)

    # Merge all inputs
    df = connectome_pc.merge(quant_meta, on="participant_id").merge(cat_meta, on="participant_id")
    
    if is_train:
        df = df.merge(targets, on="participant_id")

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

    # Drop columns with >30% missing values (only in train)
    if is_train:
        missing_ratio = X.isnull().mean()
        to_drop = missing_ratio[missing_ratio > 0.3].index
        print(f"Dropping {len(to_drop)} columns with >30% missing values")
        X = X.drop(columns=to_drop)
    else:
        # Drop same columns as train
        X = X[feature_columns]

    # Separate numeric and categorical columns
    numeric_cols = X.columns.tolist()

    # Impute numeric features with median
    if numeric_cols:
        if is_train:
            imp_num = SimpleImputer(strategy='median')
            #imp_num = IterativeImputer(estimator=LassoCV(random_state=random_state), max_iter=5, random_state=random_state)
            imp_num = imp_num.fit(X[numeric_cols])
        X[numeric_cols] = imp_num.transform(X[numeric_cols])



    # Align test with train
    if not is_train:
        X = X.reindex(columns=feature_columns, fill_value=0)

    # Scale
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Final safety check
    assert not np.isnan(X_scaled).any(), "NaNs in processed data!"
    assert not np.isinf(X_scaled).any(), "Infs in processed data!"

    return X_scaled, y_adhd, y_sex, scaler, X.columns.tolist(), pca_connectome, imp_num



# === Preprocess TRAIN ===
print("=== Preprocess TRAIN ===")
X_train_full, y_adhd, y_sex, scaler, feature_cols, train_pca_connectome, train_imp_num = preprocess_data(
    connectome=connectome_train,
    quant_meta=quant_meta_train,
    cat_meta=cat_meta_train,
    targets=targets_train,
    is_train=True, n_components=0.99
)

#=== Preprocess TEST ===
print("=== Preprocess TEST ===")
X_test, _, _, _, _, _, _ = preprocess_data(
    connectome=connectome_test,
    quant_meta=quant_meta_test,
    cat_meta=cat_meta_test,
    targets=None,
    is_train=False,
    scaler=scaler,
    feature_columns=feature_cols,
    pca_connectome=train_pca_connectome,
    imp_num=train_imp_num
)


X_train, X_val, y_train_adhd, y_val_adhd, y_train_sex, y_val_sex = train_test_split(
    X_train_full, y_adhd, y_sex, test_size=0.1, random_state=42
)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)


class BrainDataset(Dataset):
    def __init__(self, X, y1, y2):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y1 = torch.tensor(y1.values, dtype=torch.float32)
        self.y2 = torch.tensor(y2.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y1[idx], self.y2[idx]



class BrainNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), # 256 hidden units
            nn.ReLU(), # ReLU activation
            nn.Dropout(0.3), # 30% dropout
            nn.Linear(256, 128), # 128 hidden units
            nn.ReLU() # ReLU activation
        )
        self.out_adhd = nn.Linear(128, 1) # 1 output
        self.out_sex = nn.Linear(128, 1) # 1 output

    def forward(self, x):
        feat = self.net(x)
        return self.out_adhd(feat).squeeze(1), self.out_sex(feat).squeeze(1)



# 2x weight to Female ADHD cases (ADHD_Outcome=1, Sex_F=1)
def new_evaluate(true_sex, pred_sex,  true_adhd, pred_adhd):
    true = np.column_stack((true_sex, true_adhd)) 
    pred = np.column_stack((pred_sex, pred_adhd))
    weights = true.all(axis=1).astype(np.int32)
    weights = [x + 1 for x in weights]
    acc = accuracy_score(true, pred, sample_weight=weights)
    prec = precision_score(true, pred, sample_weight=weights, average='samples', zero_division=0) # in case of imbalanced data set in our case
    rec = recall_score(true, pred, sample_weight=weights, average='samples') 
    f1 = f1_score(true, pred, sample_weight=weights, average='samples')
    print(f1)
    print(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

def evaluate(true, pred, label=""):
    acc = accuracy_score(true, pred)
    prec = precision_score(true, pred, zero_division=0) # in case of imbalanced data set in our case
    rec = recall_score(true, pred) 
    f1 = f1_score(true, pred)
    print(f"{label:<6} → Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

def train_model(model, loader, val_loader, epochs=10, adhd_threshold=0.5, sex_threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Move model to GPU if available
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Use pos_weight to handle ADHD imbalance
    pos_weight = torch.tensor(
        ((y_train_adhd + y_train_sex) == 2).astype(int).value_counts()[0] /
        ((y_train_adhd + y_train_sex) == 2).astype(int).value_counts()[1]
        ).to(device)
    adhd_pos_weight = torch.tensor([y_train_adhd.value_counts()[0] / y_train_adhd.value_counts()[1]]).to(device)
    sex_pos_weight = torch.tensor([y_train_sex.value_counts()[0] / y_train_sex.value_counts()[1]]).to(device)
    # Define separate losses
    loss_adhd = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # binary classification
    loss_sex = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        total_loss = 0
        model.train() # set model to training mode
        for xb, yb1, yb2 in loader: # iterate over batches
            xb, yb1, yb2 = xb.to(device), yb1.to(device), yb2.to(device)
            pred1, pred2 = model(xb)
            loss = loss_adhd(pred1, yb1) + loss_sex(pred2, yb2) # add up losses TODO: add weights
            total_loss += loss.item() # accumulate total loss
            opt.zero_grad(set_to_none=True) # clear gradients
            loss.backward() # compute gradients
            opt.step() # update parameters

        model.eval()
        with torch.no_grad():
            all_preds, all_trues = { "ADHD": [], "Sex": [] }, { "ADHD": [], "Sex": [] }
            for xb, yb1, yb2 in val_loader:
                xb = xb.to(device)
                p1, p2 = model(xb)
                all_preds["ADHD"].extend((torch.sigmoid(p1) > adhd_threshold).cpu().numpy())
                all_preds["Sex"].extend((torch.sigmoid(p2) > sex_threshold).cpu().numpy())
                all_trues["ADHD"].extend(yb1.numpy())
                all_trues["Sex"].extend(yb2.numpy())

        print(f"\nEpoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")
        if 1==1:
            for label in ["ADHD", "Sex"]:
                evaluate(np.array(all_trues[label]), np.array(all_preds[label]), label)
        else:
            new_evaluate(
                np.array(all_trues["Sex"]), np.array(all_preds["Sex"]), 
                np.array(all_trues["ADHD"]), np.array(all_preds["ADHD"]))


train_ds = BrainDataset(X_train, y_train_adhd, y_train_sex)
val_ds = BrainDataset(X_val, y_val_adhd, y_val_sex)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = BrainNet(X_train.shape[1])
train_model(model, train_loader, val_loader, epochs=10, adhd_threshold=0.5, sex_threshold=0.5)



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
submission_path = "submission_epoch10.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path}")
submission.head()



def compute_feature_importance(model, X_batch, target_class=0):
    """
    Computes average absolute gradients of input features with respect to output
    target_class: 0 for ADHD, 1 for Sex
    """
    model.eval()
    X_batch = torch.tensor(X_batch, dtype=torch.float32, requires_grad=True).to(next(model.parameters()).device)

    out_adhd, out_sex = model(X_batch)
    output = out_adhd if target_class == 0 else out_sex

    # We take the mean of outputs to get a single scalar
    output = output.mean()
    output.backward()

    importance = X_batch.grad.abs().mean(dim=0).cpu().numpy()
    return importance

# Choose a sample batch from validation
X_sample = X_val[:256]

# ADHD importance
adhd_importance = compute_feature_importance(model, X_sample, target_class=0)
sex_importance = compute_feature_importance(model, X_sample, target_class=1)

# Create DataFrame for top features
importances_df = pd.DataFrame({
    "Feature": feature_cols,
    "ADHD_Importance": adhd_importance,
    "Sex_Importance": sex_importance
}).sort_values("ADHD_Importance", ascending=False)

import matplotlib.pyplot as plt

top_k = 15
top_adhd = importances_df.sort_values("ADHD_Importance", ascending=False).head(top_k)
top_sex = importances_df.sort_values("Sex_Importance", ascending=False).head(top_k)

plt.figure(figsize=(10, 5))
plt.barh(top_adhd["Feature"], top_adhd["ADHD_Importance"])
plt.title("Top Features for ADHD Prediction")
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(10, 5))
plt.barh(top_sex["Feature"], top_sex["Sex_Importance"])
plt.title("Top Features for Sex Prediction")
plt.gca().invert_yaxis()
plt.show()



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(model, X, labels, label_name="Label"):
    model.eval()
    with torch.no_grad():
        features = model.net(torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)).cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
    plt.title(f"t-SNE of Hidden Features Colored by {label_name}")
    plt.colorbar(scatter, label=label_name)
    plt.show()

# Example usage:
plot_tsne(model, X_train, y_train_adhd.values, label_name="ADHD")
plot_tsne(model, X_train, y_train_sex.values, label_name="Sex")








