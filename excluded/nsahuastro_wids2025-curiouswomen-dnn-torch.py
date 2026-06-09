#!pip install -q pandas numpy torch matplotlib scikit-learn imblearn 
!pip install -q pandas numpy torch matplotlib imblearn openpyxl
!pip install -q scikit-learn==1.5.0

#!pip install 
#!pip install -q openpyxl
#

# Standard libraries
import os
import json
import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import imblearn
#from imblearn.under_sampling import RandomUnderSampler

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Utility
import matplotlib
import matplotlib.pyplot as plt

# Warnings
import warnings
warnings.filterwarnings("ignore")
print("done")


print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"pytorch: {torch.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"imbearn: {imblearn.__version__}")




from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# === Load TRAIN data ===
train_path = "/kaggle/input/widsdatathon2025/TRAIN_NEW"
connectome_train = pd.read_csv(f"{train_path}/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
quant_meta_train = pd.read_excel(f"{train_path}/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
cat_meta_train = pd.read_excel(f"{train_path}/TRAIN_CATEGORICAL_METADATA_new.xlsx")
targets_train = pd.read_excel(f"{train_path}/TRAINING_SOLUTIONS.xlsx")

# Check shapes
print("Connectome:", connectome_train.shape)
print("Quantitative metadata:", quant_meta_train.shape)
print("Categorical metadata:", cat_meta_train.shape)
print("Targets:", targets_train.shape)


# === Load TEST data ===
test_path = "/kaggle/input/widsdatathon2025/TEST"
connectome_test = pd.read_csv(f"{test_path}/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
quant_meta_test = pd.read_excel(f"{test_path}/TEST_QUANTITATIVE_METADATA.xlsx")
cat_meta_test = pd.read_excel(f"{test_path}/TEST_CATEGORICAL.xlsx")



def preprocess_data(connectome, quant_meta, cat_meta, targets=None, is_train=True, scaler=None,imp_cat=None, feature_columns=None,  means=None, stds=None):
    # Merge all inputs
    df = connectome.merge(quant_meta, on="participant_id").merge(cat_meta, on="participant_id")
    
    if is_train:
        df = df.merge(targets, on="participant_id")

    # using random undersample to address class imbalance in Sex_F
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

    # Drop columns with >30% missing values (only in train)
    if is_train:
        missing_ratio = X.isnull().mean()
        to_drop = missing_ratio[missing_ratio > 0.3].index 
        print(f"Dropping {len(to_drop)} columns with >30% missing values")
        X = X.drop(columns=to_drop)
    else:
        # Drop same columns as train
        X = X[feature_columns]

    # Separate types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute
    if numeric_cols:
        imp_num = SimpleImputer(strategy='median')
        X[numeric_cols] = imp_num.fit_transform(X[numeric_cols]) if is_train else imp_num.fit(X[numeric_cols]).transform(X[numeric_cols])
    #rng = np.random.default_rng(seed=42) 
    #if numeric_cols:
    #    if is_train:
    #        means = X[numeric_cols].mean()
    #        stds = X[numeric_cols].std()
    #        X[numeric_cols] = X[numeric_cols].apply(
    #            lambda col: col.fillna(rng.normal(loc=means[col.name], scale=stds[col.name]))
    #        )
    #    else:
    #        assert means is not None and stds is not None, "Must pass means and stds from training!"
    #        X[numeric_cols] = X[numeric_cols].apply(
    #            lambda col: col.fillna(rng.normal(loc=means[col.name], scale=stds[col.name]))
    #        )


    #if categorical_cols:
    #    imp_cat = SimpleImputer(strategy='most_frequent')
    #    X[categorical_cols] = imp_cat.fit_transform(X[categorical_cols]) if is_train else imp_cat.fit(X[categorical_cols]).transform(X[categorical_cols])
    #    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    if categorical_cols:
        if is_train:
            imp_cat = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = imp_cat.fit_transform(X[categorical_cols])
        else:
            assert imp_cat is not None, "Imputer must be passed for test data!"
            X[categorical_cols] = imp_cat.transform(X[categorical_cols])

        # After imputing, encode categoricals
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

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

    return X_scaled, y_adhd, y_sex, scaler, imp_cat, X.columns.tolist(), means, stds



# === Preprocess TRAIN ===
X_train_full, y_adhd, y_sex, scaler, imp_cat_train, feature_cols, means_train, stds_train = preprocess_data(
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
X_test, _, _, _, _, _, _, _ = preprocess_data(
    connectome=connectome_test,
    quant_meta=quant_meta_test,
    cat_meta=cat_meta_test,
    targets=None,
    is_train=False,
    scaler=scaler,
    imp_cat=imp_cat_train,
    feature_columns=feature_cols,
    means=means_train,
    stds=stds_train
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
    #def __init__(self, input_dim):
    #    super().__init__()
    #    self.net = nn.Sequential(
    #        nn.Linear(input_dim, 256),
    #        nn.ReLU(),
    #        nn.Dropout(0.3),
    #        nn.Linear(256, 128),
    #        nn.ReLU()
    #    )
    #    self.out_adhd = nn.Linear(128, 1)
    #    self.out_sex = nn.Linear(128, 1)

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            #nn.Linear(input_dim, 512),
            #nn.BatchNorm1d(512),
            #nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
            #nn.LeakyReLU(negative_slope=0.01)

        )
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
    print(f"{label:<6} → Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()



import torch.nn.functional as F

def supervised_contrastive_loss(features, labels, temperature=0.5):
    """Compute supervised contrastive loss.
    
    Args:
        features: Tensor of shape [batch_size, feature_dim]
        labels: Tensor of shape [batch_size]
    """
    device = features.device
    labels = labels.contiguous().view(-1, 1)  # shape (batch_size, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # shape (batch_size, batch_size)

    features = F.normalize(features, dim=1)  # normalize feature vectors
    similarity_matrix = torch.matmul(features, features.T) / temperature  # cosine similarities

    # Mask out self-similarity
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(device)
    mask = mask * logits_mask

    exp_sim = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

    loss = -mean_log_prob_pos.mean()
    return loss



!pip install -q torch-optimizer


from torch_optimizer import RAdam

def train_model(model, loader, val_loader, epochs=200, patience=30, contrastive_weight=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt = RAdam(model.parameters(), lr=1e-3)
    #opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    #opt = Lookahead(opt)

    #scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10, verbose=True)

    all_yb1, all_yb2 = [], []
    for _, yb1, yb2 in loader:
        all_yb1.append(yb1)
        all_yb2.append(yb2)
    all_yb1 = torch.cat(all_yb1, dim=0)
    all_yb2 = torch.cat(all_yb2, dim=0)

    num_neg_adhd = (all_yb1 == 0).sum().item()
    num_pos_adhd = (all_yb1 == 1).sum().item()
    adhd_pos_weight = torch.tensor([num_neg_adhd / num_pos_adhd]).to(device)

    num_neg_sex = (all_yb2 == 0).sum().item()
    num_pos_sex = (all_yb2 == 1).sum().item()
    sex_pos_weight = torch.tensor([num_neg_sex / num_pos_sex]).to(device)
    sex_alpha = num_neg_sex / (num_neg_sex + num_pos_sex)

    loss_adhd = nn.BCEWithLogitsLoss(pos_weight=adhd_pos_weight)
    #loss_sex = FocalLoss(alpha=sex_alpha, gamma=1.5)
    loss_sex =nn.BCEWithLogitsLoss() #pos_weight=sex_pos_weight

    best_f1 = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for xb, yb1, yb2 in loader:
            xb, yb1, yb2 = xb.to(device), yb1.to(device), yb2.to(device)

            feat = model.net(xb)  # hidden shared features
            feat = F.normalize(feat, p=2, dim=1)  # <- normalize embeddings

            pred1 = model.out_adhd(feat).squeeze(1)
            pred2 = model.out_sex(feat).squeeze(1)

            cls_loss = loss_adhd(pred1, yb1) + loss_sex(pred2, yb2)

            # === ADD contrastive loss ===
            multi_target = yb1.int() * 2 + yb2.int()  # [0,1,2,3]
            contrastive_loss_joint = 0 #supervised_contrastive_loss(feat, multi_target)
            #contrastive_loss_adhd = supervised_contrastive_loss(feat, yb1)
            #contrastive_loss_sex  = supervised_contrastive_loss(feat, yb2)

            #total_batch_loss = cls_loss + contrastive_weight * (contrastive_loss_adhd + contrastive_loss_sex)
            total_batch_loss = cls_loss + contrastive_weight * (contrastive_loss_joint)

            opt.zero_grad()
            total_batch_loss.backward()
            opt.step()

            total_loss += total_batch_loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            all_preds, all_trues = { "ADHD": [], "Sex": [] }, { "ADHD": [], "Sex": [] }
            for xb, yb1, yb2 in val_loader:
                xb = xb.to(device)
                p1, p2 = model(xb)
                all_preds["ADHD"].extend((torch.sigmoid(p1) > 0.5).cpu().numpy())
                all_preds["Sex"].extend((torch.sigmoid(p2) > 0.5).cpu().numpy())
                all_trues["ADHD"].extend(yb1.numpy())
                all_trues["Sex"].extend(yb2.numpy())

        f1_adhd = f1_score(np.array(all_trues["ADHD"]), np.array(all_preds["ADHD"]))
        f1_sex = f1_score(np.array(all_trues["Sex"]), np.array(all_preds["Sex"]))
        avg_f1 = (f1_adhd + f1_sex) / 2

        #scheduler.step(avg_f1)

        print(f"\nEpoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")
        for label in ["ADHD", "Sex"]:
            evaluate(np.array(all_trues[label]), np.array(all_preds[label]), label)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                print(f"Best model was from epoch {best_epoch} with Avg F1: {best_f1:.4f}")
                model.load_state_dict(best_model_state)
                break



train_ds = BrainDataset(X_train, y_train_adhd, y_train_sex)

val_ds = BrainDataset(X_val, y_val_adhd, y_val_sex)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = BrainNet(X_train.shape[1])
train_model(model, train_loader, val_loader, epochs=100, patience=10)



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

top_k = 30
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

#X_resampled.shape, y_sex_resampled.shape, y_adhd_resampled.shape
#plot_tsne(model, X_resampled, y_adhd_resampled.values, label_name="ADHD")
#plot_tsne(model, X_resampled, y_sex_resampled.values, label_name="Sex")



# Select features with ADHD importance > 0.0008
selected_features_adhd = importances_df[importances_df["ADHD_Importance"] > 0.0008]["Feature"].tolist()

# Select features with ADHD importance > 0.00025
selected_features_sex = importances_df[importances_df["Sex_Importance"] > 0.0002]["Feature"].tolist()

# Union of both
selected_features = list(set(selected_features_sex) | set(selected_features_adhd ))  # set union
#print(f"Selected features: {selected_features}")
print(len(selected_features))


# Map feature names to indices
feature_to_idx = {feature: idx for idx, feature in enumerate(feature_cols)}
selected_indices = [feature_to_idx[feat] for feat in selected_features]


## for pandas data frame 1. Subset only the selected features
#X_train_selected = X_train[selected_features]
#X_val_selected = X_val[selected_features]

##X_resampled.shape, y_sex_resampled.shape, y_adhd_resampled.shape

# Subset X_train using integer indices
X_train_selected = X_train[:, selected_indices]
X_val_selected = X_val[:, selected_indices]

# 2. Create Datasets with selected features
train_ds_2 = BrainDataset(X_train_selected, y_train_adhd, y_train_sex)
val_ds_2 = BrainDataset(X_val_selected, y_val_adhd, y_val_sex)

# 3. Create Dataloaders
train_loader_2 = DataLoader(train_ds_2, batch_size=32, shuffle=True)
val_loader_2 = DataLoader(val_ds_2, batch_size=32)

# 4. Update BrainNet input_dim
model_2 = BrainNet(input_dim=X_train_selected.shape[1])

# 5. Train the model
train_model(model_2, train_loader_2, val_loader_2, epochs=500, patience=100)



plot_tsne(model_2, X_train_selected, y_train_adhd.values, label_name="ADHD")
plot_tsne(model_2, X_train_selected, y_train_sex.values, label_name="Sex")


# Create test loader
test_loader1 = DataLoader(torch.tensor(X_test[:, selected_indices], dtype=torch.float32), batch_size=32)

# Predict on test set
model_2.eval()
all_preds_adhd1, all_preds_sex1 = [], []

with torch.no_grad():
    for xb in test_loader1:
        xb = xb.to(next(model_2.parameters()).device)
        out_adhd, out_sex = model_2(xb)
        pred_adhd = (torch.sigmoid(out_adhd) > 0.5).cpu().numpy().astype(int)
        pred_sex = (torch.sigmoid(out_sex) > 0.5).cpu().numpy().astype(int)
        all_preds_adhd1.extend(pred_adhd)
        all_preds_sex1.extend(pred_sex)

# Combine predictions
predictions_df1 = pd.DataFrame({
    "ADHD_Outcome": all_preds_adhd1,
    "Sex_F": all_preds_sex1
})
sample_sub1 = pd.read_excel("/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")
submission1 = sample_sub1.copy()
submission1["ADHD_Outcome"] = predictions_df1["ADHD_Outcome"]
submission1["Sex_F"] = predictions_df1["Sex_F"]

# Save submission
submission_path1 = "submission_DNN_undersample_SexF.csv"
submission1.to_csv(submission_path1, index=False)
print(f"Submission saved to: {submission_path1}")
submission1.head()


