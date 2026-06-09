!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


!pip install /kaggle/input/fttransformer-and-hyper-connections-requirements/tab_transformer_pytorch-0.4.1-py3-none-any.whl
!pip install /kaggle/input/fttransformer-and-hyper-connections-requirements/hyper_connections-0.1.11-py3-none-any.whl


import pandas as pd
import numpy as np
import torch
from metric import score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tab_transformer_pytorch import FTTransformer
import joblib


class SurvivalPreprocessor:
    def __init__(self):
        self.cat_encoders = {}
        self.num_scaler = StandardScaler()
        self.feature_names = None

    def fit_transform(self, df):
        df = df.copy()
        targets = df[["efs", "efs_time"]]
        df = df.drop(columns=["efs", "efs_time"])
        
        # Identify columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.feature_names = categorical_cols + numerical_cols

        # Handle missing values in categorical columns
        for col in categorical_cols:
            # Fill missing values with a placeholder
            df[col] = df[col].fillna("MISSING")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col]).astype(int) + 1
            self.cat_encoders[col] = le

        # Handle missing values in numerical columns (e.g., fill with mean)
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        df[numerical_cols] = self.num_scaler.fit_transform(df[numerical_cols])
        
        return pd.concat([df, targets], axis=1)

    def transform(self, df):
        df = df.copy()
        targets = None
        if {"efs", "efs_time"}.issubset(df.columns):
            targets = df[["efs", "efs_time"]]
            df = df.drop(columns=["ID", "efs", "efs_time"])
        # Process categorical features
        for col, le in self.cat_encoders.items():
            df[col] = df[col].fillna("MISSING")
            df[col] = df[col].map(lambda x: le.transform([x])[0] + 1 if x in le.classes_ else 0).astype(int)

        # Process numerical features
        numerical_cols = self.num_scaler.feature_names_in_
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        df[numerical_cols] = self.num_scaler.transform(df[numerical_cols])
            
        if targets is not None:
            return pd.concat([df, targets], axis=1)
        else:
            return df


class SurvivalFTTransformer(nn.Module):
    def __init__(self, categories, num_continuous, dim=64, depth=4, heads=8):
        super().__init__()
        self.transformer = FTTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            dim_out=1,
            depth=depth,
            heads=heads,
            attn_dropout=0.3,
            ff_dropout=0.3
        )
        
    def forward(self, x_categ, x_cont):
        return self.transformer(x_categ, x_cont).squeeze()



def cox_loss(pred_risk, y_time, y_event):
    # Sort batch by descending survival time
    _, idx = torch.sort(y_time, descending=True)
    pred_risk = pred_risk[idx]
    y_event = y_event[idx]

    # Calculate log likelihood
    hr = torch.exp(pred_risk)
    log_risk = torch.log(torch.cumsum(hr, dim=0) + 1e-7)
    loss = -torch.sum((pred_risk - log_risk) * y_event)
    
    # Normalize by number of events
    return loss / (torch.sum(y_event) + 1e-7)

def stratified_split(df, stratify_col, test_size=0.2):
    groups = df.groupby(stratify_col).groups
    train_indices, val_indices = [], []

    for _, idx in groups.items():
        if len(idx) < 2:
            train_indices.extend(idx)
            continue
            
        trn, val = train_test_split(idx, test_size=test_size)
        train_indices.extend(trn)
        val_indices.extend(val)
        
    return df.iloc[train_indices], df.iloc[val_indices]


def train_model(train_df, preprocessor, device='cuda'):
    # Preprocess data
    processed_df = preprocessor.fit_transform(train_df)
    joblib.dump(preprocessor, "preprocessor.pkl")
    categorical_cols = list(preprocessor.cat_encoders.keys())
    numerical_cols = list(preprocessor.num_scaler.feature_names_in_)
    
    # Stratified split
    train_df, val_df = stratified_split(processed_df, 'race_group', 0.2)
    
    # Create datasets
    def create_tensors(df):
        X_categ = torch.tensor(df[categorical_cols].values, dtype=torch.long)
        X_numer = torch.tensor(df[numerical_cols].values, dtype=torch.float32)
        y_time = torch.tensor(df["efs_time"].values, dtype=torch.float32)
        y_event = torch.tensor(df["efs"].values, dtype=torch.float32)
        return TensorDataset(X_categ, X_numer, y_time, y_event)
    
    train_dataset = create_tensors(train_df)
    val_dataset = create_tensors(val_df)

    # Model setup
    model = SurvivalFTTransformer(
        categories=[len(preprocessor.cat_encoders[col].classes_)+1 
                   for col in categorical_cols],
        num_continuous=len(numerical_cols)
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_score = -np.inf
    for epoch in range(25):
        model.train()
        epoch_loss = 0
        
        # Training
        for X_cat, X_num, T, E in DataLoader(train_dataset, batch_size=256, shuffle=True):
            X_cat, X_num = X_cat.to(device), X_num.to(device)
            T, E = T.to(device), E.to(device) # T = survival time, E = event indicator (1, 0)
            
            optimizer.zero_grad()
            pred_risk = model(X_cat, X_num)
            loss = cox_loss(pred_risk, T, E)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_T, val_E, val_race = [], [], [], []
        with torch.no_grad():
            for X_cat, X_num, T, E in DataLoader(val_dataset, batch_size=512):
                pred_risk = model(X_cat.to(device), X_num.to(device))
                val_preds.append(pred_risk.cpu())
                val_T.append(T)
                val_E.append(E)
        
        # Calculate metrics
        val_preds = torch.cat(val_preds).numpy()
        val_df_copy = val_df.copy()  # Create a copy of the DataFrame
        val_df_copy.loc[:, "prediction"] = val_preds  # Use .loc on the copied DataFrame
        val_score = score(val_df_copy[["ID", "efs", "efs_time", "race_group"]], val_df_copy[["ID", "prediction"]], "ID")
        
        # Update scheduler and save best model
        scheduler.step(loss)
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), "best_model.pth")
            
        print(f"Epoch {epoch+1:2d} | Loss: {epoch_loss/len(train_dataset):.4f} | "
              f"Val C-index: {val_score:.4f}")

    return model



def predict(test_df, preprocessor, device='cuda'):
    model = SurvivalFTTransformer(
        categories=[len(preprocessor.cat_encoders[col].classes_)+1
                   for col in preprocessor.cat_encoders],
        num_continuous=len(preprocessor.num_scaler.feature_names_in_)
    ).to(device)

    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()

    # Preprocess test data
    processed_test = preprocessor.transform(test_df)
    X_cat = torch.tensor(processed_test[preprocessor.cat_encoders.keys()].values,
                        dtype=torch.long)
    X_num = torch.tensor(processed_test[preprocessor.num_scaler.feature_names_in_].values,
                        dtype=torch.float32)

    # Create DataLoader for batching
    dataset = TensorDataset(X_cat, X_num)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    preds = []
    with torch.no_grad():
        for x_cat_batch, x_num_batch in dataloader:
            x_cat_batch = x_cat_batch.to(device)
            x_num_batch = x_num_batch.to(device)
            batch_pred = model(x_cat_batch, x_num_batch)
            preds.append(batch_pred.cpu().numpy())

    preds = np.concatenate(preds)

    return pd.DataFrame({
        "ID": test_df["ID"],
        "prediction": preds
    })


# Initialize preprocessing
preprocessor = SurvivalPreprocessor()

# Load and process data
train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

# Train model
model = train_model(train_df, preprocessor)

# Generate predictions
submission = predict(test_df, preprocessor)
submission.to_csv("submission.csv", index=False)


sub = pd.read_csv("/kaggle/working/submission.csv")
sub.head()

