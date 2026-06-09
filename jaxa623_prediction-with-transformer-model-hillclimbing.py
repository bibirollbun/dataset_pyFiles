import os
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_log_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


data_dir = '/kaggle/input/playground-series-s5e5/'
output_dir = '/kaggle/working'


from sklearn.preprocessing import StandardScaler, LabelEncoder
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df = df.drop('id', axis=1)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # Male=1, Female=0

numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

X = df.drop('Calories', axis=1).values
y = df['Calories'].values


def compute_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))  # clip to avoid log(-)


class FTTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=64, dropout=0.132568)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [batch, 1, d_model]
        x = self.transformer(x).squeeze(1)  # [batch, d_model]
        return self.fc_out(x).squeeze(1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
epochs = 12
batch_size = 128
best_rmsle = float('inf')

fold_rmsle_dict = {}


oof_preds  = np.zeros(len(y))
test_preds_folds = []
test_preds_log = []
fold_rmsles = []
n_folds = 5

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"\n===== Fold {fold+1} =====")
    
    # Data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float()),
                            batch_size=batch_size, shuffle=False)

    model = FTTransformer(input_dim=X.shape[1]).to(device)
    criterion = nn.MSELoss()  #  RMSELoss 
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_rmsle = float('inf')
    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Fold {fold+1})")
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        scheduler.step()

        # ---------- Val ----------
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                preds.append(pred)
                targets.append(yb.numpy())
        
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        rmsle = compute_rmsle(targets, preds)
        
        if rmsle < best_rmsle:
            best_rmsle = rmsle
            torch.save(model.state_dict(),
                       os.path.join(output_dir, f'best_model_fold{fold+1}.pt'))
            print(f"✅ Saved new best model for fold {fold+1} with RMSLE: {rmsle:.4f}")
            
    
        else:
            print(f"Fold {fold+1}, Epoch {epoch+1}, RMSLE: {rmsle:.4f} (best: {best_rmsle:.4f})")
    
            print(f"Fold {fold+1}, Epoch {epoch+1}, RMSLE: {rmsle:.4f}")
            
    print(f"Fold {fold+1} complete.")
    fold_rmsles.append(best_rmsle)


df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
ids = df_test['id'].values 

df_test['Sex'] = le.transform(df_test['Sex'])

df_test[numerical_features] = scaler.transform(df_test[numerical_features])

X_test = torch.tensor(df_test.drop(columns=['id']).values).float().to(device)


# ----- Submission -----
# Load model
test_preds_log = []

for fold in range(1, n_folds + 1):
    # 1) Load model weights
    model = FTTransformer(input_dim=X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(f'best_model_fold{fold}.pt'))
    model.eval()

    # 2) Inference
    with torch.no_grad():
        pred = model(X_test).cpu().numpy().squeeze()   # shape:(n_test,)
        pred = np.clip(pred, 0, None)                  # No negative
        test_preds_log.append(np.log1p(pred))          # https://www.kaggle.com/competitions/playground-series-s5e5/discussion/576111

    print(f'Fold {fold} done.')

    # 3) Cleaning memory
    del model                                 
    torch.cuda.empty_cache()                  
    gc.collect()                             

inv = 1 / np.array(fold_rmsles)       ## Inverse
weights = inv / inv.sum()            ## to 1
print("Auto weight:", weights)

test_preds_log = np.stack(test_preds_log, axis=0)
avg_log = (test_preds_log.T @ weights).flatten()
final_pred = np.expm1(avg_log)


submission = pd.DataFrame({
    'id': ids,
    'Calories': final_pred
})

submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
print("✅ Generated submission.csv, ready to submit")

