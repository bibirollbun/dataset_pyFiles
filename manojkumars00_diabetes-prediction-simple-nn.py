import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn

from sklearn.metrics import roc_auc_score


train_csv_path = "/kaggle/input/playground-series-s5e12/train.csv"


train_df = pd.read_csv(train_csv_path)
train_df.drop(['id'], axis=1, inplace=True)


cat_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
]

train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
train_df = train_df.astype('float64')
train_df.head()


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler


standard_cols = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp',
    'heart_rate', 'cholesterol_total',
    'hdl_cholesterol', 'ldl_cholesterol'
]

robust_cols = [
    'physical_activity_minutes_per_week',
    'screen_time_hours_per_day',
    'triglycerides'
]

no_scale_cols = list(set(train_df.columns.tolist()) - {"diagnosed_diabetes"} - set(standard_cols) - set(robust_cols))

preprocessor = ColumnTransformer(
    transformers=[
        ('std', StandardScaler(), standard_cols),
        ('robust', RobustScaler(), robust_cols),
        ('none', 'passthrough', no_scale_cols)
    ],
    remainder='passthrough'
)





target_col = "diagnosed_diabetes"
y = train_df.pop(target_col)           # shape: (N,)
X = train_df           # shape: (N, num_features)

# Train / valid split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,        # keep same class balance
)



preprocessor.fit(X_train)

X_train_transformed = preprocessor.transform(X_train)
X_valid_transformed = preprocessor.transform(X_valid)


class TabularDataset(Dataset):
    def __init__(self, X, y):
        # Convert to torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # (N,) binary 0/1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

train_dataset = TabularDataset(X_train_transformed, y_train.values)
valid_dataset = TabularDataset(X_valid_transformed, y_valid.values)



batch_size = 1024*8  # tune based on GPU/CPU memory

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,   # set >0 if using in real env (not in Kaggle notebook sometimes)
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


batch_X, batch_y = next(iter(train_loader))
print(batch_X.shape)   # [batch_size, num_features]
print(batch_y.shape)   # [batch_size]
print(batch_X.dtype, batch_y.dtype)




class TabularNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            # nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            # nn.Dropout(0.2),
            nn.Linear(16, 1)   # output logit
        )
        
    def forward(self, x):
        return self.net(x).squeeze(1)  # [B]



criterion = nn.BCEWithLogitsLoss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = TabularNN(36)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

num_epochs = 20
best_val_auc = 0.0
best_val_acc = 0.0

for epoch in range(1, num_epochs + 1):
    # -------- TRAIN --------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)               # shape [B]
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

        # Metrics
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        train_correct += (preds == y_batch).sum().item()
        train_total += y_batch.size(0)

    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0
    val_total = 0
    val_correct = 0

    all_val_probs = []
    all_val_targets = []

    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            val_total += y_batch.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            val_correct += (preds == y_batch).sum().item()

            all_val_probs.append(probs.detach().cpu())
            all_val_targets.append(y_batch.detach().cpu())

    avg_val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    all_val_probs = torch.cat(all_val_probs).numpy()
    all_val_targets = torch.cat(all_val_targets).numpy()

    try:
        val_auc = roc_auc_score(all_val_targets, all_val_probs)
    except ValueError:
        # In case only one class present in validation in some fold
        val_auc = float("nan")

    # Save best model by AUC
    if not np.isnan(val_acc) and val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_tabular_diabetes_model.pth")

    print(
        f"Epoch [{epoch}/{num_epochs}] "
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} "
        f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
    )

print("Best Val ACC:", val_acc)



best_model_path = "/kaggle/working/best_tabular_diabetes_model.pth"

best_model = TabularNN(36)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)
best_model.eval()


test_data = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

cat_cols = [
"gender",
"ethnicity",
"education_level",
"income_level",
"smoking_status",
"employment_status",
]

test_data = pd.get_dummies(test_data, columns=cat_cols, drop_first=True)
test_data = test_data.astype('float64')
ids = test_data.pop('id')

test_data = preprocessor.transform(test_data)
print(f"Length of Test Data: {test_data.shape[0]}")


predictions = best_model(torch.tensor(test_data).float().to(device))
predictions = torch.sigmoid(predictions)
predictions = (predictions > 0.6).cpu().numpy().astype(int)


predictions.min(), predictions.max(), predictions.mean()


ids.shape, predictions.shape


submission = pd.DataFrame({
    "id": ids.astype('int'),
    "diagnosed_diabetes": predictions
})


submission.to_csv("submission.csv", index=False)


submission.head()




