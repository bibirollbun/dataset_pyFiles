import torch
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train_df.plot(subplots=True, layout=(10, 2), figsize=(15, 10), title='Train Data Distribution')
plt.tight_layout()
plt.show()


if (train_df.isnull().values.any()):
    train_df.fillna(train_df.mean(), inplace=True)
if (test_df.isnull().values.any()):
    test_df.fillna(test_df.mean(), inplace=True)


test_predictions = {}

original_path = "/kaggle/input/hongkongrainfall/hongkong.csv"
original_path_2 = "/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv"
    

original = pd.read_csv(original_path, encoding="gbk")
original["date"] = pd.to_datetime(original[["year", "month", "day"]])
original = original.drop(["year", "month", "day", "low visibility hour", "radiation", "evaporation"], axis=1)
original["day"] = original.date.dt.dayofyear
original = original.drop("date", axis=1)
original.rainfall = original.rainfall.apply(lambda x: 1 if str(x).replace('.', '', 1).isdigit() else x)
original.rainfall = original.rainfall.replace({'微量': 1, '-': 0}).astype(int)
original.sunshine = original.sunshine.replace('-', 0).astype(float)
original.windspeed = original.windspeed.fillna(original.windspeed.mean())
for col in original.columns:
    original[col] = original[col].astype(train_df[col].dtype)

original_2 = pd.read_csv(original_path_2)
original_2.columns = original_2.columns.str.replace(" ", "")
original_2['rainfall'] = original_2['rainfall'].map({"yes": 1, "no": 0})
original_2.winddirection = original_2.winddirection.fillna(original_2.winddirection.mean())
original_2.windspeed = original_2.windspeed.fillna(original_2.windspeed.mean())
original_2.day = original_2.index + 1
for col in original_2.columns:
    original_2[col] = original_2[col].astype(train_df[col].dtype)

original_combined = pd.concat([original, original_2], axis=0).reset_index(drop=True)
original_combined = original_combined.drop_duplicates().reset_index(drop=True)
X_original = original_combined.drop(columns=['rainfall'], axis=1)
y_original = original_combined['rainfall']

models = {}
scaler = StandardScaler()
scaler.set_output(transform='pandas')

X_full = train_df.drop(columns=['rainfall', 'id'])
X_full = scaler.fit_transform(X_full)
y_full = train_df['rainfall']

X_test = test_df.drop(columns=['id'])
X_test = scaler.transform(X_test)

xgb_params = {
    'colsample_bylevel': 0.7921066237164537,
    'colsample_bynode': 0.6431557579286489,
    'colsample_bytree': 0.33314916328121835,
    'gamma': 2.6533897486162306,
    'learning_rate': 0.0995872230739346,
    'max_depth': 488,
    'max_leaves': 313,
    'min_child_weight': 9,
    'n_estimators': 4644,
    'n_jobs': -1,
    'random_state': 2100,
    'reg_alpha': 0.07653965420877373,
    'reg_lambda': 56.09661479066265,
    'subsample': 0.987487242879055,
    'verbosity': 0
}

model = XGBClassifier(**xgb_params)
# model = KNeighborsClassifier(n_neighbors=101, p=1)

# 5fold cross-validation
kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
for train_index, val_index in kf.split(X_full, y_full):
    X_train, X_val = X_full.iloc[train_index], X_full.iloc[val_index]
    y_train, y = y_full.iloc[train_index], y_full.iloc[val_index]

    X_train = pd.concat([X_train, X_original], ignore_index=True)
    y_train = pd.concat([y_train, y_original], ignore_index=True)
    
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1] 

    # ROC area
    roc_auc = metrics.roc_auc_score(y, y_pred)
    print(f'ROC AUC: {roc_auc:.4f}')

model.fit(X_full, y_full)
# Final predictions on the test set
y_test_pred = model.predict_proba(X_test)[:, 1]
test_predictions['xgb'] = y_test_pred



from sklearn.preprocessing import StandardScaler

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, target, sequence_length=1):
        self.df = df
        self.target = target
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.df) - self.sequence_length

    def __getitem__(self, idx):
        row = self.df.iloc[idx: idx + self.sequence_length]
        # Convert to tensor
        sequence = torch.tensor(row.drop(columns=[self.target]).values, dtype=torch.float32)
        target = torch.tensor(row[self.target].values[-1], dtype=torch.float32)
        return sequence, target

split_idx = int(len(train_df) * 0.8)
X_train = train_df[train_df['id'] < split_idx]
X_val = train_df[train_df['id'] >= split_idx]

X_train.drop(columns=['id'], inplace=True)
X_val.drop(columns=['id'], inplace=True)
# Normalize X per column except id and rainfall but still keep them
to_normalize = X_train.columns.difference(['rainfall'])
scaler = StandardScaler()
X_train[to_normalize] = scaler.fit_transform(X_train[to_normalize])
X_val[to_normalize] = scaler.transform(X_val[to_normalize])

train_dataset = Dataset(X_train, 'rainfall', sequence_length=20)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = Dataset(X_val, 'rainfall', sequence_length=20)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)


class Model(torch.nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, 16, num_layers=1, dropout=0.3, batch_first=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(16, 1)


    def forward(self, x):
        x = self.lstm(x)[0][:, -1, :]  # Get the last output of the LSTM
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)  # Use sigmoid for binary classification
    
input_size = X_train.shape[1] -1 
model = Model(input_size)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

num_epochs = 150
best_roc_auc = 0.0
patience = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for X, y in train_loader:
        batch_X = X
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

    total_loss = 0.0
    all_val_outputs = []
    all_val_targets = []
    for X, y in val_loader:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X)
            val_loss = criterion(val_outputs.squeeze(), y)
            total_loss += val_loss.item()
            all_val_outputs.append(val_outputs.squeeze().numpy())
            all_val_targets.append(y.numpy())
    print(f'Validation Loss: {total_loss/len(val_loader):.4f}')
    # Calculate ROC AUC
    all_val_outputs = np.concatenate(all_val_outputs)
    all_val_targets = np.concatenate(all_val_targets)
    roc_auc = metrics.roc_auc_score(all_val_targets, all_val_outputs)
    print(f'Validation ROC AUC: {roc_auc:.4f}')

    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        patience = 10
    else:
        patience -= 1
        if patience <= 0:
            print("Early stopping triggered")
            break
    
    

# Final predictions on the test set
model.eval()
X_test = test_df.drop(columns=['id']).values
X_test = scaler.transform(X_test)

# Stack last 20 val entries to test set
X_test = np.concatenate([X_val.drop(columns=['rainfall']).tail(19).values, X_test], axis=0)

X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
test_outputs = []
for i in range(20, X_test.shape[1]+1):
    X = X_test[:, i-20:i]
    with torch.no_grad():
        output = model(X).squeeze().numpy()
        
    test_outputs.append(output)


test_predictions['lstm'] = test_outputs


class ClassificationModel(torch.nn.Module):
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 16)
        self.fc3 = torch.nn.Linear(16, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)  # Use sigmoid for binary classification
    

train_tensor_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train.drop(columns=['rainfall']).values, dtype=torch.float32), 
                                                      torch.tensor(X_train['rainfall'].values, dtype=torch.float32))
train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=16, shuffle=True)

val_tensor_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val.drop(columns=['rainfall']).values, dtype=torch.float32), 
                                                    torch.tensor(X_val['rainfall'].values, dtype=torch.float32))
val_loader = torch.utils.data.DataLoader(val_tensor_dataset, batch_size=16, shuffle=False)

input_size = X_train.shape[1] - 1  # Exclude 'rainfall'
model = ClassificationModel(input_size)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
num_epochs = 200

best_roc_auc = 0.0
patience = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for X, y in train_loader:
        optimizer.zero_grad()
        outputs = model(X.float())
        loss = criterion(outputs.squeeze(), y.float())
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

    total_loss = 0.0
    all_val_outputs = []
    all_val_targets = []

    for X, y in val_loader:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X.float())
            val_loss = criterion(val_outputs.squeeze(), y.float())
            total_loss += val_loss.item()
            all_val_outputs.append(val_outputs.squeeze().numpy())
            all_val_targets.append(y.numpy())
    
    print(f'Validation Loss: {total_loss/len(val_loader):.4f}')
    # Calculate ROC AUC
    all_val_outputs = np.concatenate(all_val_outputs)
    all_val_targets = np.concatenate(all_val_targets)
    roc_auc = metrics.roc_auc_score(all_val_targets, all_val_outputs)
    print(f'Validation ROC AUC: {roc_auc:.4f}')

    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        patience = 10
    else:
        patience -= 1
        if patience <= 0:
            print("Early stopping triggered")
            break

# Predict test data
X_test = test_df.drop(columns=['id']).values
X_test = scaler.transform(X_test)

test_tensor_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32))

test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=1, shuffle=False)
test_outputs = []

model.eval()
with torch.no_grad():
    for X in test_loader:
        preds = model(X[0].float())
        test_outputs.append(preds.squeeze().numpy())
test_predictions['mlp'] = test_outputs



# Combine predictions from all models
mean_predictions = np.mean(list(test_predictions.values()), axis=0)
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': mean_predictions
})
submission_df.to_csv('ensemble_submission.csv', index=False)

