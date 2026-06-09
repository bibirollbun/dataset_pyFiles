import numpy as np
import pandas as pd
train =  pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test =  pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')



train.columns


cols_to_remove = ['id','ethnicity', 'education_level','employment_status']
idx = test['id']
train.drop(cols_to_remove,axis=1,inplace=True)
test.drop(cols_to_remove,axis=1,inplace=True)


test_df = test


train.info()


import matplotlib.pyplot as plt
import seaborn as sns
plt.bar(["yes","no"],train['diagnosed_diabetes'].value_counts())


df_majority = train[train.diagnosed_diabetes == 1.0]
df_minority = train[train.diagnosed_diabetes == 0.0]
df_maj = df_majority.sample(len(df_minority))


df = pd.concat([df_maj,df_minority])
df = train





from sklearn.preprocessing import LabelEncoder
col_str = ['income_level','gender','smoking_status']
le = LabelEncoder()
for col in col_str:
    df[col] = le.fit_transform(df[col])
    test[col] = le.transform(test[col])
test_df = test



X = df.drop("diagnosed_diabetes",axis = 1)
y = df['diagnosed_diabetes'].astype(int).values


from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)


def clip_outliers_iqr(df, factor=1.5):
    df_clipped = df.copy()
    numeric_cols = df_clipped.select_dtypes(include="number").columns

    for col in numeric_cols:
        Q1 = df_clipped[col].quantile(0.25)
        Q3 = df_clipped[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        df_clipped[col] = df_clipped[col].clip(lower, upper)

    return df_clipped

x_train = clip_outliers_iqr(x_train)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)






# from xgboost import XGBClassifier

# neg, pos = np.bincount(y_train)
# scale_pos_weight = neg / pos

# model = XGBClassifier(
#     objective='binary:logistic',
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=4,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     scale_pos_weight=scale_pos_weight,
#     eval_metric='logloss',
#     random_state=42
# )
# model.fit(x_train, y_train)


# from sklearn.metrics import classification_report
# y_pred = model.predict(x_test)
# print(classification_report(y_test,y_pred))





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np


# df = train
# # ---------- data ----------
# X = df.drop(columns=['diagnosed_diabetes']).values
# y = df['diagnosed_diabetes'].astype(int).values

# x_train, x_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # ---------- scaling ----------
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# ---------- tensors ----------
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ---------- dataset ----------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TabularDataset(x_train, y_train)
test_ds = TabularDataset(x_test, y_test)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)


# ---------- imbalance handling ----------
neg = (y_train == 0).sum().item()
pos = (y_train == 1).sum().item()
pos_weight = torch.tensor([neg / pos])

# ---------- model ----------
class NN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
       )

    def forward(self, x):
        return self.net(x)

model = NN(x_train.shape[1])

# ---------- training setup ----------
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---------- training ----------
epochs = 100
model.train()

for epoch in range(epochs):
    epoch_loss = 0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        logits = model(xb).squeeze()
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # ---------- accuracy ----------
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {acc:.4f}")





model.eval()
y_preds = []

with torch.no_grad():
    for xb, _ in test_loader:
        probs = torch.sigmoid(model(xb))
        y_preds.append(probs)

y_preds = torch.cat(y_preds).squeeze()
y_pred_labels = (y_preds >= 0.5).int().numpy()

print(classification_report(y_test.numpy(), y_pred_labels))



X_test = test_df.values
X_test = scaler.transform(X_test)
X_test = torch.tensor(X_test, dtype=torch.float32)
class TestDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

test_ds = TestDataset(X_test)
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

model.eval()
all_probs = []

with torch.no_grad():
    for xb in test_loader:
        probs = torch.sigmoid(model(xb))
        all_probs.append(probs)

all_probs = torch.cat(all_probs).squeeze()


y_predicted = all_probs.numpy()
# y_pred = (y_predicted >= 0.5).astype(int)
# len(y_pred)


len(y_predicted)


y_predicted


df = pd.DataFrame(
    {
    'id':idx,
    'diagnosed_diabetes':y_predicted
    }
)
df.to_csv('submission.csv',index=False)



len(idx)


len(test_df)




