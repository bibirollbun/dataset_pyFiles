import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df.head()


(df.isnull().sum() / len(df)) * 100


df.head()


mean_values = {}

def handle_missing_values(df, train=False):
    df['Stage_fear'] = df['Stage_fear'].fillna('Maybe')
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('Maybe')

    cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    for col in cols:
        if train:
            mean_values[col] = df[col].mean()
        df[col] = df[col].fillna(mean_values[col])

handle_missing_values(df, True)
(df.isnull().sum() / len(df)) * 100



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

ohe = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
ohe_cols = ['Stage_fear', 'Drained_after_socializing']
ohe_df = ohe.fit_transform(df[ohe_cols])
df = df.join(ohe_df)
df = df.drop(ohe_cols, axis=1)

le = LabelEncoder()
le_target = LabelEncoder()
df['Personality'] = le_target.fit_transform(df['Personality'])

df.head()



from sklearn.model_selection import train_test_split

X = torch.tensor(df.drop(['id', 'Personality'], axis=1).values, dtype=torch.float32)
y = torch.tensor(df['Personality'].values, dtype=torch.long)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1153, stratify=y)
X_train = X
X_val = X
y_train = y
y_val = y


import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, num_features=1, num_outputs=2):
        super().__init__()

        hidden_layer_size = 16
        
        self.hidden = nn.Sequential(
            nn.Linear(num_features, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
        )
        self.classifiers = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(hidden_layer_size, num_outputs),
        )
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.classifiers(x)
        return x



model = SimpleNN(X.shape[1], 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# setup early stopping
patience = 10
patience_count = 0
best_loss = np.Inf

epochs = 2000
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        val_loss = criterion(outputs, y_val)
        y_pred = torch.argmax(outputs.data, 1)
        correct = (y_pred == y_val).sum().item()
        total = y_val.size(0)
    
    scheduler.step(val_loss)
    if (epoch+1)%10 == 0:
        print(f'[{epoch+1}/{epochs}] Loss: {val_loss.item():.8f} Acc.: {correct/total:.4f}')

    if val_loss.item() < best_loss:
        patience_count = 0
        best_loss = val_loss.item()
        best_epoch = epoch+1
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_count += 1

    if patience_count >= patience:
        print(f'[{epoch+1}/{epochs}] Loss: {val_loss.item():.8f} Acc.: {correct/total:.4f}')
        print("Early stopping")
        break

print(f'Loading model from epoch {best_epoch}')
model.load_state_dict(torch.load('best_model.pth'))



df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

handle_missing_values(df_test)

ohe_df = ohe.transform(df_test[ohe_cols])
df_test = df_test.join(ohe_df)
df_test = df_test.drop(ohe_cols, axis=1)

X_test = torch.tensor(df_test.drop('id', axis=1).values, dtype=torch.float32)



model.eval()
outputs = model(X_test)
y_pred = torch.argmax(outputs.data, 1)

df_sub = pd.DataFrame()
df_sub['id'] = df_test['id']
df_sub['Personality'] = y_pred
df_sub['Personality'] = le_target.inverse_transform(df_sub['Personality'])
df_sub.to_csv('/kaggle/working/submission.csv', index=False)


