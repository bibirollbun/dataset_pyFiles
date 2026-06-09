import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils as nn_utils






df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df.head()


df[df.isna().any(axis=1)]


df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].mean(), inplace=True)
df['Social_event_attendance'].fillna(df['Social_event_attendance'].mean(), inplace=True)
df['Going_outside'].fillna(df['Going_outside'].mean(), inplace=True)
df['Friends_circle_size'].fillna(df['Friends_circle_size'].mean(), inplace=True)
df['Post_frequency'].fillna(df['Post_frequency'].mean(), inplace=True)
df['Stage_fear'].fillna(df['Stage_fear'].fillna('No'),inplace=True)
df['Drained_after_socializing'].fillna(df['Drained_after_socializing'].fillna('No'),inplace=True)


df[df.isna().any(axis=1)]
df.head()


df['Stage_fear'] = df['Stage_fear'].astype('category').cat.codes
df['Drained_after_socializing'] = df['Drained_after_socializing'].astype('category').cat.codes
df['Personality'] = df['Personality'].astype('category').cat.codes


df.head()


df.head()


# X = df.iloc[0:, :-1]
# y = df.iloc[:, -1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X = df.iloc[:, 1:-1]
Y = df.iloc[:, -1]

X_train = X.values
y_train = Y.values



X_train.shape


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


X_train


encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)



X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))



import torch
import torch.nn as nn


class MySimpleNN(nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.network = nn.Sequential(
        nn.Linear(num_features, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(), 
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid() ,
    )

  def forward(self, features):
    out = self.network(features)
    return out


learning_rate = 0.01
epochs = 2000


loss_function = nn.BCELoss()
type(loss_function)


X_train_tensor.shape


model = MySimpleNN(X_train_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

losses = [] 
t = trange(epochs, desc='Training Progress')

for epoch in t:
    model.train()
    y_pred = model(X_train_tensor)
    loss = loss_function(y_pred, y_train_tensor.view(-1, 1))
    optimizer.zero_grad()
    loss.backward()
    nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    losses.append(loss.item())
    current_lr = optimizer.param_groups[0]['lr']
    t.set_description(f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_test.head()


df_test['Time_spent_Alone'].fillna(df_test['Time_spent_Alone'].mean(), inplace=True)
df_test['Social_event_attendance'].fillna(df_test['Social_event_attendance'].mean(), inplace=True)
df_test['Going_outside'].fillna(df_test['Going_outside'].mean(), inplace=True)
df_test['Friends_circle_size'].fillna(df_test['Friends_circle_size'].mean(), inplace=True)
df_test['Post_frequency'].fillna(df_test['Post_frequency'].mean(), inplace=True)
df_test['Stage_fear'].fillna(df_test['Stage_fear'].fillna('No'),inplace=True)
df_test['Drained_after_socializing'].fillna(df_test['Drained_after_socializing'].fillna('No'),inplace=True)


df[df.isna().any(axis=1)]


df_test['Stage_fear'] = df_test['Stage_fear'].astype('category').cat.codes
df_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].astype('category').cat.codes



df_test.head()


X = df_test.iloc[:, 1:]
X_test = X.values


scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

X_test.shape


X_test_tensor = torch.from_numpy(X_test.astype(np.float32))


y_pred = model(X_test_tensor).squeeze()
labels = ['Introvert' if val > 0.5 else 'Extrovert' for val in y_pred]



import pandas as pd

y_pred = model(X_test_tensor).squeeze()
labels = ['Introvert' if val > 0.5 else 'Extrovert' for val in y_pred]

result_df = pd.DataFrame({
    'id': df_test.iloc[:, 0],
    'Personality': labels
})

result_df.to_csv('introvertVSextrovert2.csv', index=False)
# result_df.head()


