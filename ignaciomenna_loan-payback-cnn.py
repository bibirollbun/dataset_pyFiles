# =====================================
# 1. Setup
# =====================================
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print(train_df.shape)
print(test_df.shape)
print(submission_df.shape)


# =====================================
# 2. Basic Info & Data Quality
# =====================================
train_df.info()
train_df.describe(include='all')

# Duplicates
print()
print("Train duplicate rows:", train_df.duplicated().sum())
print("Test duplicate rows:", test_df.duplicated().sum())


# =====================================
# 3. Data Investigation
# =====================================
print('Clases Distribution is: ')
class_counts = train_df['loan_paid_back'].value_counts()

plt.figure(figsize=(5, 5))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%')
plt.title('Class Distribution')
plt.show()


# =====================================
# 4. Data Split
# =====================================
train_df, val_df = train_test_split(train_df, test_size=0.3, random_state=42, stratify=train_df['loan_paid_back'])

print("Training Shape: ", train_df.shape)
print("Validation Shape: ", val_df.shape)
print("Testing Shape: ", test_df.shape)


# =====================================
# 5. Data Preprocessing
# =====================================
# Using LabelEncoder for object columns
cat_cols = train_df.select_dtypes(include=['object']).columns
for col in cat_cols:
  label = LabelEncoder()
  train_df[col] = label.fit_transform(train_df[col].astype(str))
  val_df[col] = label.fit_transform(val_df[col].astype(str))
  test_df[col] = label.fit_transform(test_df[col].astype(str))

# Using StandarScaler for num columns
num_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
num_cols = num_cols.drop(['id', 'loan_paid_back'])

scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
val_df[num_cols] = scaler.fit_transform(val_df[num_cols])
test_df[num_cols] = scaler.fit_transform(test_df[num_cols])


# =====================================
# 6. Dataset Object
# =====================================
class LoanDataset(Dataset):
  def __init__(self, X, Y=None):
    self.X = torch.tensor(X.values, dtype=torch.float32)
    if Y is not None:
      self.Y = torch.tensor(Y.values, dtype=torch.float32)
    else:
      self.Y = None

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    if self.Y is not None:
      return self.X[index], self.Y[index]
    else:
      return self.X[index]


# Create Datasets Objects
X_train = train_df.drop(columns=['loan_paid_back', 'id'])
y_train = train_df['loan_paid_back']

X_val = val_df.drop(columns=['loan_paid_back', 'id'])
y_val = val_df['loan_paid_back']

X_test = test_df.drop(columns=['id'])

train_dataset = LoanDataset(X_train, y_train)
val_dataset = LoanDataset(X_val, y_val)
test_dataset = LoanDataset(X_test, Y=None)


# =====================================
# 7. Hyper Parameters
# =====================================
BATCH_SIZE = 32
EPOCHS = 2
LR = 1e-3


# =====================================
# 8. DataLoaders
# =====================================
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =====================================
# 9. Model
# =====================================
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.input_layer = nn.Linear(X_train.shape[1], 128)
    self.dropout1 = nn.Dropout(0.3)

    self.hidden_layer = nn.Linear(128, 64)
    self.relu = nn.ReLU()
    self.dropout2 = nn.Dropout(0.25 )

    self.output_layer = nn.Linear(64, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.input_layer(x)
    x = self.relu(x)
    x = self.dropout1(x)

    x = self.hidden_layer(x)
    x = self.relu(x)
    x = self.dropout2(x)

    x = self.output_layer(x)
    x = self.sigmoid(x)
    return x


# =====================================
# 9. Model Creation
# =====================================
model = Net().to(device)
summary(model, (X_train.shape[1],))


# =====================================
# 10. Loss and Optimizer
# =====================================
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LR)


# =====================================
# 11. Training and Validation
# =====================================
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

for epoch in range(EPOCHS):
  total_acc_train = 0
  total_loss_train = 0
  total_acc_val = 0
  total_loss_val = 0

  for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    output = model(inputs).squeeze(1)
    batch_loss = criterion(output, labels)
    total_loss_train += batch_loss.item()
    acc = ((output.round() == labels).sum().item())
    total_acc_train += acc

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

  with torch.no_grad():
    for inputs, labels in val_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)

      output = model(inputs).squeeze(1)
      batch_loss = criterion(output, labels)
      total_loss_val += batch_loss.item()
      acc = ((output.round() == labels).sum().item())
      total_acc_val += acc

  total_loss_train_plot.append(round(total_loss_train / len(train_loader), 4))
  total_loss_validation_plot.append(round(total_loss_val / len(val_loader), 4))
  total_acc_train_plot.append(round(total_acc_train / (train_dataset.__len__())*100, 4))
  total_acc_validation_plot.append(round(total_acc_val / (val_dataset.__len__())*100, 4))

  print(f'''Epoch {epoch + 1} Train Loss: {total_loss_train/len(train_loader):.4f} Train Accuracy: {(total_acc_train/(train_dataset.__len__())*100):.4f} Validation Loss: {total_loss_val/len(val_loader):.4f} Validation Accuracy: {(total_acc_val/(val_dataset.__len__())*100):.4f}''')
  print("="*70)



# =====================================
# 12. Testing
# =====================================
model.eval()
all_preds = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).squeeze(1)
        probs = outputs.cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds)

submission = pd.DataFrame({
    'id': test_df['id'].values,
    'loan_paid_back': all_preds
})

submission.to_csv('submission.csv', index=False)

