# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Header di set None, karena pada dataset tidak menyediakan nama kolom
train = pd.read_csv('/kaggle/input/just-the-basics-strata-2013/train.csv', header=None)
train_y = pd.read_csv('/kaggle/input/just-the-basics-strata-2013/train_labels.csv', header=None)
test  = pd.read_csv('/kaggle/input/just-the-basics-strata-2013/test.csv', header=None)

# Normalisasi
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

column_names = train.columns if isinstance(train, pd.DataFrame) else None
train = pd.DataFrame(train, columns=column_names)
test = pd.DataFrame(test, columns=column_names)

# Set nama kolom target train
train_y.columns = ['target']


print("Train shape :",train.shape)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    display(train.head())


print("Train shape :",test.shape)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    display(test.head())


import matplotlib.pyplot as plt
import seaborn as sns

# Hitung distribusi kelas
distribution = train_y['target'].value_counts()

# Tampilkan distribusi numeriknya
print(distribution)

# Visualisasi dengan seaborn
sns.countplot(x='target', data=train_y)
plt.title('Target Distribution')
plt.xlabel('Target Class')
plt.ylabel('Count')
plt.show()


# Check : Nama kolom train dan test harusnya sama
print(train.columns)
print(test.columns)


# Contoh data target train
train_y.head()


# Isi data kosong pada train
train= train.fillna(0.0)
test = test.fillna(0.0)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, train_y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)


import torch

# Ubah jadi Tensor
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long).squeeze()  # untuk klasifikasi, label harus long
X_test  = torch.tensor(X_test.values, dtype=torch.float32)
y_test  = torch.tensor(y_test.values, dtype=torch.long).squeeze()


y_train = y_train.view(-1, 1).float()
y_test = y_test.view(-1, 1).float()


type(test)


test = torch.tensor(test.values, dtype=torch.float32)


print("Train : ")
print("X : ", X_train.shape)
print("y : ", y_train.shape)
print("Test : ")
print("X : ", X_test.shape)
print("y : ", y_test.shape)


import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

model = nn.Sequential(
    nn.Linear(100, 300),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(300, 1)
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(500):
    pred = model(X_train)  # Output berupa logit
    loss = loss_fn(pred, y_train.float())  # y_train harus float

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(pred)
            predicted = (prob > 0.5).float()
            acc = (predicted == y_train).float().mean()
            auc = roc_auc_score(y_train.cpu(), prob.cpu())
        
        print(f"Epoch {epoch}: Loss = {loss.item():.4f} | Accuracy = {acc.item():.4f} | AUC = {auc:.4f}")
        model.train()


model.eval()
with torch.no_grad():
    pred_test = model(X_test)             # Logit output
    prob_test = torch.sigmoid(pred_test)  # Ubah ke probabilitas
    preds = (prob_test > 0.5).float()     # Thresholding jika ingin akurasi

    # Pastikan y_test juga float/tensor
    y_test_float = y_test.float()

    # Akurasi
    acc = (preds == y_test_float).float().mean()

    # AUC
    auc = roc_auc_score(y_test_float.cpu(), prob_test.cpu())

    # Confusion Matrix
    cm = confusion_matrix(y_test_float.cpu(), preds.cpu())

# Print hasil
print("Contoh hasil probabilitas (5 sample):\n", prob_test[:5])
print("Contoh hasil klasifikasi biner (5 sample):\n", preds[:5])
print(f"Akurasi Test Set : {acc.item()*100:.2f}%")
print(f"ROC AUC Score   : {auc:.4f}")
print("Confusion Matrix:\n", cm)



test


# model.eval()
# with torch.no_grad():
#     logits = model(test_X)
#     probs = torch.sigmoid(logits)
#     preds = (probs > 0.5).float()
#     acc = (preds == test_y).float().mean()

# print(f"Akurasi Test: {acc.item()*100:.2f}%")



model.eval()
with torch.no_grad():
    probas = torch.sigmoid(model(test))  # hasilnya float antara 0–1

print("Contoh hasil prediksi probabilitas:")
print(probas[:5])


import pandas as pd

submit = pd.DataFrame({
    "prediction": probas.squeeze().cpu().numpy()  # pastikan bentuk 1D
})
submit.to_csv("submission.csv", index=False)











