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


data = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
data.head()


data.drop('id', axis=1, inplace=True)


data.isnull().sum()


data.groupby(['loan_status']).size()


cat_features = data.select_dtypes(include=['object'])
print(cat_features.nunique())
print(cat_features.apply(lambda x: x.unique()))


print(data['loan_intent'].unique())


import seaborn as sns
import matplotlib.pyplot as plt

num_features = data.select_dtypes(include=['float64', 'int64'])
corr_matrix = num_features.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation matrix')
plt.show()


data.drop(['cb_person_cred_hist_length'], axis=1, inplace=True)


num_features.drop(['cb_person_cred_hist_length'], axis=1, inplace=True)


for col in cat_features.columns:
    plt.figure(figsize=(8, 6))
    cat_features[col].value_counts().plot(kind='bar')
    plt.title(f'Распределение {col}')
    plt.show()


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(cat_features)
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
encoded_df


num_features.isna().sum()


data2 = pd.concat([num_features, encoded_df], axis=1)
data2.head()


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
y = data2['loan_status']
data2.drop('loan_status', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data2, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



X_train.shape


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report

train_pool = Pool(data=X_train, label=y_train)
test_pool = Pool(data=X_test, label=y_test)


param_grid = {
    'iterations': [100, 300, 500, 800],
    'learning_rate': [0.01, 0.1, 0.2],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5]
}

baseline = CatBoostClassifier(random_state=42, verbose=0)
random_search = RandomizedSearchCV(
    estimator=baseline,
    param_distributions=param_grid,
    cv=5,
    n_iter=10,
    n_jobs=-1,
    verbose=2
)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
predictions = best_model.predict(X_test)
f1 = f1_score(y_test, predictions)
print(f'Test F1-score: {f1:.4f}')
print("Best params:", random_search.best_params_)
baseline = best_model



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.d1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.d2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.d1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.d2(x)
        
        x = self.fc3(x)
        return x


from torch.utils.data import TensorDataset, DataLoader

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

batch_size = 128

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# from torch.utils.data import WeightedRandomSampler

# class_counts = np.bincount(y_train_t.numpy().astype(int))  
# weights = 1.0 / class_counts
# samples_weights = weights[y_train_t.numpy().astype(int)]

# sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


y_train_t


def train(model, optimizer, scheduler, criterion, num_epochs=10, device=torch.device('cuda')):
    print(device)
    training_loss = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        training_loss.append(epoch_loss)
        
        if epoch % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    return training_loss


from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
def evaluate_model(model, test_loader, device="cuda"):
    model.eval()
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    
            outputs = model(X_batch).squeeze(1)
            y_scores.extend(torch.sigmoid(outputs).cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    threshold = 0.4
    y_pred = (np.array(y_scores) > threshold)
    print(f"Classification Report at Threshold {threshold:.4f}:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = X_train_t.shape[1]
model = Model(input_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
num_epochs = 20
# pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)])
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
criterion = nn.BCEWithLogitsLoss()
#criterion = FocalLoss(alpha=0.7, gamma=1.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs*len(train_loader), gamma=0.1)

loss_history = train(model, optimizer, scheduler, criterion, num_epochs, device)
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



evaluate_model(model, test_loader)


test_data = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')

id_column = test_data['id']
test_data = test_data.drop(columns=['id', 'cb_person_cred_hist_length'], axis=1)
cat_features_test = test_data.select_dtypes(include=['object'])
encoded_test = encoder.transform(cat_features_test)
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out())
test_num_features = test_data.select_dtypes(include=['float64', 'int64'])
test_data_processed = pd.concat([test_num_features, encoded_test_df], axis=1)
test_data_processed = scaler.transform(test_data_processed)

test_data_t = torch.tensor(test_data_processed, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    predictions = model(test_data_t).squeeze(1)

predicted_probabilities = torch.sigmoid(predictions).cpu().numpy()

predicted_labels = (predicted_probabilities > 0.5).astype(int)
submission = pd.DataFrame({
    'id': id_column,
    'loan_status': predicted_labels.flatten()
})
submission.to_csv('submission.csv', index=False)




