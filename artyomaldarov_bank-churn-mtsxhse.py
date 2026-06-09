!pip install -q catboost ydata_profiling


import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier


df = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv')
df.head()


df = df.drop(columns=['id', 'CustomerId', 'Surname'])


from ydata_profiling import ProfileReport

ProfileReport(df)


df.EstimatedSalary = (df.EstimatedSalary - df.EstimatedSalary.mean()) / df.EstimatedSalary.std()


X = df.drop(columns=['Exited'])
y = df.Exited.values


model = CatBoostClassifier(
    cat_features=[1, 2],
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

grid = {
    'iterations': [200, 300, 500],
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'depth': [4, 5, 6, 8],
    'l2_leaf_reg': [1, 3, 5]
}

grid_search_result = model.grid_search(
    grid, X, y, plot=False
)

best_params = grid_search_result['params']
print("Best parameters:", best_params)


best_params


model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=5,
    l2_leaf_reg=3,
    cat_features=[1, 2],
    loss_function='CrossEntropy',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

model.fit(X, y)


df_pred = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')
df_pred.EstimatedSalary = (df_pred.EstimatedSalary - df_pred.EstimatedSalary.mean()) / df_pred.EstimatedSalary.std()

X_test = df_pred.drop(columns=['id', 'CustomerId', 'Surname'])


preds = model.predict(X_test) # ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ¼ĞµÑ‚ĞºĞ¸

preds_proba = model.predict_proba(X_test) # Ğ’ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ´Ğ»Ñ� ĞºĞ°Ğ¶Ğ´Ğ¾Ğ¹ Ğ¸Ğ· Ğ¼ĞµÑ‚Ğ¾Ğº


preds_proba, preds


df1 = pd.DataFrame()
df1['id'] = [15000+i for i in range(10000)]
df1['Exited'] = [i[1] for i in preds_proba]
df1.to_csv('submission.csv', index=False)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
# from imblearn.over_sampling import SMOTE # Ğ£ Kaggle ĞºĞ°ĞºĞ¸Ğµ-Ñ‚Ğ¾ Ğ¿Ñ€Ğ¾Ğ±Ğ»ĞµĞ¼Ñ‹ Ñ� imblearn, Ğ¿Ğ¾Ñ�Ñ‚Ğ¾Ğ¼Ñƒ Ğ»ÑƒÑ‡ÑˆĞµ Ğ¾Ñ‚ĞºÑ€Ñ‹Ğ²Ğ°Ñ‚ÑŒ Ğ½Ğ¾ÑƒÑ‚Ğ±ÑƒĞº Ğ² Colab
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


data = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv')

data = data.drop(['id', 'CustomerId', 'Surname'], axis=1)


categorical_cols = ['Geography', 'Gender']
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                  'IsActiveMember', 'EstimatedSalary']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])


X = data.drop('Exited', axis=1)
y = data['Exited'].values

X_processed = preprocessor.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# smote = SMOTE(random_state=42)                                  # Ñ€Ğ°Ğ±Ğ¾Ñ‚Ğ°ĞµÑ‚ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ğ² Colab
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

X_train_res, y_train_res = X_train, y_train

X_train_tensor = torch.tensor(X_train_res.toarray() if hasattr(X_train_res, 'toarray') else X_train_res, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_res, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val.toarray() if hasattr(X_val, 'toarray') else X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


class ChurnDNN(nn.Module):
    def __init__(self, input_dim):
        super(ChurnDNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.bnorm = nn.BatchNorm1d(13)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bnorm(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        return x

input_dim = X_train_tensor.shape[1]
model = ChurnDNN(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 200
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

        val_auc = roc_auc_score(y_val_tensor.numpy(), val_outputs.numpy())

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val AUC: {val_auc:.4f}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            model.load_state_dict(best_model_state)
            break

model.eval()
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    val_auc = roc_auc_score(y_val_tensor.numpy(), val_outputs.numpy())
    print(f'Final Val AUC: {val_auc:.4f}')


batch_data = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv').drop(columns=['id', 'CustomerId', 'Surname'])
processed_batch = preprocessor.fit_transform(batch_data)

batch_tensor = torch.tensor(processed_batch.toarray() if hasattr(processed_batch, 'toarray') else processed_batch, dtype=torch.float32)
with torch.no_grad():
    batch_probs = model(batch_tensor).numpy()


batch = pd.DataFrame()
batch['id'] = [15000+i for i in range(10000)]
batch['Exited'] = batch_probs


batch.to_csv('submission_nn.csv', index=False)

