# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from scipy.sparse import issparse
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
test.head()


# Display first few rows
print("Training Data Head:")
print(train.head())
print("\nTest Data Head:")
print(test.head())

# Basic info
print("\nTraining Data Info:")
print(train.info())
print("\nTest Data Info:")
print(test.info())

# Separate features and target
X = train.drop(['id', 'rainfall'], axis=1)
y = train['rainfall']
X_test = test.drop('id', axis=1)
test_ids = test['id']


# Check for missing values
print("\nMissing Values in Train:")
print(train.isnull().sum())
print("\nMissing Values in Test:")
print(test.isnull().sum())

# Target distribution
print("\nTarget Distribution (rainfall):")
print(y.value_counts(normalize=True))

# Identify feature types
cat_features = X.select_dtypes(include=['object']).columns
num_features = X.select_dtypes(exclude=['object']).columns
print("\nCategorical Features:", list(cat_features))
print("Numerical Features:", list(num_features))


X_test["winddirection"].fillna(X_test["winddirection"].mean(), inplace=True)


# Set plot style
plt.style.use('seaborn')

# Distribution of numerical features
for feature in num_features[:5]:  # Limit to first 5 for brevity
    plt.figure(figsize=(10, 4))
    sns.histplot(data=train, x=feature, hue='rainfall', multiple='stack', bins=30)
    plt.title(f'Distribution of {feature} by Rainfall')
    plt.show()

# Boxplots for numerical features
for feature in num_features[:5]:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x='rainfall', y=feature, data=train)
    plt.title(f'{feature} vs Rainfall')
    plt.show()

# Correlation matrix (numerical features only)
plt.figure(figsize=(12, 8))
sns.heatmap(X[num_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features)
    ])

# Preprocess data
X_preprocessed = preprocessor.fit_transform(X)
X_test_preprocessed = preprocessor.transform(X_test)

# Convert to dense for Keras (if sparse)
X_preprocessed_dense = X_preprocessed.toarray() if issparse(X_preprocessed) else X_preprocessed
X_test_preprocessed_dense = X_test_preprocessed.toarray() if issparse(X_test_preprocessed) else X_test_preprocessed


# LightGBM
def train_lgbm(X, y, X_test, kf):
    test_pred = np.zeros(len(X_test))
    val_scores = []
    
    # Define better hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create Dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train with early stopping
        model = lgb.train(params, 
                          train_data, 
                          valid_sets=[val_data])
        
        # Predict on validation set
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        val_scores.append(roc_auc_score(y_val, val_pred))
        
        # Predict on test set
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    
    print(f'LGBM Average Val AUC: {np.mean(val_scores):.4f}')
    return test_pred

# CatBoost
def train_catboost(X, y, X_test, kf):
    test_pred = np.zeros(len(X_test))
    val_scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = cb.CatBoostClassifier(random_seed=42, verbose=0)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
        val_pred = model.predict_proba(X_val)[:, 1]
        val_scores.append(roc_auc_score(y_val, val_pred))
        test_pred += model.predict_proba(X_test)[:, 1] / kf.n_splits
    print(f'CatBoost Average Val AUC: {np.mean(val_scores):.4f}')
    return test_pred

# XGBoost
def train_xgb(X, y, X_test, kf):
    test_pred = np.zeros(len(X_test))
    val_scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        val_pred = model.predict_proba(X_val)[:, 1]
        val_scores.append(roc_auc_score(y_val, val_pred))
        test_pred += model.predict_proba(X_test)[:, 1] / kf.n_splits
    print(f'XGBoost Average Val AUC: {np.mean(val_scores):.4f}')
    return test_pred

# Keras Neural Network
def create_keras_model(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=[keras.metrics.AUC(name='auc')])
    return model

def train_keras(X, y, X_test, kf):
    test_pred = np.zeros(len(X_test))
    val_scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = create_keras_model(X_train.shape[1])
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=10, restore_best_weights=True
        )
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, 
                  callbacks=[early_stopping], verbose=0)
        val_pred = model.predict(X_val, verbose=0).flatten()
        val_scores.append(roc_auc_score(y_val, val_pred))
        test_pred += model.predict(X_test, verbose=0).flatten() / kf.n_splits
    print(f'Keras Average Val AUC: {np.mean(val_scores):.4f}')
    return test_pred

## Pytorch

# Define the PyTorch model
class PyTorchModel(nn.Module):
    def __init__(self, input_shape):
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Training function for PyTorch
def train_pytorch(X, y, X_test, kf):
    test_pred = np.zeros(len(X_test))
    val_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model, loss, and optimizer
        model = PyTorchModel(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Early stopping
        best_val_auc = 0
        patience = 10
        epochs_no_improve = 0
        
        for epoch in range(100):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_pred = val_outputs.numpy()
                val_auc = roc_auc_score(y_val, val_pred)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        break
        
        # Load the best model
        model.load_state_dict(torch.load('best_model.pth'))
        
        # Predict on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_pred = val_outputs.numpy()
            val_scores.append(roc_auc_score(y_val, val_pred))
            
            # Predict on test set
            test_outputs = model(X_test_tensor)
            test_pred += test_outputs.numpy().flatten() / kf.n_splits
    
    print(f'PyTorch Average Val AUC: {np.mean(val_scores):.4f}')
    return test_pred


kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgbm_pred = train_lgbm(X_preprocessed, y, X_test_preprocessed, kf)
catboost_pred = train_catboost(X_preprocessed, y, X_test_preprocessed, kf)
xgb_pred = train_xgb(X_preprocessed, y, X_test_preprocessed, kf)
keras_pred = train_keras(X_preprocessed_dense, y, X_test_preprocessed_dense, kf)
pytorch_pred = train_pytorch(X_preprocessed_dense, y, X_test_preprocessed_dense, kf)


# Ensemble predictions (simple average)
final_pred = (lgbm_pred + catboost_pred + xgb_pred + keras_pred + pytorch_pred) / 5

# For local evaluation, we could split train data earlier, but here we rely on CV scores
print("Individual Model Validation AUCs printed above.")
print("Ensemble prediction generated; final evaluation via submission.")
# Create submission DataFrame
submission = pd.DataFrame({'id': test_ids, 'rainfall': final_pred})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created as 'submission.csv':")
print(submission.head())

