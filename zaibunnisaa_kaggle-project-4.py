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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


train = pd.read_csv('/kaggle/input/playground-series-s3e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s3e12/sample_submission.csv')
train.head()


test.head()


# Check Dimensions and Basic Info
print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
train.info()


# Target Distribution
sns.countplot(x='target', data=train)
plt.title("Target Variable Distribution")
plt.show()


# Feature-wise Boxplots for Target Classes
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
sns.boxplot(ax=axes[0, 0], x='target', y='gravity', data=train)
sns.boxplot(ax=axes[0, 1], x='target', y='ph', data=train)
sns.boxplot(ax=axes[0, 2], x='target', y='osmo', data=train)
sns.boxplot(ax=axes[1, 0], x='target', y='cond', data=train)
sns.boxplot(ax=axes[1, 1], x='target', y='urea', data=train)
sns.boxplot(ax=axes[1, 2], x='target', y='calc', data=train)
plt.suptitle("Feature Distributions by Target")
plt.show()


# Feature Distributions
train.drop(columns=['id', 'target']).hist(bins=20, figsize=(14, 10), layout=(3, 3))
plt.suptitle("Feature Distributions in Training Data")
plt.show()


# Correlation Analysis
corr = train.drop(columns=['id', 'target']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# Model Evaluation Function
def evaluate_model(model, X, y, test_data):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    preds = []
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(test_data)[:, 1]
        
        score = roc_auc_score(y_val, y_pred)
        scores.append(score)
        preds.append(test_pred)
    
    return np.mean(scores), np.mean(preds, axis=0)


# Model Initialization 
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Hist Gradient Boosting': HistGradientBoostingClassifier(max_iter=100),
    'LightGBM': LGBMClassifier(n_estimators=100, verbosity=-1),  # Suppress warnings
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(iterations=100, verbose=False)
}


# Prepare Data for Modeling
X = train.drop(columns=['id', 'target'])
y = train['target']
test_data = test.drop(columns=['id'])


# Model Evaluation
results = []
for name, model in models.items():
    score, test_pred = evaluate_model(model, X, y, test_data)
    results.append((name, score, test_pred))
    print(f'{name} ROC-AUC: {score:.4f}')


# Display Model Performance
results_df = pd.DataFrame(results, columns=['Model', 'ROC-AUC Score', 'Predictions'])
results_df.sort_values(by='ROC-AUC Score', ascending=False, inplace=True)
print(results_df[['Model', 'ROC-AUC Score']])


!pip install pytorch-tabnet



import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score



# Load the dataset
train_df = pd.read_csv('/kaggle/input/playground-series-s3e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s3e12/test.csv')

# Separate features and target
X = train_df.drop(columns=['id', 'target'])  # Drop ID column
y = train_df['target']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df.drop(columns=['id']))  # Apply same transformation

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)



# Convert data to PyTorch tensors
X_train_tensor, X_val_tensor = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)
y_train_tensor, y_val_tensor = torch.tensor(y_train.values, dtype=torch.long), torch.tensor(y_val.values, dtype=torch.long)

# Initialize TabNet Classifier
tabnet_model = TabNetClassifier()

# Train the model
tabnet_model.fit(
    X_train_tensor.numpy(), y_train_tensor.numpy(),
    eval_set=[(X_val_tensor.numpy(), y_val_tensor.numpy())],
    eval_metric=['auc'],
    max_epochs=50,
    patience=10,
    batch_size=256
)


