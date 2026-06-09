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


#Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_df.head()


train_df.info()


test_df.info()


# Using the median to fill the missing value is not entirely appropriate in this case. 
# Since there is only one block with a missing value, we will use it to avoid future errors.
warnings.simplefilter(action='ignore', category=FutureWarning)    
test_df['winddirection'].fillna(train_df['winddirection'].median(), inplace=True)


#drop non-feature columns 
X = train_df.drop(["id", "day", "rainfall"], axis=1)
y = train_df["rainfall"]


X.describe()


ax = train_df.rainfall.value_counts().plot.bar(figsize=(6, 4))
ax.set_xlabel("Rainfall")
ax.set_ylabel("Count")


# The dataset is completely unbalaced 
# let's see the non-rainy to rainy samples ratio
# scale_pos_weight = neg_class / pos_class
counts = train_df.rainfall.value_counts()
scale_pos_weight = counts[0] / counts[1]
print("Negative to positive sample ratio = {:.2f}".format(scale_pos_weight))


def plot_feature_distribution(train, test, features, bins=50):
    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.histplot(train[feature], bins=bins, kde=True, stat="percent", color='blue', label='Train', alpha=0.5)
        sns.histplot(test[feature], bins=bins, kde=True, stat="percent", color='red', label='Test', alpha=0.5)
        plt.title(f'Distribution of {feature}')
        plt.legend()
        plt.show()

# Call the function
plot_feature_distribution(X, test_df.drop(['id', 'day'], axis=1), X.columns)


# Seems like the distributions are identical 
# To be sure, We use Kolmogorov-Smirnov (KS) Test to check if distributions differ significantly.

def ks_test(train, test, features):
    results = {}
    for feature in features:
        stat, p_value = ks_2samp(train[feature], test[feature])
        results[feature] = {'KS Statistic': stat, 'p-value': p_value}
    return results

# If the p-value is small (e.g., < 0.05), 
# it suggests the distributions are significantly different.
ks_results = ks_test(X, test_df.drop(['id', 'day'], axis=1), X.columns)
for key in ks_results:
    print(key, ks_results[key])


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
auc_scores = []

for train_index, val_index in kf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Train LightGBM model
    model = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=100)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10)])
    
    # Predict probabilities
    preds = model.predict_proba(X_val)[:, 1]
    
    # Evaluate using AUC
    auc = roc_auc_score(y_val, preds)
    auc_scores.append(auc)

print(f"Mean AUC: {np.mean(auc_scores):.4f}")


model = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=100, scale_pos_weight=scale_pos_weight)
model.fit(X, y)
# lgb_pred = model.predict(test_df.drop(["id", "day"], axis=1))
# output = pd.DataFrame({'id': test_df.id, 
#                        'rainfall': lgb_pred})
# output.to_csv('submission.csv', index=False)
# Private score: 0.79, Public score: 0.756


# plotting the feature importance 
feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()  # Compute correlation matrix

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)
    plt.title("Feature Correlation Matrix")
    plt.show()

# Call the function for the train dataset
plot_correlation_matrix(X_train)


def creating_new_features(dataframe):
    # Keep the dewpoint and drop all temparature-related features
    new_dataframe = dataframe.drop(['maxtemp', 'mintemp', 'temparature'], axis=1)

    # Create new feature
    # 1) Temp Range to capture dairy variation
    new_dataframe['temp_range'] = dataframe['maxtemp'] - dataframe['mintemp']

    # 2) Dew Point Depression which higher values indicate drier air
    new_dataframe['dpoint_dep'] = dataframe['temparature'] - dataframe['dewpoint']

    # 3) Relative Humidity Approximation which is alternative to humidity
    new_dataframe['hum_app'] = dataframe['dewpoint'] / dataframe['temparature']

    # 4) Cloud-Sun Ratio to captures cloud dominance over sunshine
    new_dataframe['cloud-sun_ratio'] = dataframe['cloud'] / (dataframe['sunshine'] + 1e-5)

    # 5) Wind Direction Encoding
    new_dataframe['wind_x'] = np.cos(np.radians(dataframe['winddirection']))
    new_dataframe['wind_y'] = np.sin(np.radians(dataframe['winddirection']))

    return new_dataframe

X_new = creating_new_features(X)


plot_correlation_matrix(X_new)


test_new = creating_new_features(test_df)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
auc_scores = []

for train_index, val_index in kf.split(X_new, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Train LightGBM model
    model = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=100)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10)])
    
    # Predict probabilities
    preds = model.predict_proba(X_val)[:, 1]
    
    # Evaluate using AUC
    auc = roc_auc_score(y_val, preds)
    auc_scores.append(auc)

print(f"Mean AUC: {np.mean(auc_scores):.4f}")


model = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=100, scale_pos_weight=scale_pos_weight)
model.fit(X_new, y)
# lgb_pred = model.predict(test_new.drop(["id", "day"], axis=1))

# output = pd.DataFrame({'id': test_new.id, 
#                        'rainfall': lgb_pred})
# output.to_csv('submission_new.csv', index=False)
# Private score: 0.807, Public score: 0.75
# about 2 percent improvement over the basic model on private score


feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X_new.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-02.png')


# Define the model with GPU support
lgbm = LGBMClassifier(
    random_state=42,
    device='gpu',              # Use GPU
    gpu_use_dp=True,            
    scale_pos_weight=scale_pos_weight
)

# Define the hyperparameter grid
param_grid = {
    'num_leaves': [15, 31, 63],
    'max_depth': [-1, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300]
}

# Set up the grid search
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='roc_auc',         
    cv=5,                      # 5-fold cross-validation
    verbose=1,
    n_jobs=-1
)

# Fit to training data
grid_search.fit(X_new, y)

# Get best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best AUC Score:", grid_search.best_score_)


lgbm = LGBMClassifier(
    random_state=42,
    device='gpu',              # Use GPU
    gpu_use_dp=True,            
    scale_pos_weight=scale_pos_weight,
    learning_rate= 0.01,
    max_depth= 5,
    n_estimators= 300,
    num_leaves= 15
)
lgbm.fit(X_new, y)
# lgb_pred = lgbm.predict(test_new.drop(["id", "day"], axis=1))

# output = pd.DataFrame({'id': test_new.id, 
#                        'rainfall': lgb_pred})
# output.to_csv('submission_lgbm.csv', index=False)
# Private score: 0.824, Public score: 0.768


# Neural net class
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class FNN(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=50, lr=0.001, batch_size=32, device="cpu"):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = FeedforwardNN(input_dim).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        if isinstance(y, pd.Series):
            y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(device)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            
    def fit_eval(self, X, y, X_val, y_val):

        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

        
        # Validation
        self.model.eval()
        with torch.no_grad():
            val_preds = self.model(X_val).cpu().numpy()
        auc = roc_auc_score(y_val, val_preds)
        return auc

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            if(device=="cpu"):
                probs = self.model(X_tensor).numpy()
            else: 
                probs = self.model(X_tensor).cpu().numpy()
        return (probs > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            if(device=="cpu"):
                probs = self.model(X_tensor).numpy()
            else: 
                probs = self.model(X_tensor).cpu().numpy()
        return np.hstack([1 - probs, probs])  # shape: (n_samples, 2)


# Standardize for SVM & Neural Network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_new)
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for train_idx, val_idx in skf.split(X_scaled, y):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_tensor = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    model = FNN(X_scaled.shape[1], device=device)

    # Training
    auc = model.fit_eval(X_tr_tensor, y_tr_tensor, X_val_tensor, y_val_tensor)

    auc_scores.append(auc)

print("FNN Cross-Validation AUC Scores:", auc_scores)
print("Mean AUC:", np.mean(auc_scores))


fnn_model = FNN(X_scaled.shape[1], device=device)
fnn_model.fit(X_scaled, y)

# test_scaled = scaler.transform(test_new.drop(["id", "day"], axis=1))
# fnn_pred = fnn_model.predict(test_scaled)
# output = pd.DataFrame({'id': test_new.id, 
#                        'rainfall': fnn_pred})
# output.to_csv('submission_fnn.csv', index=False)
# Private score: 0.755, Public score: 0.746


# Define the SVM model
svm = SVC(probability=True)  # enable probability for things like ROC/AUC

# Define the parameter grid to search
param_grid = {
    'C': [0.1, 1, 10],                  # Regularization strength
    'kernel': ['linear', 'rbf'],        # Kernel type
    'gamma': ['scale', 'auto', 0.1, 1], # Only for 'rbf' kernel
}

# Create GridSearchCV object
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Fit the model on training data
grid_search.fit(X_scaled, y.values)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best AUC Score:", grid_search.best_score_)


svm_model = SVC(probability=True, C= 0.1, gamma= 'scale', kernel= 'linear')
svm_model.fit(X_scaled, y.values)

# test_scaled = scaler.transform(test_new.drop(["id", "day"], axis=1))
# svm_pred = svm_model.predict(test_scaled)
# output = pd.DataFrame({'id': test_new.id, 
#                        'rainfall': svm_pred})
# output.to_csv('submission_svm.csv', index=False)
# Private score: 0.773, Public score: 0.755


from scipy.stats import mode as voter

def voting_classifier(X_test, lgb, fnn, svm, mode="soft"):

    test_scaled = scaler.transform(X_test.drop(['id', 'day'], axis=1)) 
    
    if mode=="soft":
        # Make predictions (probabilities) on test set
        lgb_proba = lgb.predict_proba(X_test.drop(['id', 'day'], axis=1))[:, 1]
        fnn_proba = fnn.predict_proba(test_scaled)[:, 1]
        svm_proba = svm.predict_proba(test_scaled)[:, 1]
        
        # Combine using soft voting
        ensemble_proba = (lgb_proba + fnn_proba + svm_proba) / 3
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
    elif mode=="hard":
        # Get individual predictions
        lgb_pred = lgb.predict(X_test.drop(['id', 'day'], axis=1))
        fnn_pred = fnn.predict(test_scaled)
        svm_pred = svm.predict(test_scaled)

        # Combine using hard voting
        predictions = np.vstack([lgb_pred + fnn_pred + svm_pred])
        ensemble_pred, _ = voter(predictions, axis=0)
        
    else:
        print("The mode is not defined")

    
    output = pd.DataFrame({'id': X_test.id, 
                           'rainfall': ensemble_pred})
    output.to_csv('submission.csv', index=False)
    print("Done")


voting_classifier(test_new, lgbm, fnn_model, svm_model, mode="hard")
# Hard Mode: Private score: 0.84, Public score: 0.774
# Soft Mode: Private score: 0.773, Public score: 0.757 **** D ****

