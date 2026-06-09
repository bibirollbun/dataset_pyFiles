#%% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
import warnings as ww
ww.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#%% Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sub_id = test.copy()

# Remove 'id' column
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# Convert target to categorical
train['rainfall'] = train['rainfall'].astype('category')

# Fill missing values
test["winddirection"].fillna(test["winddirection"].mean(), inplace=True)


#%% Data Analysis
cols = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
       'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

# KDE Plots
fig, axes = plt.subplots(2, 5, figsize=(18, 9))  
axes = axes.flatten()
for i, col in enumerate(cols):
    sns.kdeplot(data=train, x=col, hue='rainfall', ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
plt.show()

# Histogram Plots
fig, axes = plt.subplots(2, 5, figsize=(18, 9))  
axes = axes.flatten()
for i, col in enumerate(cols):
    sns.histplot(data=train, x=col, hue='rainfall', ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
plt.show()

# Boxplots
z_train = pd.DataFrame(zscore(train[cols]), columns=cols)
z_train['rainfall'] = train['rainfall']

plt.figure(figsize=(10,6))
sns.boxplot(data=z_train)
plt.xticks(rotation=85)
plt.tight_layout()
plt.show()


#%% Train Baseline Model
X_train, X_test, y_train, y_test = train_test_split(z_train[cols], z_train['rainfall'], test_size=0.2, random_state=42)

model = LogisticRegressionCV()
model.fit(X_train, y_train)

# Feature Importances
pd.DataFrame(model.coef_, columns=model.feature_names_in_).T.plot(kind='bar')

# Predictions
y_prev = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_prev))


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.title("Baseline Model ROC AUC")
plt.plot(fpr, tpr, label=f'ROC AUC: {roc_auc:.4f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.text(x=0.73, y=0.01, s=f'ROC AUC: {roc_auc:.4F}')
plt.legend()
plt.show()


#%% Train Best Classifier
def train_best_classifier(X, y, test_size=0.2, random_state=42, n_iter=30, cv=5):
    """
    Trains multiple classification models, tunes hyperparameters, evaluates metrics, and plots the ROC curve.
    Returns the best model and its parameters.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    models = {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "LogisticRegression": LogisticRegression(solver='liblinear'),
        "SVC": SVC(probability=True, random_state=random_state),
        "XGBoost": XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    }
    
    params = {
        "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]},
        "GradientBoosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
        "LogisticRegression": {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]},
        "SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
        "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
    }
    
    best_model, best_score, best_params = None, 0, None
    plt.figure(figsize=(10, 8))
    
    for name in models:
        search = RandomizedSearchCV(models[name], params[name], n_iter=n_iter, cv=StratifiedKFold(), scoring='roc_auc', n_jobs=-1, random_state=random_state)
        search.fit(X_train, y_train)
        
        model = search.best_estimator_
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)
        
        if auc_score > best_score:
            best_score, best_model, best_params = auc_score, model, search.best_params_
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for All Models')
    plt.legend()
    plt.grid()
    plt.show()
    
    return best_model, best_params


#%% Train Best Model and Predict
cols = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
       'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
X_train, X_test, y_train, y_test = train_test_split(train[cols], train['rainfall'], test_size=0.2, random_state=42)
best_model, best_params = train_best_classifier(train[cols], train["rainfall"])
best_model.fit(X_train, y_train)


#%% Predict for Submission
prev = best_model.predict_proba(test)[:,1]
pd.DataFrame({'id': sub_id['id'], 'rainfall': prev}).to_csv('/kaggle/working/submission', index=False)

