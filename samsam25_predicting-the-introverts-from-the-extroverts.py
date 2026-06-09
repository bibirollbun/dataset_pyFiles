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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import catboost as cb 
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import optuna
import warnings
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy.stats import ttest_ind, ks_2samp
import shap

warnings.filterwarnings('ignore')


np.random.seed(42)
torch.manual_seed(42)


try:
    train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col='id')
    test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col='id')
    original = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")
    submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
    
    # Validate data loading
    assert not train.empty and not test.empty, "Train or test data is empty"
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    raise


feature_names = [col for col in train.columns if col != 'Personality']
print("Feature names:", feature_names)


print("\n=== Dataset Overview ===")
print(f"Train: {train.shape} | Test: {test.shape} | Original: {original.shape}")


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
target_dist = train['Personality'].value_counts(normalize=True)
ax = sns.barplot(x=target_dist.index, y=target_dist.values, palette='viridis')
plt.title('Target Class Distribution', fontsize=14)
plt.xlabel('Personality')
plt.ylabel('Proportion')


for i, v in enumerate(target_dist):
    ax.text(i, v+0.01, f"{v:.2%}", ha='center', fontsize=12, fontweight='bold')
    
plt.subplot(1, 2, 2)
sns.countplot(data=pd.concat([train.assign(source='Train'), 
                            original.assign(source='Original')]), 
             x='Personality', hue='source', palette='mako')
plt.title('Train vs Original Data Distribution', fontsize=14)
plt.tight_layout()
plt.show()


missing_data = pd.concat([
    train.isnull().sum().rename('Train'),
    test.isnull().sum().rename('Test'),
    original.isnull().sum().rename('Original')
], axis=1)

plt.figure(figsize=(16, 8))
missing_data.plot(kind='bar', width=0.8, color=['#3498db', '#e74c3c', '#2ecc71'])
plt.title('Missing Values Comparison Across Datasets', fontsize=16)
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Dataset')
plt.tight_layout()
plt.show()


num_features = [col for col in train.columns if train[col].dtype in ['int64', 'float64'] and col != 'Personality']

plt.figure(figsize=(20, 15))
for i, feature in enumerate(num_features, 1):
    plt.subplot(4, 4, i)
    
    # KDE plots for both classes
    sns.kdeplot(data=train, x=feature, hue='Personality', 
                palette=['#3498db', '#e74c3c'], fill=True, common_norm=False)
    
    # Add statistical test results
    stat, p = ks_2samp(
        train[train['Personality'] == 0][feature].dropna(),
        train[train['Personality'] == 1][feature].dropna()
    )
    plt.title(f"{feature}\n(KS p-value: {p:.3e})", fontsize=10)
    
plt.tight_layout()
plt.suptitle('Numerical Features Distribution by Personality Type', y=1.02, fontsize=16)
plt.show()


cat_features = ['Stage_fear', 'Drained_after_socializing']  # Update based on actual categorical features

if cat_features:
    plt.figure(figsize=(16, 6))
    for i, feature in enumerate(cat_features, 1):
        plt.subplot(1, len(cat_features), i)
        
        # Stacked percentage bar plot
        ct = pd.crosstab(train[feature], train['Personality'], normalize='index') * 100
        ct.plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'], ax=plt.gca())
        
        plt.title(f'{feature} Distribution by Personality', fontsize=12)
        plt.ylabel('Percentage')
        plt.legend(title='Personality', bbox_to_anchor=(1, 1))
        
    plt.tight_layout()
    plt.show()



plt.figure(figsize=(18, 12))
corr_matrix = train.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
            annot_kws={"size": 10}, fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


X_temp = train.drop('Personality', axis=1).apply(lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x)
mi_scores = mutual_info_classif(X_temp, train['Personality'], random_state=42)
mi_scores = pd.Series(mi_scores, index=X_temp.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=mi_scores.values, y=mi_scores.index, palette='viridis')
plt.title('Mutual Information Scores with Target', fontsize=16)
plt.xlabel('MI Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


def clean_data(df):
    # Handle categorical variables consistently
    for col in ['Stage_fear', 'Drained_after_socializing']:
        if col in df.columns:
            df[col] = df[col].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mode()[0])
    
    # Handle target variable if present
    if 'Personality' in df.columns:
        df['Personality'] = df['Personality'].replace({'Extrovert': 0, 'Introvert': 1, 'extrovert': 0, 'introvert': 1})
        df['Personality'] = pd.to_numeric(df['Personality'], errors='coerce').fillna(df['Personality'].mode()[0])
    
    return df

train = clean_data(train)
test = clean_data(test)
original = clean_data(original)


def create_advanced_features(df):
    """Generate powerful predictive features"""
    # Social interaction features
    df['Social_Energy_Ratio'] = (df['Social_event_attendance'] + 1) / (df['Drained_after_socializing'] + 1)
    df['Social_Recovery_Index'] = df['Time_spent_Alone'] / (df['Drained_after_socializing'] + 1)
    
    # Personality interaction features
    df['Alone_Social_Interaction'] = df['Time_spent_Alone'] * df['Social_event_attendance']
    df['Social_Media_Engagement'] = np.log1p(df['Post_frequency']) * np.log1p(df['Friends_circle_size'])
    
    # Behavioral features
    df['Isolation_Index'] = df['Time_spent_Alone'] * (1 - df['Going_outside'])
    df['Social_Confidence'] = (1 - df['Stage_fear']) * df['Social_event_attendance']
    
    # Polynomial features
    for col in ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_log'] = np.log1p(df[col])
    
    # Interaction terms
    df['Drain_Friends_Interaction'] = df['Drained_after_socializing'] * df['Friends_circle_size']
    df['Stage_Fear_Impact'] = df['Stage_fear'] * df['Social_event_attendance']
    
    return df

train = create_advanced_features(train)
test = create_advanced_features(test)
original = create_advanced_features(original)

# Update feature names
feature_names = [col for col in train.columns if col != 'Personality']
print("\nUpdated feature names after engineering:", feature_names)



print("\n=== Missing Value Handling ===")
print("Before imputation - Train:", train.isnull().sum().sum(), "| Test:", test.isnull().sum().sum())

# Multiple imputation strategies
imputer1 = IterativeImputer(max_iter=50, random_state=42, add_indicator=True)
imputer2 = SimpleImputer(strategy='median', add_indicator=True)

# Create imputed versions
X_train = train.drop('Personality', axis=1)
y_train = train['Personality'].values

X_train_imp1 = pd.DataFrame(imputer1.fit_transform(X_train), columns=X_train.columns.tolist() + 
                                                               [f'missing_{col}' for col in X_train.columns])
X_test_imp1 = pd.DataFrame(imputer1.transform(test), columns=test.columns.tolist() + 
                                                     [f'missing_{col}' for col in test.columns])

X_train_imp2 = pd.DataFrame(imputer2.fit_transform(X_train), columns=X_train.columns.tolist() + 
                                                               [f'missing_{col}' for col in X_train.columns])
X_test_imp2 = pd.DataFrame(imputer2.transform(test), columns=test.columns.tolist() + 
                                                     [f'missing_{col}' for col in test.columns])



scaler = StandardScaler()
X_train_scaled1 = scaler.fit_transform(X_train_imp1)
X_test_scaled1 = scaler.transform(X_test_imp1)

X_train_scaled2 = scaler.fit_transform(X_train_imp2)
X_test_scaled2 = scaler.transform(X_test_imp2)

# Feature selection using mutual information
selector = SelectKBest(mutual_info_classif, k=15)
X_train_selected = selector.fit_transform(X_train_scaled1, y_train)
selected_features = X_train_imp1.columns[selector.get_support()]
print("\nSelected features:", selected_features.tolist())

# PCA for dimensionality reduction
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled1)
X_test_pca = pca.transform(X_test_scaled1)
print(f"PCA reduced dimensions from {X_train_scaled1.shape[1]} to {X_train_pca.shape[1]}")


models = {
    "XGBoost": xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        enable_categorical=True,
        random_state=42,
        early_stopping_rounds=50
    ),
    "LightGBM": lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        random_state=42,
        boosting_type='dart',
        class_weight='balanced'
    ),
    "CatBoost": cb.CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        random_state=42,
        verbose=False,
        auto_class_weights='Balanced'
    ),
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
}


print("\n=== Cross-Validation Results ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled1, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    results.append((name, mean_score, std_score))
    
    print(f"{name:<20} | Mean AUC: {mean_score:.5f} ± {std_score:.5f}")


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 30),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 30),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.5),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    }
    
    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='auc', random_state=42)
    score = cross_val_score(model, X_train_scaled1, y_train, cv=skf, scoring='roc_auc', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, show_progress_bar=True)

print("\n=== Best XGBoost Parameters ===")
print(study.best_params)
best_xgb_params = study.best_params



class PersonalityClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.net(x)


# Neural Network Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PersonalityClassifier(X_train_scaled1.shape[1]).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.3]).to(device))
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)



# Convert data to PyTorch tensors
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled1), torch.LongTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# Training loop with early stopping
best_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(300):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss/len(train_loader)
    scheduler.step(avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_nn_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(torch.load('best_nn_model.pth'))


# XGBoost with tuned parameters
xgb_model = xgb.XGBClassifier(**best_xgb_params, use_label_encoder=False, eval_metric='auc')
xgb_model.fit(X_train_scaled1, y_train)
xgb_preds = xgb_model.predict_proba(X_test_scaled1)[:, 1]


# LightGBM with optimized parameters
lgbm_model = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=10,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    boosting_type='dart',
    random_state=42
)
lgbm_model.fit(X_train_scaled1, y_train)
lgbm_preds = lgbm_model.predict_proba(X_test_scaled1)[:, 1]



# CatBoost with optimized parameters
cat_model = cb.CatBoostClassifier(
    iterations=1500,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    random_strength=0.5,
    auto_class_weights='Balanced',
    verbose=False,
    random_state=42
)
cat_model.fit(X_train_scaled1, y_train)
cat_preds = cat_model.predict_proba(X_test_scaled1)[:, 1]


# Neural Network predictions
model.eval()
with torch.no_grad():
    nn_probs = torch.softmax(model(torch.FloatTensor(X_test_scaled1).to(device)), dim=1)[:, 1].cpu().numpy()


stack_models = [
    ('xgb', xgb_model),
    ('lgbm', lgbm_model),
    ('catboost', cat_model),
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
]

stacker = StackingClassifier(
    estimators=stack_models,
    final_estimator=xgb.XGBClassifier(**best_xgb_params),
    cv=5,
    passthrough=True,
    n_jobs=-1
)
stacker.fit(X_train_scaled1, y_train)
stacked_preds = stacker.predict_proba(X_test_scaled1)[:, 1]


# Calculate weights based on cross-validation performance
val_scores = {
    'xgb': roc_auc_score(y_train, xgb_model.predict_proba(X_train_scaled1)[:, 1]),
    'lgbm': roc_auc_score(y_train, lgbm_model.predict_proba(X_train_scaled1)[:, 1]),
    'cat': roc_auc_score(y_train, cat_model.predict_proba(X_train_scaled1)[:, 1]),
    'stack': roc_auc_score(y_train, stacker.predict_proba(X_train_scaled1)[:, 1])
}

total_score = sum(val_scores.values())
weights = {k: v/total_score for k, v in val_scores.items()}
weights['nn'] = 0.1  # Fixed weight for neural network

print("\nModel Weights:", weights)



# Final ensemble prediction
ensemble_probs = (
    weights['xgb'] * xgb_preds +
    weights['lgbm'] * lgbm_preds +
    weights['cat'] * cat_preds +
    weights['stack'] * stacked_preds +
    weights['nn'] * nn_probs
)



# Use Youden's J statistic to find optimal threshold
fpr, tpr, thresholds = roc_curve(y_train, xgb_model.predict_proba(X_train_scaled1)[:, 1])
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal threshold: {optimal_threshold:.4f}")


submission['Personality'] = (ensemble_probs > optimal_threshold).astype(int)
submission['Personality'] = submission['Personality'].map({0: 'Extrovert', 1: 'Introvert'})
submission.to_csv("submission.csv", index=False)

print("\n=== Final Submission Preview ===")
print(submission.head())


plt.figure(figsize=(18, 12))

# XGBoost Importance
plt.subplot(2, 2, 1)
xgb.plot_importance(xgb_model, max_num_features=20, importance_type='gain', ax=plt.gca())
plt.title('XGBoost Feature Importance (Gain)')

# LightGBM Importance
plt.subplot(2, 2, 2)
lgb.plot_importance(lgbm_model, max_num_features=20, ax=plt.gca())
plt.title('LightGBM Feature Importance')

# CatBoost Importance
plt.subplot(2, 2, 3)
importances = pd.DataFrame({
    'Feature': X_train_imp1.columns,
    'Importance': cat_model.get_feature_importance()
}).sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=importances.head(20), ax=plt.gca())
plt.title('CatBoost Feature Importance')

# SHAP Values
plt.subplot(2, 2, 4)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train_scaled1)
shap.summary_plot(shap_values, X_train_scaled1, feature_names=X_train_imp1.columns, plot_type='bar', max_display=20)
plt.title('SHAP Feature Importance')

plt.tight_layout()
plt.show()


plt.figure()
shap.initjs()
sample_idx = 0  # Change this to see different examples
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    X_train_scaled1[sample_idx],
    feature_names=X_train_imp1.columns,
    matplotlib=True
)
plt.title(f'SHAP Explanation for Sample {sample_idx}')
plt.tight_layout()
plt.show()







