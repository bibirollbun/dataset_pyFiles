import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import uniform, randint

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.svm import SVC

from catboost import CatBoostClassifier, Pool


# Load official competition data

df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df.head()


print("Total NaN count per column:")
print(df.isnull().sum(axis = 0))


# Encoding of categorical features

cat_cols = ['Stage_fear', 'Drained_after_socializing']
for col in cat_cols:
    df[col] = df[col].map({'No': 0, 'Yes': 1})

df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})


# Define numeric feature columns (excluding 'id' and 'Personality')
feature_cols = [col for col in df.columns if col not in ['id', 'Personality']]

# Get all unique pairs of features
pairs = list(combinations(feature_cols, 2))

# Set up the plot
num_plots = len(pairs)
ncols = 6
nrows = -(-num_plots // ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows))
axes = axes.flatten()

# Plot each pair
for i, (x, y) in enumerate(pairs):
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue='Personality',
        ax=axes[i],
        alpha=0.7
    )
    axes[i].set_title(f'{x} vs {y}')

# Turn off unused subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


def feature_engineering(df):
    df['Social_Outside'] = np.round((df['Social_event_attendance'] / 10) * (df['Going_outside'] / 7), 2)
    df['Social_Friends'] = np.round((df['Social_event_attendance'] / 10) * (df['Friends_circle_size'] / 15), 2)
    df['Social_Posts'] = np.round((df['Social_event_attendance'] / 10) * (df['Post_frequency'] / 10), 2)
    df['Outside_Friends'] = np.round((df['Going_outside'] / 7) * (df['Friends_circle_size'] / 15), 2)
    df['Outside_Posts'] = np.round((df['Going_outside'] / 7) * (df['Post_frequency'] / 10), 2)
    df['Friends_Posts'] = np.round((df['Friends_circle_size'] / 15) * (df['Post_frequency'] / 10), 2)

feature_engineering(df)

df.head()


X = df.drop(columns=['id', 'Personality'])
y = df['Personality']

# 5-Fold Stratified CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store OOF predictions and models
oof_preds = np.zeros(len(X))
xgb_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    xgb_models.append(model)

    # Optional: Evaluate each fold
    auc = roc_auc_score(y_val, val_pred)
    acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
    print(f"  Fold AUC: {auc:.4f}, Accuracy: {acc:.4f}")

# Final OOF metrics
final_auc = roc_auc_score(y, oof_preds)
final_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
print(f"\nOverall AUC: {final_auc:.4f}")
print(f"Overall Accuracy: {final_acc:.4f}")


X = df.drop(columns=['id', 'Personality'])
y = df['Personality'].astype(int)

# 5-Fold Stratified CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
svm_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Scale data before feeding into SVM
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=1.0,
            probability=True,  # Required for predict_proba
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    svm_models.append(pipeline)

    auc = roc_auc_score(y_val, val_pred)
    acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
    print(f"  Fold AUC: {auc:.4f}, Accuracy: {acc:.4f}")

# Final OOF metrics
final_auc = roc_auc_score(y, oof_preds)
final_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
print(f"\nOverall AUC: {final_auc:.4f}")
print(f"Overall Accuracy: {final_acc:.4f}")


X = df.drop(columns=['id', 'Personality'])
y = df['Personality'].astype(int)

# 5-Fold Stratified CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
cat_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        verbose=0,
        random_seed=42,
        early_stopping_rounds=50
    )

    model.fit(train_pool, eval_set=val_pool)

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    cat_models.append(model)

    auc = roc_auc_score(y_val, val_pred)
    acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
    print(f"  Fold AUC: {auc:.4f}, Accuracy: {acc:.4f}")

# Final OOF metrics
final_auc = roc_auc_score(y, oof_preds)
final_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
print(f"\nOverall AUC: {final_auc:.4f}")
print(f"Overall Accuracy: {final_acc:.4f}")


test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

cat_cols = ['Stage_fear', 'Drained_after_socializing']
for col in cat_cols:
    test_df[col] = test_df[col].map({'No': 0, 'Yes': 1})

feature_engineering(test_df)

X_test = test_df.drop(columns=['id'])

# Predict with XGBoost ensemble
xgb_preds = np.mean([model.predict_proba(X_test)[:, 1] for model in xgb_models], axis=0)

# Predict with SVM ensemble
svm_preds = np.mean([model.predict_proba(X_test)[:, 1] for model in svm_models], axis=0)

# Predict with CatBoost ensemble
cat_preds = np.mean([model.predict_proba(X_test)[:, 1] for model in cat_models], axis=0)

# Weighted ensemble
preds = 0.25 * xgb_preds + 0.25 * svm_preds + 0.5 * cat_preds

binary_preds = (preds > 0.5).astype(int)


# Only consider confident predictions
confidence_threshold = 0.9
confident_mask = (preds > confidence_threshold) | (preds < (1 - confidence_threshold))
X_test_confident = X_test[confident_mask]
pseudo_labels = binary_preds[confident_mask]

print(f"Number of confident predictions: {len(confident_mask)}")

X_augmented = pd.concat([X, X_test_confident], axis=0).reset_index(drop=True)

y_augmented = np.concatenate([y, pseudo_labels])


# XGBoost retraining
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_pseudo_models = []

for train_idx, val_idx in skf.split(X_augmented, y_augmented):
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_augmented.iloc[train_idx], y_augmented[train_idx],
              eval_set=[(X_augmented.iloc[val_idx], y_augmented[val_idx])],
              early_stopping_rounds=50, verbose=False)
    xgb_pseudo_models.append(model)


# SVM retraining
svm_pseudo_models = []

for train_idx, val_idx in skf.split(X_augmented, y_augmented):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    ])
    pipeline.fit(X_augmented.iloc[train_idx], y_augmented[train_idx])
    svm_pseudo_models.append(pipeline)


# Catboost retraining
cat_pseudo_models = []

for train_idx, val_idx in skf.split(X_augmented, y_augmented):
    train_pool = Pool(X_augmented.iloc[train_idx], y_augmented[train_idx])
    val_pool = Pool(X_augmented.iloc[val_idx], y_augmented[val_idx])
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50
    )
    model.fit(train_pool, eval_set=val_pool)
    cat_pseudo_models.append(model)


# Predict with updated XGBoost ensemble
xgb_preds = np.mean([model.predict_proba(X_test)[:, 1] for model in xgb_pseudo_models], axis=0)

# Predict with SVM ensemble
svm_preds = np.mean([model.predict_proba(X_test)[:, 1] for model in svm_pseudo_models], axis=0)

# Predict with CatBoost ensemble
cat_preds = np.mean([model.predict_proba(X_test)[:, 1] for model in cat_pseudo_models], axis=0)

# Weighted ensemble
preds = 0.25 * xgb_preds + 0.25 * svm_preds + 0.5 * cat_preds

binary_preds = (preds > 0.5).astype(int)

submission = pd.DataFrame({'id': test_df['id'], 'Personality': binary_preds})
submission['Personality'] = submission['Personality'].map({0: 'Introvert', 1: 'Extrovert'})
submission.to_csv('submission.csv', index=False)

print(submission.head())

