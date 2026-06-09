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


# =============================
# 1. Import libraries
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

# =============================
# 2. Load data
# =============================
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# =============================
# 3. Quick preview
# =============================
print("Train head:")
display(train.head())

print("Train info():")
print(train.info())

print("Missing values (train):")
print(train.isnull().sum())

print("Target value counts:")
print(train['Personality'].value_counts(normalize=True))

# =============================
# 4. Visualize target balance
# =============================
plt.figure(figsize=(5,4))
sns.countplot(data=train, x='Personality', palette='coolwarm')
plt.title("Target Distribution")
plt.show()

# =============================
# 5. Identify feature types
# =============================
target = 'Personality'
features = [col for col in train.columns if col != target]

num_features = train[features].select_dtypes(include=['int64', 'float64']).columns
cat_features = train[features].select_dtypes(include=['object']).columns

print("Numerical features:", list(num_features))
print("Categorical features:", list(cat_features))

# =============================
# 6. Summary stats
# =============================
print("Numerical feature summary:")
display(train[num_features].describe().T)

print("Categorical feature summary:")
for col in cat_features:
    print(f"{col} -> unique: {train[col].nunique()}, top: {train[col].value_counts().index[0]}")

# =============================
# 7. Correlation heatmap (numerical only)
# =============================
if len(num_features) > 1:
    plt.figure(figsize=(10,8))
    corr = train[num_features].corr()
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()

# =============================
# 8. Example feature-target relationship
# =============================
if len(num_features) > 0:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=train, x=target, y=num_features[0], palette='viridis')
    plt.title(f"{num_features[0]} vs {target}")
    plt.show()

# Pairplot for a quick feel (optional, may be slow if many features)
# sns.pairplot(train, hue=target, vars=num_features[:4], diag_kind='kde')
# plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings


# Data for EDA
eda_df = train.drop(columns=['id'])
num_features = ['Time_spent_Alone', 'Social_event_attendance', 
                'Going_outside', 'Friends_circle_size', 'Post_frequency']
cat_features = ['Stage_fear', 'Drained_after_socializing']

fig, axes = plt.subplots(
    nrows=len(num_features) + len(cat_features) + 1,  # +1 for missingno
    ncols=2,
    figsize=(14, 4*(len(num_features) + len(cat_features) + 1))
)

#  Numerical distributions (KDE + Box)
for i, col in enumerate(num_features):
    # KDE
    sns.kdeplot(
        data=eda_df, x=col, hue='Personality',
        fill=True, common_norm=False, palette='coolwarm', alpha=0.5,
        ax=axes[i, 0]
    )
    axes[i, 0].set_title(f'Distribution of {col} by Personality')

    # Boxplot
    sns.boxplot(
        data=eda_df, x='Personality', y=col, palette='Set2',
        ax=axes[i, 1]
    )
    axes[i, 1].set_title(f'{col} by Personality')

# Categorical countplots
start_cat = len(num_features)
for j, col in enumerate(cat_features):
    sns.countplot(
        data=eda_df, x=col, hue='Personality', palette='pastel',
        ax=axes[start_cat + j, 0]
    )
    axes[start_cat + j, 0].set_title(f'{col} vs Personality')
    axes[start_cat + j, 1].axis('off')  # Empty col for aesthetics

# 3. Missing value matrix in the last row
msno.matrix(eda_df, ax=axes[-1, 0])
axes[-1, 0].set_title('Missing Value Matrix')
axes[-1, 1].axis('off')

# Save before showing
fig.savefig("eda_dashboard.png", dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


# Drop ID column
train = train.drop(columns=['id'])
test_IDs = test['id']  # Keep for submission later
test = test.drop(columns=['id'])

# Separate features and target
y_train = train['Personality']
X_train = train.drop(columns=['Personality'])

# Identify columns by dtype
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Imputation
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
test[num_cols] = num_imputer.transform(test[num_cols])

X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
test[cat_cols] = cat_imputer.transform(test[cat_cols])

# Encoding (binary Yes/No to 1/0)
le = LabelEncoder()
for col in cat_cols + ['Personality']:
    if col in X_train.columns:
        X_train[col] = le.fit_transform(X_train[col])
        test[col] = le.transform(test[col])
    elif col == 'Personality':
        y_train = le.fit_transform(y_train)

# Final check
print("Training data shape:", X_train.shape)
print("Test data shape:", test.shape)
print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)
print("Target distribution:", np.bincount(y_train))



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Baseline model: RandomForest with imbalance handling
rf_baseline = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_baseline.fit(X_tr, y_tr)

# Validation predictions
y_pred = rf_baseline.predict(X_val)

# Metrics
bal_acc = balanced_accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average='macro')

print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Macro F1-score:   {macro_f1:.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Feature importance plot
importances = rf_baseline.feature_importances_
indices = np.argsort(importances)[::-1]
features_sorted = X_train.columns[indices]
importances_sorted = importances[indices]

plt.figure(figsize=(8, 5))
plt.barh(features_sorted, importances_sorted, color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("RandomForest Baseline - Feature Importance")
plt.tight_layout()
plt.show()

# Keep sorted feature importance in a DataFrame for later
feature_importance_df = pd.DataFrame({
    "Feature": features_sorted,
    "Importance": importances_sorted
})



from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# LightGBM baseline
lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
lgbm_model.fit(X_tr, y_tr)
y_pred_lgbm = lgbm_model.predict(X_val)

lgbm_bal_acc = balanced_accuracy_score(y_val, y_pred_lgbm)
lgbm_macro_f1 = f1_score(y_val, y_pred_lgbm, average='macro')

# XGBoost baseline
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    scale_pos_weight=(len(y_tr) - sum(y_tr)) / sum(y_tr),  # class balance
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_tr, y_tr)
y_pred_xgb = xgb_model.predict(X_val)

xgb_bal_acc = balanced_accuracy_score(y_val, y_pred_xgb)
xgb_macro_f1 = f1_score(y_val, y_pred_xgb, average='macro')

# Results table
results_df = pd.DataFrame({
    'Model': ['RandomForest', 'LightGBM', 'XGBoost'],
    'Balanced Accuracy': [bal_acc, lgbm_bal_acc, xgb_bal_acc],
    'Macro F1': [macro_f1, lgbm_macro_f1, xgb_macro_f1]
})

print(results_df)



import shap
import matplotlib.pyplot as plt

#  Feature Importances (side-by-side comparison)
def plot_feature_importance(model, model_name, feature_names):
    importances = model.feature_importances_
    sorted_idx = importances.argsort()
    plt.figure(figsize=(8, 5))
    plt.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx])
    plt.title(f"{model_name} Feature Importance")
    plt.tight_layout()
    plt.show()

plot_feature_importance(rf_baseline, "RandomForest", X_train.columns)
plot_feature_importance(lgbm_model, "LightGBM", X_train.columns)
plot_feature_importance(xgb_model, "XGBoost", X_train.columns)

#  SHAP summary for XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_val)

plt.figure()
shap.summary_plot(shap_values, X_val, feature_names=X_train.columns)
plt.show()


#  Retrain XGB on full training data
final_xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_xgb.fit(X_train, y_train)

#  Predict & save submission
test_preds = final_xgb.predict(test)

# Convert 0/1 back to original Personality labels
submission = pd.DataFrame({
    'id': test_IDs,
    'Personality': test_preds
})
submission['Personality'] = submission['Personality'].map({0: 'Extrovert', 1: 'Introvert'})

submission_path = "final_xgb_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")


# Predictions on validation set
val_preds = final_xgb.predict(X_val)

# Metrics
bal_acc = balanced_accuracy_score(y_val, val_preds)
macro_f1 = f1_score(y_val, val_preds, average='macro')

print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, val_preds))


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score

# New instance of XGBClassifier
xgb_cv_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Custom scorers
bal_acc_scorer = make_scorer(balanced_accuracy_score)
macro_f1_scorer = make_scorer(f1_score, average='macro')

# StratifiedKFold for class balance
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
bal_acc_scores = cross_val_score(xgb_cv_model, X_train, y_train, cv=cv, scoring=bal_acc_scorer)
macro_f1_scores = cross_val_score(xgb_cv_model, X_train, y_train, cv=cv, scoring=macro_f1_scorer)

print(f"Balanced Accuracy (5-fold CV): {bal_acc_scores.mean():.4f} ± {bal_acc_scores.std():.4f}")
print(f"Macro F1-score (5-fold CV): {macro_f1_scores.mean():.4f} ± {macro_f1_scores.std():.4f}")



import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, balanced_accuracy_score


# Scorer
bal_acc_scorer = make_scorer(balanced_accuracy_score)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train),
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=bal_acc_scorer)
    return np.mean(scores)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print('Best Balanced Accuracy:', study.best_value)
print('Best Parameters:', study.best_params)

# Train final tuned XGB on full training set
best_params = study.best_params
best_params.update({
    'random_state': 42,
    'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train),
    'n_jobs': -1,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
})
final_xgb_tuned = XGBClassifier(**best_params)
final_xgb_tuned.fit(X_train, y_train)



from sklearn.impute import SimpleImputer
import numpy as np

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# Numeric imputation
num_imputer = SimpleImputer(strategy='median')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_val[num_cols]   = num_imputer.transform(X_val[num_cols])
X[num_cols]       = num_imputer.fit_transform(X[num_cols])
test[num_cols]    = num_imputer.transform(test[num_cols])

# Categorical imputation (if any left in this stage)
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_val[cat_cols]   = cat_imputer.transform(X_val[cat_cols])
    X[cat_cols]       = cat_imputer.fit_transform(X[cat_cols])
    test[cat_cols]    = cat_imputer.transform(test[cat_cols])


# === Assume we already have: X_train, X_val, y_train, y_val, X, y, test, test_IDs ===

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

# Baseline RandomForest
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

# Baseline LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Tuned XGBoost from Optuna
xgb_model = XGBClassifier(
    n_estimators=990,
    max_depth=3,
    learning_rate=0.07197126871667955,
    min_child_weight=2,
    gamma=1.0661739666870234,
    subsample=0.9112147109680436,
    colsample_bytree=0.5349246085986328,
    scale_pos_weight=len(y[y==0]) / len(y[y==1]),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

# Soft-voting ensemble
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('lgbm', lgbm_model),
        ('xgb', xgb_model)
    ],
    voting='soft',
    n_jobs=-1
)

# Validation score
ensemble.fit(X_train, y_train)
y_val_pred = ensemble.predict(X_val)
print(f"Ensemble Balanced Accuracy: {balanced_accuracy_score(y_val, y_val_pred):.4f}")
print(f"Ensemble Macro F1: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

# Retrain on full set
ensemble.fit(X, y)

# Test set predictions + submission
test_preds = ensemble.predict(test)
inv_target_map = {0: 'Extrovert', 1: 'Introvert'}
final_preds = pd.Series(test_preds).map(inv_target_map)

submission = pd.DataFrame({
    'id': test_IDs,
    'Personality': final_preds
})
submission.to_csv("ensemble_submission.csv", index=False)
print("Saved ensemble_submission.csv")


