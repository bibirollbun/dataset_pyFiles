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


import pandas as pd, numpy as np

train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')

print('Train Shape:', train_df.shape)
print('Test Shape:', test_df.shape)





TARGET = 'loan_paid_back' 
CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
BASE = [col for col in train_df.columns if col not in ['id', TARGET]]

train = train_df.copy()
test = test_df.copy()



ORIG = []

for col in BASE:
    # MEAN
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # COUNT
    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

print(len(ORIG), 'Orig Features Created!!')

FEATURES = BASE + ORIG
print(len(FEATURES), 'Total Features.')


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train[FEATURES], train[TARGET].astype(int), 
    test_size=0.2, random_state=42, stratify=train[TARGET]
)

print(f'Train split (with ORIG features): {X_train.shape}')
print(f'Val split (with ORIG features): {X_val.shape}')


# numeric correlations
numeric_cols = train[BASE].select_dtypes(include=[np.number]).columns.tolist()
numeric_corr = pd.Series(dtype=float)
if numeric_cols:
    numeric_corr = train[numeric_cols].corrwith(train[TARGET].astype(int)).sort_values(ascending=False)

# categorical associations (Cramer's V)
cat_cols = [c for c in BASE if c not in numeric_cols]

# try to use scipy for chi2_contingency; fallback to a safe zero if unavailable
try:
    from scipy.stats import chi2_contingency
    def cramers_v(x, y):
        confusion = pd.crosstab(x, y)
        if confusion.size == 0:
            return 0.0
        chi2 = chi2_contingency(confusion)[0]
        n = confusion.sum().sum()
        if n == 0:
            return 0.0
        phi2 = chi2 / n
        r, k = confusion.shape
        # bias correction
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        denom = min((kcorr-1), (rcorr-1))
        return (phi2corr / denom)**0.5 if denom > 0 else 0.0
except Exception:
    def cramers_v(x, y):
        # fallback: compute association ratio by encoding categories to codes and using Pearson
        x_codes = x.astype('category').cat.codes
        return abs(x_codes.corr(y.astype(int)))

cat_assoc = {}
for c in cat_cols:
    col = train[c].fillna('NA')
    cat_assoc[c] = cramers_v(col, train[TARGET].astype(int))

cat_series = pd.Series(cat_assoc).sort_values(ascending=False)

# Display summaries
print('Numeric features (Pearson correlation with target):')
if not numeric_corr.empty:
    print(numeric_corr.to_string())
else:
    print('  No numeric features found in BASE.')

print('\nTop categorical associations (Cramer\'s V or fallback):')
if not cat_series.empty:
    print(cat_series.head(20).to_string())
else:
    print('  No categorical features found in BASE.')

# combined ranking by absolute strength
combined = pd.concat([
    numeric_corr.abs().rename('strength'),
    cat_series.abs().rename('strength')
]).sort_values(ascending=False)

print('\nTop features by absolute association strength with the target:')
print(combined.head(20).to_string())


import matplotlib.pyplot as plt
import seaborn as sns

# Visualization 1: Numeric correlations
if not numeric_corr.empty:
    fig, ax = plt.subplots(figsize=(10, max(4, len(numeric_corr) * 0.3)))
    numeric_corr.plot(kind='barh', ax=ax, color=['green' if x > 0 else 'red' for x in numeric_corr])
    ax.set_title('Pearson Correlations: Numeric Features vs Target', fontsize=14, fontweight='bold')
    ax.set_xlabel('Correlation Coefficient', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("No numeric features to visualize.")

# Visualization 2: Categorical associations
if not cat_series.empty:
    fig, ax = plt.subplots(figsize=(10, max(4, len(cat_series) * 0.3)))
    cat_series.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title("Cramer's V: Categorical Features vs Target", fontsize=14, fontweight='bold')
    ax.set_xlabel("Cramer's V (Association Strength)", fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print("No categorical features to visualize.")



# Visualization 3: Combined importance 
if not combined.empty:
    fig, ax = plt.subplots(figsize=(10, max(5, len(combined) * 0.25)))
    top_n = min(20, len(combined))
    top_combined = combined.head(top_n)
    top_combined.plot(kind='barh', ax=ax, color='coral')
    ax.set_title(f'Top {top_n} Features by Association Strength with Target', fontsize=14, fontweight='bold')
    ax.set_xlabel('Association Strength (Absolute Value)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print("No features to visualize.")



from sklearn.preprocessing import LabelEncoder

train_encoded = train[BASE].copy()

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    train_encoded[col] = le.fit_transform(train[col].fillna('NA'))
    le_dict[col] = le

train_encoded[TARGET] = train[TARGET].astype(int)

corr_matrix = train_encoded.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

target_corr = corr_matrix[TARGET].sort_values(ascending=False)
print(f'\n{TARGET} correlation:')
print(target_corr.to_string())


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# preprocess function
def preprocess_features(df, features, cat_cols, numeric_cols):
    """Prepare Categoric and Numerical Features"""
    df_processed = df[features].copy()
    
    # Kategorik kolonlar
    for col in cat_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('NA').astype('category')
    
    # Numeric kolonlar
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(0)
    
    return df_processed

# select numerical colons
numeric_cols_full = [col for col in FEATURES if col not in cat_cols]

print(f'Total features: {len(FEATURES)}')
print(f'  - BASE features: {len(BASE)}')
print(f'  - ORIG features: {len(ORIG)}')
print(f'Numeric features: {len(numeric_cols_full)}')
print(f'Categorical features: {len(cat_cols)}')


X_train_full = preprocess_features(train, FEATURES, cat_cols, numeric_cols_full)
y_train_full = train[TARGET].astype(int)

X_test = preprocess_features(test, FEATURES, cat_cols, numeric_cols_full)

X_train_split = preprocess_features(X_train, FEATURES, cat_cols, numeric_cols_full)
X_val_split = preprocess_features(X_val, FEATURES, cat_cols, numeric_cols_full)

print(f'\nTraining set: {X_train_full.shape}')
print(f'Test set: {X_test.shape}')
print(f'Train split: {X_train_split.shape}')
print(f'Val split: {X_val_split.shape}')

# XGBoost model parameters
xgb_params = {
    'n_estimators': 10000,
    'max_depth': 4,
    'learning_rate': 0.010433357477511243,
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': 42,
    'min_child_weight': 20,
    'subsample': 0.8879829126651821,
    'colsample_bytree': 0.5543148418738543,
    'gamma': 0.6845363006652688,
    'reg_alpha': 0.2399421158144976,
    'reg_lambda': 0.28254661049782354,
    'enable_categorical': True,
    'early_stopping_rounds': 100,
}

# Train XGBoost model
print('\nXGBoost training contuinue...')
model = XGBClassifier(**xgb_params)

model.fit(
    X_train_full, y_train_full,
    eval_set=[(X_train_split, y_train), (X_val_split, y_val)],
    verbose=1000
)

print('Model training completed!')

# Predict on test set
pred = model.predict_proba(X_test)[:, 1]

# Prepare submission
submission = pd.DataFrame({
    "id": test["id"],
    TARGET: pred
})


# Save submission file
submission.to_csv("submission.csv", index=False)
print(f'\nSubmission file saved!: submission.csv')
print(f'Submission shape: {submission.shape}')
submission.head()


# XGBoost feature importance ve loss visualization
import matplotlib.pyplot as plt

# Feature importance 
feature_importance = pd.DataFrame({
    'feature': X_train_full.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('Feature Importance:')
print(feature_importance.head(20).to_string())

# 2 subplot: feature importance + loss curves
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 1. Feature importance plot (colorful + logarithmic)
top_features = feature_importance.head(20)


import matplotlib.cm as cm
colors = cm.viridis(np.linspace(0, 1, len(top_features)))

# Horizontal bar plot
bars = axes[0].barh(range(len(top_features)), top_features['importance'], color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_yticks(range(len(top_features)))
axes[0].set_yticklabels(top_features['feature'])
axes[0].set_xlabel('Importance (log scale)', fontsize=12)
axes[0].set_ylabel('Feature', fontsize=12)
axes[0].set_title('Top Feature Importances (XGBoost) on log scale', fontsize=14, fontweight='bold')
axes[0].set_xscale('log')  # Logaritmik Ã¶lÃ§ek
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3, which='both')

for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
    axes[0].text(val, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', 
                ha='left', va='center', fontsize=9, fontweight='bold')

# Train & Validation Loss curves
results = model.evals_result()
train_loss = results['validation_0']['auc']
val_loss = results['validation_1']['auc']
epochs = range(len(train_loss))

axes[1].plot(epochs, train_loss, 'b-', label='Train AUC', linewidth=2, alpha=0.8)
axes[1].plot(epochs, val_loss, 'r-', label='Validation AUC', linewidth=2, alpha=0.8)
axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel('AUC Score', fontsize=12)
axes[1].set_title('XGBoost Training Progress: Train vs Val AUC', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f'\nFinal Train AUC: {train_loss[-1]:.4f}')
print(f'Final Validation AUC: {val_loss[-1]:.4f}')
print(f'Total iterations: {len(train_loss)}')

