# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Loading data
df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission_data = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


df.head()


test.head()


submission_data.head()


df = df.drop(columns=['id'], errors='ignore')



df.info()


# Checking for missing values
df.isnull().sum()


df['gender'].unique()


df['ethnicity'].unique()


df['education_level'].unique()


df['income_level'].unique()


df['smoking_status'].unique()


df['employment_status'].unique()


# Separate columns by data type for easier looping
target_col = 'diagnosed_diabetes'
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
# Add binary history columns to categorical list for analysis purposes
binary_cols = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
all_cat_cols = cat_cols + binary_cols

# Numerical columns (excluding the target and binary history)
num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
            if col not in [target_col] + binary_cols]

print(f"Numerical Features: {len(num_cols)}")
print(f"Categorical Features: {len(all_cat_cols)}")


plt.figure(figsize=(6, 4))
ax = sns.countplot(x=target_col, data=df, palette='viridis')
plt.title('Distribution of Target: Diagnosed Diabetes')
plt.xlabel('Diabetes (0 = No, 1 = Yes)')
plt.ylabel('Count')

#Percentage labels
total = len(df)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width() / 2 - 0.05
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')
plt.show()


n_cols = 3
n_rows = (len(num_cols) + n_cols - 1) // n_cols

# A. Histograms (Distributions)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=30, color='skyblue')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel('')

# Hiding empty subplots
for i in range(len(num_cols), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.suptitle('Numerical Feature Distributions', y=1.02, fontsize=16)
plt.show()



# B. Boxplots vs Target
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(x=target_col, y=col, data=df, ax=axes[i], palette='coolwarm')
    axes[i].set_title(f'{col} vs Diabetes')

for i in range(len(num_cols), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.suptitle('Numerical Features vs Diabetes Status', y=1.02, fontsize=16)
plt.show()


plt.figure(figsize=(10, 6))
# Using hexbin because scatterplot with 700k points is too heavy/slow
plt.hexbin(x=df['age'], y=df['bmi'], gridsize=20, cmap='Blues', mincnt=1)
plt.colorbar(label='Count')
plt.title('Density Plot: Age vs BMI')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()


# Calculate correlation matrix
corr_matrix = df[num_cols + [target_col]].corr()

plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Mask top half for readability
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu', 
            mask=mask, vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


cat_cols = ['gender', 'smoking_status', 'education_level', 'income_level', 'family_history_diabetes']

plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_cols):
    plt.subplot(2, 3, i+1)
    sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette='pastel')
    plt.title(col)
plt.tight_layout()
plt.show()


def plot_stack_bar(col):
    pd.crosstab(df[col], df['diagnosed_diabetes'], normalize='index').plot(
        kind='bar', stacked=True, figsize=(8, 5), colormap='viridis'
    )
    plt.title(f'{col} vs Diabetes Proportion')
    plt.ylabel('Proportion')
    plt.show()

plot_stack_bar('smoking_status')
plot_stack_bar('family_history_diabetes')


# Filter only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='bmi', hue='diagnosed_diabetes', data=df.head(1000), alpha=0.6, palette='bright')
plt.title('Interaction: Age vs BMI by Diabetes Status')
plt.show()


import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping
import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

target_col = 'diagnosed_diabetes'

# Numerical features (exclude ID and target)
num_cols = [
    col for col in df.columns
    if df[col].dtype in ['int64', 'float64']
    and col not in [target_col, 'id']
]

# Nominal categorical
nominal_cols = [
    'gender',
    'ethnicity',
    'smoking_status',
    'employment_status'
]

# Ordinal categorical
ordinal_cols = ['education_level', 'income_level']

education_order = ['No formal', 'Highschool', 'Graduate', 'Postgraduate']
income_order = ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']



target_col = 'diagnosed_diabetes'
X = df.drop(columns=[target_col, 'id'], errors='ignore')
y = df[target_col]

# Identify columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ],
    verbose_feature_names_out=False
)
X_encoded = preprocessor.fit_transform(X)
X_encoded = pd.DataFrame(X_encoded, columns=num_cols + cat_cols)
for col in cat_cols:
    X_encoded[col] = X_encoded[col].astype('category')

print("X shape:", X_encoded.shape)
print("y shape:", y.shape)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
        'class_weight': 'balanced',
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y)):
        X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)

        # We rely on Early Stopping to save time, but NOT the Pruning Callback inside fit
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_probs)
        aucs.append(fold_auc)

        # --- Pruning Strategy: Prune based on Fold Average ---
        # Instead of pruning every iteration (which is noisy), we prune if
        # the average AUC after K folds is significantly worse than other trials.
        trial.report(np.mean(aucs), step=fold)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(aucs)



study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
)

print("Starting optimization...")
study.optimize(objective, n_trials=25, show_progress_bar=True)

print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")


best_model = LGBMClassifier(
    **study.best_params,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])

final_pipeline.fit(X, y)



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(X))

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train = y.iloc[train_idx]

    final_pipeline.fit(X_train, y_train)
    oof_probs[val_idx] = final_pipeline.predict_proba(X_val)[:, 1]



thresholds = np.linspace(0.3, 0.7, 100)
best_threshold = 0.5
best_accuracy = 0

for t in thresholds:
    preds = (oof_probs >= t).astype(int)
    acc = accuracy_score(y, preds)
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = t

print(f"\nBest Threshold: {best_threshold:.3f}")
print(f"Optimized Accuracy: {best_accuracy:.4f}")



final_preds = (oof_probs >= best_threshold).astype(int)

print("\nFinal Classification Report:")
print(classification_report(y, final_preds))



'''
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# 1. XGBoost: "The Reliable Standard"
xgb_clf = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.015,  
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    enable_categorical=False 
)

# 2. LightGBM: "The Fast & Light"
lgbm_clf = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.015,
    num_leaves=31,        
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

# 3. CatBoost: "The Categorical Specialist"

cat_clf = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.015,
    depth=6,
    allow_writing_files=False, # Keeps directory clean
    verbose=0,                 # Silent training
    random_state=42
)

# 4. Voting Classifier 

voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_clf),
        ('lgbm', lgbm_clf),
        ('cat', cat_clf)
    ],
    voting='soft',
    n_jobs=-1
)

# 5. Final Pipeline Integration

final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', voting_clf)
])
'''


'''
# Stratified K-Fold Training Loop

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Arrays to store results
oof_preds = np.zeros(len(X))          # Validation predictions
test_preds = np.zeros(len(test))      # Final test set predictions
scores = []

print(f"Starting Training with {FOLDS} Folds (Voting Ensemble)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Create fold datasets
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train the full pipeline
    final_model.fit(X_train_fold, y_train_fold)
    
    # Predict Probabilities (Target=1)
    val_probs = final_model.predict_proba(X_val_fold)[:, 1]
    
    # Store OOF predictions
    oof_preds[val_idx] = val_probs
    
    # Evaluate Fold
    fold_auc = roc_auc_score(y_val_fold, val_probs)
    scores.append(fold_auc)
    
    # Predict on Test Set (Accumulate average)
    test_fold_probs = final_model.predict_proba(test)[:, 1]
    test_preds += test_fold_probs / FOLDS
    
    print(f"Fold {fold+1} ROC AUC: {fold_auc:.5f}")

print("-" * 30)
print(f"Mean ROC AUC: {np.mean(scores):.5f} (+/- {np.std(scores):.5f})")
print("-" * 30)
'''


test_probs = final_pipeline.predict_proba(test)[:, 1]

test_preds = (test_probs >= best_threshold).astype(int)

class CFG:
    sample_submission_csv = '/kaggle/input/playground-series-s5e12/sample_submission.csv'
    target = 'diagnosed_diabetes'
    output_filename = 'submission.csv'

submission = pd.read_csv(CFG.sample_submission_csv)

submission[CFG.target] = test_preds

submission.to_csv(CFG.output_filename, index=False)

print(f"Submission saved to: {CFG.output_filename}")
print(f"Submission Shape: {submission.shape}")
print("-" * 30)
print(submission.head())




