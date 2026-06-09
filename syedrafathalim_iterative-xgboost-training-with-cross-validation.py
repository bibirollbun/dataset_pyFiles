import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



# Load datasets from Kaggle's input folder
df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

df.head()



# Basic info
print("Shape of training data:", df.shape)
print("Shape of test data:", test_df.shape)
print("\nColumn Info:")
df.info()



# Preview the data
df.head()



# Target distribution
print(df['y'].value_counts(normalize=True))  # Show proportion of each class
df['y'].value_counts().plot(kind='bar', title='Target Class Distribution')
plt.show()



# Check for missing values
missing_values = df.isnull().sum().sort_values(ascending=False)
print(missing_values)



# Separate categorical and numerical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=['y']).columns.tolist()

print("Categorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)



# Re-confirm categorical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical Columns:", cat_cols)



from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()
test_encoded = test_df.copy()

for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    # Handle unseen labels in test set
    test_encoded[col] = test_encoded[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<unknown>')
    test_encoded[col] = le.transform(test_encoded[col])



# Features and target
X = df_encoded.drop(columns=['y', 'id'])  # Drop target + ID
y = df_encoded['y']

# Split with stratification to preserve class distribution
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)



# Prepare test features (drop only ID column)
X_test = test_encoded.drop(columns=['id'])

print("Test shape:", X_test.shape)



from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Get categorical feature indices
cat_feature_indices = [X_train.columns.get_loc(col) for col in cat_cols]

# Define GPU model
model = CatBoostClassifier(
    iterations=30000,  # GPU-friendly, early stopping will stop earlier if needed
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    cat_features=cat_feature_indices,
    task_type='GPU',
    devices='0-1',
    max_bin=254,
    bootstrap_type='Bayesian',
    verbose=100,
    early_stopping_rounds=100
)

# Fit
model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

# Predict
val_preds = model.predict_proba(X_val)[:, 1]

# Evaluate
auc = roc_auc_score(y_val, val_preds)
print("Validation ROC AUC:", auc)



# Predict probabilities on the test set
test_preds = model.predict_proba(X_test)[:, 1]



# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'y': test_preds
})

# Save as CSV
submission.to_csv('/kaggle/working/submission12.csv', index=False)
print("Submission file saved as submission12.csv")



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import numpy as np

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
val_auc_scores = []
test_preds = np.zeros(len(X_test))

# Convert categorical column names to indices
cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n Fold {fold + 1}")
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=30000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=4,
        border_count=254,
        bagging_temperature=0.5,
        bootstrap_type='Bayesian',
        random_strength=1.5,
        eval_metric='AUC',
        cat_features=cat_feature_indices,
        random_seed=42,
        task_type='GPU',
        devices='0-1',
        verbose=200,
        early_stopping_rounds=300
    )

    model.fit(X_train_fold, y_train_fold,
              eval_set=(X_val_fold, y_val_fold),
              use_best_model=True)

    val_pred = model.predict_proba(X_val_fold)[:, 1]
    auc = roc_auc_score(y_val_fold, val_pred)
    print(f"Fold {fold + 1} ROC AUC: {auc:.5f}")
    val_auc_scores.append(auc)

    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

print(f"\n Average CV ROC AUC: {np.mean(val_auc_scores):.5f}")



# Create submission file from averaged predictions
submission = pd.DataFrame({
    'id': test_df['id'],
    'y': test_preds
})
submission.to_csv('submission13.csv', index=False)

print("Submission saved as submission13.csv")



import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Convert categorical column names to indices
cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

# Split training data for tuning
X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

def objective(trial):
    params = {
        'iterations': 20000,  # smaller for faster tuning
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 254),
        'bootstrap_type': 'Bayesian',
        'eval_metric': 'AUC',
        'cat_features': cat_feature_indices,
        'task_type': 'GPU',
        'devices': '0-1',
        'verbose': 0,
        'random_seed': 42,
        'early_stopping_rounds': 300
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train_tune, y_train_tune,
              eval_set=(X_val_tune, y_val_tune),
              use_best_model=True)
    preds = model.predict_proba(X_val_tune)[:, 1]
    return roc_auc_score(y_val_tune, preds)

# Run Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("Best parameters:", study.best_params)



# Convert categorical column names to indices for GPU training
cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

best_params = study.best_params
best_params.update({
    'iterations': 30000,
    'eval_metric': 'AUC',
    'cat_features': cat_feature_indices,  # indices instead of names for GPU
    'verbose': 100,
    'random_seed': 42,
    'task_type': 'GPU',   # enable GPU training
    'devices': '0-1'        # specify GPU device
})

# Remove early_stopping_rounds for full-data training
best_params.pop('early_stopping_rounds', None)

# Train final model on full data
model = CatBoostClassifier(**best_params)
model.fit(X, y)





test_preds = model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({'id': test_df['id'], 'y': test_preds})
submission.to_csv('submission14.csv', index=False)
print("Submission saved as submission14.csv")

