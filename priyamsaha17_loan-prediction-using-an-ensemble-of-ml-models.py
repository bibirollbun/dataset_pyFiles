# Load libraries
import pandas as pd, numpy as np
# Load data (works for Kaggle environment or local directory)
train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

# Preserve test IDs for submission
test_ids = test_df['id']

# Drop ID columns (not useful as features)
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


# 1. Datatype and Missing Count
print("--- TRAIN DATAFRAME INFO ---")
train_df.info() # Gives data type and non-null count

# Calculate and display missing percentage
missing_percentage = (train_df.isnull().sum() / len(train_df)) * 100
print("\n--- TRAIN DATAFRAME MISSING PERCENTAGES ---")
# print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))
print(missing_percentage)
# Use a plotting library like Matplotlib/Seaborn to plot missing_percentage

# 2. Descriptive Statistics
print("\n--- TRAIN DATAFRAME NUMERICAL STATS ---")
print(train_df.describe())

print("\n--- TRAIN DATAFRAME CATEGORICAL STATS ---")
for col in train_df.select_dtypes(include='object').columns:
    print(f"\n{col} Value Counts:")
    print(train_df[col].value_counts(normalize=True).head())
    # Use a plotting library like Matplotlib/Seaborn to plot distributions (histograms/box plots)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Step 0: Convert all categorical columns into numeric labels ----
df = train_df.copy()

from sklearn.preprocessing import LabelEncoder

for col in df.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ---- Step 1: Correlation series ----
correlation_series = df.corr()['loan_paid_back'].sort_values(ascending=False)

# Optional: remove self-correlation
correlation_series = correlation_series.drop('loan_paid_back', errors='ignore')

print("--- Correlation with loan_paid_back ---")
print(correlation_series)

# ---- Step 2: Plot ----
plt.figure(figsize=(10, len(correlation_series) * 0.4))
sns.barplot(x=correlation_series.values, y=correlation_series.index, palette='vlag')
plt.title('Feature Correlation with loan_paid_back', fontsize=16)
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


test_df.head()


'''cols_to_drop = [
    'annual_income',
    'marital_status',
    'loan_purpose',
    'loan_amount',
    'gender'
]

train_df = train_df.drop(columns=cols_to_drop)
test_df = test_df.drop(columns=cols_to_drop)
print("Dropped:", cols_to_drop)
'''


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Identify categorical columns (after dropping ID and low-corr features)
cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()

# Label-encode categories using combined data (to handle unseen categories):contentReference[oaicite:7]{index=7}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# Separate target from features
y = train_df['loan_paid_back'].values
X = train_df.drop(columns=['loan_paid_back'])

# Create train/validation split (stratified to preserve class ratio)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale numeric features for models that need scaling
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler().fit(X_train[num_cols])
# Prepare scaled copies
X_train_scaled = X_train.copy()
X_val_scaled   = X_val.copy()
X_train_scaled[num_cols] = scaler.transform(X_train[num_cols])
X_val_scaled[num_cols]   = scaler.transform(X_val[num_cols])


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm

# Detect if GPU is available for LightGBM/XGBoost/CatBoost
GPU_FLAG = torch.cuda.is_available() if 'torch' in globals() else False

# Instantiate models (with class weights and GPU options where applicable)
models = []
# LightGBM (with class_weight balanced)
lgbm_params = {'random_state': 42, 'class_weight': 'balanced', 'n_estimators': 200}
if GPU_FLAG:
    lgbm_params['device'] = 'gpu'
models.append(('lgbm', LGBMClassifier(**lgbm_params)))

# XGBoost (with updated GPU settings for version >= 2.0)
xgb_params = {'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss', 'n_estimators': 200}
if GPU_FLAG:
    xgb_params.update({'tree_method': 'hist', 'device': 'cuda'})
models.append(('xgb', XGBClassifier(**xgb_params)))

# CatBoost (use GPU if available)
cat_params = {'random_state': 42, 'verbose': 0}
if GPU_FLAG:
    cat_params['task_type'] = 'GPU'
models.append(('cat', CatBoostClassifier(**cat_params)))

# Other tree-based models
models.append(('rf', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200, n_jobs=-1)))
models.append(('et', ExtraTreesClassifier(random_state=42, class_weight='balanced', n_estimators=200, n_jobs=-1)))
models.append(('histgb', HistGradientBoostingClassifier(random_state=42, max_iter=200, class_weight='balanced')))
models.append(('gb', GradientBoostingClassifier(random_state=42, n_estimators=200)))

# Linear models
models.append(('logistic', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)))
models.append(('logistic_l1', LogisticRegression(random_state=42, penalty='l1', solver='saga',
                                                  class_weight='balanced', max_iter=1000)))

# Distance-based
models.append(('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)))

# Probabilistic
models.append(('nb', GaussianNB()))

# Neural network
models.append(('mlp', MLPClassifier(random_state=42, max_iter=200)))

# Others
models.append(('qda', QuadraticDiscriminantAnalysis()))
models.append(('ada', AdaBoostClassifier(random_state=42, n_estimators=100)))
models.append(('bag', BaggingClassifier(random_state=42, n_estimators=100, n_jobs=-1)))

# Train base models and collect validation predictions
val_meta_features = []
model_names = []
print("\nTraining base models:")
for name, model in tqdm(models):
    print(f"Training {name.upper()}...")
    # Choose scaled or unscaled features
    if name in ['logistic', 'logistic_l1', 'knn', 'mlp', 'qda']:
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_val_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    print(f"{name.upper()} AUC: {auc:.4f}")
    val_meta_features.append(proba)
    model_names.append(name)


from sklearn.linear_model import LogisticRegression

# Prepare meta-model training data from validation set
X_meta_val = np.column_stack(val_meta_features)
meta_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
meta_model.fit(X_meta_val, y_val)

# Retrain all base models on full training data (100%)
X_full = X.copy()
y_full = y.copy()
# Refit scaler on full data for scaled models
scaler_full = StandardScaler().fit(X_full[num_cols])
X_full_scaled = X_full.copy()
X_full_scaled[num_cols] = scaler_full.transform(X_full[num_cols])
# Apply same transformation to test set
X_test = test_df.copy()
X_test_scaled = test_df.copy()
X_test_scaled[num_cols] = scaler_full.transform(test_df[num_cols])

# Collect test-set predictions from each base model
test_meta_features = []
for name, model in models:
    # Reset the model by re-instantiating (to avoid reusing partially-trained model)
    # (Alternatively, one could clone; here we simply reuse `models` by retraining.)
    if name in ['logistic', 'logistic_l1', 'knn', 'mlp', 'qda']:
        model.fit(X_full_scaled, y_full)
        proba_test = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_full, y_full)
        proba_test = model.predict_proba(X_test)[:, 1]
    test_meta_features.append(proba_test)

# Stack test predictions and apply meta-model
X_meta_test = np.column_stack(test_meta_features)
final_probs = meta_model.predict_proba(X_meta_test)[:, 1]


# Create submission DataFrame and save to CSV
submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': final_probs
})
submission.to_csv('submission.csv', index=False)

