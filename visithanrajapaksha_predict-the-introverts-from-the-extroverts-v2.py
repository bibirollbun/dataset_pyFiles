# Install required packages with compatible versions
!pip uninstall -y numpy pandas scikit-learn xgboost optuna matplotlib seaborn scipy
!pip install numpy==1.26.4 pandas==2.2.3 scipy==1.15.2
!pip install scikit-learn==1.2.2 imbalanced-learn==0.12.3 xgboost==2.0.3 optuna==4.3.0 matplotlib==3.7.2 seaborn==0.12.2 lightgbm==4.5.0 catboost==1.2.7


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import os

# Set plot style
plt.style.use('seaborn')
sns.set_palette('husl')


print("--- Starting Data Loading ---")
DATA_DIR = os.path.join(os.getcwd(), '/kaggle/input/playground-series-s5e7/')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')


# Initialize dataframes
train_df = pd.DataFrame()
test_df = pd.DataFrame()

try:
    print(f"Loading train data from: {TRAIN_FILE}")
    train_df = pd.read_csv(TRAIN_FILE)
    print(f"Train data loaded. Shape: {train_df.shape}")
    print(f"Loading test data from: {TEST_FILE}")
    test_df = pd.read_csv(TEST_FILE)
    print(f"Test data loaded. Shape: {test_df.shape}")
    if 'Personality' in test_df.columns:
        test_df = test_df.drop('Personality', axis=1)
        print("Dropped 'Personality' column from test_df as it is the target variable.")
    if len(test_df) != 6175:
        print(f"Warning: Test set has {len(test_df)} rows, expected 6175. Check test.csv.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Store IDs
train_id = train_df['id']
test_id = test_df['id']


print("\n" + "="*50)
print("--- Adding Constant Feature to Synthetic Datasets ---")
print("="*50)
CONSTANT_FEATURE_NAME = 'constant_zero_feature'
CONSTANT_FEATURE_VALUE = 0
train_df[CONSTANT_FEATURE_NAME] = CONSTANT_FEATURE_VALUE
test_df[CONSTANT_FEATURE_NAME] = CONSTANT_FEATURE_VALUE
print(f"Added '{CONSTANT_FEATURE_NAME}' with value {CONSTANT_FEATURE_VALUE} to train_df and test_df.")
print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


print("\n" + "="*50)
print("--- Applying Missing Value Imputation ---")
print("="*50)

# Numerical Imputation
numerical_cols = train_df.select_dtypes(include=np.number).columns.drop(['id', 'Personality', CONSTANT_FEATURE_NAME], errors='ignore').tolist()
imputer = IterativeImputer(max_iter=10, random_state=42)
for df_name, df_obj in [('train_df', train_df), ('test_df', test_df)]:
    current_numerical_cols = [col for col in numerical_cols if col in df_obj.columns and df_obj[col].isnull().any()]
    if current_numerical_cols:
        print(f"Imputing numerical missing values in {df_name} for columns: {current_numerical_cols}")
        if df_name == 'train_df':
            df_obj[current_numerical_cols] = imputer.fit_transform(df_obj[current_numerical_cols])
        else:
            df_obj[current_numerical_cols] = imputer.transform(df_obj[current_numerical_cols])
    else:
        print(f"No numerical missing values to impute in {df_name}.")

# Categorical Imputation
categorical_cols = train_df.select_dtypes(include='object').columns.drop('Personality', errors='ignore').tolist()
for col in categorical_cols:
    if train_df[col].isnull().any():
        train_df[col] = train_df[col].fillna('Missing')
        print(f"Filled missing values in train_df[{col}] with 'Missing'.")
    if test_df[col].isnull().any():
        test_df[col] = test_df[col].fillna('Missing')
        print(f"Filled missing values in test_df[{col}] with 'Missing'.")


print("\n" + "="*50)
print("--- Feature Engineering ---")
print("="*50)

# Preserve original categorical columns for target encoding and EDA
train_df_original = train_df.copy()
test_df_original = test_df.copy()

# Target encoding for categorical features
for col in categorical_cols:
    target_mean = train_df_original.groupby(col)['Personality'].apply(lambda x: (x == 'Extrovert').mean())
    train_df[f'{col}_target_enc'] = train_df_original[col].map(target_mean)
    test_df[f'{col}_target_enc'] = test_df_original[col].map(target_mean).fillna(target_mean.mean())
    numerical_cols.append(f'{col}_target_enc')

# One-Hot Encoding
print("Applying One-Hot Encoding to categorical features.")
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=False)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=False)

# Align columns
train_feature_cols = [col for col in train_df.columns if col not in ['id', 'Personality']]
test_feature_cols = [col for col in test_df.columns if col not in ['id']]
all_features_union = sorted(list(set(train_feature_cols) | set(test_feature_cols)))
original_train_id = train_df['id']
original_train_personality = train_df['Personality']
original_test_id = test_df['id']
train_df = train_df.reindex(columns=all_features_union + ['id', 'Personality'], fill_value=0)
test_df = test_df.reindex(columns=all_features_union + ['id'], fill_value=0)
train_df['id'] = original_train_id
train_df['Personality'] = original_train_personality
test_df['id'] = original_test_id
print("Columns aligned after One-Hot Encoding.")

# Interaction terms for top numerical features
numerical_cols_no_target = [col for col in numerical_cols if col != 'Personality']
if len(numerical_cols_no_target) >= 2:
    for i in range(min(2, len(numerical_cols_no_target))):
        for j in range(i + 1, min(2, len(numerical_cols_no_target))):
            col1, col2 = numerical_cols_no_target[i], numerical_cols_no_target[j]
            train_df[f'{col1}_{col2}_inter'] = train_df[col1] * train_df[col2]
            test_df[f'{col1}_{col2}_inter'] = test_df[col1] * test_df[col2]
            numerical_cols.append(f'{col1}_{col2}_inter')

# Polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
if len(numerical_cols_no_target) >= 2:
    top_cols = numerical_cols_no_target[:2]
    poly_features = poly.fit_transform(train_df[top_cols])
    poly_feature_names = poly.get_feature_names_out(top_cols)
    train_df[poly_feature_names] = poly_features
    test_df[poly_feature_names] = poly.transform(test_df[top_cols])
    numerical_cols.extend(poly_feature_names)

# Scale numerical features
scaler = StandardScaler()
train_df[numerical_cols_no_target] = scaler.fit_transform(train_df[numerical_cols_no_target])
test_df[numerical_cols_no_target] = scaler.transform(test_df[numerical_cols_no_target])


print("\n" + "="*50)
print("--- Data Description ---")
print("="*50)

print("\n1. Train Dataset Overview:")
print(f"Shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print("\n2. Test Dataset Overview:")
print(f"Shape: {test_df.shape}")
print(f"Columns: {list(test_df.columns)}")

print("\n3. Column Data Types (Train):")
print(train_df.dtypes)
print("\n4. Column Data Types (Test):")
print(test_df.dtypes)

print("\n5. Missing Values (Train):")
print(train_df.isnull().sum())
print(f"Total missing values: {train_df.isnull().sum().sum()}")
print("\n6. Missing Values (Test):")
print(test_df.isnull().sum())
print(f"Total missing values: {test_df.isnull().sum().sum()}")

print("\n7. Summary Statistics for Numerical Features (Train):")
print(train_df[numerical_cols].describe())
print("\n8. Summary Statistics for Numerical Features (Test):")
print(test_df[[col for col in numerical_cols if col in test_df.columns]].describe())

print("\n9. Categorical Features and Unique Values (Train):")
for col in categorical_cols:
    print(f"{col}: {train_df_original[col].nunique()} unique values")
    print(f"Sample values: {train_df_original[col].unique()[:5]}")
print("\n10. Categorical Features and Unique Values (Test):")
for col in categorical_cols:
    print(f"{col}: {test_df_original[col].nunique()} unique values")
    print(f"Sample values: {test_df_original[col].unique()[:5]}")

print("\n11. Target Variable Distribution (Train):")
print(train_df['Personality'].value_counts(normalize=True))
print(f"Unique target values: {train_df['Personality'].unique()}")


plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=train_df)
plt.title('Distribution of Personality (Train)')
plt.xlabel('Personality Type')
plt.ylabel('Count')
plt.show()
print("\nClass Imbalance Check:")
print(train_df['Personality'].value_counts(normalize=True))


print("\nNumerical Features:", numerical_cols)
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_no_target, 1):
    plt.subplot((len(numerical_cols_no_target)//3 + 1), 3, i)
    sns.histplot(train_df[col], bins=30, kde=True)
    plt.title(f'{col} (Train)')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.suptitle('Distribution of Numerical Features (Train)')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_no_target, 1):
    plt.subplot((len(numerical_cols_no_target)//3 + 1), 3, i)
    sns.histplot(test_df[col], bins=30, kde=True)
    plt.title(f'{col} (Test)')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.suptitle('Distribution of Numerical Features (Test)')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# 5. Pair Plots of Numerical Features
print("\nGenerating Pair Plots for Top Numerical Features (Train and Test)")
# Select top 4 numerical features to avoid excessive computation
top_numerical_cols = numerical_cols_no_target[:5]
if len(top_numerical_cols) > 0:
    # Pair plots for training set
    pair_plot_df_train = train_df[top_numerical_cols + ['Personality']].copy()
    sns.pairplot(pair_plot_df_train, diag_kind='kde')
    plt.suptitle('Pair Plots of Top Numerical Features by Personality (Train)', y=1.02)
    plt.show()
    
    # Pair plots for test set
    pair_plot_df_test = test_df[top_numerical_cols].copy()
    sns.pairplot(pair_plot_df_test, diag_kind='kde')
    plt.suptitle('Pair Plots of Top Numerical Features (Test)', y=1.02)
    plt.show()
else:
    print("No numerical features available for pair plots.")


def detect_outliers(df, columns):
    outliers_count = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outliers_count[col] = len(outliers)
    return outliers_count

# Detect outliers in train and test sets
print("\nOutliers in Training Set:")
train_outliers = detect_outliers(train_df, numerical_cols_no_target)
for col, count in train_outliers.items():
    print(f"{col}: {count} outliers")

print("\nOutliers in Test Set:")
test_outliers = detect_outliers(test_df, numerical_cols_no_target)
for col, count in test_outliers.items():
    print(f"{col}: {count} outliers")

# Box plots for training set with hue by Personality
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_no_target, 1):
    plt.subplot((len(numerical_cols_no_target)//3 + 1), 3, i)
    sns.boxplot(x='Personality', y=col, data=train_df)
    plt.title(f'{col} (Train)')
plt.suptitle('Box Plots of Numerical Features by Personality (Train)')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_no_target, 1):
    plt.subplot((len(numerical_cols_no_target)//3 + 1), 3, i)
    sns.boxplot(y=train_df[col])
    plt.title(f'{col} (Train)')
plt.suptitle('Box Plots of Numerical Features (Train)')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_no_target, 1):
    plt.subplot((len(numerical_cols_no_target)//3 + 1), 3, i)
    sns.boxplot(y=test_df[col])
    plt.title(f'{col} (Test)')
plt.suptitle('Box Plots of Numerical Features (Test)')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


print("\nCategorical Features:", categorical_cols)
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, hue='Personality', data=train_df_original)
    plt.title(f'Distribution of {col} by Personality (Train)')
    plt.xticks(rotation=45)
    plt.show()

for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=test_df_original)
    plt.title(f'Distribution of {col} (Test)')
    plt.xticks(rotation=45)
    plt.show()


encoded_train = train_df.copy()
encoded_test = test_df.copy()

# Use a dictionary to store a separate LabelEncoder for each column
label_encoders = {}
for col in categorical_cols + ['Personality']:
    if col in train_df_original.columns:
        le = LabelEncoder()
        encoded_train[col] = le.fit_transform(train_df_original[col].astype(str))
        label_encoders[col] = le

for col in categorical_cols:
    if col in test_df_original.columns:
        le = label_encoders.get(col, LabelEncoder())
        # Get unique values in train and test
        train_values = set(train_df_original[col].astype(str).unique())
        test_values = set(test_df_original[col].astype(str).unique())
        unseen_values = test_values - train_values
        if unseen_values:
            print(f"Warning: Unseen values {unseen_values} in test_df_original[{col}]. Mapping to most common training value.")
            most_common = train_df_original[col].mode()[0]
            encoded_test[col] = test_df_original[col].astype(str).map(lambda x: x if x in train_values else most_common)
        else:
            encoded_test[col] = test_df_original[col].astype(str)
        encoded_test[col] = le.transform(encoded_test[col])

plt.figure(figsize=(12, 8))
sns.heatmap(encoded_train.drop('id', axis=1).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix (Train)')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(encoded_test.drop('id', axis=1).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix (Test)')
plt.show()


TARGET = 'Personality'
if TARGET not in train_df.columns:
    raise KeyError(f"'{TARGET}' not found in train_df.")
y = train_df[TARGET]
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"\nTarget variable encoded: {le_target.classes_} mapped to {le_target.transform(le_target.classes_)}")

FEATURES = [col for col in train_df.columns if col not in ['id', TARGET]]
X = train_df[FEATURES]
X_test = test_df[FEATURES]

# Handle class imbalance
class_ratio = train_df['Personality'].value_counts(normalize=True)
class_weights = {0: 1 / class_ratio['Extrovert'], 1: 1 / class_ratio['Introvert']}

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


def optimize_threshold(model, X_val, y_val):
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_threshold = 0.5
    best_score = 0
    for threshold in thresholds:
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > threshold).astype(int)
        score = accuracy_score(y_val, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=[class_weights[y] for y in y_train])
    best_threshold, best_score = optimize_threshold(model, X_val, y_val)
    return best_score


def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, sample_weight=[class_weights[y] for y in y_train])
    best_threshold, best_score = optimize_threshold(model, X_val, y_val)
    return best_score


def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_seed': 42,
        'verbose': 0
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, sample_weight=[class_weights[y] for y in y_train])
    best_threshold, best_score = optimize_threshold(model, X_val, y_val)
    return best_score


study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=30)
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30)
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=30)

# Train models with best parameters
best_params_xgb = study_xgb.best_params
best_params_xgb.update({'random_state': 42})
best_params_lgb = study_lgb.best_params
best_params_lgb.update({'random_state': 42})
best_params_cat = study_cat.best_params
best_params_cat.update({'random_seed': 42, 'verbose': 0})

model_xgb = XGBClassifier(**best_params_xgb)
model_lgb = LGBMClassifier(**best_params_lgb)
model_cat = CatBoostClassifier(**best_params_cat)

# Optimize ensemble weights
def objective_ensemble(trial):
    w_xgb = trial.suggest_float('w_xgb', 0, 1)
    w_lgb = trial.suggest_float('w_lgb', 0, 1)
    w_cat = trial.suggest_float('w_cat', 0, 1)
    w_sum = w_xgb + w_lgb + w_cat
    if w_sum == 0:
        return 0
    w_xgb, w_lgb, w_cat = w_xgb / w_sum, w_lgb / w_sum, w_cat / w_sum
    y_pred_proba = (w_xgb * model_xgb.predict_proba(X_val)[:, 1] +
                    w_lgb * model_lgb.predict_proba(X_val)[:, 1] +
                    w_cat * model_cat.predict_proba(X_val)[:, 1])
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_score = 0
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        score = accuracy_score(y_val, y_pred)
        if score > best_score:
            best_score = score
    return best_score


# Train models for ensemble optimization
model_xgb.fit(X_train, y_train, sample_weight=[class_weights[y] for y in y_train])
model_lgb.fit(X_train, y_train, sample_weight=[class_weights[y] for y in y_train])
model_cat.fit(X_train, y_train, sample_weight=[class_weights[y] for y in y_train])

study_ensemble = optuna.create_study(direction='maximize')
study_ensemble.optimize(objective_ensemble, n_trials=30)
best_weights = study_ensemble.best_params
w_sum = best_weights['w_xgb'] + best_weights['w_lgb'] + best_weights['w_cat']
best_weights = {
    'w_xgb': best_weights['w_xgb'] / w_sum,
    'w_lgb': best_weights['w_lgb'] / w_sum,
    'w_cat': best_weights['w_cat'] / w_sum
}
print("\nBest Ensemble Weights:", best_weights)

# Optimize ensemble threshold
def optimize_ensemble_threshold(X_val, y_val):
    y_pred_proba = (best_weights['w_xgb'] * model_xgb.predict_proba(X_val)[:, 1] +
                    best_weights['w_lgb'] * model_lgb.predict_proba(X_val)[:, 1] +
                    best_weights['w_cat'] * model_cat.predict_proba(X_val)[:, 1])
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_threshold = 0.5
    best_score = 0
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        score = accuracy_score(y_val, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score

threshold_ensemble, score_ensemble = optimize_ensemble_threshold(X_val, y_val)
print(f"Ensemble: Threshold={threshold_ensemble:.2f}, Accuracy={score_ensemble:.4f}")


print("\n" + "="*50)
print("--- Final Model Training and Evaluation ---")
print("="*50)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_encoded))
test_preds_folds = []
fold_accuracies = []
feature_importances_sum = np.zeros(len(FEATURES))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded)):
    print(f"Training Fold {fold+1}/5...")
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y_encoded[train_idx], y_encoded[val_idx]
    
    model_xgb.fit(X_train_cv, y_train_cv, sample_weight=[class_weights[y] for y in y_train_cv])
    model_lgb.fit(X_train_cv, y_train_cv, sample_weight=[class_weights[y] for y in y_train_cv])
    model_cat.fit(X_train_cv, y_train_cv, sample_weight=[class_weights[y] for y in y_train_cv])
    
    y_pred_proba = (best_weights['w_xgb'] * model_xgb.predict_proba(X_val_cv)[:, 1] +
                    best_weights['w_lgb'] * model_lgb.predict_proba(X_val_cv)[:, 1] +
                    best_weights['w_cat'] * model_cat.predict_proba(X_val_cv)[:, 1])
    y_pred = (y_pred_proba > threshold_ensemble).astype(int)
    oof_preds[val_idx] = y_pred
    fold_acc = accuracy_score(y_val_cv, y_pred)
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold+1} Accuracy: {fold_acc:.4f}")
    
    test_pred_proba = (best_weights['w_xgb'] * model_xgb.predict_proba(X_test)[:, 1] +
                       best_weights['w_lgb'] * model_lgb.predict_proba(X_test)[:, 1] +
                       best_weights['w_cat'] * model_cat.predict_proba(X_test)[:, 1])
    test_preds_folds.append(test_pred_proba)
    
    feature_importances_sum += model_xgb.feature_importances_  # Use XGBoost for importance

print(f"\nOverall CV Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")


# Feature Importance
avg_feature_importances = feature_importances_sum / 5
feature_importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': avg_feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance from XGBoost (Mean across CV folds)')
plt.show()
print("\nFeature Importances:")
print(feature_importance_df)

# Save OOF predictions
oof_df = pd.DataFrame({'id': train_id, 'oof_preds': oof_preds, 'target': y_encoded})
oof_df.to_csv('oof_predictions.csv', index=False)
print(f"\nOOF predictions saved to: oof_predictions.csv")
print(f"Sample OOF Head:\n{oof_df.head()}")

# Train final models
model_xgb.fit(X, y_encoded, sample_weight=[class_weights[y] for y in y_encoded])
model_lgb.fit(X, y_encoded, sample_weight=[class_weights[y] for y in y_encoded])
model_cat.fit(X, y_encoded, sample_weight=[class_weights[y] for y in y_encoded])

# Generate test predictions
test_pred_proba = (best_weights['w_xgb'] * model_xgb.predict_proba(X_test)[:, 1] +
                   best_weights['w_lgb'] * model_lgb.predict_proba(X_test)[:, 1] +
                   best_weights['w_cat'] * model_cat.predict_proba(X_test)[:, 1])
test_predictions = (test_pred_proba > threshold_ensemble).astype(int)
test_predictions_labels = le_target.inverse_transform(test_predictions)

# Create submission
submission_df = pd.DataFrame({'id': test_id, 'Personality': test_predictions_labels})
submission_df.to_csv('submission.csv', index=False)
print(f"\nSubmission file created at: submission.csv")
print(f"Shape: {submission_df.shape}")
print(f"Sample Submission Head:\n{submission_df.head()}")
print(f"Personality Value Counts:\n{submission_df['Personality'].value_counts()}")


val_pred_proba = (best_weights['w_xgb'] * model_xgb.predict_proba(X_val)[:, 1] +
                  best_weights['w_lgb'] * model_lgb.predict_proba(X_val)[:, 1] +
                  best_weights['w_cat'] * model_cat.predict_proba(X_val)[:, 1])
val_predictions = (val_pred_proba > threshold_ensemble).astype(int)
val_predictions_labels = le_target.inverse_transform(val_predictions)
val_submission_df = pd.DataFrame({'id': train_df.loc[X_val.index, 'id'], 'Personality': val_predictions_labels})
val_submission_df.to_csv('submission_validation.csv', index=False)
print(f"\nValidation submission file created at: submission_validation.csv")
print(f"Shape: {val_submission_df.shape}")
print(f"Sample Validation Submission Head:\n{val_submission_df.head()}")


from sklearn.preprocessing import LabelEncoder
import joblib

# Calculate target means for categorical features
target_means = {}
for col in categorical_cols:
    target_mean = train_df_original.groupby(col)['Personality'].apply(lambda x: (x == 'Extrovert').mean())
    target_means[f'{col}_target_enc'] = target_mean

# Initialize and fit polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
if len(numerical_cols_no_target) >= 2:
    top_cols = numerical_cols_no_target[:2]
    poly.fit(train_df[top_cols])
    poly_feature_names = poly.get_feature_names_out(top_cols)

# Save all components
ensemble_model_components = {
    'model_xgb': model_xgb,
    'model_lgb': model_lgb,
    'model_cat': model_cat,
    'best_weights': best_weights,
    'threshold_ensemble': threshold_ensemble,
    'le_target': le_target,
    'numerical_cols_no_target': numerical_cols_no_target,
    'categorical_cols': categorical_cols,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'all_features_union': all_features_union,
    'poly_feature_names': poly_feature_names,
    'poly': poly,
    'target_means': target_means
}

joblib.dump(ensemble_model_components, 'final_ensemble_model.pkl')


# Create a function to preprocess new input data and make predictions
def predict_personality(input_data):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Add constant feature
    input_df[CONSTANT_FEATURE_NAME] = CONSTANT_FEATURE_VALUE
    
    # Handle missing values (if any)
    for col in numerical_cols_no_target:
        if col in input_df.columns and input_df[col].isnull().any():
            input_df[col] = imputer.transform(input_df[[col]])
    
    for col in categorical_cols:
        if col in input_df.columns and input_df[col].isnull().any():
            input_df[col] = 'Missing'
    
    # Target encoding for categorical features
    for col in categorical_cols:
        if f'{col}_target_enc' in numerical_cols:
            target_mean = train_df_original.groupby(col)['Personality'].apply(lambda x: (x == 'Extrovert').mean())
            input_df[f'{col}_target_enc'] = input_df[col].map(target_mean).fillna(target_mean.mean())
    
    # One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)
    
    # Ensure all expected columns are present
    for col in all_features_union:
        if col not in input_df.columns and col != 'id':
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[all_features_union]
    
    # Interaction terms
    if len(numerical_cols_no_target) >= 2:
        for i in range(min(2, len(numerical_cols_no_target))):
            for j in range(i + 1, min(2, len(numerical_cols_no_target))):
                col1, col2 = numerical_cols_no_target[i], numerical_cols_no_target[j]
                if col1 in input_df.columns and col2 in input_df.columns:
                    input_df[f'{col1}_{col2}_inter'] = input_df[col1] * input_df[col2]
    
    # Polynomial features
    if len(numerical_cols_no_target) >= 2:
        top_cols = numerical_cols_no_target[:2]
        if all(col in input_df.columns for col in top_cols):
            poly_features = poly.transform(input_df[top_cols])
            for i, name in enumerate(poly_feature_names):
                input_df[name] = poly_features[:, i]
    
    # Scale numerical features
    input_df[numerical_cols_no_target] = scaler.transform(input_df[numerical_cols_no_target])
    
    # Make prediction
    xgb_proba = model_xgb.predict_proba(input_df)[:, 1]
    lgb_proba = model_lgb.predict_proba(input_df)[:, 1]
    cat_proba = model_cat.predict_proba(input_df)[:, 1]
    
    ensemble_proba = (best_weights['w_xgb'] * xgb_proba + 
                     best_weights['w_lgb'] * lgb_proba + 
                     best_weights['w_cat'] * cat_proba)
    
    prediction = (ensemble_proba > threshold_ensemble).astype(int)
    personality = le_target.inverse_transform(prediction)[0]
    
    # Get confidence score
    confidence = ensemble_proba[0] if personality == 'Extrovert' else 1 - ensemble_proba[0]
    
    return personality, confidence

# Example usage with your input data
example_input = {
    'Time_spent_Alone': 8.5,        # Higher value - prefers more alone time
    'Social_event_attendance': 1,    # Lower value - attends fewer social events
    'Going_outside': 2,              # Lower value - goes outside less
    'Friends_circle_size': 5,        # Smaller friend circle
    'Post_frequency': 1,             # Posts less frequently
    'Stage_fear': 'Yes',             # Has stage fear
    'Drained_after_socializing': 'Yes'
}

# Make prediction
prediction, confidence = predict_personality(example_input)
print(f"\nPredicted Personality: {prediction}")
print(f"Confidence Score: {confidence:.2f}")

