import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the data
INPUT_DATA_PATH = '/kaggle/input/playground-series-s5e7/'
train = pd.read_csv(INPUT_DATA_PATH + 'train.csv')
test = pd.read_csv(INPUT_DATA_PATH + 'test.csv')

print(f"Dataset Shape: {train.shape}")
print("\nData Info:")
train.info()

print("\nFirst 5 Rows of the Dataset:")
display(train.head(5))

# Drop ID column from train
train = train.drop(columns='id')
train.head()


# Check for missing values
print("\nMissing values in train:")
print(train.isnull().sum())

# Check for Feature types and unique values in each column
numerical_features = train.select_dtypes(include=['number']).columns
categorical_cols = train.select_dtypes(exclude=['number']).columns

pd.concat([
    pd.DataFrame(train.dtypes, columns=['dtypes']),
    pd.DataFrame(train.nunique(), columns=['n_unique'])
], axis=1).iloc[1:]


# Target distribution
plt.figure(figsize=(5,3))
sns.countplot(data=train, x='Personality')
plt.title("Target Distribution")
plt.show()

# Summary statistics for numerical features
print("\nSummary statistics:")
display(train.describe())


# Feature correlation (numerical features only)
numerical_features = train.select_dtypes(include=[np.number]).columns.tolist()

plt.figure(figsize=(5,4))
sns.heatmap(train[numerical_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Numerical Feature Correlation")
plt.show()

for feature in numerical_features:
    plt.figure(figsize=(8, 3))
    plt.suptitle(f"Distribution of {feature}", fontsize=14, y=0.92)

    plt.subplot(1, 2, 1)
    sns.histplot(train[feature], kde=True, bins=30)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    
    skew_val = train[feature].skew()
    missing_val = train[feature].isnull().sum()
    plt.text(
        0.95, 0.95,
        f"Skewness: {skew_val:.2f}\nMissing: {missing_val}",
        ha='right', va='top',
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray')
    )

    plt.subplot(1, 2, 2)
    sns.boxplot(x=train['Personality'], y=train[feature])
    plt.tight_layout()
    plt.show()


# Check categorical/object features (if any)
cat_features = train.select_dtypes(include=['object']).columns.tolist()
cat_features = [f for f in cat_features if f not in ['Personality']]

if cat_features:
    for col in cat_features:
        plt.figure(figsize=(6,3))
        sns.countplot(data=train, x=col, hue='Personality')
        plt.title(f"{col} by Personality")
        plt.show()
        display(pd.DataFrame(train[col].value_counts()).reset_index())


# Identify numeric and non-numeric features
num_features = train.select_dtypes(include=[np.number]).columns.tolist()
num_features = [col for col in num_features if col not in ['id']]

cat_features = [col for col in train.columns if col not in num_features + ['id', 'Personality']]

# Fill numeric features with median
for col in num_features:
    median_value = train[col].median()
    train[col] = train[col].fillna(median_value)
    test[col] = test[col].fillna(median_value)


!pip install xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

print("\n=== DATA PREPROCESSING ===")

# Separate features and target
X_train = train.drop(['id', 'Personality'], axis=1, errors='ignore')
y_train = train['Personality'] if 'Personality' in train.columns else None
X_test = test.drop(['id'], axis=1, errors='ignore')

print(f"Features shape: {X_train.shape}")
print(f"Target shape: {y_train.shape if y_train is not None else 'None'}")

# Encode categorical variables
label_encoders = {}
categorical_columns = X_train.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {le.classes_}")

# Encode target variable
if y_train is not None:
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)
    print(f"Target classes: {target_encoder.classes_}")
else:
    y_train_encoded = np.random.choice([0, 1], X_train.shape[0])  # For demo


!pip install --upgrade xgboost


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Define parameters (put eval_metric here)
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',  # <- keep it here
    'max_leaves': 30,
    'n_estimators': 10000,
    'random_state': 42,
    'tree_method': 'hist',
    'device': "cuda"
}

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"Performing {n_splits}-fold Stratified Cross Validation...")

cv_scores = []
fold_num = 1

for train_idx, val_idx in skf.split(X_train, y_train_encoded):
    print(f"\nTraining Fold {fold_num}/{n_splits}...")

    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_encoded[train_idx], y_train_encoded[val_idx]

    fold_model = xgb.XGBClassifier(**xgb_params)

    # Fit without eval_metric in fit() for older versions
    fold_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )

    fold_predictions = fold_model.predict(X_fold_val)
    fold_accuracy = accuracy_score(y_fold_val, fold_predictions)
    cv_scores.append(fold_accuracy)

    print(f"Fold {fold_num} Accuracy: {fold_accuracy:.4f}")
    fold_num += 1

cv_scores = np.array(cv_scores)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")



best_iterations = []
fold_num = 1

print("Extracting best iterations from each CV fold...")

for train_idx, val_idx in skf.split(X_train, y_train_encoded):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
    
    temp_model = xgb.XGBClassifier(**xgb_params)
    
    # Fit the model WITHOUT early stopping
    temp_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )
    
    # Use total number of boosting rounds used (approximate)
    best_iteration = xgb_params["n_estimators"]  # fallback since no early stopping
    best_iterations.append(best_iteration)
    
    print(f"Fold {fold_num} assumed best iteration: {best_iteration}")
    fold_num += 1

# Use average of best_iterations (though all are likely same due to no early stopping)
optimal_n_estimators = int(np.mean(best_iterations))
print(f"\nOptimal n_estimators (average): {optimal_n_estimators}")
print(f"Range: {min(best_iterations)} - {max(best_iterations)}")

# Final model with adjusted estimator count
print(f"\nTraining final model on full dataset with {optimal_n_estimators} estimators...")
xgb_params_final = xgb_params.copy()
xgb_params_final['n_estimators'] = optimal_n_estimators

xgb_model_final = xgb.XGBClassifier(**xgb_params_final)
xgb_model_final.fit(X_train, y_train_encoded)

print("âœ… Final model trained on 100% of training data!")

# Feature importances
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("ğŸ”� Top Important Features:")
display(feature_importance.head(10))


test_predictions = xgb_model_final.predict(X_test)
test_pred_proba = xgb_model_final.predict_proba(X_test)

test_pred_labels = target_encoder.inverse_transform(test_predictions)

print("\n=== CREATING SUBMISSION FILE ===")
submission_df = pd.DataFrame({
    'id': test['id'],
    'Personality': test_pred_labels
})
submission_df.to_csv('submission_xgb.csv', index=False)
print("Submission file saved as 'submission_xgb.csv'")


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Encode categorical features using LabelEncoder
for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Encode target variable
target_col = 'Personality'
id_col = 'id'
le_target = LabelEncoder()
train[target_col] = le_target.fit_transform(train[target_col])  # Extrovert/Introvert to 1/0

# Features for modeling
features = [col for col in train.columns if col not in [id_col, target_col]]

X = train[features].values
y = train[target_col].values
X_test = test[features].values

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((test.shape[0], 5))
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nTraining fold {fold+1}/5...")
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]


from scipy.stats import mode

# Majority voting (row-wise mode over the 5 folds)
test_preds_majority = mode(test_preds, axis=1, keepdims=False)[0].astype(int)


test_preds_labels = le_target.inverse_transform(test_preds_majority)


submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_preds_labels
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")

