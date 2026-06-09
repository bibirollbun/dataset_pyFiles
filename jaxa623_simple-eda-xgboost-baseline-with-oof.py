import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


# Load the data
INPUT_DATA_PATH = '/kaggle/input/playground-series-s5e7/'
train = pd.read_csv(INPUT_DATA_PATH + 'train.csv')
test = pd.read_csv(INPUT_DATA_PATH + 'test.csv')


# Basic info
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:", train.columns.tolist())


# Display the first few rows
print("\nFirst 5 rows of train:")
display(train.head())


# Check for missing values
print("\nMissing values in train:")
print(train.isnull().sum())


# Visualize the target distribution
plt.figure(figsize=(5,3))
sns.countplot(data=train, x='Personality')
plt.title("Target Distribution")
plt.show()


# Feature types
print("\nFeature data types:")
print(train.dtypes)

# Check for unique values in each column (to spot categorical features)
print("\nUnique value count in each column:")
print(train.nunique())


# Summary statistics for numerical features
print("\nSummary statistics:")
display(train.describe())

# Feature correlation (numerical features only)
numerical_features = train.select_dtypes(include=[np.number]).columns.tolist()
numerical_features = [f for f in numerical_features if f not in ['id']]  # exclude id

plt.figure(figsize=(10,8))
sns.heatmap(train[numerical_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Numerical Feature Correlation")
plt.show()


# Distribution plots for numerical features
for col in numerical_features:
    plt.figure(figsize=(6,3))
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# Boxplot of numerical features by target class
for col in numerical_features:
    plt.figure(figsize=(6,3))
    sns.boxplot(data=train, x='Personality', y=col)
    plt.title(f"{col} by Personality")
    plt.show()

# Check categorical/object features (if any)
cat_features = train.select_dtypes(include=['object']).columns.tolist()
cat_features = [f for f in cat_features if f not in ['Personality']]
if cat_features:
    print("\nCategorical features:", cat_features)
    for col in cat_features:
        print(f"\nValue counts for {col}:")
        print(train[col].value_counts())
        plt.figure(figsize=(6,3))
        sns.countplot(data=train, x=col, hue='Personality')
        plt.title(f"{col} by Personality")
        plt.show()
else:
    print("\nNo categorical features except target.")

print("\nEDA complete.")


# Calculate missing percentage for each feature
missing_percent = train.isnull().mean() * 100
print("\nMissing value percentage per column:")
print(missing_percent)

# Visualize missing values as a bar chart
plt.figure(figsize=(8,4))
missing_percent[:-1].sort_values(ascending=False).plot(kind='bar')  # exclude target if desired
plt.title("Missing Value Percentage per Feature")
plt.ylabel("Percent (%)")
plt.show()


# Identify numeric and non-numeric features
num_features = train.select_dtypes(include=[np.number]).columns.tolist()
num_features = [col for col in num_features if col not in ['id']]
cat_features = [col for col in train.columns if col not in num_features + ['id', 'Personality']]

# Fill numeric features with median
for col in num_features:
    median_value = train[col].median()
    train[col] = train[col].fillna(median_value)
    test[col] = test[col].fillna(median_value)

# Fill categorical features with mode (most frequent value)
for col in cat_features:
    mode_value = train[col].mode().iloc[0]
    train[col] = train[col].fillna(mode_value)
    test[col] = test[col].fillna(mode_value)


# Encode categorical features, if any, using LabelEncoder
for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Encode target variable
target_col = 'Personality'
id_col = 'id'
le_target = LabelEncoder()
train[target_col] = le_target.fit_transform(train[target_col])  # Extrovert/Introvert to 1/0


xgb_params = {
    'eval_metric': 'logloss',
    'max_leaves': 25,
    'min_child_weight': np.float64(0.003440906647223279),
    'learning_rate': np.float64(0.09470087254583547),
    'n_estimators': 10000,
    'subsample': np.float64(0.8025291728808135),
    'colsample_bylevel': np.float64(0.8360122952647302),
    'colsample_bytree': np.float64(0.87329448975438),
    'reg_alpha': np.float64(0.002926163798802797),
    'reg_lambda': np.float64(27.126259438996986),
    'random_state': 42,
    'tree_method': 'hist',
    'device': "cpu"
}


# Features for modeling
features = [col for col in train.columns if col not in [id_col, target_col]]

X = train[features].values
y = train[target_col].values
X_test = test[features].values

# 3. 5-Fold Stratified Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((test.shape[0], 5))
oof_pred = np.zeros((len(train),))             # OOF hard label
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nTraining fold {fold+1}/5...")
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=20
    )
    val_pred_logit = model.predict_proba(X_val)
    val_pred = np.argmax(val_pred_logit, -1)
    acc = accuracy_score(y_val, val_pred)
    oof_pred[val_idx] = model.predict_proba(X_val)[:, 1]
    print(f"Fold {fold+1} Validation Accuracy: {acc:.4f}")
    val_scores.append(acc)

    # Test prediction for this fold
    # test_preds[:, fold] = model.predict_proba(X_test)[:, 1]

val_scores = np.array(val_scores)
print(f"Cross-validation scores: {val_scores}")
print("\nCV Mean Accuracy: {:.4f} | Std: {:.4f}".format(np.mean(val_scores), np.std(val_scores)))


best_iterations = []
fold_num = 1

print("Extracting best iterations from each CV fold...")
for train_idx, val_idx in kf.split(X, y):
    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]
    
    # Create temporary model to find best iteration
    temp_model = XGBClassifier(**xgb_params)
    temp_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    best_iterations.append(temp_model.best_iteration)
    print(f"Fold {fold_num} best iteration: {temp_model.best_iteration}")
    fold_num += 1

# Use average best iteration for final model
optimal_n_estimators = int(np.mean(best_iterations))
print(f"\nOptimal n_estimators (average): {optimal_n_estimators}")
print(f"Range: {min(best_iterations)} - {max(best_iterations)}")

# Train final model on full dataset with optimal n_estimators
print(f"\nTraining final model on full dataset with {optimal_n_estimators} estimators...")
xgb_params_final = xgb_params.copy()
xgb_params_final['n_estimators'] = optimal_n_estimators

xgb_model_final = XGBClassifier(**xgb_params_final)
xgb_model_final.fit(X, y)

print("Final model trained on 100% of training data!")


feature_importance = pd.DataFrame({
    'feature': features,
    'importance': xgb_model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))


test_predictions = xgb_model_final.predict(X_test)
test_pred_proba = xgb_model_final.predict_proba(X_test)

if train[target_col] is not None:
    test_pred_labels = le_target.inverse_transform(test_predictions)
else:
    test_pred_labels = ['Introvert' if pred == 0 else 'Extrovert' for pred in test_predictions]


np.save("oof.npy", oof_pred)
np.save("pred.npy", test_pred_proba[:,-1])


# Create Submission File
print("\n=== CREATING SUBMISSION FILE ===")
submission_df = pd.DataFrame({
    'id': test['id'],
    'Personality': test_pred_labels
})
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")




