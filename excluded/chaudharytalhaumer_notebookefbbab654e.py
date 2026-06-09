# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Show first few rows
print("Training Data:")
display(train_df.head())
print("\nTest Data:")
display(test_df.head())

# Check data types
print("\nData Types:")
display(train_df.dtypes)
sample_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


# Check for missing values
print("Missing values in train set:")
display(train_df.isna().sum())
print("\nMissing values in test set:")
display(test_df.isna().sum())




# Fill numerical missing values with median
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 
           'Going_outside', 'Friends_circle_size', 'Post_frequency']
for col in num_cols:
    test_df[col] = test_df[col].fillna(test_df[col].median())

# Fill categorical missing values with mode
cat_cols = ['Stage_fear', 'Drained_after_socializing']
for col in cat_cols:
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

# Verify no missing values remain
print("\nAfter cleaning - missing values in test set:")
display(test_df.isna().sum())

# Fill numerical missing values with median
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 
           'Going_outside', 'Friends_circle_size', 'Post_frequency']
for col in num_cols:
    train_df[col] = train_df[col].fillna(train_df[col].median())

for col in cat_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])

# Verify no missing values remain
print("\nAfter cleaning - missing values in test set:")
display(test_df.isna().sum())



train_df[train_df.duplicated(keep=False)]


# Create interaction feature
train_df['Social_Activity_Ratio'] = train_df['Social_event_attendance'] / (train_df['Time_spent_Alone'] + 1e-6)
test_df['Social_Activity_Ratio'] = test_df['Social_event_attendance'] / (test_df['Time_spent_Alone'] + 1e-6)

# Bin post frequency
bins = [0, 25, 50, 75, 100]
train_df['Post_Freq_Bin'] = pd.cut(train_df['Post_frequency'], bins=bins)
test_df['Post_Freq_Bin'] = pd.cut(test_df['Post_frequency'], bins=bins)


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 
           'Going_outside', 'Friends_circle_size', 'Post_frequency']
corr_matrix = train_df[num_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Between Numerical Features')
plt.show()


for col in cat_cols:
    plt.figure(figsize=(12,5))
    
    # Proportion plot
    ax1 = plt.subplot(121)
    prop_df = train_df.groupby(col)['Personality'].value_counts(normalize=True).unstack()
    prop_df.plot(kind='bar', stacked=True, ax=ax1)
    plt.title(f'{col} vs Personality Proportions')
    
    # Count plot
    ax2 = plt.subplot(122)
    sns.countplot(data=train_df, x=col, hue='Personality')
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
    sns.boxplot(data=train_df, x='Personality', y=col)
    plt.title(f"{col} Distribution by Personality")
    plt.show()


# Encode categorical features
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Encode target variable
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df['Personality'])
print("Class mapping:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

# Prepare final feature sets
X_train = train_df[num_cols + cat_cols + ['Social_Activity_Ratio']]
X_test = test_df[num_cols + cat_cols + ['Social_Activity_Ratio']]


# Initialize and evaluate
rf_model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"RandomForest CV Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

# Feature importance
rf_model.fit(X_train, y_train)
feat_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top Important Features')
plt.show()


# Cross-validation
xgb_model = XGBClassifier()
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
print(f"XGBoost CV Accuracy: {np.mean(xgb_scores):.2f} ± {np.std(xgb_scores):.2f}")

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_


# Train final model with best parameters
final_model = XGBClassifier(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    random_state=42
)
final_model.fit(X_train, y_train)

# Evaluate
train_preds = final_model.predict(X_train)
print("=== Training Performance ===")
print(classification_report(y_train, train_preds, target_names=le_target.classes_))

# If test labels available
if 'Personality' in test_df.columns:
    y_test = le_target.transform(test_df['Personality'])
    test_preds = final_model.predict(X_test)
    print("\n=== Test Performance ===")
    print(classification_report(y_test, test_preds, target_names=le_target.classes_))
    
    # Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, test_preds), 
                annot=True, fmt='d', 
                xticklabels=le_target.classes_, 
                yticklabels=le_target.classes_)
    plt.title('Confusion Matrix')
    plt.show()


# Generate final predictions
test_predictions = final_model.predict(X_test)
test_predictions_decoded = le_target.inverse_transform(test_predictions)

# Prepare submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_predictions_decoded
})
submission.to_csv('final_submission.csv', index=False)




# Identify categorical features (by column index)
cat_features = [X_train.columns.get_loc(col) for col in cat_cols] 

# Initialize
cat_model = CatBoostClassifier(
    random_seed=42,
    verbose=0,  # Set to 100 for training logs
    auto_class_weights='Balanced'
)

# Train with categorical features
cat_model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test) if 'Personality' in test_df.columns else None
)



cv_scores = []
cv = StratifiedKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    cat_model.fit(
        X_tr, y_tr,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    score = cat_model.score(X_val, y_val)
    cv_scores.append(score)
    print(f"Fold {fold+1} Accuracy: {score:.4f}")

print(f"\nCatBoost CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")


param_grid = {
    'iterations': [100, 200],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1],
    'l2_leaf_reg': [1, 3, 5]
}

grid_search = GridSearchCV(
    CatBoostClassifier(random_seed=42, silent=True),
    param_grid,
    cv=3,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train, cat_features=cat_features)

print("Best CatBoost params:", grid_search.best_params_)
best_cat = grid_search.best_estimator_


plt.figure(figsize=(15,5))

# CatBoost Importance
plt.subplot(121)
cat_importance = pd.Series(best_cat.feature_importances_, index=X_train.columns)
cat_importance.sort_values().plot.barh(title='CatBoost Feature Importance')

# XGBoost Importance
plt.subplot(122)
xgb_importance = pd.Series(final_model.feature_importances_, index=X_train.columns)
xgb_importance.sort_values().plot.barh(title='XGBoost Feature Importance')

plt.tight_layout()
plt.show()


from sklearn.metrics import accuracy_score, roc_auc_score

models = {
    'XGBoost': final_model,
    'CatBoost': best_cat
}

results = []
for name, model in models.items():
    # Training evaluation
    train_preds = model.predict(X_train)
    train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
    
    results.append({
        'Model': name,
        'Train Accuracy': accuracy_score(y_train, train_preds),
        'Train ROC AUC': roc_auc_score(y_train, train_proba) if train_proba is not None else None,
        'CV Accuracy': np.mean(cv_scores[name]) if name in cv_scores else None
    })

# Create comparison table
performance_df = pd.DataFrame(results).set_index('Model')
display(performance_df)


# Generate predictions from both models
xgb_preds = final_model.predict(X_test)
cat_preds = best_cat.predict(X_test)

# Ensemble predictions (simple voting)
ensemble_preds = np.where((xgb_preds + cat_preds) > 1, 1, 0)

# Decode back to original labels
final_predictions = le_target.inverse_transform(ensemble_preds)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': final_predictions
})

# Add confidence scores
if hasattr(final_model, 'predict_proba'):
    submission['confidence'] = np.max(final_model.predict_proba(X_test), axis=1)

submission.to_csv('final_ensemble_submission.csv', index=False)
print("Submission saved with ensemble predictions!")




