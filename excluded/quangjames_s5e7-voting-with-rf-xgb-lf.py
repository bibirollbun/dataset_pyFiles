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


import pandas as pd
import numpy as np
from IPython.display import display, HTML

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance


train_file = '/kaggle/input/playground-series-s5e7/train.csv'
test_file = '/kaggle/input/playground-series-s5e7/test.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
print("-----"*10 + "Overview Train Dataset" + "------"*10)
display(HTML("<span style = 'color: blue; font-weight:bold;'> Train dataset\'s Information</span>"))
display(train_data.info())
print("-----"*10 + "Overview Test Dataset" + "------"*10)
display(HTML("<span style = 'color: red; font-weight:bold;'> Test dataset\'s Information</span>"))
display(test_data.info())


print(train_data['Personality'].value_counts())
plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=train_data, palette='Set2')
plt.title('Personality Distribution on Train set', fontsize=14, pad=15)
plt.xlabel('Personality', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


target_col = 'Personality'
id_col = 'id'
numerical_cols = [col for col in train_data.select_dtypes(include=['int64', 'float64']).columns.tolist() if col != id_col and col != target_col]

display(HTML("<span style = 'color: blue; font-weight:bold;'> Numerical features vs Target column/span>"))
for col in numerical_cols:
    plt.figure(figsize=(4, 3))
    groups = [group[col].dropna() for name, group in train_data.groupby(target_col)]    
    sns.boxplot(data = train_data, x=target_col, y=col)
    plt.title(f"{col} vs {target_col}")
    plt.show()


# Feature Engineering
train_data['Alone_to_Social_Ratio'] = train_data['Time_spent_Alone'] / (train_data['Social_event_attendance'] + 1e-5)
test_data['Alone_to_Social_Ratio'] = test_data['Time_spent_Alone'] / (test_data['Social_event_attendance'] + 1e-5)

train_data['Drained_Interaction'] = train_data['Drained_after_socializing'].astype(str) + '_' + \
                                         train_data['Time_spent_Alone'].apply(lambda x: 'High' if x > train_data['Time_spent_Alone'].median() else 'Low')
test_data['Drained_Interaction'] = test_data['Drained_after_socializing'].astype(str) + '_' + \
                                        test_data['Time_spent_Alone'].apply(lambda x: 'High' if x > train_data['Time_spent_Alone'].median() else 'Low')

# Feature and label extraction
X = train_data.drop(columns=['id', 'Personality'])
y = train_data['Personality']
test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])

# Encode the categories columns
le = LabelEncoder()
#categorical_cols = ['Stage_fear', 'Drained_after_socializing']
categorical_cols = [col for col in train_data.select_dtypes(include='object').columns.tolist() if col != id_col and col != target_col]
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test_data[col] = test_data[col].astype(str).map(lambda x: x if x in le.classes_ else 'unknown')
    if 'unknown' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'unknown')
    X_test[col] = le.transform(test_data[col])

# Encode Personality label
y = le.fit_transform(y)

# Handling missing values
#numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
numerical_cols = [col for col in train_data.select_dtypes(include=['int64', 'float64']).columns.tolist() if col != id_col and col != target_col]
imputer = SimpleImputer(strategy='mean')
X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
X_test[numerical_cols] = imputer.transform(X_test[numerical_cols])

# Normalize data for Logistic Regression
scaler = StandardScaler()
X_scaled = X.copy()
X_test_scaled = X_test.copy()
X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Split train/test data for evaluation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_unscaled, X_val_unscaled, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

display(HTML("<span style = 'color: red; font-weight:bold;'> Data preprocessing is completed</span>"))
# print("Shape of training set:", X_train.shape)
# print("Shape of validation set:", X_val.shape)
# print("Shape of unscaled training set:", X_train_unscaled.shape)
# print("Shape of unscaled validation set:", X_val_unscaled.shape)
# print("Featues of training:", X_train.columns.tolist())


# # Tinh chỉnh XGBoost
# param_grid_xgb = {
#     'n_estimators': [50, 100, 200, 300],
#     'max_depth': [3, 5, 7, 9],
#     'learning_rate': [0.01, 0.05, 0.1, 0.3],
#     'subsample': [0.6, 0.8, 1.0]
# }
# grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search_xgb.fit(X_train_unscaled, y_train)
# best_xgb_model = grid_search_xgb.best_estimator_
# print("Best XGBoost parameters:", grid_search_xgb.best_params_)



# Define models with the best hyperparameters
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    min_samples_split=5, 
    class_weight='balanced', 
    random_state=42
)
xgb_model = XGBClassifier(
    n_estimators=300, 
    max_depth=3, 
    learning_rate=0.01,
    scale_pos_weight=len(y[y == 0]) / len(y[y == 1]), 
    subsample = 0.8,
    random_state=42
)
lr_model = LogisticRegression(
    C=0.1, solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=42
)

# Create Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lr', lr_model)
    ],
    voting='soft'
)

# StratifiedKFold
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store predicted probability on test set
test_probs = np.zeros((X_test_scaled.shape[0], 2))  # 2 lớp: Introvert, Extrovert
val_accuracies = []

# Training and evaluation on each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    print(f"\nTraining on Fold {fold + 1}/{n_splits}...")
    
    # Chia dữ liệu (sử dụng .iloc để chọn hàng)
    X_train_fold = X_scaled.iloc[train_idx]
    X_val_fold = X_scaled.iloc[val_idx]
    y_train_fold = y[train_idx]
    y_val_fold = y[val_idx]
    X_train_fold_unscaled = X.iloc[train_idx]
    X_val_fold_unscaled = X.iloc[val_idx]
    
    # Train Voting Classifier
    # Use unnormalized data for Random Forest and XGBoost
    rf_model.fit(X_train_fold_unscaled, y_train_fold)
    xgb_model.fit(X_train_fold_unscaled, y_train_fold)
    lr_model.fit(X_train_fold, y_train_fold)
    voting_clf.fit(X_train_fold, y_train_fold)
    
    # Evaluation on validation set
    y_val_pred = voting_clf.predict(X_val_fold)
    fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
    val_accuracies.append(fold_accuracy)
    print(f"Fold {fold + 1} Validation Accuracy: {fold_accuracy:.4f}")
    
    # Predict probability on test set
    test_probs += voting_clf.predict_proba(X_test_scaled) / n_splits

# Average Accuracy over folds
mean_val_accuracy = np.mean(val_accuracies)
print(f"\nMean Validation Accuracy across {n_splits} folds: {mean_val_accuracy:.4f}")


# Inspect and visualize Feature Importance
feature_names = X.columns

# Feature Importance from Random Forest
rf_importances = rf_model.feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importances
}).sort_values(by='Importance', ascending=False)

# Feature Importance from XGBoost
xgb_importances = xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_importances
}).sort_values(by='Importance', ascending=False)

# Calculation for Logistic Regression
lr_coefficients = np.abs(lr_model.coef_[0])
lr_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': lr_coefficients
}).sort_values(by='Importance', ascending=False)

# Permutation Importance for Voting Classifier
perm_importance = permutation_importance(voting_clf, X_val_fold, y_val_fold, n_repeats=10, random_state=42, n_jobs=-1)
perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

# Feature Importance Visualization
plt.figure(figsize=(12, 10))

# Random Forest
plt.subplot(2, 2, 1)
sns.barplot(x='Importance', y='Feature', data=rf_importance_df, palette='viridis')
plt.title('Random Forest Feature Importance', fontsize=12)
plt.xlabel('Importance', fontsize=10)
plt.ylabel('Feature', fontsize=10)

# XGBoost
plt.subplot(2, 2, 2)
sns.barplot(x='Importance', y='Feature', data=xgb_importance_df, palette='magma')
plt.title('XGBoost Feature Importance', fontsize=12)
plt.xlabel('Importance', fontsize=10)
plt.ylabel('Feature', fontsize=10)

# Logistic Regression
plt.subplot(2, 2, 3)
sns.barplot(x='Importance', y='Feature', data=lr_importance_df, palette='coolwarm')
plt.title('Logistic Regression Coefficients (Absolute)', fontsize=12)
plt.xlabel('Importance (Absolute Coefficient)', fontsize=10)
plt.ylabel('Feature', fontsize=10)

# Permutation Importance (Voting Classifier)
plt.subplot(2, 2, 4)
sns.barplot(x='Importance', y='Feature', data=perm_importance_df, palette='Blues')
plt.title('Voting Classifier Permutation Importance', fontsize=12)
plt.xlabel('Importance', fontsize=10)
plt.ylabel('Feature', fontsize=10)

plt.tight_layout()
#plt.savefig('feature_importance_comparison.png')
plt.show()

# Feature Importance values for each model
print("\nRandom Forest Feature Importance:")
print(rf_importance_df)
print("\nXGBoost Feature Importance:")
print(xgb_importance_df)
print("\nLogistic Regression Coefficients (Absolute):")
print(lr_importance_df)
print("\nVoting Classifier Permutation Importance:")
print(perm_importance_df)


# Prediction on test set
y_test_pred = voting_clf.predict(X_test_scaled)
y_test_pred = le.inverse_transform(y_test_pred)

# Create a submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': y_test_pred
})
submission.to_csv('submission_voting.csv', index=False)
display(HTML("<span style = 'color: blue; font-weight:bold;'> File submission.csv was created!</span>"))

# Submission file checking
print(submission.head())

