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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train_data.info()


from sklearn.preprocessing import LabelEncoder


train_data


numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
for col in numerical_cols:
    median_val = train_data[col].median()
    train_data[col] = train_data[col].fillna(median_val)

categorical_cols_to_impute = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols_to_impute:
    mode_val = train_data[col].mode()[0]
    train_data[col] = train_data[col].fillna(mode_val)


print(train_data.isnull().sum())


print(f"Unique values for Stage_fear: {train_data['Stage_fear'].unique()}")
print(f"Unique values for Drained_after_socializing: {train_data['Drained_after_socializing'].unique()}")


le = LabelEncoder()
train_data['Stage_fear'] = le.fit_transform(train_data['Stage_fear'])
train_data['Drained_after_socializing'] = le.fit_transform(train_data['Drained_after_socializing'])


train_data['Personality']


personality_le = LabelEncoder()
train_data['Personality'] = personality_le.fit_transform(train_data['Personality'])



print(f"\n'Personality' (target) encoded. Mapping: {list(personality_le.classes_)} -> {list(range(len(personality_le.classes_)))}")


train_data = train_data.drop('id', axis=1)


from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


X = train_data.drop('Personality', axis=1)
y = train_data['Personality'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y ensures that the proportion of Introvert/Extrovert is the same in train and test sets.
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")


model = CatBoostClassifier(
    iterations=1000,           # Number of boosting iterations (trees)
    learning_rate=0.05,        # Step size shrinkage to prevent overfitting
    depth=6,                   # Depth of the tree (max depth is 16)
    loss_function='Logloss',   # Loss function for binary classification
    eval_metric='Accuracy',    # Metric to monitor during training
    random_seed=42,            # For reproducibility
    verbose=0,                 # Suppress output during training
    early_stopping_rounds=50   # Stop if validation metric doesn't improve for 50 iterations
)


model.fit(X_train, y_train,
          eval_set=(X_test, y_test),
          verbose=False) 


print(f"Best iteration: {model.get_best_iteration()} out of {model.get_param('iterations')} iterations.")


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]


target_names = personality_le.inverse_transform([0, 1])


print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


feature_importances = model.get_feature_importance(Pool(X_train, y_train))
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('CatBoost Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


test_data


numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
for col in numerical_cols:
    median_val = test_data[col].median()
    test_data[col] = test_data[col].fillna(median_val)

categorical_cols_to_impute = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols_to_impute:
    mode_val = test_data[col].mode()[0]
    test_data[col] = test_data[col].fillna(mode_val)


print(test_data.isnull().sum())


print(f"Unique values for Stage_fear: {test_data['Stage_fear'].unique()}")
print(f"Unique values for Drained_after_socializing: {test_data['Drained_after_socializing'].unique()}")


le = LabelEncoder()
test_data['Stage_fear'] = le.fit_transform(test_data['Stage_fear'])
test_data['Drained_after_socializing'] = le.fit_transform(test_data['Drained_after_socializing'])


test_data


test_features = test_data.drop(columns='id', axis = 1)


pred = model.predict(test_features)
pred_proba = model.predict_proba(test_features)[:, 1]


pred


pred_proba


target_names = personality_le.inverse_transform(pred)


submission = pd.DataFrame({
    'id': test_data['id'],
    'Personality': target_names
})
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("Prediction file has been created")

