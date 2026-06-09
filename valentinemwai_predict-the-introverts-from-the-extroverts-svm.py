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
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")



print(train.head())
print(train.info())


print(train.isnull().sum())


train.describe()


train_df=train.dropna()
train_df.info()


#visualizing Personality
plt.figure(figsize=(10, 6))
sns.countplot(x='Personality', data=train_df, palette='viridis')

# Add title and labels
plt.title('Distribution of Personality Types', fontsize=16)
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Display the plot
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Set style
sns.set(style="whitegrid")

# Numerical columns
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
            'Friends_circle_size', 'Post_frequency']

# Plot boxplots for each numerical feature by Personality
plt.figure(figsize=(15, 20))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x='Personality', y=col, data=train_df, palette='Set3')
    plt.title(f'{col} by Personality', fontsize=12)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



# Categorical columns
cat_cols = ['Stage_fear', 'Drained_after_socializing']

# Plot countplots for each categorical feature by Personality
plt.figure(figsize=(12, 10))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(2, 1, i)
    sns.countplot(x=col, hue='Personality', data=train_df, palette='muted')
    plt.title(f'{col} by Personality')
    plt.legend(title='Personality')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



#Distribution of time spent alone



train_df.head()


df = train_df.drop(columns=['id'])



from sklearn.preprocessing import LabelEncoder

label_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])


df.head()


from sklearn.preprocessing import StandardScaler

num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


X = df.drop(columns=['Personality'])  # features
y = df['Personality']                # target



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)



import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split



model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



print(test.head())
print(test.info())


print(test.isnull().sum())


test_df=test

test_df.info()


test_new = test_df.drop(columns=['id'])


from sklearn.preprocessing import LabelEncoder

label_cols = ['Stage_fear', 'Drained_after_socializing']
le = LabelEncoder()
for col in label_cols:
    test_new[col] = le.fit_transform(test_new[col])


from sklearn.preprocessing import StandardScaler

num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']

scaler = StandardScaler()
test_new[num_cols] = scaler.fit_transform(test_new[num_cols])


prediction = model.predict(test_new)
prediction


# Creating submission with model predictions
submission = pd.DataFrame({'id': test_df['id'], "Personality": prediction})

# Converting 1s back to Extrovert and 0s back to Introvert
submission['Personality'].replace({0: 'Extrovert', 1: 'Introvert'}, inplace=True)


submission


submission.to_csv("submission.csv", index=False)


import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd


train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 4. Set LightGBM parameters
params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

# 5. Train the model
model_lgb = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

# 6. Predict probabilities and convert to binary labels
y_pred_proba = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC Score: {auc:.4f}")


from sklearn.metrics import classification_report, confusion_matrix

# Predict binary labels
y_pred = (y_pred_proba > 0.5).astype(int)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted: Class 0', 'Predicted: Class 1'],
            yticklabels=['Actual: Class 0', 'Actual: Class 1'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()


#Classification Report
class_report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("\nClassification Report:")
print(class_report)



prediction_lgb = model_lgb.predict(test_new)
pred_lgb = (prediction_lgb > 0.5).astype(int)
pred_lgb


# Creating submission with model predictions
submission_lgb = pd.DataFrame({'id': test_df['id'], "Personality": pred_lgb})

# Converting 1s back to Extrovert and 0s back to Introvert
submission_lgb['Personality'].replace({0: 'Extrovert', 1: 'Introvert'}, inplace=True)


submission_lgb


submission_lgb.to_csv("submission.csv", index=False)


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Predict binary output
y_pred = rf_model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Predicted: 0', 'Predicted: 1'],
            yticklabels=['Actual: 0', 'Actual: 1'])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 4. Predict binary labels
y_pred = svm_model.predict(X_test)

# 5. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Predicted: 0', 'Predicted: 1'],
            yticklabels=['Actual: 0', 'Actual: 1'])
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()



from sklearn.impute import SimpleImputer

# Impute missing values with the mean (or use 'median', 'most_frequent')
imputer = SimpleImputer(strategy='mean')
test_new_imputed = imputer.fit_transform(test_new)

# Predict using SVM
prediction_svm = svm_model.predict(test_new_imputed)
prediction_svm


# Creating submission with model predictions
submission_svm = pd.DataFrame({'id': test_df['id'], "Personality": prediction_svm})

# Converting 1s back to Extrovert and 0s back to Introvert
submission_svm['Personality'].replace({0: 'Extrovert', 1: 'Introvert'}, inplace=True)


submission_svm.head()


submission_svm.to_csv("submission.csv", index=False)

