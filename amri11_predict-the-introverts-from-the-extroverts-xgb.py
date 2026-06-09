


import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import optuna
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


%%time

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head()


test.head()


train.info()


test.info()


train.isna().sum()


test.isna().sum()


train.fillna(train.median(numeric_only=True), inplace=True)

train['Stage_fear'] = train.groupby('Friends_circle_size')['Stage_fear'].transform(
    lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
train['Drained_after_socializing'] = train.groupby('Friends_circle_size')['Drained_after_socializing'].transform(
    lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
train.isna().sum()


test.fillna(test.median(numeric_only=True), inplace=True)

test['Stage_fear'] = test.groupby('Friends_circle_size')['Stage_fear'].transform(
    lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
test['Drained_after_socializing'] = test.groupby('Friends_circle_size')['Drained_after_socializing'].transform(
    lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
test.isna().sum()


train.describe()


train.describe(include = 'object')


test.describe()


test.describe(include = 'object')


personality_numeric = train.select_dtypes(include=['number'])

# Calculate correlation
corr_matrix = personality_numeric.corr()

# Plot heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Correlation")
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Personality'], y=train['Time_spent_Alone'], palette='coolwarm')
plt.xlabel('Personality')
plt.ylabel('Time Spent Alone')
plt.title('Boxplot Time Spent Alone vs Personality')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Personality'], y=train['Social_event_attendance'], palette='coolwarm')
plt.xlabel('Personality')
plt.ylabel('Social Event Attendence')
plt.title('Boxplot Social Event Attendence vs Personality')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Personality'], y=train['Going_outside'], palette='coolwarm')
plt.xlabel('Personality')
plt.ylabel('Going Outside')
plt.title('Boxplot Going Outside vs Personality')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Personality'], y=train['Friends_circle_size'], palette='coolwarm')
plt.xlabel('Personality')
plt.ylabel('Friends Circle Size')
plt.title('Boxplot Friends Circle Size vs Personality')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Personality'], y=train['Post_frequency'], palette='coolwarm')
plt.xlabel('Personality')
plt.ylabel('Post Frequency')
plt.title('Boxplot Post Frequency vs Personality')
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(x='Personality', hue='Stage_fear', data=train)
plt.title('Personality by Stage Fear')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(x='Personality', hue='Drained_after_socializing', data=train)
plt.title('Personality by Drained After Socializing')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.show()


train['Stage_fear'] = train['Stage_fear'].map({'Yes': 1, 'No': 0, 'Unknown': -1})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'Yes': 1, 'No': 0, 'Unknown': -1})
train['Alone_to_Social_Ratio'] = train['Time_spent_Alone'] / (train['Social_event_attendance'] + 1)
train['Activity_Level'] = train['Going_outside'] + train['Social_event_attendance']
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency', 'Alone_to_Social_Ratio', 'Activity_Level']

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])

test['Stage_fear'] = test['Stage_fear'].map({'Yes': 1, 'No': 0, 'Unknown': -1})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes': 1, 'No': 0, 'Unknown': -1})
test['Alone_to_Social_Ratio'] = test['Time_spent_Alone'] / (test['Social_event_attendance'] + 1)
test['Activity_Level'] = test['Going_outside'] + test['Social_event_attendance']
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency', 'Alone_to_Social_Ratio', 'Activity_Level']

scaler = StandardScaler()
test[num_cols] = scaler.fit_transform(test[num_cols])


train['Personality'] = train['Personality'].map({'Introvert': 1, 'Extrovert': 0})

X = train.drop(['Personality', 'id'], axis=1)
y = train['Personality']

test_final = test.drop('id', axis=1)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X):
    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


model = XGBClassifier(
    n_estimators=2000,
    max_depth=15,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=101
)

# Train model
model.fit(x_train, y_train)


# Feature Importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight')
plt.title("Feature Importance - XGBoost")
plt.show()


y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

print("Accuracy of Conversion Prediction Model:", accuracy_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("Report Classification:")
print(classification_report(y_test, y_pred))


y_test_pred = model.predict(test_final)
label_map = {0: 'Extrovert', 1: 'Introvert'}
y_test_label = [label_map[val] for val in y_test_pred]

submission = pd.DataFrame({
    'id': test['id'],        # X_test_raw = sebelum drop kolom
    'Personality': y_test_label
})

submission.to_csv('submission.csv', index=False)
submission.head()

