


import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import optuna
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


%%time

train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train.head()


train.info()


test.head()


test.info()


train.isna().sum()


test.isna().sum()


train.describe()


train.describe(include = 'object')


bank_numeric = train.select_dtypes(include=['number'])

# Calculate correlation
corr_matrix = bank_numeric.corr()

# Plot heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Correlation")
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['y'], y=train['age'], palette='coolwarm')
plt.xlabel('Y')
plt.ylabel('Age')
plt.title('Boxplot Age vs Y')
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(x='job', hue='y', data=train)
plt.title('Job by Y')
plt.xlabel('Job')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(x='marital', hue='y', data=train)
plt.title('Marital by Y')
plt.xlabel('Marital')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(x='education', hue='y', data=train)
plt.title('Education by Y')
plt.xlabel('Education')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(x='default', hue='y', data=train)
plt.title('Default by Y')
plt.xlabel('Default')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['y'], y=train['balance'], palette='coolwarm')
plt.xlabel('Y')
plt.ylabel('Balance')
plt.title('Boxplot Balance vs Y')
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(x='housing', hue='y', data=train)
plt.title('Housing by Y')
plt.xlabel('Housing')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(x='loan', hue='y', data=train)
plt.title('Loan by Y')
plt.xlabel('Loan')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(x='contact', hue='y', data=train)
plt.title('Contact by Y')
plt.xlabel('Contact')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['y'], y=train['day'], palette='coolwarm')
plt.xlabel('Y')
plt.ylabel('Day')
plt.title('Boxplot Day vs Y')
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(x='month', hue='y', data=train)
plt.title('Month by Y')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['y'], y=train['duration'], palette='coolwarm')
plt.xlabel('Y')
plt.ylabel('Duration')
plt.title('Boxplot Duration vs Y')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['y'], y=train['campaign'], palette='coolwarm')
plt.xlabel('Y')
plt.ylabel('Campaign')
plt.title('Boxplot Campaign vs Y')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['y'], y=train['pdays'], palette='coolwarm')
plt.xlabel('Y')
plt.ylabel('Pdays')
plt.title('Boxplot Pdays vs Y')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['y'], y=train['previous'], palette='coolwarm')
plt.xlabel('Y')
plt.ylabel('Previous')
plt.title('Boxplot Previous vs Y')
plt.show()


train['y'].value_counts()


cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
le = LabelEncoder()

for col in cat_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])


X = train.drop(['y', 'id'], axis=1)
y = train['y']

test_final = test.drop('id', axis=1)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X):
    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


model = XGBClassifier(
    n_estimators=2000,
    max_depth=15,
    learning_rate=0.001,
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


y_pred_proba = model.predict_proba(test_final)[:, 1]

submission = pd.DataFrame({
    'id': sample['id'],         
    'y': y_pred_proba           
})

submission.to_csv("submission.csv", index=False)
submission.head()

