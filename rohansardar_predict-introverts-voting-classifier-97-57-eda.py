import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col="id")


train.head()


cat_cols = train.select_dtypes(include=['object']).columns
num_cols = train.select_dtypes(include=['int64', 'float64']).columns

train.replace([np.inf, -np.inf], np.nan, inplace=True)
train[num_cols] = train[num_cols].fillna(train[num_cols].mean())

for col in cat_cols:
    if train[col].isnull().any():
        train[col] = train[col].fillna(train[col].mode()[0])


cat_cols = test.select_dtypes(include=['object']).columns
num_cols = test.select_dtypes(include=['int64', 'float64']).columns

test.replace([np.inf, -np.inf], np.nan, inplace=True)
test[num_cols] = test[num_cols].fillna(test[num_cols].mean())

for col in cat_cols:
    if test[col].isnull().any():
        test[col] = test[col].fillna(test[col].mode()[0])


print(f"The categorical value columns are: {cat_cols.values}")


le = LabelEncoder()
for col in cat_cols.values:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])


train.head()


train[num_cols] = train[num_cols].astype(int)
test[num_cols] = test[num_cols].astype(int)


sns.histplot(train['Personality'])
plt.title('Count of Personality')
plt.show()


plt.figure(figsize=(8, 2 * len(cat_cols)))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(1, len(cat_cols), i)  
    sns.countplot(x=train[col], hue=train['Personality'], palette='Set2')
    plt.title(f"{col} vs Personality count") 
    
plt.tight_layout()
plt.show()


n_plots = len(num_cols)
cols_per_row = math.ceil(n_plots / 2)

plt.figure(figsize=(4 * cols_per_row, 6))

for i, col in enumerate(num_cols, 1):
    plt.subplot(2, cols_per_row, i)
    sns.histplot(x=col, hue='Personality', data=train, fill=True, palette='Set2')
    plt.title(f"{col} vs Personality count")

plt.tight_layout()
plt.show()



X = train.drop('Personality', axis=1)
y = train['Personality']
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22, stratify=y)

xgb = XGBClassifier(enable_categorical=True, verbosity=0)
lgbm = LGBMClassifier(verbose=-1)
cat = CatBoostClassifier(verbose=0, allow_writing_files=False)
rf = RandomForestClassifier()
logreg = LogisticRegression()

model = VotingClassifier(estimators=[
    ('xgb', xgb),
    ('lgbm', lgbm),
    ('cat', cat),
    ('rf', rf),
    ('logreg', logreg)
], voting='soft') 

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"VotingClassifier Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Classification Report: {classification_report(y_test, y_pred)}")


test_pred = model.predict(test)
test_pred_labels = le.inverse_transform(test_pred)

sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission = pd.DataFrame({
    'id': sub['id'],
    'Personality': test_pred_labels
})

submission.to_csv('submission.csv', index=False)




