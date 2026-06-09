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


test_data = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
train_data = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train_data.head()


test_data.head()


# Importing the libreries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split


train_data.isnull().sum()


train_data.describe()


train_data.info()


catagorical_col = train_data.select_dtypes(include = 'object')
numerical_col = train_data.select_dtypes(include = 'int64')

print("Catagorical columns = ", catagorical_col.columns)
print("Numerical columns = ", numerical_col.columns)


y_counts = train_data['y'].value_counts()
plt.figure(figsize=(4, 4))

cmap = plt.get_cmap('flare')
colors = cmap(np.linspace(0, 1, len(y_counts)))

plt.pie(
    y_counts.values,
    labels=y_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    counterclock=False,
    textprops={'color': 'white'}
)
plt.title('Distribution of y')
plt.tight_layout()
plt.show()


# ---  Target Distribution ---
plt.figure(figsize=(5,4))
sns.countplot(x='y', data=train_data, palette="Set2")   # 'y' = target column
plt.title("Target Class Distribution")
plt.show()

# ---  Numerical Feature Distribution ---
plt.figure(figsize=(6,4))
sns.histplot(train_data['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Boxplot of age vs target
plt.figure(figsize=(6,4))
sns.boxplot(x='y', y='age', data=train_data, palette="Set3")
plt.title("Age vs Subscription")
plt.show()


le = LabelEncoder()
for i in catagorical_col:
    train_data[i] = le.fit_transform(train_data[i])
    test_data[i] = le.fit_transform(test_data[i])


train_data['y'] = train_data['y'].fillna(train_data['y'].mode()[0]) 
train_data['y'].shape


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X = train_data.drop(columns = ["y"], axis=1)
y = train_data["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=2)

xgb_clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Fit model
xgb_clf.fit(X_train, y_train)

# Predictions
y_pred = xgb_clf.predict(X_test)
y_prob = xgb_clf.predict_proba(X_test)[:,1]

# Evaluation
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
X_test.shape


print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Predictions length:", len(y_pred))



xgb_clf.fit(X, y)
final_pred = xgb_clf.predict_proba(test_data)[:, 1]  # returns floats
submission = pd.DataFrame({
    "id": test_data["id"],
    "y": final_pred.astype(float)
})

submission.to_csv("submission_x.csv", index=False)




