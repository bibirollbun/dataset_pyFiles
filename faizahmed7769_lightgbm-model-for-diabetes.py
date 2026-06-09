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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_auc_score,RocCurveDisplay
import lightgbm as lgb


train_data = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train_data.info()


train_data.describe()


corr = train_data.corr(numeric_only=True)
plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')


train_data.columns


x = train_data.drop(['id','diagnosed_diabetes'], axis=1)
y = train_data['diagnosed_diabetes'] 


X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

numerical_features = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = x.select_dtypes(include=['object']).columns.tolist()

preprocessing = ColumnTransformer(transformers=[
    ('num', 'passthrough', numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])


model2 = lgb.LGBMClassifier(
    boosting_type='gbdt',
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42
)

pipe2 = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('model', model2)
])


pipe2.fit(X_train, y_train)


y_pred_proba2 = pipe2.predict_proba(X_val)[:, 1]


roc = roc_auc_score(y_val, y_pred_proba2)
print(f'ROC AUC Score: {roc:.4f}')


disp = RocCurveDisplay.from_predictions(
    y_val,
    pipe2.predict_proba(X_val)[:, 1],
    name='LightGBM Classifier'
)

plt.title("ROC Curve - LightGBM")
plt.show()



X_test = test_data.drop(columns=['id'])
y_test_proba = pipe2.predict_proba(X_test)[:, 1]

pipe2.fit(x,y)


test_proba = pipe2.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    'id': test_data['id'],
    'diagnosed_diabetes': test_proba
})


plt.hist(y_test_proba, bins=50)
plt.title("Test Prediction Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.show()


submission.to_csv("submission.csv", index=False)

