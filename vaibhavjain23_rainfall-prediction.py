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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score , classification_report , roc_auc_score , confusion_matrix
import xgboost as xgb
import catboost as cb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_df.head()


print("Null values in train_data")
print(train_df.isnull().sum())


print("Null values in test_data")
print(test_df.isnull().sum())


imputer = SimpleImputer(strategy = 'most_frequent')
test_df['winddirection'] = imputer.fit_transform(test_df[['winddirection']])


train_df.describe()


plt.figure(figsize=(6,4))
sns.countplot(x = 'rainfall' , data = train_df, palette = 'coolwarm')
plt.title("Target Distribution")
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(train_df.corr(),cmap='coolwarm',annot=True,fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()


X = train_df.drop(columns=['rainfall','id'])
y = train_df['rainfall']
X_test = test_df.drop(columns=['id'])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.fit_transform(X_test)


X_train,X_val,y_train,y_val = train_test_split(X_scaled,y,test_size=0.2,random_state=42,stratify=y)


models = {
    'Random Forest' : RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42),

    'Gradient Boosting' : GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=5,
                                                    random_state=42),

    'XGBoost' : xgb.XGBClassifier(n_estimators=200,learning_rate=0.05,max_depth=5,use_label_encoder=
                                 False,eval_metrics='logloss',random_state=42),

    'CatBoost' : cb.CatBoostClassifier(iterations=200,learning_rate=0.05,depth=5,verbose=0,
                                      random_state=42),

    'LightGBM' : LGBMClassifier(n_estimators=200,learning_rate=0.05,max_depth=5,random_state=42)
}


for name,model in models.items():
    print(f"\n Training {name}...")
    model.fit(X_train,y_train)
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    print(f"Accuracy: {accuracy_score(y_val,y_pred)}")
    print(f"ROC AUC Score: {roc_auc_score(y_val, y_pred_proba)}")
    print(f"Classification Report:\n{classification_report(y_val, y_pred)}")
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_val,y_pred),annot=True,fmt='d',cmap='coolwarm',cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()


best_model = max(models.items(), key=lambda x: roc_auc_score(y_val, x[1].predict_proba(X_val)[:, 1]))[1]


print(best_model)


test_predictions = best_model.predict_proba(X_test_scaled)[:, 1]


submission = pd.DataFrame({'id': test_df['id'], 'rainfall': test_predictions})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

