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
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

sns.set(style="whitegrid")

train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)




train_df.head()
train_df.describe()



plt.figure(figsize=(8, 6))
sns.histplot(train_df['efs'], kde=True)
plt.title("Distribution of Target Variable (efs)")
plt.xlabel("efs")
plt.ylabel("Frequency")
plt.show()



num_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    train_df[col].fillna(train_df[col].median(), inplace=True)


cat_cols = train_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    train_df[col].fillna("NaN", inplace=True)


for col in train_df.select_dtypes(include=['float64', 'int64']).columns:
    if col in test_df.columns:
        test_df[col].fillna(train_df[col].median(), inplace=True)
        
for col in train_df.select_dtypes(include=['object']).columns:
    if col in test_df.columns:
        test_df[col].fillna("NaN", inplace=True)


train_df.columns = train_df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
test_df.columns = test_df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)



X_train = train_df.drop(columns=['ID', 'efs'], errors='ignore')
y_train = train_df['efs']



X_test = test_df.drop(columns=['ID'], errors='ignore')


low_card_cols = [col for col in X_train.select_dtypes(include=['object']).columns if X_train[col].nunique() <= 10]
print("Low cardinality columns for one-hot encoding:", low_card_cols)



X_train = pd.get_dummies(X_train, columns=low_card_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=low_card_cols, drop_first=True)


X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


for col in X_train.select_dtypes(include='object').columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')



for col in X_test.select_dtypes(include='object').columns:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')



X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)



for col in X_train.columns:
    if X_train[col].isnull().any():
        X_train[col].fillna(X_train[col].median(), inplace=True)
        
for col in X_test.columns:
    if X_test[col].isnull().any():
        X_test[col].fillna(X_test[col].median(), inplace=True)



X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("Training set shape:", X_tr.shape)
print("Validation set shape:", X_val.shape)



illegal_chars = r"[,\[\]<]"


X_train.columns = X_train.columns.str.replace(illegal_chars, "_", regex=True)
X_val.columns = X_val.columns.str.replace(illegal_chars, "_", regex=True)
X_tr.columns = X_tr.columns.str.replace(illegal_chars, "_", regex=True)  

X_test.columns = X_test.columns.str.replace(illegal_chars, "_", regex=True)


print("Cleaned feature names:")
print(X_train.columns.tolist())



X_test = X_test.astype(np.float32)




import xgboost as xgb
from sklearn.metrics import roc_auc_score


model = xgb.XGBRegressor(objective='reg:squarederror',
                         n_estimators=300,
                         max_depth=6,
                         learning_rate=0.01)


model.fit(X_tr, y_tr)  


y_val_pred = model.predict(X_val)
val_auc = roc_auc_score(y_val, y_val_pred)
print("Validation ROC AUC:", val_auc)



predictions = model.predict(X_test)
print("Prediction shape:", predictions.shape)




predictions = model.predict(X_test)
print("Prediction shape:", predictions.shape)

submission = pd.DataFrame({
    'ID': test_df['ID'],  
    'prediction': predictions
})

submission.to_csv('submission.csv', index=False, encoding='utf-8')
print("Submission file saved successfully!")























