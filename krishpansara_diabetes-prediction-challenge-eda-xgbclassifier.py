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


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df.head()


df.info()


df.describe()


df.isnull().sum()


TARGET = 'diagnosed_diabetes'
BASE = [col for col in df.columns if col not in ['id', TARGET]]
CATS = df.select_dtypes('object').columns.to_list()
NUMS = [col for col in BASE if col not in CATS]
print(f'{len(BASE)} Base Features:{BASE}')


df[CATS].nunique()


df[NUMS].nunique()





import matplotlib.pyplot as plt
import seaborn as sns


sns.countplot(data=df, x=TARGET )


# sns.kdeplot(data=df, x=TARGET)


corr_features = NUMS + [TARGET]
corr_matrix = df[corr_features].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
            cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix (Numerical Features vs Target)')
plt.show()


BASE





from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score


x = pd.get_dummies(df[BASE], columns=CATS, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(x, df[TARGET], random_state=42, test_size=0.2)
X_train.head()


y_train


model = XGBClassifier()
    
model.fit(
    X_train, y_train,
    # early_stopping_rounds=200,
    verbose=500
)


y_pred = model.predict(X_test)


y_pred


y_test


roc_auc_score(y_test, y_pred)


# gender: object, ethnicity: object, education_level: object, income_level: object, smoking_status: object, employment_status: object


test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv') 


test_df.head()


tst_X = pd.get_dummies(test_df, columns=CATS, drop_first=True)



tst_X.head()


test_X_data = tst_X.iloc[:,1:]
test_X_data.head()


# Make predictions
sub_y_pred = model.predict(test_X_data)


sub_y_pred


# Create submission file
submission = pd.DataFrame({
    "id": tst_X['id'],
    "target": sub_y_pred 
})

submission.to_csv("submission.csv", index=False)





