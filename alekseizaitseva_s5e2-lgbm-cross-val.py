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


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test =  pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train.head()


train.info()


train.describe(include ='O')


train.describe()


import matplotlib.pyplot as plt
import seaborn as sns
pd.option_context('mode.use_inf_as_na', True)
plt.figure(figsize=(14,7))
sns.lineplot(data=train, x='Brand', y='Price', alpha=0.5, label='Daily Sales', color='blue')



def create_features(df):
    object_columns = df.dtypes[df.dtypes == 'object'].index
    df = pd.get_dummies(data = df, columns = object_columns,  drop_first=True)
    return df


train = create_features(train)
test = create_features(test)


test = test.reindex(columns=train.columns, fill_value=0)


train.head()


test.head()


train = train.fillna(train.mean())


test.isna().sum()

test = test.fillna(test.mean())


X = train.drop(columns=["id", "Price"])
y = train["Price"]
X_test = test.drop(columns=["id"])




import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from lightgbm import early_stopping
from lightgbm import log_evaluation


common_columns = X.columns.intersection(X_test.columns)
X = X[common_columns]
X_test = X_test[common_columns]

print("Number of features after alignment:", len(common_columns))
print("Common columns:", common_columns.tolist())


kf = KFold(n_splits=5, shuffle=True, random_state=42)


final_predictions = np.zeros(X_test.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=100)
        ]
    )

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    final_predictions += test_preds / kf.n_splits

    print(f'Fold {fold + 1} MAE: {mean_absolute_error(y_val, val_preds):.4f}')
    print()

print("Training completed.")


test["Price"] = final_predictions
submission = test[["id", "Price"]]
submission.to_csv("submission.csv", index=False)
print("Submission file created!")


fi = pd.DataFrame(data=model.feature_importances_,
             index=model.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()

