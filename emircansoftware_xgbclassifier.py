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


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train.info()


train.drop("id",axis=1,inplace=True)


train.info()


test.info()


test.drop("id",axis=1,inplace=True)


cat_cols = train.select_dtypes(include="object").columns.tolist()
cat_cols.remove("Personality")


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
for i in cat_cols:
    encoder = OrdinalEncoder()
    train[cat_cols] = encoder.fit_transform(train[cat_cols])
    test[cat_cols] = encoder.transform(test[cat_cols])


le=LabelEncoder()
X=train.drop("Personality",axis=1)
y=train["Personality"]
y=le.fit_transform(y)


params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}
# params are from Can Özensoy


from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(test)

    model = xgb.train(params, dtrain, num_boost_round=100,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=10, verbose_eval=False)
    
    oof_preds[val_idx] = model.predict(dval) > 0.5

    test_preds += model.predict(dtest) / skf.n_splits


from sklearn.metrics import accuracy_score
cv_acc = accuracy_score(y, oof_preds)
print(f"Cross-Validation Accuracy: {cv_acc:.4f}")

submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

final_preds = (test_preds > 0.5).astype(int)
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
submission.head()




