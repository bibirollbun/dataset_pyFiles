# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to loads

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


df=pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")



original_data=pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


df = pd.concat([df, original_data], ignore_index=True)


df.head()


df.info()


num_cols=["Temparature","Humidity","Moisture","Nitrogen","Potassium","Phosphorous"]
cat_cols=["Soil Type","Crop Type"]


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
le=LabelEncoder()


df.drop("id",axis=1,inplace=True)


from xgboost import XGBClassifier



test=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


test.info()


test.drop("id",axis=1,inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

df["Fertilizer Name"]=le.fit_transform(df["Fertilizer Name"])

X=df.drop(columns=["Fertilizer Name"])
y=df["Fertilizer Name"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(), cat_cols)
    ]
)


params = {
        'objective': 'multi:softprob',  
        'num_class': len(np.unique(y)), 
        'max_depth': 7,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'max_bin': 128,
        'colsample_bytree': 0.3, 
        'colsample_bylevel': 1,  
        'colsample_bynode': 1,  
        'tree_method': 'hist',  
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'device': "cuda",
        'enable_categorical':True,
        'n_estimators':10000,
        'early_stopping_rounds':50,
    }
#Parameters taken from https://www.kaggle.com/code/hahahaj/single-xgb


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import time

FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X), dtype=int)
test_preds_proba = np.zeros((len(test), len(np.unique(y)))) 

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nğŸ”� Fold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    test_scaled=preprocessor.transform(test)


    model = XGBClassifier(**params)

    start = time.time()

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=100,
    )

    val_preds = model.predict(X_val_scaled)
    oof_preds[val_idx] = val_preds

    test_preds_proba += model.predict_proba(test_scaled)

    acc = accuracy_score(y_val, val_preds)
    print(f"âœ… Fold {fold} Accuracy: {acc:.4f}")
    print(f"â�±ï¸� Time: {time.time() - start:.1f} sec")

test_preds_proba /= FOLDS

oof_acc = accuracy_score(y, oof_preds)
print(f"\n Final OOF Accuracy: {oof_acc:.4f}")


top_3_preds = np.argsort(test_preds_proba, axis=1)[:, -3:][:, ::-1]  


top3_labels = np.array([le.inverse_transform(row) for row in top_3_preds])

top3_joined = [" ".join(row) for row in top3_labels]


sub=pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


sub["Fertilizer Name"]=top3_joined


sub.head()


sub.to_csv("submission.csv",index=False)




