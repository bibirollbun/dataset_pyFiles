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


train=pd.read_csv("/kaggle/input/tabular-playground-series-feb-2021/train.csv")
test=pd.read_csv("/kaggle/input/tabular-playground-series-feb-2021/test.csv")
sub=pd.read_csv("/kaggle/input/tabular-playground-series-feb-2021/sample_submission.csv")


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# 1. VERİYİ SIFIRDAN YÜKLE (Hafıza hatalarını silmek için)


# 2. ÖN İŞLEME
target = train["target"]
test_id = test["id"]

# Sütunları temizle ve hizala
train.drop(["id", "target"], axis=1, inplace=True)
test.drop(["id"], axis=1, inplace=True)

cat_features = [f'cat{i}' for i in range(10)]
cont_features = [f'cont{i}' for i in range(14)]

# A. Row-wise İstatistikler
for df in [train, test]:
    df["cont_mean"] = df[cont_features].mean(axis=1)
    df["cont_std"] = df[cont_features].std(axis=1)
    df["cont_min"] = df[cont_features].min(axis=1)
    df["cont_max"] = df[cont_features].max(axis=1)

# B. KMeans (Düzeltilmiş Mantık)
kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
train["cluster"] = kmeans.fit_predict(train[cont_features])
test["cluster"] = kmeans.predict(test[cont_features])

# C. Label Encoding
for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# D. KATEGORİK SÜTUNLARI TYPE OLARAK BELİRLE (LGBM için kritik)
for col in cat_features + ["cluster"]:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# 3. EĞİTİM PARAMETRELERİ
lgbm_params = {
    'metric': 'rmse',
    'n_estimators': 10000,
    'learning_rate': 0.005,
    'num_leaves': 128,
    'max_depth': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'random_state': 42
}

# 4. CROSS VALIDATION
kf = KFold(n_splits=10, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train))
test_predictions = np.zeros(len(test)) # SIFIRLANDIĞINDAN EMİN OLDUK
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train, target)):
    # .iloc ile veriyi hizalı alıyoruz
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
    
    model = LGBMRegressor(**lgbm_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=0)
        ]
    )
    
    # Validasyon tahmini
    preds_val = model.predict(X_val)
    oof_predictions[val_idx] = preds_val
    
    # Test tahmini (DİKKAT: Sütun sırası X_train ile aynı olmalı)
    test_predictions += model.predict(test[X_train.columns]) / 10 

    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    scores.append(rmse)
    print(f"Fold {fold+1}: RMSE = {rmse:.5f}")

print(f"\nGenel OOF RMSE: {np.sqrt(mean_squared_error(target, oof_predictions)):.5f}")

# 5. SUBMISSION
submission = pd.DataFrame({'id': test_id, 'target': test_predictions})
submission.to_csv('submission.csv', index=False)




