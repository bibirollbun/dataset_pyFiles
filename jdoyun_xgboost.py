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


df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')


X = df.drop('label', axis=1)
y = df['label']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from xgboost import XGBRegressor

# ëª¨ë�¸ ì„ ì–¸ ì˜ˆì‹œ
model = XGBRegressor(n_estimators=500, learning_rate=0.2, max_depth=4, random_state =0)
model.fit(X_train, y_train)


fscore = model.get_booster().get_fscore()
sorted_dic = dict(sorted(fscore.items(), key=lambda item: item[1], reverse=True))
sorted_dic


features = [k for k, v in sorted_dic.items() if v >= 20]
features


new_X = X[features]
new_X


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.3, random_state = 0)


from xgboost import XGBRegressor

# ëª¨ë�¸ ì„ ì–¸ ì˜ˆì‹œ
model = XGBRegressor(n_estimators=500, learning_rate=0.2, max_depth=4, random_state =0)
model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

y_pred = model.predict(X_test)
print(f'mse:{mean_squared_error(y_test, y_pred)}')
print(f'r2_score:{r2_score(y_test, y_pred)}')


# train_pred = model.predict(X)
# pd.DataFrame(train_pred).to_csv('train_pred.csv')


test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')


X = test.drop('label',axis=1)[features]
y = test['label']


prediction = model.predict(X)
prediction = pd.DataFrame(prediction)
prediction['ID'] = range(1,len(prediction)+1)
prediction = prediction.rename(columns={0:'prediction'})[['ID','prediction']]


prediction[['ID','prediction']]


prediction.to_csv('prediction.csv')


import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# X, yëŠ” ì£¼ì–´ì§„ ë…¸íŠ¸ë¶� ê¸°ì¤€
# ì˜ˆì‹œ: X = new_X, y = y
# ì•„ë�˜ ë�¼ì�¸ì�€ ìƒ�ë�µ ê°€ëŠ¥
X = df.drop('label', axis=1)[features]
y = df['label']

# ì��ë�™ n_splits ì„¤ì •
min_val_size = 20  # í•œ validation setì�˜ ìµœì†Œ í�¬ê¸°
n_samples = len(X)
max_splits = n_samples // min_val_size

if max_splits < 2:
    raise ValueError("ë�°ì�´í„°ê°€ ë„ˆë¬´ ì �ì–´ êµ�ì°¨ê²€ì¦�ì�„ í•  ìˆ˜ ì—†ìŠµë‹ˆë‹¤.")

n_splits = min(5, max_splits)  # ìµœëŒ€ 5ê°œ í�´ë“œê¹Œì§€ë§Œ ì‚¬ìš©
tscv = TimeSeriesSplit(n_splits=n_splits)

print(f"ì´� ìƒ˜í”Œ ìˆ˜: {n_samples}, ì‚¬ìš© í�´ë“œ ìˆ˜: {n_splits}\n")

# ê²°ê³¼ ì €ì�¥ìš©
rmse_list = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    if max(val_idx) >= len(X):
        print(f"âš ï¸� Fold {fold} skipped: val_idx out of bounds")
        continue

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print(f"Fold {fold} | Train: {train_idx[0]}~{train_idx[-1]} | Val: {val_idx[0]}~{val_idx[-1]}")

    # ëª¨ë�¸ í›ˆë ¨
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ì˜ˆì¸¡ ë°� í�‰ê°€
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_list.append(rmse)

    print(f"  ğŸ“Š Fold {fold} RMSE: {rmse:.4f}\n")

# ì „ì²´ í�‰ê·  RMSE
print(f"âœ… í�‰ê·  RMSE: {np.mean(rmse_list):.4f}")


X = test.drop('label',axis=1)[features]

prediction = model.predict(X)
prediction = pd.DataFrame(prediction)
prediction['ID'] = range(1,len(prediction)+1)
prediction = prediction.rename(columns={0:'prediction'})[['ID','prediction']]

prediction[['ID','prediction']]


prediction.to_csv('fold_prediction.csv')

