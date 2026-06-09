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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor



train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')



target = 'accident_risk'
id_col = 'id'

features = [
    'road_type', 'num_lanes', 'curvature', 'speed_limit',
    'lighting', 'weather', 'road_signs_present', 'public_road',
    'time_of_day', 'holiday', 'school_season', 'num_reported_accidents'
]

X = train[features]
y = train[target]
X_test = test[features]



categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']
numeric_features = [
    'num_lanes', 'curvature', 'speed_limit',
    'road_signs_present', 'public_road',
    'holiday', 'school_season', 'num_reported_accidents'
]



kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(train))
cat_oof = np.zeros(len(train))
lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))



import lightgbm as lgb
for col in categorical_features:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')
    
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"===== Fold {fold + 1} =====")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # --- LightGBM ---
    lgb_model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Use callbacks for early stopping and logging
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100)
        ]
    )

    lgb_oof[valid_idx] = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration_)
    lgb_preds += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration_) / kf.n_splits

    # --- CatBoost ---
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=8,
        loss_function='RMSE',
        cat_features=categorical_features,
        random_seed=42,
        verbose=200
    )
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100)
    cat_oof[valid_idx] = cat_model.predict(X_valid)
    cat_preds += cat_model.predict(X_test) / kf.n_splits



from sklearn.metrics import mean_squared_error

lgb_rmse = mean_squared_error(y, lgb_oof, squared=False)
cat_rmse = mean_squared_error(y, cat_oof, squared=False)

print(f"LightGBM CV RMSE: {lgb_rmse:.5f}")
print(f"CatBoost CV RMSE: {cat_rmse:.5f}")



final_preds = 0.5 * lgb_preds + 0.5 * cat_preds  # You can tune weights (e.g., 0.6 / 0.4)



submission = sample_submission.copy()
submission['accident_risk'] = final_preds
submission.to_csv('submission.csv', index=False)

print("✅ Submission file saved as submission.csv")





