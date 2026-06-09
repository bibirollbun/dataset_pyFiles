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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import early_stopping,log_evaluation
import lightgbm as lgb

# ğŸ“¥ 2. Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")




# ğŸ§¹ 3. Handle Categorical Columns
categorical_cols = train.select_dtypes(include='object').columns.tolist()

# Convert to category dtype
for col in categorical_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")
    test[col] = test[col].cat.set_categories(train[col].cat.categories)

# ğŸ�¯ 4. Features & Target
X = train.drop(columns=["y"])
y = train["y"]

# âœ‚ï¸� 5. Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)





# Prepare datasets
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_cols)

params = {
    "objective": "binary",
    "metric": "auc",   # or 'auc' if you want AUC
    "boosting_type": "gbdt",
    "max_depth": 4,
    "learning_rate": 0.03,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "subsample": 0.8,
    "verbosity": -1,
    "seed": 42
}

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=20000,
    valid_sets=[val_data],
     callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(period=200)
    ],
    
)





val_pred = model.predict(X_val)
auc = roc_auc_score(y_val, val_pred)
print(f"AUC-ROC: {auc:.4f}")



y_pred_test = model.predict(test)
submission['y'] = y_pred_test


submission


submission.to_csv("submission.csv", index=False)




