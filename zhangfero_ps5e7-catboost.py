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


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from catboost import CatBoostClassifier


# -------- 1. 加载数据 -------- #
DATA_DIR = '/kaggle/input/playground-series-s5e7'
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")


# -------- 2. 特征工程 -------- #
def create_ambivert_features(df):
    X = df.copy()
    # 对数变换
    for col in ("Friends_circle_size", "Post_frequency"):
        if col in X:
            X[f"{col}_log"] = np.log1p(X[col])
    # 分箱
    bin_edges = {
        "Time_spent_Alone": [-0.1, 3, 7, 11],
        "Post_frequency":   [-0.1, 2, 5, 15]
    }
    for col, edges in bin_edges.items():
        if col in X:
            X[f"{col}_bin"] = pd.cut(
                X[col], bins=edges, labels=[0,1,2], include_lowest=True
            ).astype("Int64")
    # 交互特征
    if {"Social_event_attendance","Friends_circle_size"} <= set(X.columns):
        X["attendance_per_friend"] = X["Social_event_attendance"]/(X["Friends_circle_size"]+1)
    if {"Going_outside","Time_spent_Alone"} <= set(X.columns):
        X["outside_alone_time"] = X["Going_outside"] * X["Time_spent_Alone"]
    return X


train_df = create_ambivert_features(train_df)
test_df  = create_ambivert_features(test_df)

# -------- 3. 分离特征与目标 -------- #
TARGET = 'Personality'
y = train_df[TARGET]
X = train_df.drop(columns=[TARGET, 'id'], errors='ignore')
X_test = test_df.drop(columns=['id'], errors='ignore')

# -------- 4. 类别特征预处理 -------- #
cat_cols = ['Stage_fear', 'Drained_after_socializing']
for c in cat_cols:
    X[c]      = X[c].fillna('missing').astype(str)
    X_test[c] = X_test[c].fillna('missing').astype(str)
cat_idx = [X.columns.get_loc(c) for c in cat_cols]


# -------- 5. 超参数搜索：RandomizedSearchCV -------- #
param_dist = {
    'depth':              [4, 6, 8, 10],
    'learning_rate':      [0.01, 0.03, 0.05, 0.1],
    'l2_leaf_reg':        [1, 3, 5, 7, 9],
    'bagging_temperature':[0, 1, 2, 3, 4],
    'border_count':       [32, 50, 100, 200],
    'iterations':         [200, 500, 1000]
}

base_model = CatBoostClassifier(
    random_seed=42,
    eval_metric='Accuracy',
    verbose=0
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring='accuracy',
    cv=skf,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# 内部 CV 会自动调用 fit
search.fit(X, y, cat_features=cat_idx)

print(f"Best CV accuracy: {search.best_score_:.4f}")
print("Best hyperparameters:")
for k, v in search.best_params_.items():
    print(f"  - {k}: {v}")


# -------- 6. 直接用最优参数在全量数据上训练 -------- #
final_model = search.best_estimator_

final_model.fit(
    X, y,
    cat_features=cat_idx,
    verbose=0
)

# -------- 7. 预测测试集 & 保存提交 -------- #
preds_test = final_model.predict(X_test)
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': preds_test
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved: submission.csv")

