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


import numpy as np
import pandas as pd
import os
import random
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from lightgbm import LGBMClassifier
import xgboost as xgb
import optuna
import shap
import missingno as msno


np.random.seed(42)
random.seed(42)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train_df.head()


train_df.info()


train_df.isnull().sum()


test_df.isnull().sum()


train_unique = {
    col: dict(sorted(train_df[col].value_counts().to_dict().items()))
    for col in train_df.drop(columns=["id"]).columns
}

train_unique


# print(train_df.drop(columns=["id"]).apply(lambda col: col.value_counts()))


import pandas as pd
import numpy as np

# ② 安全対策：欠損を埋めて型をそろえる（必要ないならそのままでOK）
test_df["curvature"] = test_df["curvature"].astype(float).fillna(0.0)
for col in ["lighting","weather"]:
    test_df[col] = test_df[col].astype(str).str.lower().fillna("")

test_df["speed_limit"] = pd.to_numeric(test_df["speed_limit"], errors="coerce").fillna(0).astype(int)
test_df["num_reported_accidents"] = pd.to_numeric(test_df["num_reported_accidents"], errors="coerce").fillna(0).astype(int)

# ③ ルールに基づくスコア（base_risk）
base_risk = (
    0.3 * test_df["curvature"].values
    + 0.2 * (test_df["lighting"].values == "night").astype(int)
    + 0.1 * (test_df["weather"].values != "clear").astype(int)
    + 0.2 * (test_df["speed_limit"].values >= 60).astype(int)
    + 0.1 * (test_df["num_reported_accidents"].values > 2).astype(int)
)

# ④ ノイズなしでクリップ（0〜1）。丸めは任意（提出時は小数6桁くらいで十分）
pred = np.clip(base_risk, 0.0, 1.0)

# ⑤ 提出ファイル作成（カラム名はコンペ仕様に合わせる）
#   - 一般的には id 列がある想定。なければ index を id にする
id_col = "id" if "id" in test_df.columns else None
sub = pd.DataFrame({
    "id": test_df[id_col] if id_col else np.arange(len(test_df)),
    "accident_risk": pred
})

sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved: submission_rule_only.csv")


