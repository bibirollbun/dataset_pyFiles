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


import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# 学習の再現性と警告非表示
np.random.seed(42)
warnings.filterwarnings("ignore")

# 可視化のテーマ設定
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)

# データ分割・クロスバリデーション
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)

# 前処理
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

# モデル本体
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 特徴量選択
from sklearn.feature_selection import SelectFromModel, RFE

# 評価指標
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


# データ構造の確認
display(train.head())        # 先頭5行
print(train.info())          # 各列の型と欠損値チェック
print(train.describe())      # 数値列の基本統計量

# テストデータも同様にチェック
display(test.head())
print(test.info())


train['sunshine'] = train['sunshine'].replace(0, train['sunshine'].mean())
test['sunshine'] = test['sunshine'].replace(0, test['sunshine'].mean())


import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(8, 15))

sns.regplot(data = train, x = "humidity", y = "cloud", ax = axes[0], scatter_kws = {'alpha': 0.5})
axes[0].set_title("Humidity vs Cloud")
axes[0].grid(True)

sns.regplot(data = train, x = "humidity", y = "temparature", ax = axes[1], scatter_kws = {'alpha': 0.5}, color = "green")
axes[1].set_title("Humidity vs Temperature")
axes[1].grid(True)

sns.regplot(data = train, x = "cloud", y = "temparature", ax = axes[2], scatter_kws = {'alpha': 0.5}, color = "orange")
axes[2].set_title("Cloud vs Temperature")
axes[2].grid(True)

plt.tight_layout()
plt.show()


# 列ごとの欠損件数
print(train.isnull().sum())

# 欠損パターンの可視化（seaborn 版ヒートマップなど）
sns.heatmap(train.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()


# 数値変数一覧（id, day, rainfall は除外）
num_cols = train.select_dtypes(include=["int64","float64"]) \
                .drop(["id","day","rainfall"], axis=1).columns.tolist()

# レイアウト設定
n = len(num_cols) # 変数の数
cols_per_row = 4  # 1行あたりのプロット数
rows = math.ceil(n / cols_per_row)

# ヒストグラム
fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 5, rows * 4))
axes = axes.flatten()

for idx, col in enumerate(num_cols):
    ax = axes[idx]
    sns.histplot(train[col], kde=True, ax=ax)
    ax.set_title(f"{col}")
    ax.set_xlabel(col)
    ax.set_ylabel("頻度")

# 余ったサブプロットをオフに
for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()


# 数値変数一覧（id, day, rainfall は除外）
num_cols = train.select_dtypes(include=["int64","float64"]) \
                .drop(["id","day","rainfall"], axis=1).columns.tolist()

# レイアウト設定
n = len(num_cols) # 変数の数
cols_per_row = 4  # 1行あたりのプロット数
rows = math.ceil(n / cols_per_row)

# 箱ひげ図
fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 5, rows * 4))
axes = axes.flatten()

for idx, col in enumerate(num_cols):
    ax = axes[idx]
    sns.boxplot(x=train[col], ax=ax)
    ax.set_title(f"{col}")
    ax.set_xlabel(col)

# 余ったサブプロットをオフに
for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()


# rainfall の 0/1 件数を確認
print(train['rainfall'].value_counts())
sns.countplot(x='rainfall', data=train)
plt.title('Rainfall')
plt.show()


# 数値特徴量＋目的変数の相関行列
corr = train.drop(['id','day'], axis=1).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('heatmap')
plt.show()


# day を日付に変換
date_origin = '2020-01-01'
dates = pd.to_datetime(train['day'], unit='D', origin=date_origin)
train['month']      = dates.dt.month
train['weekofyear'] = dates.dt.isocalendar().week
train['dayofweek']  = dates.dt.dayofweek  # 0=月,6=日

# 同じ処理を test にも
dates_test = pd.to_datetime(test['day'], unit='D', origin=date_origin)
test['month']      = dates_test.dt.month
test['weekofyear'] = dates_test.dt.isocalendar().week
test['dayofweek']  = dates_test.dt.dayofweek


model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('clf',     LogisticRegression(class_weight='balanced', random_state=42))
])

# 特徴量・目的変数の定義
num_features = [c for c in train.columns if c not in ['id','day','rainfall']]
X = train[num_features]
y = train['rainfall']


# モデルは既存の pipeline + clf を使って再学習
model.fit(X, y)

# 予測確率の取得
y_proba = model.predict_proba(X)[:, 1]

# ROC 曲線用の指標計算
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

# ROC 曲線
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()



# 特徴量と目的変数
num_features = [col for col in train.columns if col not in ['id', 'day', 'rainfall']]
X = train[num_features]
y = train['rainfall']

# 学習用と訓練用のデータを分割
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,      # クラス比を保ったまま分割
    random_state=42
)

# モデル学習
model.fit(X_train, y_train)

# 予測確率の取得
y_val_proba = model.predict_proba(X_val)[:, 1]
y_val_pred  = model.predict(X_val)

# ROC 曲線用の指標計算
fpr, tpr, _ = roc_curve(y_val, y_val_proba)
roc_auc = auc(fpr, tpr)

# ROC 曲線
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


