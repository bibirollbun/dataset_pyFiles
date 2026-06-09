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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
train_df.tail(10)


print("Train:", train_df.shape)
print("Test :", test_df.shape)


# データ構造の確認
display(train_df.head())        # 先頭5行
print(train_df.info())          # 各列の型と欠損値チェック
print(train_df.describe())      # 数値列の基本統計量

display(train_df.tail())        # 最後5行
print(train_df.info())          # 各列の型と欠損値チェック
print(train_df.describe())      # 数値列の基本統計量


# テストデータも同様にチェック
display(test_df.head())
print(test_df.info())

display(test_df.tail())
print(test_df.info())


print(train_df.corr())


# 列ごとの欠損件数
print(train_df.isnull().sum())

# 欠損パターンの可視化（seaborn 版ヒートマップなど）
sns.heatmap(train_df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()


# 数値変数一覧（id, day, rainfall は除外）
num_cols = train_df.select_dtypes(include=["int64","float64"]) \
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
    sns.histplot(train_df[col], kde=True, ax=ax)
    ax.set_title(f"{col}")
    ax.set_xlabel(col)
    ax.set_ylabel("頻度")

# 余ったサブプロットをオフに
for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()


# 数値変数一覧（id, day, rainfall は除外）
num_cols = train_df.select_dtypes(include=["int64","float64"]) \
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
    sns.boxplot(x=train_df[col], ax=ax)
    ax.set_title(f"{col}")
    ax.set_xlabel(col)

# 余ったサブプロットをオフに
for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()


print(train_df['rainfall'].value_counts())
sns.countplot(x='rainfall', data=train_df)
plt.title('Rainfall')
plt.show()


corr = train_df.drop(['id','day'], axis=1).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('heatmap')
plt.show()


# day を日付に変換
date_origin = '2022-01-01'
dates = pd.to_datetime(train_df['day'], unit='D', origin=date_origin)
train_df['month']      = dates.dt.month
train_df['weekofyear'] = dates.dt.isocalendar().week
train_df['dayofweek']  = dates.dt.dayofweek  # 0=月,6=日

# 同じ処理を test にも
dates_test_df = pd.to_datetime(test_df['day'], unit='D', origin=date_origin)
test_df['month']      = dates_test_df.dt.month
test_df['weekofyear'] = dates_test_df.dt.isocalendar().week
test_df['dayofweek']  = dates_test_df.dt.dayofweek


model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('clf',     LogisticRegression(class_weight='balanced', random_state=42))
])

# 特徴量・目的変数の定義
num_features = [c for c in train_df.columns if c not in ['id','day','rainfall']]
X = train_df[num_features]
y = train_df['rainfall']


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


# 説明変数
X = train_df.drop(['rainfall'], axis=1)
# 目的変数
y = train_df['rainfall']



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
#標準化
X = scaler.fit_transform(X)



from sklearn.model_selection import train_test_split

X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train_df.shape, X_test_df.shape)



from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_df, y_train_df)



print(model.intercept_)
print(model.coef_)



X_some = X_train_df[:4]
y_some = y_train_df[:4]
print(f'予測結果{np.round(model.predict(X_some), 3)}')
print(f'正解ラベル{list(y_some)}')



model_pred = model.predict(X_train_df)
err_sum = ((y_train_df - model_pred) ** 2).sum()
mse = err_sum / len(y_train_df)
rmse = np.sqrt(mse)

print(f'誤差の合計:{err_sum}')
print(f'誤差の平均値(MSE):{mse}')
print(f'RMSE:{rmse}')


