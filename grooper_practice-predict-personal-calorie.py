!pip install -U scikit-learn -q


!pip install -U autogluon -q


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')
from datetime import datetime

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from autogluon.tabular import TabularPredictor


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


print("train data size:", train.shape)
print("test  data size:", test.shape)


train.head()


test.head()


print("target:", set(train.columns)-set(test.columns))


train.info()


test.info()


train.isnull().sum()


test.isnull().sum()


train.describe().T


test.describe().T


train.nunique()


test.nunique()


for col in train.columns:
    print(f"{col}: {train[col].unique()}")
    print("")


train.duplicated().sum()


test.duplicated().sum()


effective_feature = [c for c in train.columns if c not in ['id', 'Calories']]

dup = train[train.duplicated(subset=effective_feature, keep=False)].sort_values(effective_feature)

print(f" pseudoduplicate rows: {len(dup)}")

display(dup.head(10))


# 同じ特徴なのにtargetの値が異なる→学習時に悪影響を及ぼす恐れがあるため、平均値を入れた1行のみ残す
train = (
    train.groupby([col for col in train.columns if col not in ['id', 'Calories']])
         .agg({'Calories': 'mean', 'id': 'first'})
         .reset_index()
)


# 750000から少し減っている
train.shape


plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.show()


df_all = pd.concat([train, test], ignore_index=True)


# BMIを計算
df_all['BMI'] = df_all['Weight'] / (df_all['Height'] / 100)**2


# 体温/心拍 を計算 
df_all['Temp_HR_ratio'] = df_all['Body_Temp'] / df_all['Heart_Rate']


df_all = pd.get_dummies(df_all, columns=['Sex'], prefix='Sex', drop_first=True, dtype=int)


df_all.info()


df_train = df_all[df_all['Calories'].notnull()]  
df_test = df_all[df_all['Calories'].isnull()] 


X = df_train.drop(columns=['Calories'])    
y_log = np.log1p(df_train['Calories'])   


X_train, X_val, y_train_log, y_val_log = (
    train_test_split(X, y_log, test_size=0.2, random_state=42)
)


y_val = df_train.loc[y_val_log.index, 'Calories']


def rmsle(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(
        mean_squared_error(
            np.log1p(y_true),
            np.log1p(y_pred)
        )
    )


#LGBM Regressor
lgbm_model = lgb.LGBMRegressor()
lgbm_model.fit(X_train, y_train_log)
y_pred_log_lgb = lgbm_model.predict(X_val)
y_pred_lgb = np.expm1(y_pred_log_lgb)
print(f"LightGBM RMSLE: {rmsle(y_val, y_pred_lgb):.5f}")


X_test = df_test.drop(columns=['Calories'], errors='ignore')

X_test = X_test[X_train.columns]

y_pred_log_test = lgbm_model.predict(X_test)

y_pred_test = np.expm1(y_pred_log_test)


submission = pd.DataFrame({
    'id': test['id'],  
    'Calories': y_pred_test  
})


date = datetime.today().strftime("%Y%m%d")
# submission.to_csv(f"submission_{date}.csv", index=False)


# 特徴量の重要度を取得
booster = lgbm_model.booster_
importance = booster.feature_importance()
feature_names = booster.feature_name()



# DataFrameにまとめてソート
feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print(feature_importance_df.head(10))  # 上位10件を表示



# 可視化
plt.figure(figsize=(10, 6))
topn = 20
plt.barh(feature_importance_df["feature"][:topn], feature_importance_df["importance"][:topn])
plt.xlabel("Gain Importance")
plt.title("Top 20 Feature Importances (Gain)")
plt.gca().invert_yaxis()
plt.show()


import numpy as np
from sklearn.metrics import mean_squared_log_error
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor


predictor = TabularPredictor(
    label='Calories',        # 目的変数
    eval_metric='rmse'     # 自動で最適な評価指標を選択
).fit(
    df_train,            # 学習用 DataFrame
    time_limit=600         # 制限時間（秒）
)


# 1) RMSLE を計算する関数（y_true, y_pred は numpy.ndarray）
def rmsle_func(y_true, y_pred, sample_weight=None):
    # mean_squared_log_error の squared=False で直接 RMSLE を返しても OK
    return mean_squared_log_error(y_true, y_pred, squared=False)


# 2) AutoGluon 用の Scorer を作成
rmsle_scorer = make_scorer(
    name='rmsle',              # メトリック名
    score_func=rmsle_func,     # スコア計算用関数
    optimum=0,                 # 最適値（RMSLE は 0 が最良なので）
    greater_is_better=False,   # 小さいほど良いメトリックなので False
    needs_pred=True            # Default でも OK （何を渡すかを明示したい場合）
)


predictor = TabularPredictor(
    label='Calories',
    eval_metric=rmsle_func
).fit(
    df_train,
    time_limit=600
)

# テストデータへの予測
preds = predictor.predict(test_data)

