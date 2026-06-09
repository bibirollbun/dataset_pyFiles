import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split


df_sub=pd.read_csv("/kaggle/input/playground-series-s4e9/sample_submission.csv")
df_train=pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")


df_test.shape,df_train.shape, df_sub.shape


df_train.describe(include="all")


df_train.info()


df_test.info()


# 適当な欠損値埋め
df_train.fillna("nan").info()


# 目的変数 (trainデータのみ)
# 適当な欠損値埋め
y_train = df_train.fillna("nan")["price"]

features = [col for col in df_train.columns if col not in ['id', 'price']]

X_train = df_train.fillna("nan")[features]
X_test = df_test.fillna("nan")[features]

categorical_features_indices = X_train.select_dtypes(include='object').columns.tolist()

# 分割
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1, # 検証データの割合を10%に設定
    random_state=42 # 分割の再現性を確保するための乱数シード
)


# 学習
model = CatBoostRegressor(
    iterations=1000, # 学習のイテレーション数 (少ないと学習不足、多いと過学習の可能性)
    learning_rate=0.05, # 学習率
    depth=6, # ツリーの深さ
    loss_function='RMSE', # 回帰問題の損失関数 (RMSEが一般的)
    eval_metric='RMSE', # 評価指標
    random_state=42, # 乱数シード
    verbose=100 # 学習中のログ表示間隔
)

print("\n--- Start Model Training ---")

model.fit(
    X_train_split,
    y_train_split,
    cat_features=categorical_features_indices,
    eval_set=(X_val, y_val), # 検証データがある場合、ここに指定すると学習中の評価が見れます
    early_stopping_rounds=100, # 検証データでの性能が改善しなくなった場合に早期停止
)

print("\n--- Model Training Finished ---")


predictions = model.predict(X_test)


df_sub["price"] = predictions
df_sub.describe()


VER = "0.0.1"
df_sub.to_csv(f"submission_v{VER}.csv",index=False)


feature_importances = model.get_feature_importance(Pool(X_train_split, label=y_train_split, cat_features=categorical_features_indices), type='PredictionValuesChange')

# 特徴量名と重要度をDataFrameにまとめます。
feature_importance_df = pd.DataFrame({
    'feature': X_train_split.columns,
    'importance': feature_importances
})

# 重要度で降順にソートします。
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
feature_importance_df




