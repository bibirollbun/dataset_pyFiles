!pip3 install -U category_encoders


import numpy as np
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_object_dtype,
    is_string_dtype,
    is_bool_dtype
)
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly import offline


ce.__version__


df_sub=pd.read_csv("/kaggle/input/playground-series-s4e9/sample_submission.csv")
df_train=pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")


df_test.shape,df_train.shape, df_sub.shape


df_train.describe(include="all")


df_train.info()


df_test.info()


# 前処理関数
def preprocessing_train(df: pd.DataFrame) -> dict[any]:
    """trainとvalidationの前処理

    test dataのfuel_typeにはtrainに含まれていない要素がありそう
    ValueError: Unexpected categories found in column fuel_type
    また実際には外れ値の処理やエンジンの種類などいろいろ前処理やる
    """
    # 1. 欠損値埋め(ここでは適当)
    df = df.fillna("nan")

    # 2. カテゴリカル変数をエンコード
    # カテゴリ扱いしたい列を一気に抽出
    categorical_columns = [
        col
        for col in df.columns
        if (
            is_categorical_dtype(df[col])
            or is_string_dtype(df[col])
            or is_object_dtype(df[col])
            or is_bool_dtype(df[col])
        )
    ]
    oe = ce.OrdinalEncoder(
        cols=categorical_columns,
        handle_unknown="value",  # 未知の値は-1
        # handle_unknown="error",  # 未知の値があったら確認する為
        handle_missing="value",  # 欠損値は-2
    )
    df_ce = oe.fit_transform(df)
    
    # 本当は色々やる
    # df.do_something

    return {
        "preprocessed": df_ce,
        "encoder": oe,
    }


def preprocessing_test(df: pd.DataFrame, encoder) -> dict[any]:
    """testの前処理
    """
    # 1. 欠損値埋め(ここでは適当)
    df = df.fillna("nan")

    # 1.5 カラム数を合わせるためにpriceを空でつくる
    df["price"] = np.nan

    # 2. カテゴリカル変数をエンコード(事前にfit済のものを使用)
    df_ce = encoder.transform(df)

    # 適当なのでハードコーディング
    exclude = {"id", "price"}
    features = [col for col in df_ce.columns if col not in exclude]
    df_out = df_ce[features]
    
    # 本当は色々やる
    # df.do_something

    return {
        "preprocessed": df_out,
        "encoder": encoder,
    }


# 目的変数と特徴量の抽出
def extract_target_and_features(df: pd.DataFrame, target_column: str) -> dict[any]:
    """目的変数と特徴量を抽出して返す
    """
    y_train = df[target_column]

    # 適当なのでハードコーディング
    exclude = {"id", target_column}
    features = [col for col in df.columns if col not in exclude]
    x_train = df[features]
    
    return {
        "y": y_train,
        "x": x_train,
    }

# def split_train_df(x: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.1, seed: int = 42) -> dict:
#     """学習の検証用に学習データを分割（これは勾配ブースティング系用）
#     """
#     x_train_split, x_val, y_train_split, y_val = train_test_split(
#         x,
#         y,
#         test_size=test_size, # 検証データの割合を設定
#         random_state=seed # 分割の再現性を確保するための乱数シード
#     )
#     return {
#         "x_train_split": x_train_split,
#         "x_validation": x_val,
#         "y_train_split": y_train_split,
#         "y_validation": y_val,
#     }


# trainの前処理と分割の実行
preprocessed_train: dict = preprocessing_train(df_train)
target_and_features: dict = extract_target_and_features(
    preprocessed_train["preprocessed"],
    "price"
)
# splited_data: dict = split_train_df(
#     x=target_and_features["x"],
#     y=target_and_features["y"],
# )

target_and_features["x"].shape, target_and_features["y"].shape


# 学習設定
rf_regr = RandomForestRegressor(
    # n_estimators=100,  # 木の数 増やすと性能向上だが過学習してる場合は落とす
    max_depth=20,  # 過学習してたら10-30程度に制限
    max_features=0.5,  # sqrt(特徴量数)程度がいいらしい
    min_samples_split=10,  # 2-20程度がいいらしい
    min_samples_leaf=5,  # 1-10程度
    random_state=42,  # 再現性のために値を固定  
)

# n_estimators（木の数）の効果を可視化
param_name="n_estimators"
param_range=[50, 100, 200, 300]

train_scores, val_scores = validation_curve(
    rf_regr,
    target_and_features["x"], 
    target_and_features["y"], 
    param_name=param_name,
    param_range=param_range,
    cv=5, 
    scoring="neg_mean_squared_error",  # RMSEの負の値
    n_jobs=-1,  # 並列化
)

print("Validation curve calculation finished!")


# 可視化
plt.plot(param_range, -train_scores.mean(axis=1), "o-", label="Training RMSE")
plt.plot(param_range, -val_scores.mean(axis=1), "o-", label="Validation RMSE")
plt.xlabel(param_name)
plt.ylabel("RMSE")
plt.legend()
plt.show()


np.mean(val_scores, axis=1)


# 一応max_depthもも見てみる
# 学習設定
rf_regr = RandomForestRegressor(
    n_estimators=100,  # 木の数 増やすと性能向上だが過学習してる場合は落とす
    # max_depth=20,  # 過学習してたら10-30程度に制限
    max_features=0.5,  # sqrt(特徴量数)程度がいいらしい
    min_samples_split=10,  # 2-20程度がいいらしい
    min_samples_leaf=5,  # 1-10程度
    random_state=42,  # 再現性のために値を固定  
)

# n_estimators（木の数）の効果を可視化
param_name="max_depth"
param_range=[3, 5, 7, 10, 15, 20, 30]

train_scores, val_scores = validation_curve(
    rf_regr,
    target_and_features["x"], 
    target_and_features["y"], 
    param_name=param_name,
    param_range=param_range,
    cv=5, 
    scoring="neg_mean_squared_error",  # RMSEの負の値
    n_jobs=-1,  # 並列化
)

print("Validation curve calculation finished!")


# 可視化
plt.plot(param_range, -train_scores.mean(axis=1), "o-", label="Training RMSE")
plt.plot(param_range, -val_scores.mean(axis=1), "o-", label="Validation RMSE")
plt.xlabel(param_name)
plt.ylabel("RMSE")
plt.legend()
plt.show()


np.mean(val_scores, axis=1)


# 決定したparamsで学習し重要度の可視化とテストデータの予測を行う
rf_regr = RandomForestRegressor(
    n_estimators=100,  # 木の数 増やすと性能向上だが過学習してる場合は落とす
    max_depth=15,  # 過学習してたら10-30程度に制限
    max_features=0.5,  # sqrt(特徴量数)程度がいいらしい
    min_samples_split=10,  # 2-20程度がいいらしい
    min_samples_leaf=5,  # 1-10程度
    random_state=42,  # 再現性のために値を固定
    n_jobs=-1,  # CPU並列化
)
rf_regr.fit(target_and_features["x"], target_and_features["y"])
print("Training finished!")


# テストデータの処理
preprocessed_test: dict = preprocessing_test(df_test, preprocessed_train["encoder"])
preprocessed_test["preprocessed"].shape


# testの予測
predictions = rf_regr.predict(preprocessed_test["preprocessed"])
df_sub["price"] = predictions
df_sub.describe()


VER = "0.0.1"
df_sub.to_csv(f"submission_v{VER}.csv",index=False)


# 重要度の可視化
importances = pd.DataFrame({
    "Feature": target_and_features["x"].columns,
    "Importance": rf_regr.feature_importances_
}).sort_values("Importance", ascending=False) # 重要度順にソート

# 上位20件を可視化
plt.style.use("ggplot")
plt.figure(figsize=(10, 8))
plt.barh(
    importances["Feature"][::-1], 
    importances["Importance"][::-1]
)
plt.title("Feature Importance", fontsize=16)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.show()


# 予測値と実測値の差
y_pred = rf_regr.predict(target_and_features["x"])
y_actual = target_and_features["y"]
plot_df = pd.DataFrame({
    "actual": y_actual,
    "predicted": y_pred
})
# 描画用サンプリング
plot_df_sample = plot_df.sample(frac=0.05, random_state=42)

# --- ここからが Plotly による可視化 ---
# 1. 散布図のトレースを作成
scatter_trace = go.Scatter(
    x=plot_df_sample["actual"],
    y=plot_df_sample["predicted"],
    mode="markers",
    name="sample",
    marker=dict(
        size=3,
        opacity=0.6,
    ),
    # ホバーした際に表示されるテキストをカスタム
    # text=[f"Actual: {a:.2f}<br>Predicted: {p:.2f}" for a, p in zip(plot_df_sample["actual"], plot_df_sample["predicted"])],
    # hoverinfo="text"
)

# 2. y=x 対角線のトレースを作成
max_val = max(plot_df_sample["actual"].max(), plot_df_sample["predicted"].max())
min_val = min(plot_df_sample["actual"].min(), plot_df_sample["predicted"].min())

line_trace = go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode="lines",  # "lines"で線グラフを指定
    name="y=x",
    line=dict(
        color="red",
        width=2,
        dash="dash"
    )
)

# 3. Figureオブジェクトを作成し、トレースを追加
fig = go.Figure(data=[scatter_trace, line_trace])

# 4. レイアウトを更新
fig.update_layout(
    title=dict(
        text="予測値 vs 実測値",
        x=0.5, # タイトルを中央に配置
        font=dict(size=20)
    ),
    xaxis_title="実測値 (Actual Values)",
    yaxis_title="予測値 (Predicted Values)",
    width=700,
    height=700,
    # 縦横比を1:1に固定して正方形にする
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain="domain"),
    # 見やすいようにテンプレートを指定 (例: "plotly_white", "ggplot2", "seaborn")
    template="plotly_white", 
    legend=dict(
        x=0.01,
        y=0.99,
        bordercolor="Black",
        borderwidth=1
    )
)

# 5. グラフを表示
offline.iplot(fig)





