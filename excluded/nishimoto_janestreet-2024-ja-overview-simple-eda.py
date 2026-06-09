import gc
import glob
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pylab as plt
from collections import defaultdict
warnings.simplefilter("ignore")


targets = [f"responder_{i}" for i in range(9)]
columns = ["date_id", "time_id", "symbol_id"]
columns.extend(targets)
df_targets = pd.read_parquet("/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet", columns=columns)


print("All data length:", len(df_targets))


df = pd.read_parquet("/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=0/part-0.parquet")


df


print("date_id", df_targets["date_id"].unique().min(), df_targets["date_id"].unique().max())
print("time_id", df_targets["time_id"].unique().min(), df_targets["time_id"].unique().max())
print("symbol_id", df_targets["symbol_id"].unique().min(), df_targets["symbol_id"].unique().max())


# 日付ごとに株のIDの個数が違うのでプロット
df_targets.groupby("date_id")["symbol_id"].count().plot()


# 計算時間すごいので、手元で試すならサンプリングしてください
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    sns.distplot(df_targets[f"responder_{i}"])


df_targets.corr("spearman")["responder_6"].sort_values(ascending=False)


plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
df_targets.groupby("time_id")["responder_6"].mean().plot()
plt.ylabel("responder_6 mean")

plt.subplot(1, 2, 2)
df_targets.groupby("date_id")["responder_6"].median().plot()
plt.ylabel("responder_6 median")
plt.show()


# 初日, 銘柄1の様子を見てみます
df00 = df.query("date_id == 0 and symbol_id == 1")


df00


# 予測対象はresponder_6
df00["responder_6"].plot()


# 一応全部見てみる
df00["responder_0"].plot()
df00["responder_1"].plot()
df00["responder_2"].plot()
df00["responder_3"].plot()
df00["responder_4"].plot()
df00["responder_5"].plot()
df00["responder_7"].plot()
df00["responder_8"].plot()


corr_mean = {
    "feature_00": 0.0097,
    "feature_01": -0.0320,
    "feature_02": 0.0214,
    "feature_03": 0.0157,
    "feature_04": -0.0460,
    "feature_05": -0.0129,
    "feature_06": -0.0540,
    "feature_07": -0.0338,
    "feature_08": 0.0106,
    "feature_09": -0.0018,
    "feature_10": 0.0039,
    "feature_11": 0.0010,
    "feature_12": -0.0114,
    "feature_13": -0.0037,
    "feature_14": -0.0114,
    "feature_15": np.nan,
    "feature_16": 0.0035,
    "feature_17": np.nan,
    "feature_18": -0.0046,
    "feature_19": 0.0036,
    "feature_20": 0.0020,
    "feature_21": -0.0047,
    "feature_22": 0.0014,
    "feature_23": 0.0065,
    "feature_24": -0.0028,
    "feature_25": -0.0034,
    "feature_26": 0.0055,
    "feature_27": 0.0055,
    "feature_28": 0.0044,
    "feature_29": 0.0051,
    "feature_30": 0.0055,
    "feature_31": -0.0049,
    "feature_32": np.nan,
    "feature_33": np.nan,
    "feature_34": 0.0134,
    "feature_35": 0.0090,
    "feature_36": -0.0370,
    "feature_37": -0.0205,
    "feature_38": -0.0216,
    "feature_39": np.nan,
    "feature_40": 0.0108,
    "feature_41": np.nan,
    "feature_42": np.nan,
    "feature_43": 0.0118,
    "feature_44": np.nan,
    "feature_45": 0.0149,
    "feature_46": 0.0178,
    "feature_47": 0.0316,
    "feature_48": 0.0017,
    "feature_49": 0.0273,
    "feature_50": np.nan,
    "feature_51": 0.0185,
    "feature_52": np.nan,
    "feature_53": np.nan,
    "feature_54": 0.0167,
    "feature_55": np.nan,
    "feature_56": 0.0170,
    "feature_57": 0.0237,
    "feature_58": np.nan,
    "feature_59": 0.0164,
    "feature_60": 0.0417,
    "feature_61": np.nan,
    "feature_62": 0.0031,
    "feature_63": 0.0011,
    "feature_64": 0.0032,
    "feature_65": -0.0066,
    "feature_66": 0.0048,
    "feature_67": -0.0148,
    "feature_68": -0.0040,
    "feature_69": -0.0166,
    "feature_70": -0.0065,
    "feature_71": -0.0046,
    "feature_72": -0.0058,
    "feature_73": np.nan,
    "feature_74": np.nan,
    "feature_75": -0.0042,
    "feature_76": 0.0017,
    "feature_77": -0.0064,
    "feature_78": -0.0007,
}


pd.Series(corr_mean).dropna().sort_values(ascending=False)


pd.Series(corr_mean).dropna().sort_values(ascending=False).plot(kind="bar", fontsize="x-small")


d_corr = defaultdict(lambda: defaultdict(int))
feats = [f"feature_{str(col).zfill(2)}" for col in range(79)]
for i in range(10):
    train = pd.read_parquet(f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet")
    for date_id in train["date_id"].unique():
        train_date = train[train["date_id"] == date_id]
        for feat in feats:
            train_date_feat = train_date[[feat, "responder_6"]].dropna()
            if len(train_date_feat) == 0:
                d_corr[feat][date_id] = np.nan
            else:
                d_corr[feat][date_id] = st.spearmanr(train_date_feat[feat], train_date_feat["responder_6"])[0]


df_corr = pd.DataFrame(d_corr)
df_corr.to_csv("corr_date_id.csv")
df_corr.fillna(0, inplace=True)


s_corr_diff = (df_corr.head(50).mean() - df_corr.tail(50).mean()).dropna().sort_values()


plt.figure(figsize=(15, 9))
for idx, feat in enumerate(s_corr_diff.head(5).keys(), start=1):
    plt.subplot(2, 3, idx)
    df_corr[feat].rolling(20).mean().plot()
    plt.title(feat)
plt.show()


plt.figure(figsize=(15, 9))
for idx, feat in enumerate(s_corr_diff.tail(5).keys(), start=1):
    plt.subplot(2, 3, idx)
    df_corr[feat].rolling(20).mean().plot()
    plt.title(feat)
plt.show()

