# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_regression
from category_encoders import TargetEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df.pop("id")
df


for col in df.columns:
    print(f'ã€�{col}ã€‘')
    print(df[col].dtypes)
    print("æ¬ æ¸¬å€¤", df[col].isnull().sum())
    if df[col].dtypes in ['object', 'category']:
        print(df[col].value_counts())
    else:
        print(df[col].describe())
    print()


# å�„ç‰¹å¾´é‡�ã�®ç›¸é–¢ä¿‚æ•°ã€�MIå€¤
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

X = df.drop(columns=["accident_risk"])
y = df["accident_risk"]

# Label Encoding
for col in X.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Mutual Informationã‚’è¨ˆç®—
mi = mutual_info_regression(X, y, random_state=0)

# ç›¸é–¢ä¿‚æ•°ã�®è¨ˆç®—
corr = []
for col in X.columns:
    corr_value = np.corrcoef(X[col], y)[0, 1]
    corr.append(corr_value)

# è¡¨ç¤ºç”¨
result_df = pd.DataFrame({
    "Feature": X.columns,
    "Mutual_Information": mi,
    "Correlation": corr
})

# MIã�®é™�é †ã�§ä¸¦ã�¹æ›¿ã�ˆ
result_df = result_df.sort_values(by="Mutual_Information", ascending=False)

print(result_df)


# æ•°å€¤ãƒ‡ãƒ¼ã‚¿ã� ã�‘ã�«çµ�ã�£ã�¦ç›¸é–¢è¡Œåˆ—ã‚’ä½œæˆ�
num_df = df.loc[:, [col for col in df.columns if df[col].dtype not in ["object", "category", "bool"]]]
#num_df.pop("id")
corr = num_df.corr(numeric_only=True)

# ãƒ’ãƒ¼ãƒˆãƒ�ãƒƒãƒ—ã�®æ��ç”»
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr,
    annot=True,        # ç›¸é–¢ä¿‚æ•°ã‚’æ•°å€¤ã�§è¡¨ç¤º
    fmt=".2f",         # å°�æ•°ç‚¹ä»¥ä¸‹2æ¡�
    cmap="coolwarm",   # è‰²ã�®ã‚¹ã‚¿ã‚¤ãƒ«
    center=0           # 0ã‚’ä¸­å¿ƒã�«ã�—ã�Ÿã‚«ãƒ©ãƒ¼ãƒ�ãƒƒãƒ—
)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()


cat_cols = [col for col in df.columns if df[col].dtype in ["object", "category", "bool"]]

n_cols = 2  # æ¨ªã�«ä¸¦ã�¹ã‚‹æ•°
n_rows = (len(cat_cols) + n_cols - 1) // n_cols  # è¡Œæ•°ã‚’è¨ˆç®—

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
axes = axes.flatten()  # 2æ¬¡å…ƒé…�åˆ—ã‚’ãƒ•ãƒ©ãƒƒãƒˆåŒ–

for i, col in enumerate(cat_cols):
    sns.violinplot(x=col, y="accident_risk", data=df, ax=axes[i])
    axes[i].set_title(col)

# ä½™ã�£ã�Ÿã‚µãƒ–ãƒ—ãƒ­ãƒƒãƒˆã�¯é��è¡¨ç¤ºã�«
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# weather * time_of_day, weather * lightning
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_regression


class CrossFoldEncoder:
    def __init__(self, encoder, n_splits=5, random_state=42, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs
        self.cv_ = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def fit_transform(self, X, y, cols):
        """(é›†è¨ˆ+å¤‰æ�›)*5folds + MIã�¨ç›¸é–¢ä¿‚æ•°ã�®ç®—å‡º"""
        self.fitted_encoders_ = []
        self.cols = cols
        X_encoded_list = []
        self.feature_scores_ = []  # MIã�¨corrã‚’æ ¼ç´�ã�™ã‚‹

        for col in cols:
            fold_encoded_parts = []

            # KFoldã�§OOFã‚¨ãƒ³ã‚³ãƒ¼ãƒ‰
            for idx_encode, idx_train in self.cv_.split(X):
                fitted_encoder = self.encoder_(cols=[col], **self.kwargs_)
                fitted_encoder.fit(X.iloc[idx_encode][[col]], y.iloc[idx_encode])

                transformed = fitted_encoder.transform(X.iloc[idx_train][[col]])
                transformed.index = X.iloc[idx_train].index
                fold_encoded_parts.append(transformed[[col]])

                self.fitted_encoders_.append(fitted_encoder)

            # foldã�”ã�¨ã‚’concatã�—ã�¦ä¸¦ã�¹æ›¿ã�ˆ
            col_encoded_df = pd.concat(fold_encoded_parts).sort_index()
            encoded_col_name = f"{col}_encoded"
            col_encoded_df.columns = [encoded_col_name]
            X_encoded_list.append(col_encoded_df)

            # --- ğŸ’¡ã�“ã�“ã�§MIã�¨ç›¸é–¢ä¿‚æ•°ã‚’è¨ˆç®— ---
            corr_value = np.corrcoef(col_encoded_df[encoded_col_name], y)[0, 1]
            mi_value = mutual_info_regression(
                col_encoded_df[[encoded_col_name]], y, random_state=42
            )[0]
            self.feature_scores_.append(
                {"feature": encoded_col_name, "corr": corr_value, "MI": mi_value}
            )

        # æœ€çµ‚çš„ã�«OOFç‰¹å¾´é‡�ã‚’è¿”ã�™
        return pd.concat(X_encoded_list, axis=1)

    def transform(self, X):
        """ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�«é�©ç”¨"""
        X_encoded_list = []

        for i, col in enumerate(self.cols):
            encoded_versions = []
            for fitted_encoder in self.fitted_encoders_[i::len(self.cols)]:
                transformed = fitted_encoder.transform(X[[col]])
                encoded_versions.append(transformed[col])

            averaged = sum(encoded_versions) / len(encoded_versions)
            X_encoded_list.append(pd.DataFrame({f"{col}_encoded": averaged}))

        return pd.concat(X_encoded_list, axis=1)

    def get_feature_scores(self):
        """MIã�¨ç›¸é–¢ä¿‚æ•°ã‚’ã�¾ã�¨ã‚�ã�ŸDataFrameã‚’è¿”ã�™"""
        return pd.DataFrame(self.feature_scores_).sort_values(
            by="MI", ascending=False
        ).reset_index(drop=True)


#------------------------------------------------------------------------------------------------
new_df = df.copy()
new_df["wea_tim"] = new_df["weather"].astype(str) + "_" + new_df["time_of_day"].astype(str) 
new_df["wea_lig"] = new_df["weather"].astype(str) + "_" + new_df["lighting"].astype(str) 

new_cols = [
    "wea_tim",
    "wea_lig",
]

n_cols = 2  # æ¨ªã�«ä¸¦ã�¹ã‚‹æ•°
n_rows = (len(new_cols) + n_cols - 1) // n_cols  # è¡Œæ•°ã‚’è¨ˆç®—

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
axes = axes.flatten()  # 2æ¬¡å…ƒé…�åˆ—ã‚’ãƒ•ãƒ©ãƒƒãƒˆåŒ–

for i, col in enumerate(new_cols):
    sns.boxplot(x=col, y="accident_risk", data=new_df, ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', labelrotation=45)  # â†� ã�“ã‚Œã‚’è¿½åŠ ï¼�
    
# ä½™ã�£ã�Ÿã‚µãƒ–ãƒ—ãƒ­ãƒƒãƒˆã�¯é��è¡¨ç¤ºã�«
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------
# ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‰ï¼‹MIï¼†ç›¸é–¢ä¿‚æ•°
encoder = CrossFoldEncoder(TargetEncoder, n_splits=5)
X_encoded = encoder.fit_transform(new_df, df["accident_risk"], new_cols)

# ã‚¹ã‚³ã‚¢ã‚’ç¢ºèª�
scores_df = encoder.get_feature_scores()
print(scores_df)

# å…ƒã�®ãƒ‡ãƒ¼ã‚¿ã�«è¿½åŠ 
df = pd.concat([df, X_encoded], axis=1)


def show_effect_of_combination(base, group_cat):
    results = []
    for category, group in df.groupby(group_cat):
        if len(group) < 4:
            # ã‚µãƒ³ãƒ—ãƒ«æ•°ã�Œå°‘ã�ªã�™ã��ã‚‹ã�®ã�§ã‚¹ã‚­ãƒƒãƒ—
            continue
        
        corr = group[base].corr(group["accident_risk"])
        X = group[[base]].values
        y = group["accident_risk"].values
        mi = mutual_info_regression(X, y, random_state=0)[0]

        results.append({
            group_cat: category,
            "n_samples": len(group),
            "Pearson Correlation": corr,
            "Mutual Information": mi
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Mutual Information", ascending=False)
    print(df_results)



# curvature * speed_limit(binned)
df["speed_cat"] = pd.cut(df["speed_limit"], 3, labels=False)
sns.scatterplot(x="curvature", y="accident_risk", hue="speed_cat", data=df)
show_effect_of_combination("curvature", "speed_cat")

#                    Feature  Mutual_Information  Correlation
# 3                curvature            0.287497     0.543946
# 4              speed_limit            0.148435     0.430898


# curvature * num_reported_accidents
df["reported_cat"] = pd.cut(df["num_reported_accidents"], 4, labels=False)
sns.scatterplot(x="curvature", y="accident_risk", hue="reported_cat", data=df)
show_effect_of_combination("curvature", "reported_cat")

#                    Feature  Mutual_Information  Correlation
# 3                curvature            0.287497     0.543946
# 12  num_reported_accidents            0.072854     0.213891

