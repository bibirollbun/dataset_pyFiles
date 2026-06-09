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
warnings.filterwarnings("ignore")


input_file_path = '/kaggle/input/ultimate-customer-churn-prediction-challenge/train.csv'
src_df = pd.read_csv(input_file_path)
print(src_df.info())


src_df.sample(3)


pd.reset_option('display.max_rows')
src_df['Customer_ID'].value_counts()


# Customer_ID をインデックスに設定する
src_df.set_index("Customer_ID", inplace=True)


src_df["Location"].value_counts()


src_df["Subscription_Type"].value_counts()


from matplotlib import pyplot as plt
import seaborn as sns


src_df.describe()


sns.pairplot(src_df)
plt.show()


sns.heatmap(
    src_df.select_dtypes(include=["number"]).corr(),
    annot=True,
    fmt=".1f"
)
plt.show()


cat_cols = ['Gender', 'Location', 'Subscription_Type', 'Last_Interaction_Type']

# カテゴリ値をワンホットエンコーディングする
df_with_dummies = pd.get_dummies(src_df, columns=cat_cols, drop_first=False, dtype=int)

# カラムの表示数を制限なしに設定
pd.set_option('display.max_columns', None)

# カテゴリ値について、サブスクリプションタイプ以外は一様分布
display(
    df_with_dummies.describe()
)

pd.reset_option('display.max_columns')


from itertools import combinations

# 統合用の空のDataFrame
expected_df = pd.DataFrame()

# ペアごとに期待度数を計算して統合
for col1, col2 in combinations(cat_cols, 2):
    observed = pd.crosstab(src_df[col1], src_df[col2])
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    grand_total = observed.values.sum()

    # 期待度数の計算
    for row_cat in observed.index:
        for col_cat in observed.columns:
            expected_count = row_totals[row_cat] * col_totals[col_cat] / grand_total
            expected_df = pd.concat([
                expected_df,
                pd.DataFrame({
                    col1: [row_cat],
                    col2: [col_cat],
                    'expected_count': [expected_count],
                    'pair': [f"{col1} × {col2}"]
                })
            ], ignore_index=True)

# 全組み合わせの期待度数を昇順で表示する
display(expected_df.sort_values(by="expected_count"))


from scipy.stats import chi2_contingency

# 空のDataFrameを作成（行と列は変数名）
pval_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)

# ペアごとにカイ二乗検定を実施
for col1, col2 in combinations(cat_cols, 2):
    table = pd.crosstab(src_df[col1], src_df[col2])
    _, p, _, _ = chi2_contingency(table)
    pval_matrix.loc[col1, col2] = round(p, 4)
    pval_matrix.loc[col2, col1] = round(p, 4)

# 対角成分は1.0（同じ変数同士）
for col in cat_cols:
    pval_matrix.loc[col, col] = 1.0

# 結果表示
print("カテゴリ変数同士の独立性の検定結果（p値）")
display(pval_matrix)


# 統合用の空のDataFrame
expected_df = pd.DataFrame()

col1 = "Churn"

# ペアごとに期待度数を計算して統合
for col2 in cat_cols:
    observed = pd.crosstab(src_df[col1], src_df[col2])
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    grand_total = observed.values.sum()

    # 期待度数の計算
    for row_cat in observed.index:
        for col_cat in observed.columns:
            expected_count = row_totals[row_cat] * col_totals[col_cat] / grand_total
            expected_df = pd.concat([
                expected_df,
                pd.DataFrame({
                    col1: [row_cat],
                    col2: [col_cat],
                    'expected_count': [expected_count],
                    'pair': [f"{col1} × {col2}"]
                })
            ], ignore_index=True)

# 全組み合わせの期待度数を昇順で表示する
display(expected_df.sort_values(by="expected_count"))


col1 = "Churn"

# 空のDataFrameを作成（行と列は変数名）
pval_matrix = pd.DataFrame(index=[col1], columns=cat_cols)

# ペアごとにカイ二乗検定を実施
for col2 in cat_cols:
    table = pd.crosstab(src_df[col1], src_df[col2])
    _, p, _, _ = chi2_contingency(table)
    pval_matrix.loc[col1, col2] = round(p, 4)

# 結果表示
print("カテゴリ変数同士の独立性の検定結果（p値）")
display(pval_matrix)


df_with_dummies.describe()


from scipy.stats import mannwhitneyu

num_cols = [
    "Age", "Account_Age_Months", "Monthly_Spending", "Total_Usage_Hours", "Support_Calls", "Late_Payments", 
    "Streaming_Usage", "Discount_Used", "Satisfaction_Score", "Complaint_Tickets", "Promo_Opted_In"
]

rslt_df = pd.DataFrame(index=["Gender"], columns=num_cols)

for col in num_cols:
    # 例：Gender × Monthly_Spend
    group1 = src_df[src_df['Gender'] == 'Male'][col]
    group2 = src_df[src_df['Gender'] == 'Female'][col]
    
    # Mann–Whitney U検定の実行
    stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
    rslt_df.loc["Gender", col] = round(p, 4)

display(rslt_df)


from scipy.stats import kruskal

def run_kruskal_wallis(df, cat_var, num_var):
    """
    Kruskal–Wallis検定を実行する関数

    Parameters:
        df (pd.DataFrame): 対象のデータフレーム
        cat_var (str): カテゴリ変数名
        num_var (str): 数値変数名

    Returns:
        stat (float): 検定統計量
        p_value (float): p値
    """
    # カテゴリごとに数値データを抽出
    groups = [group[num_var].values for _, group in df.groupby(cat_var)]

    # 検定実行
    stat, p_value = kruskal(*groups)
    # print(f"Kruskal–Wallis検定: H={stat:.3f}, p={p_value:.4f}")
    # return stat, p_value
    return p_value


rslt_df = pd.DataFrame(columns=num_cols)

for cat_col in cat_cols[1:]:
    for num_col in num_cols:
        p = run_kruskal_wallis(src_df, cat_col, num_col)
        rslt_df.loc[cat_col, num_col] = round(p, 4)

display(rslt_df)


# 描画スタイル設定
sns.set(style="whitegrid")
plt.figure(figsize=(18, 15))

# 1. Location × Account_Age_Months
plt.subplot(3, 1, 1)
sns.violinplot(x='Location', y='Account_Age_Months', data=src_df, inner='box', palette='Set2')
plt.title('Account Age by Location')
plt.xlabel('Location')
plt.ylabel('Account Age (Months)')

# 2. Subscription_Type × Support_Calls
plt.subplot(3, 1, 2)
sns.violinplot(x='Subscription_Type', y='Support_Calls', data=src_df, inner='box', palette='Set3')
plt.title('Support Calls by Subscription Type')
plt.xlabel('Subscription Type')
plt.ylabel('Number of Support Calls')

# 3. Last_Interaction_Type × Support_Calls
plt.subplot(3, 1, 3)
sns.violinplot(x='Last_Interaction_Type', y='Support_Calls', data=src_df, inner='box', palette='Pastel1')
plt.title('Support Calls by Last Interaction Type')
plt.xlabel('Last Interaction Type')
plt.ylabel('Number of Support Calls')

# レイアウト調整と表示
plt.tight_layout()
plt.show()




