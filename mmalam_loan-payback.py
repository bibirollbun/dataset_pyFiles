import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as ws
from scipy.stats import chi2_contingency


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df.head()


len(df)


df.isnull().any()


sns.displot(data=df, x="annual_income", hue="loan_paid_back", height=5,aspect=3  )


df_annual_income_1 = df[df["annual_income"] <= 12000]
df_annual_income_2 = df[df["annual_income"] > 12000]


sns.displot(data=df_annual_income_1, x="annual_income", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_annual_income_2, x="annual_income", hue="loan_paid_back", height=5,aspect=3  )


df_annual_income_3 = df_annual_income_2[df_annual_income_2["annual_income"] <= 120000]
df_annual_income_4 = df_annual_income_2[df_annual_income_2["annual_income"] > 120000]


sns.displot(data=df_annual_income_3, x="annual_income", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_annual_income_4, x="annual_income", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df, x="debt_to_income_ratio", hue="loan_paid_back", height=5,aspect=3  )


df_debt_to_income_ratio_1 = df[df["debt_to_income_ratio"] <= 0.18]
df_debt_to_income_ratio_2 = df[df["debt_to_income_ratio"] > 0.18]


sns.displot(data=df_debt_to_income_ratio_1, x="debt_to_income_ratio", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_2, x="debt_to_income_ratio", hue="loan_paid_back", height=5,aspect=3  )


df_debt_to_income_ratio_3 = df_debt_to_income_ratio_2[df_debt_to_income_ratio_2["debt_to_income_ratio"] <= 0.4]
df_debt_to_income_ratio_4 = df_debt_to_income_ratio_2[df_debt_to_income_ratio_2["debt_to_income_ratio"] > 0.4]


sns.displot(data=df_debt_to_income_ratio_3, x="debt_to_income_ratio", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_4, x="debt_to_income_ratio", hue="loan_paid_back", height=5,aspect=3  )


df["dti_category"] = pd.cut(
    df["debt_to_income_ratio"],
    bins=[-float("inf"), 0.175, 0.4, float("inf")],
    labels=["Low (≤0.175)", "Medium (0.175–0.4)", "High (≥0.4)"]
)

conf_matrix = pd.crosstab(df["dti_category"], df["loan_paid_back"])

conf_matrix["% Defaulters"] = (
    conf_matrix[0] / (conf_matrix[0] + conf_matrix[1]) * 100
).round(2)
print(conf_matrix)


sns.displot(data=df, x="credit_score", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_1, x="credit_score", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_3, x="credit_score", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_4, x="credit_score", hue="loan_paid_back", height=5,aspect=3  )


df_debt_to_income_ratio_3_cs_670_1 = df_debt_to_income_ratio_3[df_debt_to_income_ratio_3["credit_score"] >= 670]
df_debt_to_income_ratio_3_cs_670_2 = df_debt_to_income_ratio_3[df_debt_to_income_ratio_3["credit_score"] < 670]


sns.displot(data=df_debt_to_income_ratio_3_cs_670_1, x="credit_score", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_3_cs_670_2, x="credit_score", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df, x="loan_amount", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_1, x="loan_amount", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_3, x="loan_amount", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_4, x="loan_amount", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df, x="interest_rate", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_1, x="interest_rate", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_3, x="interest_rate", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_4, x="interest_rate", hue="loan_paid_back", height=5,aspect=3  )


df_interest_rate_13_dt1_3 = df_debt_to_income_ratio_3[df_debt_to_income_ratio_3["interest_rate"] >= 670]


sns.displot(data=df_interest_rate_13_dt1_3, x="interest_rate", hue="loan_paid_back", height=5,aspect=3  )


ids_to_move = df_interest_rate_13_dt1_3["id"]
df_debt_to_income_ratio_4 = pd.concat([
    df_debt_to_income_ratio_4,
    df_debt_to_income_ratio_3[df_debt_to_income_ratio_3["id"].isin(ids_to_move)]
], ignore_index=True)
df_debt_to_income_ratio_3= df_debt_to_income_ratio_3[~df_debt_to_income_ratio_3["id"].isin(ids_to_move)]


sns.displot(data=df_debt_to_income_ratio_1, x="debt_to_income_ratio", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_3, x="debt_to_income_ratio", hue="loan_paid_back", height=5,aspect=3  )


sns.displot(data=df_debt_to_income_ratio_4, x="debt_to_income_ratio", hue="loan_paid_back", height=5,aspect=3  )




