import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Display settings
pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")


# Load dataset
df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")

# Display first rows
df.head()


df.tail()


df.shape


df.info()


df.isna().sum().sort_values(ascending=False)


df.nunique()


df.columns


df.isna().sum().sort_values(ascending=False)


df.isna().mean() * 100


df.duplicated().sum()


df["id"].nunique(), df.shape[0]


df.dtypes


categorical_cols = df.select_dtypes(include="object").columns
categorical_cols


for col in categorical_cols:
    print(f"\n{col}")
    print(df[col].unique())


df_clean = df.drop(columns=["id"])


num_cols = df_clean.select_dtypes(exclude="object").columns
cat_cols = df_clean.select_dtypes(include="object").columns

num_cols, cat_cols


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

