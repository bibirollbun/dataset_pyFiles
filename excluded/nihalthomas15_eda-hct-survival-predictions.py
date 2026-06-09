import pandas as pd

df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")


import warnings
warnings.filterwarnings("ignore")

df.head()


df.info


df.describe()


print("\n".join(f"{col} = {df[col].unique()}" for col in df.select_dtypes(include='object')))


df.columns.tolist()


df.isnull().sum()


df.nunique()

