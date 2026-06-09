import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


!pip install ydata-profiling -q


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


train_df = pd.concat([train, train_extra], axis="rows")
train_df.shape


train_df.head()


train_df = train_df.drop(columns="id")
train_df.shape


train_df = train_df.reset_index(drop=True)


train_df = train_df.drop_duplicates()


train_df.isna().sum().plot(kind="barh")


pd.Series(train_df["Price"]).plot(kind="hist")


pd.Series(train_df["Weight Capacity (kg)"]).plot(kind="hist")


sample_df = train_df.sample(n=100000)


from ydata_profiling import ProfileReport


profile = ProfileReport(sample_df, title="Profiling Report")


profile.to_notebook_iframe()

