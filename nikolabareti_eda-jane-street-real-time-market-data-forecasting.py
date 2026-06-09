import pandas as pd
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns


ROOT_DIR = "/kaggle/input/jane-street-real-time-market-data-forecasting"


features = pd.read_csv(f"{ROOT_DIR}/features.csv")
features


plt.figure(figsize=(20, 10))
plt.imshow(features.iloc[:, 1:].T.values, cmap="gray")
plt.xlabel("feature_00  ~  feature_78")
plt.ylabel("tag_0  ~  tag_16")
plt.yticks(np.arange(17))
plt.xticks(np.arange(79))
plt.grid()
plt.show()


# corr between feature_XX and feature_YY
plt.figure(figsize=(10, 10))
sns.heatmap(features[[ f"tag_{no}" for no in range(0,17,1) ] ].T.corr(), square=True, cmap="jet")
plt.show()


responders = pd.read_csv(f"{ROOT_DIR}/responders.csv")
responders


# corr between responder_XX and responder_YY
sns.heatmap(responders[[ f"tag_{no}" for no in range(0,5,1) ] ].T.corr(),  annot=True, square=True, cmap="jet")
plt.xlabel("responder_0  ~  responder_8")
plt.ylabel("responder_0  ~  responder_8")
plt.show()


sub = pd.read_csv(f"{ROOT_DIR}/sample_submission.csv")
print( f"sub.shape = {sub.shape}" )
sub


!tree /kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/


train = (
    pl.read_parquet(f"{ROOT_DIR}/train.parquet/partition_id=0/part-0.parquet")
)
train.shape


train.head()


print(str(train.columns))


for partition_id in range(10):
    print(f"> train.parquet/partition_id={partition_id}/part-0.parquet")
    train = pl.read_parquet(f"{ROOT_DIR}/train.parquet/partition_id={partition_id}/part-0.parquet")
    supervised_usable = (
    train
    .filter(pl.col('responder_6').is_not_null())
    )
    
    missing_count = (
        supervised_usable
        .null_count()
        .transpose(include_header=True,
                   header_name='feature',
                   column_names=['null_count'])
        .sort('null_count', descending=True)
        .with_columns((pl.col('null_count') / len(supervised_usable)).alias('null_ratio'))
    )
    
    plt.figure(figsize=(6, 20))
    plt.title(f'Missing values over the {len(supervised_usable)} samples which have a target')
    plt.barh(np.arange(len(missing_count)), missing_count.get_column('null_ratio'), color='coral', label='missing')
    plt.barh(np.arange(len(missing_count)), 
             1 - missing_count.get_column('null_ratio'),
             left=missing_count.get_column('null_ratio'),
             color='darkseagreen', label='available')
    plt.yticks(np.arange(len(missing_count)), missing_count.get_column('feature'))
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.xlim(0, 1)
    plt.legend()
    plt.show()


supervised_usable = (
    train
    .filter(pl.col('responder_6').is_not_null())
)

missing_count = (
    supervised_usable
    .null_count()
    .transpose(include_header=True,
               header_name='feature',
               column_names=['null_count'])
    .sort('null_count', descending=True)
    .with_columns((pl.col('null_count') / len(supervised_usable)).alias('null_ratio'))
)

plt.figure(figsize=(6, 20))
plt.title(f'Missing values over the {len(supervised_usable)} samples which have a target')
plt.barh(np.arange(len(missing_count)), missing_count.get_column('null_ratio'), color='coral', label='missing')
plt.barh(np.arange(len(missing_count)), 
         1 - missing_count.get_column('null_ratio'),
         left=missing_count.get_column('null_ratio'),
         color='darkseagreen', label='available')
plt.yticks(np.arange(len(missing_count)), missing_count.get_column('feature'))
plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
plt.xlim(0, 1)
plt.legend()
plt.show()


plt.figure(figsize=(15, 15))
sns.heatmap(train[[ f"feature_{target:02d}" for target in range(79)]].corr(), square=True, cmap="jet")
plt.xlabel("feature_00  ~  feature_78")
plt.ylabel("feature_00  ~  feature_78")
plt.grid()
plt.show()


for target in range(9):
    col = f"responder_{target}"
    mean_, sgm_ = train[col].mean(), np.sqrt(train[col].var())
    min_, max_ = train[col].min(), train[col].max()
    print("- " * 30)
    print( f"column = {col}" )
    print( f" - mean  : {mean_:.4f}",  )
    print( f" - sigma : {sgm_:.4f}",  )
    print( f" - min  : {min_:.4f}",  )
    print( f" - max  : {max_:.4f}",  )
    
    plt.hist(train[col], bins=20)
    plt.xlabel(col)
    plt.ylabel("frequency / records")
    #plt.yscale("log")
    plt.grid()
    plt.show()


plt.figure(figsize=(8, 8))
sns.heatmap(train[[ f"responder_{target}" for target in range(9)]].corr(),  annot=True, square=True, cmap="jet")
plt.xlabel("responder_0  ~  responder_8")
plt.ylabel("responder_0  ~  responder_8")
plt.show()


for partition_id in range(10):
    print(f"> train.parquet/partition_id={partition_id}/part-0.parquet")
    train_data = pl.read_parquet(f"{ROOT_DIR}/train.parquet/partition_id={partition_id}/part-0.parquet")

    print( f"symbol_id: ", train_data["symbol_id"].min(), "-", train_data["symbol_id"].max())
    bins = train_data["symbol_id"].max() - train_data["symbol_id"].min() + 1
    plt.hist(train_data["symbol_id"], bins=bins)
    plt.xlabel("symbol_id")
    plt.ylabel("frequency / records")
    plt.grid()
    plt.show()


for partition_id in range(10):
    print(f"> train.parquet/partition_id={partition_id}/part-0.parquet")
    train_data = pl.read_parquet(f"{ROOT_DIR}/train.parquet/partition_id={partition_id}/part-0.parquet")

    print( f"date_id: ", train_data["date_id"].min(), "-", train_data["date_id"].max())
    bins = train_data["date_id"].max() - train_data["date_id"].min() + 1
    plt.hist(train_data["date_id"], bins=bins)
    plt.xlabel("date_id")
    plt.ylabel("frequency / records")
    plt.grid()
    plt.show()

