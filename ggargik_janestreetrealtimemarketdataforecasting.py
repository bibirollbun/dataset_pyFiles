!pip install rtdl_num_embeddings -q --no-index --find-links=/kaggle/input/jane-street-import/rtdl_num_embeddings


import os , gc
import pandas as pd
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)

from sklearn.metrics import r2_score

import torch.optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm
from collections import OrderedDict
from my_utility import Model, make_parameter_groups

import warnings
import joblib
from pytorch_lightning.callbacks import Callback

import lightgbm as lgb
from lightgbm import LGBMRegressor, Booster
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None


ROOT_DIR = "/kaggle/input/jane-street-real-time-market-data-forecasting/"


#listall files and directories
for root, dirs, files in os.walk(ROOT_DIR):
    print(f"Directory: {root}")
    for file in files:
        print(f"  File: {file}")


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


plt.figure(figsize=(10, 10))
sns.heatmap(features[[ f"tag_{no}" for no in range(0,17,1) ] ].T.corr(), square=True, cmap="jet")
plt.xlabel("feature_0  ~  feature_78")
plt.ylabel("feature_0  ~  feature_78")
plt.show()


responders = pd.read_csv(f"{ROOT_DIR}/responders.csv")
responders


sns.heatmap(responders[[ f"tag_{no}" for no in range(0,5,1) ] ].T.corr(), annot=True, square=True, cmap="jet")
plt.xlabel("responder_0  ~  responder_8")
plt.ylabel("responder_0  ~  responder_8")
plt.show()



sub = pd.read_csv(f"{ROOT_DIR}/sample_submission.csv")
print( f"sub.shape = {sub.shape}")
sub


train = (
    pl.read_parquet(f"{ROOT_DIR}/train.parquet/partition_id=0/part-0.parquet")
)
train.shape


print(train.head())
print(str(train.columns))


plt.figure(figsize=(15, 15))
sns.heatmap(train[[ f"feature_{target:02d}" for target in range(79)]].corr(), square=True, cmap="jet")
plt.xlabel("feature_00  ~  feature_78")
plt.ylabel("feature_00  ~  feature_78")
plt.grid()
plt.show()
plt.figure(figsize=(15, 15))
sns.heatmap(train[[ f"responder_{target}" for target in range(9)]].corr(), square=True, cmap="jet")
plt.xlabel("responder_0  ~  responder_8")
plt.ylabel("responder_0  ~  responder_8")
plt.grid()
plt.show()


for target in range(9):
    col = f"responder_{target}"
    mean_, sgm_ = train[col].mean(), np.sqrt(train[col].var())
    min_, max_ = train[col].min(), train[col].max()
    print("-" * 30)
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


test = (
    pl.read_parquet(f"{ROOT_DIR}/test.parquet/date_id=0/part-0.parquet")
)
print(test.shape)
test


#missin val
supervised_usable = (
    test
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


lags = (
    pl.read_parquet(f"{ROOT_DIR}/lags.parquet/date_id=0/part-0.parquet")
)
print(lags.shape)
print(lags.columns)
lags


plt.plot(lags["responder_6_lag_1"])
plt.grid()
plt.xlabel("symbol_id")
plt.ylabel("responder_6_lag_1")
plt.show()


subset_features = [f"feature_{i:02d}" for i in range(10)]  

train = train.with_columns([pl.col(f).cast(pl.Float32) for f in subset_features])

correlation_matrix = (
    train.select(subset_features)
    .to_pandas()
    .corr()
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    square=True,
    cmap="coolwarm", 
    annot=False, 
    cbar_kws={"label": "Correlation Coefficient"}
)
plt.title("Feature Correlation Heatmap (Subset)")
plt.show()


numeric_cols = [f"feature_{i:02d}" for i in range(79)] 
numeric_data = train.select(numeric_cols).to_pandas()

numeric_data.hist(
    figsize=(20, 15), bins=30, color='darkblue', alpha=0.7, grid=False
)

plt.suptitle("Distribution of Features", fontsize=16)
plt.show()


import plotly.express as px

pairplot_data_sample = train.sample(n=2000, seed=42).select([f"feature_{i:02d}" for i in range(5)] + ["responder_6"])

fig = px.scatter_matrix(
    pairplot_data_sample,
    dimensions=[f"feature_{i:02d}" for i in range(5)],
    color="responder_6",
    title="Scatter Matrix (Pairplot Alternative with Plotly)"
)
fig.show()


correlation_matrix = train.corr()
print(correlation_matrix)



import polars as pl

train_pandas = train.to_pandas()

correlations = train_pandas.corr()["responder_6"].sort_values(ascending=False)

print("Top positively correlated features:")
print(correlations.head(10))  

print("\nTop negatively correlated features:")
print(correlations.tail(10))  

threshold = 0.1 
top_features = correlations[correlations.abs() > threshold].index.tolist()

top_features.remove("responder_6")  

print(f"\nTop features selected based on correlation: {top_features}")

test_pandas = test.to_pandas()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


correlation_matrix = train_pandas.corr()

plt.figure(figsize=(15, 12))
sns.heatmap(
    correlation_matrix, 
    annot=False,  
    cmap="coolwarm",
    vmin=-1, 
    vmax=1
)
plt.title("Correlation Matrix Heatmap")
plt.show()




#print(train.collect_schema())  


import polars as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class CONFIG:
    target_col = "responder_6"
    lag_cols_original = ["date_id", "symbol_id"] + [f"responder_{idx}" for idx in range(9)]
    lag_cols_rename = {f"responder_{idx}": f"responder_{idx}_lag_1" for idx in range(9)}
    valid_ratio = 0.05
    start_dt = 1100


train = pl.scan_parquet(f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet") \
    .select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"),
        pl.all(),
    ) \
    .with_columns(
        (pl.col(CONFIG.target_col) * 2).cast(pl.Int32).alias("label"),
    ) \
    .filter(
        pl.col("date_id").gt(CONFIG.start_dt)
    )


''''lags = train.select(pl.col(CONFIG.lag_cols_original))
lags = lags.rename(CONFIG.lag_cols_rename)
lags = lags.with_columns(
    date_id = pl.col('date_id') + 1,  # lag by 1day
    )
lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last()  
lags'''


lags = train.select(pl.col(CONFIG.lag_cols_original))
lags = lags.rename(CONFIG.lag_cols_rename)
lags = lags.with_columns(
    date_id=pl.col('date_id') + 1,  # lag by 1 day
)

for col in lags.collect_schema().names():
        if col not in ["date_id", "symbol_id"]:
            lags = lags.with_columns(
                pl.col(col).shift(1).alias(f"{col}_lag_1")
            )


#train/validationsplit
len_train = train.select(pl.col("date_id")).collect().shape[0]
valid_records = int(len_train * CONFIG.valid_ratio)
len_ofl_mdl = len_train - valid_records
last_tr_dt = train.select(pl.col("date_id")).collect().row(len_ofl_mdl)[0]

print(f"\nlen_train = {len_train}")
print(f"len_ofl_mdl = {len_ofl_mdl}")
print(f"---> Last offline train date = {last_tr_dt}\n")


training_data = train.filter(pl.col("date_id").le(last_tr_dt))
validation_data = train.filter(pl.col("date_id").gt(last_tr_dt))


train = train.collect()  
train = train.join(lags, on=["date_id", "symbol_id"], how="left")
print("Schema after join:", train.schema())


scaler = MinMaxScaler()
feature_cols = [col for col in train.collect_schema().names() if col not in ["date_id", "symbol_id", "label"]]

train_df = train.collect()
train_features = train_df.select(fearure_cols[:91]).to_numpy()

train_features_scaled = scaler.fit_transform(train_features())
for idx, col in enumerate(feature_cols):
    train_df = train_df.with_columns(
        pl.Series(name=f"{col}_scaled", values=train_features_scaled[:, idx])
    )

validation_df = validation_data.collect()
validation_features = validation_df[feature_cols].to_numpy()
validation_features_scaled = scaler.transform(validation_features)
for idx, col in enumerate(feature_cols):
    validation_df= validation_df.with_columns(
        pl.Series(name=f"{col}_scaled", values=validation_features_scaled[:, idx])
    )


train_scaled_df.write_parquet("train_scaled.parquet", partition_by="date_id")
validation_scaled_df.write_parquet("validation_scaled.parquet", partition_by="date_id")

print("\nPreprocessing completed. Processed data saved as:")
print("- train_scaled.parquet")
print("- validation_scaled.parquet")



train = pd.concat([train, valid]).reset_index(drop=True)
train.shape


print(training_data.explain())
print(validation_data.explain())


#--------------
train = pl.scan_parquet("/kaggle/input/js24-preprocessing-create-lags/training.parquet").collect().to_pandas()
valid = pl.scan_parquet("/kaggle/input/js24-preprocessing-create-lags/validation.parquet").collect().to_pandas()
train.shape, valid.shape


def get_model(seed):
    # XGBoost parameters
    XGB_Params = {
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1,
        'reg_lambda': 5,
        'random_state': seed,
        'tree_method': 'gpu_hist',
        'device' : 'cuda',
        'n_gpus' : 2,
    }
    
    XGB_Model = XGBRegressor(**XGB_Params)
    return XGB_Model


X_train = train[ CONFIG.feature_cols ]
y_train = train[ CONFIG.target_col ]
w_train = train[ "weight" ]
X_valid = valid[ CONFIG.feature_cols ]
y_valid = valid[ CONFIG.target_col ]
w_valid = valid[ "weight" ]

X_train.shape, y_train.shape, w_train.shape, X_valid.shape, y_valid.shape, w_valid.shape


%%time
model = get_model(CONFIG.seed)
model.fit( X_train, y_train, sample_weight=w_train)


y_pred_train1 = model.predict(X_train.iloc[:X_train.shape[0]//2])
y_pred_train2 = model.predict(X_train.iloc[X_train.shape[0]//2:])
train_score = r2_score(y_train, np.concatenate([y_pred_train1, y_pred_train2], axis=0), sample_weight=w_train )
train_score


y_pred_valid = model.predict(X_valid)
valid_score = r2_score(y_valid, y_pred_valid, sample_weight=w_valid )
valid_score


y_means = { symbol_id : -1 for symbol_id in range(39) }
for symbol_id, gdf in train[["symbol_id", CONFIG.target_col]].groupby("symbol_id"):
    y_mean = gdf[ CONFIG.target_col ].mean()
    y_means[symbol_id] = y_mean
    print(f"symbol_id = {symbol_id}, y_means = {y_mean:.5f}")


cv_detail = { symbol_id : 0 for symbol_id in range(39) }
for symbol_id, gdf in valid.groupby("symbol_id"):
    X_valid = gdf[ CONFIG.feature_cols ]
    y_valid = gdf[ CONFIG.target_col ]
    w_valid = gdf[ "weight" ]
    y_pred_valid = model.predict(X_valid)
    score = r2_score(y_valid, y_pred_valid, sample_weight=w_valid )
    cv_detail[symbol_id] = score
    
    print(f"symbol_id = {symbol_id}, score = {score:.5f}")


sids = list(cv_detail.keys())
plt.bar(sids, [cv_detail[sid] for sid in sids])
plt.grid()
plt.xlabel("symbol_id")
plt.ylabel("CV score")
plt.show()


result = {
    "model" : model,
    "cv" : valid_score,
    "cv_detail" : cv_detail,
    "y_mean" : y_means,
}
with open("result.pkl", "wb") as fp:
    pickle.dump(result, fp)











#correlation w responder_6
if "responder_6" in correlation_matrix.columns:
    correlation_with_responder_6 = correlation_matrix["responder_6"].drop("responder_6").sort_values(ascending=False)
    top_features = correlation_with_responder_6.head(10).index 
    valid_top_features = [feature for feature in top_features if feature in test.columns]
    print(f"Valid top features: {valid_top_features}")
    print(f"Top features correlated with 'responder_6': {list(top_features)}")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=correlation_with_responder_6.loc[valid_top_features],
        y=valid_top_features,
        palette="coolwarm",
    )
    plt.title("Top Features Correlated with Responder_6", fontsize=14)
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Features")
    plt.show()

    train_data = train_pandas[list(valid_top_features) + ["responder_6"]]

    X = train_data.drop(columns=["responder_6"]) 
    y = train_data["responder_6"] 
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy='mean')  # Use 'median' if preferred
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)


    print(f"R2 Score (Train): {r2_train:.4f}")
    print(f"R2 Score (Test): {r2_test:.4f}")

    if hasattr(model, 'coef_'):
        feature_importance = pd.Series(model.coef_, index=X.columns)
        feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), title='Feature Importance')
        plt.show()
    else:
        print("Model does not support feature importance.")





test_data = test.select(valid_top_features).to_pandas()

print(test_data.head())



test_data_imputed = imputer.transform(test_data)
test_data_imputed = pd.DataFrame(test_data_imputed, columns=test_data.columns)

test_data_scaled = scaler.transform(test_data_imputed)

test_predictions = model.predict(test_data_scaled)

submission = pd.DataFrame({
    'row_id': test_pandas['row_id'], 
    'responder_6': test_predictions
})
submission.to_parquet('submission.parquet', index=False)
print("Submission file created: submission.parquet")





