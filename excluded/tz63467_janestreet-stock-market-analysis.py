import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import optuna
plt.style.use('ggplot')


import gc
import xgboost as xgb
# import lightgbm as lgb
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# !pip install xgboost


    # def reduce_memory_usage(df: pl.DataFrame, name) -> pl.DataFrame:
    #     print(f"Memory usage of dataframe \"{name}\" is {round(df.estimated_size('mb'), 4)} MB,")
        
    #     int_types = [
    #         pl.Int8,
    #         pl.Int16,
    #         pl.Int32,
    #         pl.Int64,
    #         pl.UInt8,
    #         pl.UInt16,
    #         pl.UInt32,
    #         pl.UInt64,
    #     ]
    #     float_types = [pl.Float32, pl.Float64]
        
    #     for col in df.columns:
    #         col_type = df[col].dtype
    #         if col_type in int_types:
    #             c_min = df[col].min()
    #             c_max = df[col].max()
                
    #             if c_min is not None and c_max is not None:
                    
    #                 if col_type in int_types:
    #                     if c_min >= 0:
    #                         if (c_min >= np.iinfo(np.uint8).min
    #                             and c_max <= np.iinfo(np.uint8).max):
                                
    #                             df = df.with_columns(df[col].cast(pl.UInt8))
                                
    #                         elif (c_min >= np.iinfo(np.uint16).min
    #                             and c_max <= np.iinfo(np.uint16).max):
                                
    #                             df = df.with_columns(df[col].cast(pl.UInt16))
                                
    #                         elif (
    #                             c_min >= np.iinfo(np.uint32).min
    #                             and c_max <= np.iinfo(np.uint32).max
    #                         ):
    #                             df = df.with_columns(df[col].cast(pl.UInt32))
    #                         elif (
    #                             c_min >= np.iinfo(np.uint64).min
    #                             and c_max <= np.iinfo(np.uint64).max
    #                         ):
    #                             df = df.with_columns(df[col].cast(pl.UInt64))
    #                     else:
    #                         if (
    #                             c_min >= np.iinfo(np.int8).min
    #                             and c_max <= np.iinfo(np.int8).max
    #                         ):
    #                             df = df.with_columns(df[col].cast(pl.Int8))
    #                         elif (
    #                             c_min >= np.iinfo(np.int16).min
    #                             and c_max <= np.iinfo(np.int16).max
    #                         ):
    #                             df = df.with_columns(df[col].cast(pl.Int16))
    #                         elif (
    #                             c_min >= np.iinfo(np.int32).min
    #                             and c_max <= np.iinfo(np.int32).max
    #                         ):
    #                             df = df.with_columns(df[col].cast(pl.Int32))
    #                         elif (
    #                             c_min >= np.iinfo(np.int64).min
    #                             and c_max <= np.iinfo(np.int64).max
    #                         ):
    #                             df = df.with_columns(df[col].cast(pl.Int64))
    #                 elif col_type in float_types:
    #                     if (
    #                         c_min > np.finfo(np.float32).min
    #                         and c_max < np.finfo(np.float32).max
    #                     ):
    #                         df = df.with_columns(df[col].cast(pl.Float32))
                                
                                
    #     print(
    #         f"Memory usage of dataframe \"{name}\" became {round(df.estimated_size('mb'), 4)} MB."
    #     )

    #     return df
    
    
    # def to_pandas(df: pl.DataFrame, cat_cols: list[str] = None) -> (pd.DataFrame, list[str]):
    #     df: pd.DataFrame = df.to_pandas()

    #     if cat_cols is None:
    #         cat_cols = list(df.select_dtypes("object").columns)

    #     df[cat_cols] = df[cat_cols].astype("str")                       

    #     return df, cat_cols




# def reduce_memory_usage() -> pl.Expr:
#     expressions = [
#         pl.col(pl.Float64).cast(pl.Float32),
#         pl.col("date_id", "time_id").cast(pl.Int16),
#         pl.col("symbol_id").cast(pl.Int8),
#         ]
#     return expressions


sample_pl = []

for i in range(9):
    df_pl = pl.read_parquet(f'/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet')
    sample_pl.append(df_pl)


df_pl = pl.concat(sample_pl)
df_pl.head()


df_pl.shape


df_pl.glimpse()


null_counts = df_pl.null_count().transpose(include_header=True)
null_counts = null_counts.to_pandas()
null_counts.rename(columns={'column':'Feature', 'column_0':'Count'}, inplace=True)
null_counts = null_counts[null_counts['Count']>0]
null_counts = null_counts.sort_values('Count',ascending=False)
null_counts['Percentage'] = round((null_counts['Count']/40852762)*100,2)
null_counts


plt.figure(figsize=(70, 10))
plt.bar(null_counts['Feature'], null_counts['Count'])
plt.ylabel('count (million)', fontsize=25)
plt.xlabel('Features', fontsize=25)
plt.suptitle('Missing value counts', fontsize=40)
plt.yticks(fontsize=25)
plt.xticks(fontsize=13)
plt.show()


# df_pl_sample = df_pl[:10000000]
# df_pl_sample_symbl14 = df_pl_sample.filter(pl.col('symbol_id')==14)
# df_pl_sample_symbl14


# sns.barplot(x=df_pl.group_by('symbol_id').count()['symbol_id'],y=df_pl.group_by('symbol_id').count()['count'])
# kk = df_pl.group_by('symbol_id').count().to_pandas()
# sns.lineplot(x='symbol_id',y='count', data=kk)


# Plot distribution of a few features
features_to_plot = ['feature_00', 'feature_10', 'feature_20']
for feature in features_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_pl[feature], bins=50, kde=True)
    plt.title(f"Distribution of {feature}")
    plt.show()


features_to_plot = ['feature_04', 'feature_08', 'feature_17','feature_21','feature_32','feature_46','feature_66']
for feature in features_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_pl[feature], bins=50, kde=True)
    plt.title(f"Distribution of {feature}")
    plt.show()


features_to_plot = ['responder_2', 'responder_6', 'responder_8']
for feature in features_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_pl[feature], bins=50, kde=True)
    plt.title(f"Distribution of {feature}")
    plt.show()


date_responder6 = df_pl.select(['date_id','time_id', 'responder_6']).to_pandas()
date_responder6


# date_responder6['date'] = pd.to_datetime(date_responder6['date'])
# date_responder6.set_index('date_id', inplace=True)

plt.figure(figsize=(16, 5))
date_responder6['responder_6'].rolling(window=1000).mean().plot(color='blue',linewidth =0.05)
plt.title("Rolling Mean of Responder_6 Over Time")
plt.xlabel("Date")
plt.ylabel("Responder_6")
# plt.grid(color = 'lightgrey' , linewidth=0.8)
plt.axhline(0, color='green', linestyle='-', linewidth=1.2)
plt.show()


df_pl.corr()


df_pl_small_corr


df_pl_small_corr


# trying to get a rough idea of the features correlation by using a small subset of data
# df_pl_small = df_pl[:9000000].to_pandas()
# df_pl_small_corr = df_pl_small.corr()
# df_pl_small_corr.drop(['date_id', 'time_id', 'symbol_id'], axis=1, inplace=True)
# df_pl_small_corr.drop(['date_id', 'time_id', 'symbol_id'], axis=0, inplace=True)


# Plot heatmap
plt.figure(figsize=(96, 48))
sns.heatmap(df_pl_small_corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True, linewidth=0.05)
plt.title("All Feature Correlation Heatmap")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


#After identifying correlated features from a sample subset of the dataset, a heatmap is applied to assess whether the correlation holds.

corr_matrix = df_pl.select(['feature_12','feature_67','feature_70','feature_68', \
                           'feature_73','feature_74','feature_77','feature_78','feature_69',]).to_pandas().corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, cmap="crest", center=0, vmin=-1, vmax=1,annot=True, linewidth=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


#After identifying correlated features from a sample subset of the dataset, a heatmap is applied to assess whether the correlation holds.

corr_matrix = df_pl.select(['feature_00','feature_02','feature_21','feature_31', \
                           'feature_32','feature_34','feature_35','feature_15','feature_17','feature_29']).to_pandas().corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, cmap="coolwarm",  center=0, vmin=-1, vmax=1,annot=True, linewidth=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# df_pl.filter((pl.col('date_id')>500) & (pl.col('date_id')<=700))
# df_pl.filter(pl.col('date_id')>1500)



# Removing Highly correlated Features
"""Feature_73 - Feature_74 (.96)
- Feature_77 - Feature_78 (.96)
- Feature_67 - Feature_12 (.93)
- Feature_70 - Feature_12 (.91)
- Feature_68 - Feature_13 (.81)
- Feature_72 - Feature_14 (.86)
- Feature_69 - Feature_14 (.89)
- Feature_15 - Feature_30 (.89)
- Feature_15 - Feature_17 (.92)
- Feature_15 - Feature_29 (.8)
- Feature_31 - Feature_21 (.97)
- Feature_00 - Feature_02 (.92)
- Feature_22 - weight (.88)
- Feature_40 - Feature_43 (.89)
- Feature_40 - Feature_41 (.81)
- Feature_35 - Feature_34 (.91)
- Feature_35 - Feature_32 (.86)
- Feature_34 - Feature_32 (.84)"""

# Will add more features to it after the first model run

Features_to_remove = ['feature_74','feature_78','feature_12','feature_70','feature_13','feature_14','feature_69','feature_30',\
                     'feature_17','feature_29','feature_21','feature_02','feature_43',\
                      'feature_41','feature_34','feature_32']


 df_pl.filter(pl.col('date_id')>1520)


"""Keeping some data for test untouched. We have 1529 days of stock data. 
Out of which I am keeping 1400 days for train and validation and 129 days keeping untouched for test data."""

Test_data = df_pl.filter(pl.col('date_id')>1520)
Test_data = Test_data.sample(n=1000, seed = 0)

Train_data = df_pl.filter(pl.col('date_id')<=1520)

# spliting the train data into train and validation data.
feature_names = [f"feature_{i:02d}" for i in range(79)]
# new_features = [x for x in feature_names if x not in Features_to_remove]


X_train = Train_data.filter(pl.col('date_id')<=1200).select(feature_names)
Y_train = Train_data.filter(pl.col('date_id')<=1200).select(['responder_6'])
W_train = Train_data.filter(pl.col('date_id')<=1200).select(['weight'])


X_val = Train_data.filter((pl.col('date_id')>1200) & (pl.col('date_id')<=1520)).select(feature_names)
Y_val = Train_data.filter((pl.col('date_id')>1200) & (pl.col('date_id')<=1520)).select(['responder_6'])
W_val = Train_data.filter((pl.col('date_id')>1200) & (pl.col('date_id')<=1520)).select(['weight'])


X_test = Test_data.select(feature_names)
Y_test = Test_data.select(['responder_6'])
W_test = Test_data.select(['weight'])


# Deleteing the main data and Train data to save memory
del  Train_data, Test_data
gc.collect()


# Custom R2 metric for XGBoost
def r2_xgb(y_true, y_pred, sample_weight):    
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return -r2


model = xgb.XGBRegressor(n_estimators=1600, \
                 learning_rate=0.05,\
                 max_depth=12, \
                 # tree_method='gpu_hist',\
                 # device="cuda",\
                 objective='reg:squarederror',\
                 eval_metric=r2_xgb,\
                 verbosity=3, \
                 # disable_default_eval_metric=True, 
          early_stopping_rounds=50)


# Train XGBoost model with early stopping and verbose logging
model.fit(X_train, Y_train, sample_weight=W_train, 
          eval_set=[(X_val, Y_val)], 
          sample_weight_eval_set=[W_val], 
          verbose=True)


from sklearn.metrics import mean_squared_error

# Define the Optuna objective function
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": 'rmse',
        # "tree_method": "gpu_hist",  # Use GPU if available
        "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.4, 1.0),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1, 10),
        "gamma": trial.suggest_loguniform("gamma", 1e-3, 5.0),
    }

    # Train model
    model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
    model.fit(
        X_train, Y_train,
        sample_weight=W_train,  # Include weight
        eval_set=[(X_val, Y_val)],
        sample_weight_eval_set=[W_val],
        verbose=True
    )

    # Predict & Evaluate
    preds = model.predict(X_val)
    rmse = mean_squared_error(Y_val, preds, squared=False)
    r2_xgb = r2_xgb(Y_val, preds,W_val)
    return rmse, r2_xgb

# Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)  # Increase for better tuning

# pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")

# Best parameters
best_params = study.best_params
print("Best parameters:", best_params)

# Train final model with best parameters
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train, sample_weight=W_train)

# Save model
best_model.save_model("xgb_best_model.json")


# model.get_xgb_params()
feature_imp = pd.DataFrame({'Feature':feature_names,
    'value':model.feature_importances_})
feature_imp.sort_values(by=['value'], ascending =False, inplace=True)


plt.figure(figsize=(16, 6))
sns.barplot(x=feature_imp['Feature'], y=feature_imp['value'], alpha=0.8)
plt.xlabel("Features")
plt.ylabel("data")
plt.xticks(fontsize=8, rotation=90)
plt.title("Feature Importances")
plt.show()


# Make predictions
Y_pred = model.predict(X_test)
Y_pred = pl.from_numpy(Y_pred, schema=["Pred"])


# Evaluate model
r2_xgb = r2_xgb(Y_test, Y_pred,W_test)
print(f"Baseline Xgboost R2 score: {r2_xgb}")


Y_pred.to_series().shape


# Plot predicted vs. actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=Y_test.to_series(), y=Y_pred.to_series(), alpha=0.5)
plt.xlabel("Actual Responder_6")
plt.ylabel("Predicted Responder_6")
plt.title("Actual vs Predicted Responder_6")
plt.show()








# # Create dataset format for LightGBM
train_data = lgb.Dataset(X_train, label=Y_train,weight = W_train)
test_data = lgb.Dataset(X_val, label=Y_val,weight = W_val)

# Custom R2 metric for LightGBM
def r2_lgb(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return 'r2', r2, True


# Define parameters (basic tuning)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 256,
    'max_depth': -1,
    'device': 'gpu',  # Enable GPU
    'gpu_platform_id': 0,  # Adjust if multiple GPUs
    'gpu_device_id': 0
}

# Train model
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000)

# model_lgb = lgb.LGBMRegressor(n_estimators=500, device='gpu', gpu_use_dp=True, objective='l2')

# model_lgb.fit(train_data ,eval_set=[test_data], 
#           callbacks=[
#               lgb.early_stopping(100), 
#               lgb.log_evaluation(10)
#           ])

# model_lgb.fit(X_train, Y_train, W_train.to_numpy(),  
#           eval_metric=[r2_lgb],
#           eval_set=[(X_val, Y_val, W_val.to_numpy())], 
#           callbacks=[
#               lgb.early_stopping(100), 
#               lgb.log_evaluation(10)
#           ])


# # Make predictions
# # y_pred = model.predict(X_test)

# # Evaluate model
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# print(f"Baseline LightGBM RMSE: {rmse}")





