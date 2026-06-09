import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import time
import warnings
warnings.filterwarnings('ignore')


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')


train.shape
test.shape


train.head()


train.info()


def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

train = reduce_memory_usage(train)
test = reduce_memory_usage(test)


train.isnull().sum().sum()


train_infs = np.isinf(train.values).sum()
test_infs = np.isinf(test.values).sum()



print(f"Infs in Train: {train_infs}, Infs in Test: {test_infs}")


train_inf_counts = np.isinf(train).sum(axis=0)
test_inf_counts = np.isinf(test).sum(axis=0)

inf_summary = pd.DataFrame({
    'Train Inf Count': train_inf_counts,
    'Test Inf Count': test_inf_counts
})

columns_with_inf = inf_summary[(inf_summary['Train Inf Count'] > 0) | (inf_summary['Test Inf Count'] > 0)]

print(" Columns containing inf values:")
print(columns_with_inf.sort_values(by='Train Inf Count', ascending=False))



columns_with_inf_values = inf_summary[
    (inf_summary['Train Inf Count'] > 0) | (inf_summary['Test Inf Count'] > 0)
].index.tolist()


train.drop(columns=columns_with_inf_values, inplace=True)
test.drop(columns=columns_with_inf_values, inplace=True)


train = train.sample(frac=0.5, random_state=42)


plt.figure(figsize=(8, 5))
sns.histplot(train['label'], bins=100, kde=True, color='skyblue')
plt.title('Distribution of Target Label')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


train_df, valid_df = train_test_split(train, test_size=0.2, random_state=42)

X_train = train_df.drop(columns=['label'])
y_train = train_df['label']

X_valid = valid_df.drop(columns=['label'])
y_valid = valid_df['label']


selector = SelectKBest(score_func=f_regression, k=100)
X_train = selector.fit_transform(X_train, y_train)
X_valid = selector.transform(X_valid)


dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)


params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'gpu_hist',  
    'eval_metric': 'rmse',
    'seed': 42
}

evallist = [(dtrain, 'train'), (dvalid, 'valid')]


model = xgb.train(
    params,
    dtrain,
    num_boost_round=10000,
    evals=evallist,
    early_stopping_rounds=30,
    verbose_eval=100
)


lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid, label=y_valid)


lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'verbose': -1,
    'seed': 42
}


lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=['train', 'valid'],
    num_boost_round=10000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=100)  # Ø¨Ø¯Ù„ verbose_eval
    ]
)



def plot_rmse_and_time(results):
    df = pd.DataFrame(results)

    # Training Time
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Model', y='Train Time (s)', data=df, palette='Blues_d')
    plt.title(" Training Time Comparison")
    plt.ylabel("Seconds")
    plt.grid(True)
    plt.show()

    # RMSE
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Model', y='RMSE', data=df, palette='Reds_d')
    plt.title("RMSE Comparison on Validation Set")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.show()


print("Training XGBoost...")
start_xgb = time.time()

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

xgb_results = {}
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=10000,
    evals=[(dtrain, 'train'), (dvalid, 'valid')],
    early_stopping_rounds=30,
    evals_result=xgb_results,
    verbose_eval=False
)

end_xgb = time.time()
xgb_time = end_xgb - start_xgb

xgb_preds = xgb_model.predict(dvalid)
xgb_rmse = mean_squared_error(y_valid, xgb_preds, squared=False)

# -------------------- LightGBM ------------------------
print(" Training LightGBM...")
start_lgb = time.time()

lgb_train_data = lgb.Dataset(X_train, label=y_train)
lgb_valid_data = lgb.Dataset(X_valid, label=y_valid)

lgb_results = {}
lgb_model = lgb.train(
    lgb_params,
    lgb_train_data,
    valid_sets=[lgb_train_data, lgb_valid_data],
    valid_names=['train', 'valid'],
    num_boost_round=10000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=0),
        lgb.record_evaluation(lgb_results)
    ]
)

end_lgb = time.time()
lgb_time = end_lgb - start_lgb

lgb_preds = lgb_model.predict(X_valid)
lgb_rmse = mean_squared_error(y_valid, lgb_preds, squared=False)

# -------------------- Results Table + Plot ------------------------
results = [
    {'Model': 'XGBoost', 'Train Time (s)': xgb_time, 'RMSE': xgb_rmse},
    {'Model': 'LightGBM', 'Train Time (s)': lgb_time, 'RMSE': lgb_rmse},
]

plot_rmse_and_time(results)


plt.figure(figsize=(12, 5))
plt.plot(xgb_results['train']['rmse'], label='XGBoost Train RMSE')
plt.plot(xgb_results['valid']['rmse'], label='XGBoost Valid RMSE')
plt.plot(lgb_results['train']['rmse'], label='LightGBM Train RMSE')
plt.plot(lgb_results['valid']['rmse'], label='LightGBM Valid RMSE')
plt.title("ðŸ“Š RMSE Over Boosting Rounds")
plt.xlabel("Boosting Round")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# XGBoost Importance
xgb.plot_importance(xgb_model, max_num_features=20, importance_type='gain', title='XGBoost Feature Importance (Top 20)', height=0.5)
plt.tight_layout()
plt.show()


# LightGBM Importance
lgb.plot_importance(lgb_model, max_num_features=20, importance_type='gain', title='LightGBM Feature Importance (Top 20)', height=0.5)
plt.tight_layout()
plt.show()



Submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')



columns_to_drop = [col for col in columns_with_inf_values if col in test.columns]

test.drop(columns=columns_to_drop, inplace=True)

X_test = test.drop(columns=['label'], errors='ignore')

X_test_selected = selector.transform(X_test)
dtest = xgb.DMatrix(X_test_selected)
predictions = model.predict(dtest)

submission = pd.DataFrame({
    'ID': test.index + 1,
    'prediction': predictions
})

submission.to_csv('submission.csv', index=False)


