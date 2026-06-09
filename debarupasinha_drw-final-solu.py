import sys
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def feature_engineering(df):

    df = df.copy()
    
    df['bid_ask_interaction'] = df['ask_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)

    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume'])
    
    # Price Pressure Indicators
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']

    # Liquidity Depth Measures
    df['total_depth'] = df['ask_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['ask_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['ask_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])

     # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)

    # Market Activity Indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['ask_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Microstructure Volatility Proxies
    df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    
    # Complex Interaction Terms
    df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['ask_qty'] - df['ask_qty'])

    
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df


train_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
train_df.head(10)


train_df.shape


train_df.isnull().sum()


test_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")


test_df.shape


#importing optuna and train test split

import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

base_feature = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]

top_100_feature = ["X683", "X140", "X758", "X425", "X752", "X344", "X292", "X646",
                   "X287", "X134", "X385", "X647", "X682", "X279", "X345", "X466",
                   "X381", "X778", "X283", "X739", "X427", "X272", "X684", "X301",
                   "X198", "X465", "X608", "X738", "X384", "X137", "X386", "X581",
                   "X734", "X180", "X589", "X421", "X610", "X780", "X387", "X772",
                   "X654", "X428", "X96", "X779", "X426", "X98", "X591", "X650",
                   "X613", "X566", "X605", "X181", "X750", "X174", "X288", "X607",
                   "X579", "X176", "X508", "X178", "X419", "X219", "X343", "X89",
                   "X678", "X588", "X40", "X293", "X411", "X757", "X337", "X285",
                   "X295", "X341", "X443", "X179", "X575", "X751", "X92", "X562",
                   "X769", "X776", "X501", "X298", "X375", "X95", "X590", "X611",
                   "X94", "X270", "X424", "X86", "X587", "X434", "X638", "X170",
                   "X297", "X136", "X97", "X572"]


top_30_feature = top_100_feature[:30]

FEATURES = [*base_feature, *top_30_feature]

LABEL_COLUMN = "label"

#X=train_df[FEATURES]

#X_test = test_df[FEATURES]

X =  feature_engineering(train_df[FEATURES])
X_test =  feature_engineering(test_df[FEATURES])
y = train_df[LABEL_COLUMN]





from sklearn.model_selection import train_test_split

# Split data 
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



X_train.shape


X_valid.shape


y_train.shape


y_valid.shape


# Hyperparameter tuning

def objective(trial):
    
    params = {
        "tree_method": "gpu_hist",
        "device": "cuda",
        "verbosity": 0,
        "random_state": 42,
        "n_jobs": -1,
        
        # Learning parameters
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000), 
        
        # Tree structure
        "max_depth": trial.suggest_int("max_depth", 4, 12),  # avoid too shallow
        "max_leaves": trial.suggest_int("max_leaves", 16, 64),  
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),  # smaller range
        
        # Subsampling (avoid starving trees)
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
        
        # Regularization (smaller ranges to prevent over-penalizing)
        "gamma": trial.suggest_float("gamma", 0, 3.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10.0)
    }
    

    
    

    model = xgb.XGBRegressor(early_stopping_rounds=50,**params)    
    model.fit(
            X_train, 
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False)
        
    preds = model.predict(X_valid)
    r, _ = pearsonr(y_valid, preds)

    return r


# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best Pearson correlation:", study.best_value)
print("Best Parameters:", study.best_params)



best_params = study.best_params

# Add required non-tuned params

best_params.update({
    "tree_method": "hist",
    "device": "cuda",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    "eval_metric": "rmse"
})

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# predictions on validation data
y_pred = final_model.predict(X_valid)

# evaluating accuracy metrics
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")



# actual vs predicted graph 

import matplotlib.pyplot as plt

plt.scatter(y_valid, y_pred, alpha=0.6)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.show()


# predicting on test data

preds = final_model.predict(X_test)


# gain: total gain of splits using the feature (better measure of usefulness)
importance_gain = final_model.get_booster().get_score(importance_type='gain')

# cover: relative number of observations impacted by splits on the feature
importance_cover = final_model.get_booster().get_score(importance_type='cover')

importance_gain_df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain']).sort_values('Gain', ascending=False)
importance_cover_df = pd.DataFrame(list(importance_cover.items()), columns=['Feature', 'Cover']).sort_values('Cover', ascending=False)

print("Top 20 Features by Gain:")
display(importance_gain_df.head(20))

print("Top 20 Features by Cover:")
display(importance_cover_df.head(20))



submission_df = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
if 'label' in submission_df.columns:
    submission_df = submission_df.drop(columns=['label'])
submission_df["prediction"] = preds
submission_df.to_csv("submission.csv", index=False)
    
print("\nSubmission file 'submission.csv' created successfully.")
print(submission_df.head())


#timeseriessplit 

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)
splits = list(tscv.split(X))
train_idx, valid_idx = splits[-1]
X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]


# Hyperparameter tuning
def objective1(trial):
    

    params = {
        "tree_method": "gpu_hist",
        "device": "cuda",
        "verbosity": 0,
        "random_state": 42,
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "max_leaves": trial.suggest_int("max_leaves", 8, 64),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 50.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 80.0),
        "n_estimators": 1667
    }

    scores = []
    
    for train_idx, valid_idx in tscv.split(X):
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = xgb.XGBRegressor(early_stopping_rounds=50, **params)
        
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        
        preds = model.predict(X_valid)
        r, _ = pearsonr(y_valid, preds)
        scores.append(r)

    return np.mean(scores)
    


# Create and run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective1, n_trials=20)

#  results
print("Best Pearson correlation:", study.best_value)
print("Best Parameters:", study.best_params)


best_params = study.best_params

# Add required non-tuned params
best_params.update({
    "tree_method": "hist",
    "device": "cuda",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    "eval_metric": "rmse"
})

# Suppose last fold train indices:
tscv = TimeSeriesSplit(n_splits=3)
splits = list(tscv.split(X))
train_idx, val_idx = splits[-1]

X_final_train = X.iloc[train_idx]
y_final_train = y.iloc[train_idx]

final_model = xgb.XGBRegressor(**study.best_params)
final_model.fit(X_final_train, y_final_train, verbose=False)



y_pred = final_model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")



import matplotlib.pyplot as plt

plt.scatter(y_valid, y_pred, alpha=0.6)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.show()


preds = final_model.predict(X_test)
preds


submission_df = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
if 'label' in submission_df.columns:
    submission_df = submission_df.drop(columns=['label'])
submission_df["prediction"] = preds
submission_df.to_csv("submission1.csv", index=False)
    
print("\nSubmission file 'submission1.csv' created successfully.")
print(submission_df.head())




