import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression as PLS


train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")


 def transform_df(df, n=6):
    x=[i for i in df.columns if i not in ["label"]]#"bid_qty","ask_qty", "buy_qty","sell_qty", "volume", 
    temp=df[x]
    pls=PLS(n_components=n)
    xpls=pls.fit_transform(temp, df["label"])
    df1=pd.DataFrame()
    df1[[f"c{i}" for i in range(1, n+1)]]=xpls[0]
    df1=pd.concat([df1, df.reset_index()[["bid_qty","ask_qty", "buy_qty","sell_qty", "volume","label"]]], axis=1)
    return df1, pls
    
def transform_test(df, pls, n=6):
    x=[i for i in df.columns if i not in ["label"]]
    temp=df[x]
    xpls=pls.transform(temp)
    print(xpls)
    df1=pd.DataFrame()
    df1[[f"c{i}" for i in range(1, n+1)]]=xpls
    df1=pd.concat([df1, df.reset_index()[["bid_qty","ask_qty", "sell_qty", "buy_qty","volume"]]], axis=1)
    return df1


train.head()


temp, pls=transform_df(train, n=7)


best=sorted(list(zip(abs(pls.coef_), [i for i in train.columns if i not in ["label"]])), key=lambda x: x[0], reverse=True)


#features by weights from PLS
best_sel=[i[1] for i in best[:25]]


best_sel


base_features=best_sel+["ask_qty", "buy_qty", "sell_qty", "volume", "bid_qty", "label"]


weights=[i[1] for i in best[:25]]+[1,1,1,1,1]


train = train[base_features]
test = test[base_features]
train


def add_features(df):
    # Original features
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']

    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume'])

    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
    # === NEW MICROSTRUCTURE FEATURES ===
    
    # Price Pressure Indicators
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
    # Liquidity Depth Measures
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    # Market Activity Indicators
   # df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Microstructure Volatility Proxies
    df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    
    # Complex Interaction Terms
  #  df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
  #  df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
 #   df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    
    # Information Asymmetry Measures
   # df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    #df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10)
    #df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10) * df['volume']
    
    # Market Efficiency Indicators
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
   # df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
   # df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10)
    
    # Non-linear Transformations
    #df['sqrt_volume'] = np.sqrt(df['volume'])
   # df['sqrt_depth'] = np.sqrt(df['total_depth'])
   # df['volume_squared'] = df['volume'] ** 2
   # df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
    # Relative Measures
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    # Market Stress Indicators
   # df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
   # df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    
    # Directional Indicators
   ## df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    #df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
   # df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df


#x_train = add_features(train)
x_train=train
x_train.shape

import warnings 

warnings.filterwarnings('ignore')


#scaler = StandardScaler()
#x_train.iloc[:, :] = scaler.fit_transform(X_train)


#x_test = add_features(test)
x_test=test
x_test.shape


# Ensure the index is a datetime type
x_train.index = pd.to_datetime(x_train.index)

# Sort the dataset by timestamp (index)
x_train = x_train.sort_index()

# Define the split point (80% train, 20% validation)
split_index = int(len(x_train) * 0.8)

# Slice by index â€” NO shuffling
train_split = x_train.iloc[:split_index]
val_split = x_train.iloc[split_index:]

# Training features and labels
X_train = train_split.drop(columns=['label'])
y_train = train_split['label']

# Validation features and labels
X_val = val_split.drop(columns=['label'])
y_val = val_split['label']


y_train.head()


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

XGB_PARAMS = {
    "tree_method": "hist",
    "device": "gpu",
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 1,#39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": 42,
    "n_jobs": -1,
    "feature_weights": weights
}

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS}
]


from xgboost import XGBRegressor
import numpy as np

model = XGBRegressor(**XGB_PARAMS)
model.fit(X_train, y_train)


def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()


y_val_pred = model.predict(X_val)

from scipy.stats import pearsonr

# Pearson correlation
pearson_corr, _ = pearsonr(y_val, y_val_pred)

print(f"âœ… Pearson Correlation: {pearson_corr:.4f}")


X = x_train.drop(columns=['label'])
y = x_train['label']
feature_names = X.columns.tolist()  # âœ… Save exact feature names

x_test = x_test[feature_names]  # âœ… Ensure same columns, same order



#x_test=x_test.drop(columns=['label'])


y_pred = model.predict(x_test)


submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
submission["prediction"] = y_pred
submission.to_csv("submission.csv", index=False)
print("ğŸ“� Submission file saved as 'submission.csv'")
submission.head()




