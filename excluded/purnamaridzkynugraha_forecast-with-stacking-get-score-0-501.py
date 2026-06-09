import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



top_feature =[
 'amount_new_house_transactions',
 'area_per_unit_new_house_transactions',
 'area_new_house_available_for_sale',
 'period_new_house_sell_through',
 'amount_pre_owned_house_transactions',
 'transaction_amount',
 'transaction_amount_nearby_sectors',
 'trend_cluster'
]


land_transactions = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv')
land_transactions_nearby_sectors = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv')
new_house_transactions = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
new_house_transactions_nearby_sectors = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv')
pre_owned_house_transactions= pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv')


def fill_by_month_median(df, cols):
    df[cols] = df.groupby("month")[cols].transform(lambda x: x.fillna(x.median()))

fill_by_month_median(new_house_transactions, [
    "num_new_house_available_for_sale",
    "area_new_house_available_for_sale",
    "period_new_house_sell_through"
])

fill_by_month_median(pre_owned_house_transactions, [
    "price_pre_owned_house_transactions"
])



def ensure_all_sectors(df, month_col="month", sector_col="sector", n_sectors=96):
    df = df.copy()
    df[month_col] = pd.to_datetime(df[month_col], errors="coerce")

    df[sector_col] = df[sector_col].astype(str).str.extract(r"(\d+)").astype(float).astype("Int64")

    all_months = df[month_col].dt.to_period("M").unique()
    full_index = pd.MultiIndex.from_product([all_months, range(1, n_sectors + 1)],
                                            names=[month_col, sector_col])

    df = df.set_index([df[month_col].dt.to_period("M"), sector_col])
    df.index.names = [month_col, sector_col]

    df_full = df.reindex(full_index)

    for col in [month_col, sector_col]:
        if col in df_full.columns:
            df_full = df_full.drop(columns=[col])

    df_full = df_full.reset_index()
    df_full[month_col] = df_full[month_col].dt.to_timestamp()
    df_full.set_index(month_col, inplace=True)

    return df_full



datasets = {
    "land_transactions": land_transactions,
    "land_transactions_nearby_sectors": land_transactions_nearby_sectors,
    "new_house_transactions": new_house_transactions,
    "new_house_transactions_nearby_sectors": new_house_transactions_nearby_sectors,
    "pre_owned_house_transactions": pre_owned_house_transactions,
}
datasets_filled = {}

for name, df in datasets.items():
    datasets_filled[name] = ensure_all_sectors(df)
    datasets_filled[name].fillna(0, inplace=True)



land_transactions = datasets_filled['land_transactions']
land_transactions_nearby_sectors = datasets_filled['land_transactions_nearby_sectors']
new_house_transactions = datasets_filled['new_house_transactions']
new_house_transactions_nearby_sectors = datasets_filled['new_house_transactions_nearby_sectors']
pre_owned_house_transactions = datasets_filled['pre_owned_house_transactions']


dfs = {
    "land_transactions": land_transactions,
    "land_transactions_nearby_sectors": land_transactions_nearby_sectors,
    "new_house_transactions": new_house_transactions,
    "new_house_transactions_nearby_sectors": new_house_transactions_nearby_sectors,
    "pre_owned_house_transactions": pre_owned_house_transactions,
}


for name, df in dfs.items():
    print(f"\n=== {name} ===")
    print(df.isna().sum())
    print(f"Total NaN: {df.isna().sum().sum()}")


df_merged = new_house_transactions.copy()

datasets_to_merge = [
    land_transactions,
    land_transactions_nearby_sectors,
    new_house_transactions_nearby_sectors,
    pre_owned_house_transactions
]

for df in datasets_to_merge:
    df_merged = df_merged.merge(
        df,
        on=["month", "sector"],   
        how="left"
    )

df_merged.index.name = "month"



import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

target = "amount_new_house_transactions"
df = df_merged.copy()
df.index = pd.to_datetime(df.index)
df = df.sort_index()

sectors = sorted(df["sector"].unique())
idx = pd.period_range(df.index.min(), df.index.max(), freq="M").to_timestamp()

trend = {}
for sec in sectors:
    s = df.loc[df["sector"]==sec, target].reindex(idx).astype(float).interpolate(limit_direction="both")
    if s.dropna().size >= 24:  
        res = seasonal_decompose(s, model="additive", period=12, extrapolate_trend="freq")
        trend[sec] = res.trend

trend_df = pd.DataFrame(trend) 



var_rank = trend_df.var().sort_values(ascending=False).index[:16]
import matplotlib.pyplot as plt
import math

n = len(var_rank); r = math.ceil(n/4)
fig, axes = plt.subplots(r, 4, figsize=(16, 3*r), sharex=True, sharey=False)
axes = axes.ravel()

for i, sec in enumerate(var_rank):
    axes[i].plot(trend_df[sec])
    axes[i].set_title(f"Sector {sec}")
    axes[i].axhline(trend_df[sec].mean(), lw=0.6)
for j in range(i+1, len(axes)): axes[j].axis("off")
fig.suptitle("Top-variance sectors (trend)", y=1.02); plt.tight_layout(); plt.show()



from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

Z = pd.DataFrame(
    StandardScaler(with_mean=True, with_std=True).fit_transform(trend_df.fillna(method="ffill").fillna(method="bfill")),
    index=trend_df.index, columns=trend_df.columns
)

k = 3
km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(Z.T) 
labels = pd.Series(km.labels_, index=Z.columns, name="cluster")

plt.figure(figsize=(12,6))
for c in range(k):
    median_path = Z.loc[:, labels[labels==c].index].median(axis=1)
    plt.plot(median_path, label=f"Cluster {c} (n={sum(labels==c)})")
plt.title("Cluster medians of sector trends (z-score)"); plt.xlabel("Time"); plt.ylabel("z-trend")
plt.legend(); plt.show()

cluster_members = {c: labels[labels==c].index.tolist() for c in range(k)}



import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Z_data = Z.T  

inertia = []
K = range(1, 10)  
for k in K:
    km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(Z_data)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'o-', color='blue')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia (sum of squared distances)')
plt.title('Elbow Method for Optimal k')
plt.show()



df_merged["trend_cluster"] = df_merged["sector"].map(labels)


import pandas as pd

selected_features = [
    'amount_new_house_transactions',
    'amount_new_house_transactions_ma13', 
    'area_per_unit_new_house_transactions_ma13', 
    'area_new_house_available_for_sale_ma13', 
    'period_new_house_sell_through_ma13', 
    'amount_pre_owned_house_transactions_ma18', 
    'transaction_amount_ma24', 
    'transaction_amount_nearby_sectors_lag24', 
    'amount_new_house_transactions_lag6', 
    'amount_new_house_transactions_lag9', 
    'trend_cluster',
    'sector'
]

df_features = df_merged.copy()

for feature in selected_features:
    # Create lag features
    if 'lag' in feature:
        base_col, lag_num = feature.rsplit('_lag', 1)
        lag_num = int(lag_num)
        df_features[feature] = df_features.groupby('sector')[base_col].shift(lag_num)

    # Create moving average features
    elif 'ma' in feature:
        base_col, window = feature.rsplit('_ma', 1)
        window = int(window)
        df_features[feature] = df_features.groupby('sector')[base_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    # Check if the feature exists in the dataset
    else:
        if feature not in df_features.columns:
            print(f"⚠ Column {feature} not found in df_merged")

# Keep only the final selected features
df_final = df_features[selected_features]

# Remove rows with any missing values
df_final.dropna(inplace=True)



from xgboost import XGBRegressor
X = df_final.drop(columns=['amount_new_house_transactions'])
y = df_final['amount_new_house_transactions']



import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)

param_dist = {
    "n_estimators": [100, 300, 500, 800],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5],
    "min_child_weight": [1, 3, 5],
}

tscv = TimeSeriesSplit(n_splits=5)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring="neg_mean_absolute_error",  
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X, y)

print("Best params:", random_search.best_params_)
print("Best MAE (CV):", -random_search.best_score_)  


from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

lgb_model = LGBMRegressor(random_state=42,force_col_wise= True,verbose=-1 )

param_dist = {
    "n_estimators": [100, 300, 500, 800],
    "max_depth": [3, 5, 7, 10],   
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "num_leaves": [31, 50, 70, 100],
    "min_child_samples": [5, 10, 20],
}

tscv = TimeSeriesSplit(n_splits=5)

random_search_lgb = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search_lgb.fit(X, y)

print("Best params:", random_search_lgb.best_params_)
print("Best MAE (CV):", -random_search_lgb.best_score_)



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

rf_model = RandomForestRegressor(random_state=42)

param_dist_rf = {
    "n_estimators": [100, 300, 500, 800],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [1.0, "sqrt", 0.8]
}

tscv = TimeSeriesSplit(n_splits=5)

random_search_rf = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist_rf,
    n_iter=30,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search_rf.fit(X, y)

print("Best params:", random_search_rf.best_params_)
print("Best MAE (CV):", -random_search_rf.best_score_)


from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

en_model = ElasticNet(random_state=42, max_iter=10000)

param_dist_en = {
    "alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}

tscv = TimeSeriesSplit(n_splits=5)

random_search_en = RandomizedSearchCV(
    estimator=en_model,
    param_distributions=param_dist_en,
    n_iter=30,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search_en.fit(X, y)

print("Best params:", random_search.best_params_)
print("Best MAE (CV):", -random_search.best_score_)


models = {
    "xgb": random_search.best_estimator_,
    "lgb": random_search_lgb.best_estimator_,
    "rf": random_search_rf.best_estimator_,
    "elasticnet": random_search_en.best_estimator_
}



import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

models = {
    "xgb": random_search.best_estimator_,
    "lgb": random_search_lgb.best_estimator_,
    "rf": random_search_rf.best_estimator_,
    "elasticnet": random_search_en.best_estimator_
}

df_meta = pd.DataFrame(index=X.index)

for name, model in models.items():
    df_meta[f"{name}_pred"] = model.predict(X)

# Tambahkan target asli
df_meta["y_true"] = y

X_final = df_meta.drop(columns="y_true")
y_final = df_meta["y_true"]

final_model =  KNeighborsRegressor(n_neighbors=5)

tscv = TimeSeriesSplit(n_splits=5)

train_scores = []
test_scores = []

for train_idx, test_idx in tscv.split(X_final):
    X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
    y_train, y_test = y_final.iloc[train_idx], y_final.iloc[test_idx]

    final_model.fit(X_train, y_train)

    y_pred_train = final_model.predict(X_train)
    y_pred_test = final_model.predict(X_test)

    train_scores.append(mean_absolute_error(y_train, y_pred_train))
    test_scores.append(mean_absolute_error(y_test, y_pred_test))

for i, (tr, te) in enumerate(zip(train_scores, test_scores), 1):
    print(f"Fold {i} → Train MAE: {tr:.4f} | Test MAE: {te:.4f}")

print("\nRata-rata Train MAE:", np.mean(train_scores))
print("Rata-rata Test MAE:", np.mean(test_scores))

df_meta["y_pred_final"] = final_model.predict(X_final)


test_range = pd.date_range("2024-08-01", "2025-07-01", freq="MS")
sectors = df_merged["sector"].unique()

df_test = pd.DataFrame([
    {"month": m, "sector": s} for m in test_range for s in sectors
])

df_all = pd.concat([df_merged, df_test], ignore_index=True)

rolling_features = {
    "amount_new_house_transactions": 13,
    "area_per_unit_new_house_transactions": 13,
    "area_new_house_available_for_sale": 13,
    "period_new_house_sell_through": 13,
    "amount_pre_owned_house_transactions": 18,
    "transaction_amount": 24
}

for col, window in rolling_features.items():
    df_all[f"{col}_ma{window}"] = (
        df_all.groupby("sector")[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

lag_features = {
    "transaction_amount_nearby_sectors": [24],
    "amount_new_house_transactions": [6, 9]
}

for col, lags in lag_features.items():
    for lag in lags:
        df_all[f"{col}_lag{lag}"] = df_all.groupby("sector")[col].shift(lag)

df_all["trend_cluster"] = df_all["sector"].map(labels)

df_all.set_index("month", inplace=True)

feature_cols = [
    "amount_new_house_transactions_ma13",
    "area_per_unit_new_house_transactions_ma13",
    "area_new_house_available_for_sale_ma13",
    "period_new_house_sell_through_ma13",
    "amount_pre_owned_house_transactions_ma18",
    "transaction_amount_ma24",
    "transaction_amount_nearby_sectors_lag24",
    "amount_new_house_transactions_lag6",
    "amount_new_house_transactions_lag9",
    "trend_cluster",
    "sector"   
]

df_test_final = df_all.loc[test_range, feature_cols]

df_test_final.head()



def predict_meta(df, base_models, final_model):
    df_meta = pd.DataFrame(index=df.index)
    for name, model in base_models.items():
        df_meta[f"{name}_pred"] = model.predict(df)
    df["y_pred_final"] = final_model.predict(df_meta)
    return df

def split_test(df_test, start="2024-08-01", months=6):
    start = pd.Timestamp(start)
    mid = start + pd.DateOffset(months=months)
    end = mid + pd.DateOffset(months=months)
    df_1 = df_test[(df_test.index >= start) & (df_test.index < mid)]
    df_2 = df_test[(df_test.index >= mid) & (df_test.index < end)]
    return df_1, df_2

models = {
    "xgb": random_search.best_estimator_,
    "lgb": random_search_lgb.best_estimator_,
    "rf": random_search_rf.best_estimator_,
    "elasticnet": random_search_en.best_estimator_
}

df_test_1, df_test_2 = split_test(df_test_final, start="2024-08-01", months=6)
df_test_1 = predict_meta(df_test_1, models, final_model)


y_full = pd.concat([
    df_final[['sector','amount_new_house_transactions']],
    df_test_1[['sector','y_pred_final']].rename(columns={'y_pred_final':'amount_new_house_transactions'})
])

y_full = y_full.rename_axis('month').reset_index()

last_month = y_full['month'].max()
new_months = pd.date_range(last_month + pd.DateOffset(months=1), df_test_2.index.max(), freq='MS')
sectors = y_full['sector'].unique()
df_new = pd.DataFrame([{'month': m, 'sector': s} for m in new_months for s in sectors])

y_extended = pd.concat([y_full, df_new]).sort_values(['sector','month'])

y_extended['amount_new_house_transactions_lag6'] = y_extended.groupby('sector')['amount_new_house_transactions'].shift(6)
y_extended['amount_new_house_transactions_lag9'] = y_extended.groupby('sector')['amount_new_house_transactions'].shift(9)

df_test_2 = df_test_2.rename_axis('month').reset_index()  
df_test_2 = df_test_2.drop(columns=['amount_new_house_transactions_lag6','amount_new_house_transactions_lag9'])

df_test_2_with_lag = df_test_2.reset_index().merge(
    y_extended[['month','sector','amount_new_house_transactions_lag6','amount_new_house_transactions_lag9']],
    on=['month','sector'],
    how='left'
).set_index('month')


df_test_2_with_lag =df_test_2_with_lag[X.columns]
df_test_2_with_lag= predict_meta(df_test_2_with_lag, models, final_model)
test = pd.concat([df_test_1, df_test_2_with_lag])


df_pred = test[["y_pred_final","sector"]].copy()

df_pred = df_pred.rename_axis('month').reset_index()  
df_pred['id'] = df_pred.apply(
    lambda row: f"{row['month'].strftime('%Y %b')}_sector {int(row['sector'])}", axis=1
)

df_csv = df_pred[['id', 'y_pred_final']].rename(columns={'y_pred_final': 'new_house_transaction_amount'})

df_csv.to_csv("Predict_kaggle.csv", index=False)

