import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import log_evaluation, early_stopping, LGBMRegressor

sns.set_style('darkgrid')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=["date"])
original_train_df = train_df.copy()
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=["date"])
sub = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


gdp_per_capita_df = pd.read_csv("/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv")

years =  ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
gdp_per_capita_filtered_df = gdp_per_capita_df.loc[gdp_per_capita_df["Country Name"].isin(train_df["country"].unique()), ["Country Name"] + years].set_index("Country Name")
for year in years:
    gdp_per_capita_filtered_df[f"{year}_ratio"] = gdp_per_capita_filtered_df[year] / gdp_per_capita_filtered_df.sum()[year]
gdp_per_capita_filtered_ratios_df = gdp_per_capita_filtered_df[[i+"_ratio" for i in years]]
gdp_per_capita_filtered_ratios_df.columns = [int(i) for i in years]
gdp_per_capita_filtered_ratios_df = gdp_per_capita_filtered_ratios_df.unstack().reset_index().rename(columns = {"level_0": "year", 0: "ratio", "Country Name": "country"})
gdp_per_capita_filtered_ratios_df['year'] = pd.to_datetime(gdp_per_capita_filtered_ratios_df['year'], format='%Y')

# For plotting purposes
gdp_per_capita_filtered_ratios_df_2 = gdp_per_capita_filtered_ratios_df.copy()
gdp_per_capita_filtered_ratios_df_2["year"] = pd.to_datetime(gdp_per_capita_filtered_ratios_df_2['year'].astype(str)) + pd.offsets.YearEnd(1)
gdp_per_capita_filtered_ratios_df = pd.concat([gdp_per_capita_filtered_ratios_df, gdp_per_capita_filtered_ratios_df_2]).reset_index()


gdp_per_capita_filtered_ratios_df_2["year"] = gdp_per_capita_filtered_ratios_df_2["year"].dt.year


train_df_imputed = train_df.copy()
missing_value_ids = train_df.loc[train_df["num_sold"].isna(), "id"].values
print(f"Missing values remaining: {train_df_imputed['num_sold'].isna().sum()}")

train_df_imputed["year"] = train_df_imputed["date"].dt.year
for year in train_df_imputed["year"].unique():
    # Impute Time Series 1 (Canada, Discount Stickers, Holographic Goose)
    target_ratio = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Norway"), "ratio"].values[0] # Using Norway as should have the best precision
    current_raito = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Canada"), "ratio"].values[0]
    ratio_can = current_raito / target_ratio
    train_df_imputed.loc[(train_df_imputed["country"] == "Canada") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year), "num_sold"] * ratio_can).values
    
    # Impute Time Series 2 (Only Missing Values)
    current_ts =  train_df_imputed.loc[(train_df_imputed["country"] == "Canada") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_df_imputed.loc[(train_df_imputed["country"] == "Canada") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_can).values

    # Impute Time Series 3 (Only Missing Values)
    current_ts =  train_df_imputed.loc[(train_df_imputed["country"] == "Canada") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_df_imputed.loc[(train_df_imputed["country"] == "Canada") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_can).values
    
    # Impute Time Series 4 (Kenya, Discount Stickers, Holographic Goose)
    current_raito = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Kenya"), "ratio"].values[0]
    ratio_ken = current_raito / target_ratio
    train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Holographic Goose")& (train_df_imputed["year"] == year), "num_sold"] * ratio_ken).values

    # Impute Time Series 5 (Only Missing Values)
    current_ts = train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values

    # Impute Time Series 6 (Only Missing Values)
    current_ts = train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values

    # Impute Time Series 7 (Only Missing Values)
    current_ts = train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Kerneler") & (train_df_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Kerneler") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Kerneler") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values
    
print(f"Missing values remaining: {train_df_imputed['num_sold'].isna().sum()}")



missing_rows = train_df_imputed.loc[train_df_imputed["num_sold"].isna()]
display(missing_rows)
train_df_imputed.loc[train_df_imputed["id"] == 23719, "num_sold"] = 4
train_df_imputed.loc[train_df_imputed["id"] == 207003, "num_sold"] = 195

print(f"Missing values remaining: {train_df_imputed['num_sold'].isna().sum()}")


store_weights = train_df_imputed.groupby("store")["num_sold"].sum()/train_df_imputed["num_sold"].sum()
store_weights



product_df = train_df_imputed.groupby(["date","product"])["num_sold"].sum().reset_index()
product_ratio_df = product_df.pivot(index="date", columns="product", values="num_sold")
product_ratio_df = product_ratio_df.apply(lambda x: x/x.sum(),axis=1)
product_ratio_df = product_ratio_df.stack().rename("ratios").reset_index()
product_ratio_df.head(4)


train_df_imputed.head()


original_train_df_imputed = train_df_imputed.copy()
train_df_imputed = train_df_imputed.groupby(["date"])["num_sold"].sum().reset_index()


train_df_imputed["day_of_week"] = train_df_imputed["date"].dt.dayofweek
day_of_week_ratio = (train_df_imputed.groupby("day_of_week")["num_sold"].mean() / train_df_imputed.groupby("day_of_week")["num_sold"].mean().mean()).rename("day_of_week_ratios")
display(day_of_week_ratio)
train_df_imputed = pd.merge(train_df_imputed, day_of_week_ratio, how="left", on="day_of_week")
train_df_imputed["num_sold"] = train_df_imputed["num_sold"] / train_df_imputed["day_of_week_ratios"]
train_df_imputed = train_df_imputed.drop("day_of_week_ratios",axis=1)# we don't need it in training.


test_total_sales_df = test_df.groupby(["date"])["id"].first().reset_index().drop(columns="id")
test_total_sales_dates = test_total_sales_df[["date"]]


def feature_engineer(df):
    new_df = df.copy()
    new_df["month"] = df["date"].dt.month
    new_df["month_sin"] = np.sin(new_df['month'] * (2 * np.pi / 12))
    new_df["month_cos"] = np.cos(new_df['month'] * (2 * np.pi / 12))
    new_df["day_of_week"] = df["date"].dt.dayofweek
    new_df["day_of_week"] = new_df["day_of_week"].apply(lambda x: 0 if x<=3 else(1 if x==4 else (2 if x==5 else (3))))    
    new_df["day_of_year"] = df['date'].apply(
        lambda x: x.timetuple().tm_yday if not (x.is_leap_year and x.month > 2) else x.timetuple().tm_yday - 1
    )

    new_df['day_sin4'] = np.sin(new_df['day_of_year'] * (8 * np.pi /  365.0))
    new_df['day_cos4'] = np.cos(new_df['day_of_year'] * (8 * np.pi /  365.0))
    new_df['day_sin3.5t'] = np.sin(7 * np.pi/365 * new_df['day_of_year'])
    new_df['day_cos3.5t'] = np.cos(7 * np.pi/365 * new_df['day_of_year'])
    new_df['day_sin3'] = np.sin(new_df['day_of_year'] * (6 * np.pi /  365.0))
    new_df['day_cos3'] = np.cos(new_df['day_of_year'] * (6 * np.pi /  365.0))
    new_df['day_sin2'] = np.sin(new_df['day_of_year'] * (4 * np.pi /  365.0))
    new_df['day_cos2'] = np.cos(new_df['day_of_year'] * (4 * np.pi /  365.0))
    new_df['day_sin'] = np.sin(new_df['day_of_year'] * (2 * np.pi /  365.0))
    new_df['day_cos'] = np.cos(new_df['day_of_year'] * (2 * np.pi /  365.0)) 
    new_df['day_sin_0.5'] = np.sin(new_df['day_of_year'] * (1 * np.pi /  365.0))
    new_df['day_cos_0.5'] = np.cos(new_df['day_of_year'] * (1 * np.pi /  365.0))    
    new_df['day_sin1/4t'] = np.sin(1/2 * np.pi/365 * new_df['day_of_year'])
    new_df['day_cos1/4t'] = np.cos(1/2 * np.pi/365 * new_df['day_of_year'])
    new_df["important_dates"] = new_df["day_of_year"].apply(lambda x: x if x in [1,2,3,4,5,6,7,8,9,10,99, 100, 101, 125,126,355,256,356,357,358,359,360,361,362,363,364,365,366] else 0)
    
    new_df = new_df.drop(columns=["date","month","day_of_year"])
    new_df = pd.get_dummies(new_df, columns = ["important_dates","day_of_week"], drop_first=True)
    
    return new_df


train_total_sales_df = feature_engineer(train_df_imputed)
test_total_sales_df = feature_engineer(test_total_sales_df)


import holidays
train_df_tmp = train_df.copy()
test_df_tmp = test_df.copy()
alpha2 = dict(zip(np.sort(train_df.country.unique()), ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']))
h = {c: holidays.country_holidays(a, years=range(2010, 2020)) for c, a in alpha2.items()}
train_df_tmp['is_holiday'] = 0
test_df_tmp['is_holiday'] = 0
for c in alpha2:
    train_df_tmp.loc[train_df_tmp.country==c, 'is_holiday'] = train_df_tmp.date.isin(h[c]).astype(int)
    test_df_tmp.loc[test_df_tmp.country==c, 'is_holiday'] = test_df_tmp.date.isin(h[c]).astype(int)


train_total_sales_df['is_holiday'] = (train_df_tmp.groupby(["date"])["is_holiday"].sum().reset_index())["is_holiday"]
test_total_sales_df['is_holiday'] = (test_df_tmp.groupby(["date"])["is_holiday"].sum().reset_index())["is_holiday"]


y = train_total_sales_df["num_sold"]
X = train_total_sales_df.drop(columns="num_sold")
X_test = test_total_sales_df


import numpy as np

class LinearModel1:
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return self.nonlinear(X)

    def nonlinear(self, X):
        y = (
            X.important_dates_1 * 14514.153142903406 +
            X.important_dates_6 +
            X.important_dates_99 -
            X.important_dates_359 +
            X.important_dates_363 * 23245.947607750375 +
            64795.970182750174 -
            ((X.important_dates_362 + np.sqrt(X.important_dates_364 + X.important_dates_365)) *
             ((X.important_dates_2 - X.important_dates_10) * (X.important_dates_101 + X.month_sin) +
              X.important_dates_362 - 19142.637752625015))
        )
        return y

class LinearModel2:
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return self.nonlinear(X)

    def nonlinear(self, X):
        y = X.month_sin*210.148+\
            (X.important_dates_1+X.important_dates_362+X.important_dates_363+X.important_dates_364+X.important_dates_365)*19472.7071+\
            X.important_dates_4*X.month_sin+\
            65141.2073
            
        return y


regressors = {
    "Ridge" : Ridge(tol=1e-2, max_iter=2000000, random_state=0),
    "Lasso" : Lasso(tol=1e-2, max_iter=2000000, random_state=0),
    #"nonlinear1" : LinearModel1(),
    #"nonlinear2" : LinearModel2(),
    "LGBM1": LGBMRegressor(**{'objective': 'regression_l2',
                               'metric': 'mape', 
                               'max_depth': 7,
                               'num_leaves': 123, 
                               'min_child_samples': 21,
                               'min_child_weight': 24,
                               'colsample_bytree': 0.3641261996760593, 
                               'reg_alpha': 0.03632800166349373, 
                               'reg_lambda': 0.5287861861476272,
                               'random_state': 42,
                               'early_stopping_round':200,
                               'verbose': -1,
                               'boosting_type': 'gbdt',
                               'n_estimators': 3000,
                               'learning_rate': 0.01,
                               }),
    "LGBM2": LGBMRegressor(**{'objective': 'regression_l2',
                               'metric': 'mape',
                               'max_depth': 6,
                               'num_leaves': 502,
                               'min_child_samples': 23,
                               'min_child_weight': 18, 
                               'colsample_bytree': 0.4714820876493163, 
                               'reg_alpha': 0.054972003081022576, 
                               'reg_lambda': 0.5774608955362155,
                               'random_state': 42,
                               'early_stopping_round': 200,
                               'verbose': -1,
                               'boosting_type': 'goss',
                               'n_estimators': 3000,
                               'learning_rate': 0.01,
                              }),
    "LGBM3": LGBMRegressor(**{'objective': 'regression_l2', 
                               'metric': 'mape',
                               'max_depth': 14,
                               'num_leaves': 279,
                               'min_child_samples': 7,
                               'min_child_weight': 24, 
                               'colsample_bytree': 0.43218993309765835,
                               'reg_alpha': 0.42757392987472964,
                               'reg_lambda': 0.9039762787446107,
                               'random_state': 42,
                               'early_stopping_round': 200,
                               'verbose': -1,
                               'boosting_type': 'goss',
                               'n_estimators': 3000,
                               'learning_rate': 0.01,
                               }),

}


import time
from sklearn.model_selection import KFold
n_folds = 5

for key, reg in regressors.items():
    test_preds = np.zeros(len(X_test))
    oof_full = y.copy()

    start = time.time()

    cv = KFold(n_splits=n_folds, shuffle=False)
    score=0
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]
        start = time.time()
        if "LGBM" in key:
            reg.fit(X_train, y_train,
                   eval_set = [(X_valid, y_valid)], 
                                       callbacks = [log_evaluation(0),
                                                    early_stopping(200, verbose = False)
                                                   ])
            
        else:
            reg.fit(X_train, y_train)
        oof_preds = reg.predict(X_valid)
        score += mean_absolute_percentage_error(y_valid, oof_preds)/n_folds
        oof_full[val_idx] = oof_preds
        
        test_preds += reg.predict(X_test)/n_folds
    
    # Stop timer
    stop = time.time()
    
    print('Model:', key)
    print('Average validation MAPE:', score)
    print('Training time (mins):', np.round((stop - start)/60,2))
    print('')
    

    oof_full.to_csv(f"{key}_oof_preds.csv", index=False)
    ss = pd.DataFrame()
    ss["num_sold"] = test_preds
    ss.to_csv(f"{key}_test_preds.csv", index=False)



oof_df = pd.DataFrame(index=np.arange(len(y)))
for i in regressors.keys():
    df = pd.read_csv(f"/kaggle/working/{i}_oof_preds.csv")
    df.rename(columns={"num_sold": i}, inplace=True)
    oof_df = pd.concat([oof_df,df], axis=1)
    
# Join test preds
test_preds = pd.DataFrame(index=np.arange(len(X_test)))
for i in regressors.keys():
    df = pd.read_csv(f"/kaggle/working/{i}_test_preds.csv")
    df.rename(columns={"num_sold": i}, inplace=True)
    test_preds = pd.concat([test_preds,df], axis=1)
    
oof_df.head(3)


scores = {}
for col in oof_df.columns:
    scores[col] = mean_absolute_percentage_error(y, oof_df[col])

# Sort scores
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=False)}

# Sort oof_df and test_preds
oof_df = oof_df[list(scores.keys())]
test_preds = test_preds[list(scores.keys())]

scores


STOP = False
current_best_ensemble = oof_df.iloc[:,0]
current_best_test_preds = test_preds.iloc[:,0]
MODELS = oof_df.iloc[:,1:]
weight_range = np.arange(-1,1,0.001)  
history = [mean_absolute_percentage_error(y, current_best_ensemble)]
i=0

# Hill climbing
while not STOP:
    i+=1
    potential_new_best_cv_score = mean_absolute_percentage_error(y, current_best_ensemble)
    k_best, wgt_best = None, None
    for k in MODELS:
        for wgt in weight_range:
            potential_ensemble = (1-wgt) * current_best_ensemble + wgt * MODELS[k]
            cv_score = mean_absolute_percentage_error(y, potential_ensemble)
            if cv_score < potential_new_best_cv_score:
                potential_new_best_cv_score = cv_score
                k_best, wgt_best = k, wgt
            
    if k_best is not None:
        current_best_ensemble = (1-wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
        current_best_test_preds = (1-wgt_best) * current_best_test_preds + wgt_best * test_preds[k_best]
        MODELS.drop(k_best, axis=1, inplace=True)
        if MODELS.shape[1]==0:
            STOP = True
        print(f'Iteration: {i}, Model added: {k_best}, Best weight: {wgt_best:}, Best MAPE: {potential_new_best_cv_score:.5f}')
        history.append(potential_new_best_cv_score)
    else:
        STOP = True


plt.figure(figsize=(10,4))
plt.plot(np.arange(len(history))+1, history, marker="x")
plt.title("CV MAPE vs. Number of Models with Hill Climbing")
plt.xlabel("Number of models")
plt.ylabel("MAPE")
plt.show()


test_total_sales_dates["num_sold"] = current_best_test_preds*0.5+LinearModel2().predict(X_test)*0.5


product_ratio_2017_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2015].copy()
product_ratio_2018_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2016].copy()
product_ratio_2019_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2015].copy()

product_ratio_2017_df["date"] = product_ratio_2017_df["date"] + pd.DateOffset(years=2)
product_ratio_2018_df["date"] = product_ratio_2018_df["date"] + pd.DateOffset(years=2)
product_ratio_2019_df["date"] =  product_ratio_2019_df["date"] + pd.DateOffset(years=4)

forecasted_ratios_df = pd.concat([product_ratio_2017_df, product_ratio_2018_df, product_ratio_2019_df])


store_weights_df = store_weights.reset_index()
test_sub_df = pd.merge(test_df, test_total_sales_dates, how="left", on="date")
test_sub_df = test_sub_df.rename(columns = {"num_sold":"day_num_sold"})
# Adding in the product ratios
test_sub_df = pd.merge(test_sub_df, store_weights_df, how="left", on="store")
test_sub_df = test_sub_df.rename(columns = {"num_sold":"store_ratio"})
# Adding in the country ratios
test_sub_df["year"] = test_sub_df["date"].dt.year
test_sub_df = pd.merge(test_sub_df, gdp_per_capita_filtered_ratios_df_2, how="left", on=["year", "country"])
test_sub_df = test_sub_df.rename(columns = {"ratio":"country_ratio"})
# Adding in the product ratio
test_sub_df = pd.merge(test_sub_df, forecasted_ratios_df, how="left", on=["date", "product"])
test_sub_df = test_sub_df.rename(columns = {"ratios":"product_ratio"})

# Adding in the week ratio
test_sub_df["day_of_week"] = test_sub_df["date"].dt.dayofweek
test_sub_df = pd.merge(test_sub_df, day_of_week_ratio.reset_index(), how="left", on="day_of_week")


# Disaggregating the forecast
test_sub_df.loc[test_sub_df['country'] == 'Kenya', 'country_ratio'] -= 0.0007/2

test_sub_df["num_sold"] = test_sub_df["day_num_sold"] *test_sub_df["day_of_week_ratios"]* test_sub_df["store_ratio"] * test_sub_df["country_ratio"] * test_sub_df["product_ratio"]
#test_sub_df["num_sold"] = test_sub_df["num_sold"].round()
display(test_sub_df.head(2))



import gc
gc.collect()
submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
lb_best = pd.read_csv("/kaggle/input/transformer-starter-lb-0-052/submission_v1.csv")
tf_best = pd.read_csv("/kaggle/input/pg501-model-2-decomposition-country-doy-factor/submission.csv")
xgboost_best = pd.read_csv("/kaggle/input/prediction-using-xgboost/submission1.csv")
submission["num_sold"] = (test_sub_df["num_sold"]*0.2403 + (tf_best.num_sold*0.5+xgboost_best.num_sold*0.5)*0.2587 + lb_best.num_sold*0.5102).round()
display(submission.head(2))


submission.to_csv('submission2.csv', index = False)

