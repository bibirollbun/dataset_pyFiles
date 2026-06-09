import pandas as pd
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import holidays
train = pd.read_csv('/kaggle/input/train-filled/TrainFilled.csv',index_col = 0)
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train.head()


train.info()


df = train.copy()
df['date'] = pd.to_datetime(df['date'])


df = df.dropna(subset=['num_sold'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month


monthly_agg = df.groupby(['year', 'month'])['num_sold'].sum().reset_index()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)


sns.barplot(
    data=df.melt(id_vars=['num_sold'], value_vars=['country', 'store', 'product']),
    x='value', y='num_sold', hue='variable', ax=axes[0]
)
axes[0].set_title('Number of Stickers Sold by Features', fontsize=14)
axes[0].set_ylabel('Number of Stickers Sold', fontsize=12)
axes[0].set_xlabel('Features', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(title='Feature Type', fontsize=10)


palette = sns.color_palette("tab10", n_colors=len(monthly_agg['year'].unique()))
sns.lineplot(
    data=monthly_agg, x='month', y='num_sold', hue='year', marker='o', ax=axes[1], palette=palette
)
axes[1].set_title('Monthly Aggregated Number of Stickers Sold', fontsize=14)
axes[1].set_ylabel('Aggregated Number Sold', fontsize=12)
axes[1].set_xlabel('Month', fontsize=12)
axes[1].set_xticks(range(1, 13))

# Move legend outside the plot
axes[1].legend(title='Year', fontsize=10, loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.tight_layout()
plt.show()



train = train.dropna(subset=['num_sold'])

train['date'] = pd.to_datetime(train['date'])
train = train.sort_values(by='date')
train['dayofweek'] = train['date'].dt.dayofweek
train['dayofyear'] = train['date'].dt.dayofyear
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['quarter'] = train['date'].dt.quarter
train['year'] = train['date'].dt.year
train['year'] = train['year'] - 2010


train['country_store'] = train['country'] + '_' + train['store']
train['country_product'] = train['country'] + '_' + train['product']
train['store_product'] = train['store'] + '_' + train['product']

# Here we add in feature interaction between categories in our dataset to month. This can also be extended to monthyear
train['country_month'] = train['country'] + '_' + train['month'].astype(str)
train['store_month'] = train['store'] + '_' + train['month'].astype(str)
train['product_month'] = train['product'] + '_' + train['month'].astype(str)


test['date'] = pd.to_datetime(test['date'])
test['dayofweek'] = test['date'].dt.dayofweek
test['dayofyear'] = test['date'].dt.dayofyear
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['quarter'] = test['date'].dt.quarter
test['year'] = test['date'].dt.year
test['year'] = test['year'] - 2010


test['country_store'] = test['country'] + '_' + test['store']
test['country_product'] = test['country'] + '_' + test['product']
test['store_product'] = test['store'] + '_' + test['product']

test['country_month'] = test['country'] + '_' + test['month'].astype(str)
test['store_month'] = test['store'] + '_' + test['month'].astype(str)
test['product_month'] = test['product'] + '_' + test['month'].astype(str)



def get_gdp_per_capita(country, year):
    alpha3 = {
        'Canada': 'CAN', 'Finland': 'FIN', 'Italy': 'ITA',
        'Kenya': 'KEN', 'Norway': 'NOR', 'Singapore': 'SGP'
    }
    url = f"https://api.worldbank.org/v2/country/{alpha3[country]}/indicator/NY.GDP.PCAP.CD?date={year}&format=json"
    response = requests.get(url).json()
    try:
        return response[1][0]['value']
    except (IndexError, TypeError):
        return None

countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']
years = range(2010, 2020)
gdp_data = {}

for country in countries:
    for year in years:
        gdp_data[(country, year)] = get_gdp_per_capita(country, year)

def add_gdp_feature(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year  
    df['gdp'] = df.apply(lambda row: gdp_data.get((row['country'], row['year']), None), axis=1)
    return df


def add_holiday_feature(df):
    country_holidays = {
        'Canada': holidays.CountryHoliday('CA'),
        'Finland': holidays.CountryHoliday('FI'),
        'Italy': holidays.CountryHoliday('IT'),
        'Kenya': holidays.CountryHoliday('KE'),
        'Norway': holidays.CountryHoliday('NO'),
        'Singapore': holidays.CountryHoliday('SG')
    }

    df['date'] = pd.to_datetime(df['date'])

    df['is_holiday'] = df.apply(
        lambda row: row['date'] in country_holidays.get(row['country'], []), axis=1
    )
    return df



train = add_holiday_feature(train)
test = add_holiday_feature(test)
train = add_gdp_feature(train)
test = add_gdp_feature(test)




from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Define categorical features
categorical_features = [
    'country', 'store', 'product','country_store', 'country_product', 'store_product',
    'country_month', 'store_month', 'product_month' 
]

for col in categorical_features:
    if col in train.columns:
        train[col] = train[col].astype('category')
    if col in test.columns:
        test[col] = test[col].astype('category')


# Splitting data
X = train.drop(['num_sold', 'date', 'id'], axis=1)
y = np.log1p(train['num_sold'])

print ('Training on columns', X.columns)
# Time series split
tscv = TimeSeriesSplit(n_splits=5)
categorical_feature_names = ','.join(categorical_features) 

# CatBoost parameters
catboost_params = {
    'iterations': 832,
    'depth': 12,
    'learning_rate': 0.08217351837593317,
    'l2_leaf_reg': 9.811590810357941,
    'subsample': 0.7437966338164083,
    'bagging_temperature': 0.7845202554017578,
    'colsample_bylevel': 0.8336579366308383,
    'min_data_in_leaf': 17,
    'loss_function': 'MAPE',
    'eval_metric': 'MAPE',
    'verbose': False
}


# LightGBM parameters https://www.kaggle.com/code/abdmental01/lgbm-single-model-forecasting-sticker
lgbm_params = {
    'n_estimators': 802,
    'max_depth': 4,
    'colsample_bytree': 0.4087726844027313,
    'subsample': 0.5150029934968837,
    'learning_rate': 0.0885280505784011,
    'min_child_samples': 98,
    'verbose':-1
}


catboost_mape = []
lgbm_mape = []
lgbm_mape_act = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    print(f"Fold {fold}:")
    

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
 
    # print("Training CatBoost...")
    # cat_model = CatBoostRegressor(**catboost_params)
    # cat_model.fit(
    #     X_train,
    #     y_train,
    #     cat_features=categorical_features,
    #     eval_set=(X_test, y_test),
    #     early_stopping_rounds=100
    # )
    # y_pred_cat = cat_model.predict(X_test)
    # mape_cat = mean_absolute_percentage_error(y_test, y_pred_cat)
    # catboost_mape.append(mape_cat)
    # print(f"CatBoost MAPE for Fold {fold}: {mape_cat:.4f}")

    print("Training LightGBM...")
    lgbm_model = LGBMRegressor(**lgbm_params)
    lgbm_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='mape',
        categorical_feature=f'name:{categorical_feature_names}'
    )
    y_pred_lgbm = lgbm_model.predict(X_test)
    mape_lgbm = mean_absolute_percentage_error(y_test, y_pred_lgbm)
    lgbm_mape.append(mape_lgbm)
    
    mape_lgbm_exp = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(y_pred_lgbm))
    lgbm_mape_act.append(mape_lgbm_exp)
    
    print(f"LightGBM MAPE for Fold {fold}: {mape_lgbm:.4f}")
    print(f"LightGBM MAPE for Fold {fold} (actual): {mape_lgbm_exp:.4f}")


# print("\nCatBoost MAPE per fold:", catboost_mape)
print("LightGBM MAPE per fold:", lgbm_mape_act)


results = {
    # "CatBoost_MAPE": catboost_mape,
    "LightGBM_MAPE": lgbm_mape_act
}




# catboost_mean_mape = np.mean(results["CatBoost_MAPE"])
lgbm_mean_mape = np.mean(results["LightGBM_MAPE"])

# print(f"Mean MAPE for CatBoost: {catboost_mean_mape:.4f}")
print(f"Mean MAPE for LightGBM: {lgbm_mean_mape:.4f}")


X.columns


import matplotlib.pyplot as plt
import seaborn as sns

def calibrate_num_sold(predicted_df, train_df):
    """
    Calibrate predicted num_sold based on store, country, and product ratios, 
    with all ratios calculated within the function using both train and test data.
    
    Product ratios for future dates in the test data are inferred using historical patterns.
    
    Parameters:
    - predicted_df: DataFrame containing 'id', 'date', 'store', 'country', 'product', 'gdp', and predicted 'num_sold'.
    - train_df: DataFrame with historical 'num_sold' and 'gdp' values for calculating ratios.
    
    Returns:
    - DataFrame with calibrated 'num_sold'.
    """
    # Initialize debugging plots and descriptions
    def debug_plot(data, column, title):
        plt.figure(figsize=(12, 6))
        sns.histplot(data[column], bins=50, kde=True)
        plt.title(title)
        plt.show()
    
    def debug_describe(data, column, label):
        print(f"\n{label} Description:")
        print(data[column].describe())
        
    print(f"Are IDs unique in predicted_df before calibration? {predicted_df['id'].is_unique}")
    print(f"Initial duplicate IDs count: {predicted_df.duplicated(subset=['id']).sum()}")

    # Debug: Initial Predictions
    debug_describe(predicted_df, "num_sold", "Predicted num_sold")
    debug_plot(predicted_df, "num_sold", "Initial Predicted num_sold Distribution")

    # Step 1: Calculate Store Ratios
    store_weights = train_df.groupby("store")["num_sold"].sum() / train_df["num_sold"].sum()
    store_weights = store_weights.reset_index().rename(columns={"num_sold": "store_ratio"})
    debug_describe(store_weights, "store_ratio", "Store Ratios")
    debug_plot(store_weights, "store_ratio", "Store Ratios Distribution")

    # Step 2: Calculate GDP Ratios
    train_df["year"] = train_df["date"].dt.year
    gdp_ratios_df = train_df.groupby(["year", "country"])["gdp"].mean().reset_index()
    gdp_ratios_df["gdp_ratio"] = gdp_ratios_df.groupby("year")["gdp"].transform(lambda x: x / x.sum())
    debug_describe(gdp_ratios_df, "gdp_ratio", "GDP Ratios")
    debug_plot(gdp_ratios_df, "gdp_ratio", "GDP Ratios Distribution")

    # Step 3: Calculate Historical Product Ratios
    product_ratios_df = train_df.groupby(["date", "product"])["num_sold"].sum().reset_index()
    product_totals_by_date = train_df.groupby("date")["num_sold"].sum().reset_index().rename(columns={"num_sold": "total_sold"})
    product_ratios_df = pd.merge(product_ratios_df, product_totals_by_date, on="date")
    product_ratios_df["product_ratio"] = product_ratios_df["num_sold"] / product_ratios_df["total_sold"]
    product_ratios_df = product_ratios_df[["date", "product", "product_ratio"]]
    debug_describe(product_ratios_df, "product_ratio", "Historical Product Ratios")
    debug_plot(product_ratios_df, "product_ratio", "Historical Product Ratios Distribution")

    # Step 4: Forecast Product Ratios for Test Dates
    test_years = predicted_df["date"].dt.year.unique()
    max_train_year = train_df["date"].dt.year.max()

    forecasted_product_ratios = []
    for year in test_years:
        if year > max_train_year:
            historical_year = year - 2 if (year - 2) in train_df["date"].dt.year.unique() else max_train_year - 2
            historical_ratios = product_ratios_df.loc[product_ratios_df["date"].dt.year == historical_year].copy()
            historical_ratios["date"] = historical_ratios["date"] + pd.DateOffset(years=(year - historical_year))
            forecasted_product_ratios.append(historical_ratios)

    if forecasted_product_ratios:
        forecasted_product_ratios = pd.concat(forecasted_product_ratios)
        product_ratios_df = pd.concat([product_ratios_df, forecasted_product_ratios]).drop_duplicates()

    # Debug: Combined Product Ratios
    debug_describe(product_ratios_df, "product_ratio", "Combined Product Ratios")
    debug_plot(product_ratios_df, "product_ratio", "Combined Product Ratios Distribution")

    # Step 5: Merge Ratios with Predicted Data
    predicted_df["year"] = predicted_df["date"].dt.year

    # Merge Store Ratios
    predicted_df = pd.merge(predicted_df, store_weights, how="left", on="store")
    debug_describe(predicted_df, "store_ratio", "Merged Store Ratios")
    debug_plot(predicted_df, "store_ratio", "Merged Store Ratios Distribution")

    # Merge GDP Ratios
    predicted_df = pd.merge(predicted_df, gdp_ratios_df, how="left", on=["year", "country"])
    debug_describe(predicted_df, "gdp_ratio", "Merged GDP Ratios")
    debug_plot(predicted_df, "gdp_ratio", "Merged GDP Ratios Distribution")

    # Merge Product Ratios
    predicted_df = pd.merge(predicted_df, product_ratios_df, how="left", on=["date", "product"])
    debug_describe(predicted_df, "product_ratio", "Merged Product Ratios")
    debug_plot(predicted_df, "product_ratio", "Merged Product Ratios Distribution")

    # Step 6: Calibrate num_sold
    predicted_df["calibrated_num_sold"] = (
        predicted_df["num_sold"]
        * predicted_df["store_ratio"]
        * predicted_df["gdp_ratio"]
        * predicted_df["product_ratio"]
    )

    # Handle Missing Ratios by Defaulting to Model Predictions
    predicted_df["calibrated_num_sold"] = np.where(
        predicted_df["store_ratio"].isnull()
        | predicted_df["gdp_ratio"].isnull()
        | predicted_df["product_ratio"].isnull(),
        predicted_df["num_sold"],
        predicted_df["calibrated_num_sold"],
    )

    # Round to integers and clip to ensure no negative values
    predicted_df["calibrated_num_sold"] = np.clip(predicted_df["calibrated_num_sold"].round(), 0, None)

    # Debug: Final Calibrated Values
    debug_describe(predicted_df, "calibrated_num_sold", "Final Calibrated num_sold")
    debug_plot(predicted_df, "calibrated_num_sold", "Final Calibrated num_sold Distribution")
    print(f"Final duplicate IDs count: {predicted_df.duplicated(subset=['id']).sum()}")

    # Remove duplicates from final output
    calibrated_predictions = predicted_df[["id", "calibrated_num_sold"]].rename(columns={"calibrated_num_sold": "num_sold"})
    calibrated_predictions = calibrated_predictions.drop_duplicates(subset="id", keep="first")
    print(f"Are IDs unique in calibrated_predictions? {calibrated_predictions['id'].is_unique}")


    # Return calibrated results
    return predicted_df[["id", "calibrated_num_sold"]].rename(columns={"calibrated_num_sold": "num_sold"})



test_X = test.drop(['date', 'id'], axis=1)


test_predictions = np.expm1(lgbm_model.predict(test_X))
test_predictions = np.clip(test_predictions, 0, None).astype(int)
test['num_sold'] = test_predictions
predicted_df = test.copy() 

calibrated_predictions = calibrate_num_sold(predicted_df=predicted_df, train_df=train)

calibrated_predictions.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")


