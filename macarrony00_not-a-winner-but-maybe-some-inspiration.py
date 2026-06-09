import copy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Do not truncate the display of DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

tqdm.pandas()


"""
Note 1:
    I am validating my results using validation set as similar as our test itself.
    Train with recent data only, because I will calculate 2 year lags, I don't want to have null values
"""
training_dates = ["2023-08-01", "2024-05-19"]
validation_dates = ["2024-05-20", "2024-06-02"]
test_dates = ["2024-06-03", "2024-06-16"]


lag_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
lag_features += [21, 28, 56, 56*2, 365, 365*2]


df_train = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")
df_test = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
df_calendar = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
df_inventory = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
df_solution = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv")
df_test_weights = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv")


# Lets take a look at the data
df_train.head()


"""
Note 2:
    We want to maximize number of similar use cases for training.
    Although each unique_id has difference range of values, we can scale each product to have same values.
    The model will probabily generalized a lot better this way.
"""

from sklearn.preprocessing import StandardScaler, RobustScaler

scalers = {}

for unique_id in tqdm(df_train["unique_id"].unique()):
    scaler = StandardScaler()
    sales = df_train.loc[df_train["unique_id"] == unique_id, "sales"].values.reshape(-1, 1)
    scaler.fit(sales)
    scalers[unique_id] = scaler
    
    df_train.loc[df_train["unique_id"] == unique_id, "sales"] = scaler.transform(sales).flatten()


"""
Note 3:
    Fixing outliers didn't improve solution
"""

# Fix outliers for training data
# df_train.loc[df_train["date"].between(*training_dates), "sales"] = np.clip(df_train.loc[df_train["date"].between(*training_dates), "sales"], -5, 5)


"""
Note 4:
    I tested using the same approach as before, to scale down total order for each warehouse
    But it didn't improve solution
"""

# # Standard scaler on the total_orders column for each warehouse
# scalers_total_orders = {}

# for warehouse in tqdm(df_train["warehouse"].unique()):
#     scaler = StandardScaler()
#     total_orders = df_train.loc[df_train["warehouse"] == warehouse, "total_orders"].values.reshape(-1, 1)
#     scaler.fit(total_orders)
#     scalers_total_orders[warehouse] = scaler
    
#     df_train.loc[df_train["warehouse"] == warehouse, "total_orders"] = scaler.transform(total_orders).flatten()


# At the end we still need to reverse back to not scaled values
def inverse_norm(df_, indexes, y_pred):
    df_.loc[indexes, "prediction_norm"] = y_pred
    df_.loc[indexes, "y_pred"] = df_.groupby("unique_id")["prediction_norm"].transform(lambda x: scalers[x.name].inverse_transform(x.values.reshape(-1, 1)).flatten())
    y_pred = df_.loc[indexes, "y_pred"].values
    y_pred = np.where(y_pred < 0, 0, y_pred)
    df_ = df_.drop(columns=["prediction_norm", "y_pred"])
    return y_pred


# Create an index for warehouse demand for each day
df = pd.concat([df_train, df_test], axis=0)


from datetime import datetime
czech_holiday = [ 
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),#loss
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"), #loss
]
brno_holiday = [
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),#loss
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"), #loss
]

budapest_holidays = []
munich_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),#loss
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),#loss
]

frank_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),#loss
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),#loss
]

def fill_loss_holidays(df_fill, warehouses, holidays):
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df

df_calendar = fill_loss_holidays(df_fill=df_calendar, warehouses=['Prague_1', 'Prague_2', 'Prague_3'], holidays=czech_holiday)
df_calendar = fill_loss_holidays(df_fill=df_calendar, warehouses=['Brno_1'], holidays=brno_holiday)
df_calendar = fill_loss_holidays(df_fill=df_calendar, warehouses=['Munich_1'], holidays=munich_holidays)
df_calendar = fill_loss_holidays(df_fill=df_calendar, warehouses=['Frankfurt_1'], holidays=frank_holidays)
df_calendar = fill_loss_holidays(df_fill=df_calendar, warehouses=['Budapest_1'], holidays=budapest_holidays)

Frankfurt_1 = df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Frankfurt_1"')
Prague_2 = df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_2"')
Brno_1 = df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Brno_1"')
Munich_1 = df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Munich_1"')
Prague_3 = df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_3"')
Prague_1 = df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_1"')
Budapest_1 = df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Budapest_1"')

def process_calendar(df):
    df = df.sort_values('date').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['date'])
    df['next_holiday_date'] = df.loc[df['holiday'] == 1, 'datetime'].shift(-1)
    df['next_holiday_date'] = df['next_holiday_date'].bfill()
    df['days_to_holiday'] = (df['next_holiday_date'] - df['datetime']).dt.days
    df.drop(columns=['next_holiday_date'], inplace=True)
    df['next_shops_closed_date'] = df.loc[df['shops_closed'] == 1, 'datetime'].shift(-1)
    df['next_shops_closed_date'] = df['next_shops_closed_date'].bfill()
    df['days_to_shops_closed'] = (df['next_shops_closed_date'] - df['datetime']).dt.days
    df.drop(columns=['next_shops_closed_date'], inplace=True)
    df['day_after_closing'] = (
        (df['shops_closed'] == 0) & (df['shops_closed'].shift(1) == 1)
    ).astype(int)
    
    df['long_weekend'] = (
        (df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)
    ).astype(int)
    return df.drop(columns=['datetime'])

dfs = ['Frankfurt_1', 'Prague_2', 'Brno_1', 'Munich_1', 'Prague_3', 'Prague_1', 'Budapest_1']
processed_dfs = [process_calendar(globals()[df]) for df in dfs]
df_calendar_extended = pd.concat(processed_dfs).sort_values('date').reset_index(drop=True)

df = df.merge(df_calendar_extended, on=['date', 'warehouse'], how='left')


"""
Note 6:
    I tried to mine some extra info from total_orders
    If I understand correctly the total_orders for that day is mainly the order for the day
    But there were some cases where on some products the delivery is made on another days of something like that
    I was expecting that some value could be extracted from those cases
    For that reason I calculated: 
        total_order_median - the actual total orders for that day
        total_order_med_diff - difference between median and specific product
        warehouse_demand - calculated demand between 0 and 1 based on total_order_median (if 1 warehouse has the maximum demand to date)
"""

# Create a new column named total_orders_median
df["total_orders_median"] = df.groupby(["date", "warehouse"])["total_orders"].transform("median")
df["total_orders_med_diff"] = df["total_orders"] - df["total_orders_median"]

# Calculate all time max and min demand for each warehouse
df_warehouse_limits = df.groupby("warehouse")["total_orders"].agg(["max", "min"])

# warehouse demand should be between 0 and 1 based on the max and min demand of the warehouse
df["warehouse_demand"] = df.progress_apply(
    lambda x: (x["total_orders_median"] - df_warehouse_limits.loc[x["warehouse"], "min"]) / (
        df_warehouse_limits.loc[x["warehouse"], "max"] - df_warehouse_limits.loc[x["warehouse"], "min"]
    ),
    axis=1,
)


"""
Note 7:
    Calculate average demand across all warehouses for each day
"""

# Create a new dataframe with 2 columns
# Dates and mean of total orders for each day
df_date = df.groupby("date")["total_orders"].mean().reset_index()
df_date = df_date.rename(columns={"total_orders": "daily_demand"})
df = df.merge(df_date, on="date", how="left")


df["max_discount"] = df[
    ["type_0_discount", "type_1_discount", "type_2_discount", "type_3_discount", "type_4_discount", "type_5_discount", "type_6_discount"]
].max(axis=1)


# Extract month, day, weekday, etc.
df["datetime"] = pd.to_datetime(df["date"])
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["weekday"] = df["datetime"].dt.weekday
df["quarter"] = df["datetime"].dt.quarter
df["week_of_year"] = df["datetime"].dt.isocalendar().week
df["day_of_year"] = df["datetime"].dt.dayofyear
df["is_weekend"] = df["datetime"].dt.weekday.isin([5, 6]).astype(int)
df["is_month_start"] = df["datetime"].dt.is_month_start.astype(int)
df["is_month_end"] = df["datetime"].dt.is_month_end.astype(int)
df['year_sin'] = np.sin(2 * np.pi * df['datetime'].dt.year)
df['year_cos'] = np.cos(2 * np.pi * df['datetime'].dt.year)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12) 
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

df.drop(columns=["datetime"], inplace=True)


"""
Note 8:
    In the next cells I will calculate some lags
    I tried some approaches to fix missing values in between existing values for specific products
    But it didn't improve the solution
"""

# # Add columns with sales from previous periods
# df_all_dates = df[["unique_id", "date", "sales"]].copy()

# # Convert date to datetime
# df_all_dates["date"] = pd.to_datetime(df_all_dates["date"])

# # Create all dates for each unique_id
# all_dates = pd.date_range(df_all_dates["date"].min(), df_all_dates["date"].max(), freq="D")
# all_ids = df_all_dates["unique_id"].unique()

# # Create a complete DataFrame with all combinations of dates and unique_ids
# df_complete = pd.DataFrame(
#     [(date, id) for date in all_dates for id in all_ids],
#     columns=["date", "unique_id"]
# )

# # Merge the complete DataFrame with the original data
# df_all_dates = pd.merge(df_complete, df_all_dates, on=["date", "unique_id"], how="left")

# # Sort the DataFrame
# df_all_dates = df_all_dates.sort_values(["unique_id", "date"])

# # Perform forward fill and backward fill for each unique_id
# df_all_dates["sales"] = df_all_dates.groupby(["unique_id", "date"])["sales"].transform(lambda x: x.ffill().bfill())

# # Convert date back to string if needed
# df_all_dates["date"] = df_all_dates["date"].astype(str)
# df_all_dates

df_all_dates = df.sort_values("date").copy()


"""
Note 9:
    Calculate all the lags for sales in of the product in specific warehouse
    Note that the sales here are still scaled :)
"""

df_grouped = df_all_dates.groupby(["unique_id", "warehouse"])

for i in lag_features:
    df_all_dates[f"sales_item_warehouse_lag_{i}"] = df_grouped["sales"].shift(i)
    # shift but keep last value if there is no value
    df_all_dates[f"sales_item_warehouse_lag_{i}"] = df_all_dates[f"sales_item_warehouse_lag_{i}"].fillna(
        df_grouped["sales"].transform("last")
    )

# Merge the lag features back to the original dataframe
df = df.merge(
    df_all_dates[["date", "unique_id"] + [f"sales_item_warehouse_lag_{i}" for i in lag_features]],
    on=["date", "unique_id"],
    how="left"
)


"""
Note 10:
    Calculate all the lags for sales in of the product and product category for all warehouses combined
"""

df = df.merge(df_inventory, on=["unique_id", "warehouse"], how="left")

category_features = ["product_unique_id", "name", "L1_category_name_en", "L2_category_name_en", "L3_category_name_en", "L4_category_name_en"]

# Group by product_unique_id
for i in lag_features:
    df[f"sales_item_lag_{i}"] = df.groupby("product_unique_id")[f"sales_item_warehouse_lag_{i}"].transform("mean")
    df[f"sales_l3_lag_{i}"] = df.groupby("L3_category_name_en")[f"sales_item_warehouse_lag_{i}"].transform("mean")
    df[f"sales_l4_lag_{i}"] = df.groupby("L4_category_name_en")[f"sales_item_warehouse_lag_{i}"].transform("mean")
    
df = df.drop(columns=category_features)


"""
Note 11:
    Tried to mine some inside info with weights as well
    Maybe the weights were calculated based on our sales target
"""
# Add weights
df = df.merge(df_test_weights, on="unique_id", how="left")


# Keep same order of rows, for the final prediction
df = df.sort_values(["date", "unique_id"])


# Feature encoding using Label Encoding
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for feature in ["warehouse", "holiday_name"]:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le


"""
Note 12:
    Not ideal for all solutions, but didn't explore this in depth yet
"""
df["sales"] = df["sales"].fillna(0)
df["warehouse_demand"] = df["warehouse_demand"].fillna(0)
df = df.fillna(0)


not_features = ["unique_id", "date", "sales", "availability"] 


"""
Note 13:
    This is the main train code
    The only aspect that I would like to share is the following idea:
    - We can use distinct range of lags based on the day of test
    For example for the day 0 of test set, we can use data until day -1 (lag=1)
    But for day 14, we still can use data until day -1 (lag=14)
    We can train a model for each day of training using more lags if possible.
"""
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

def train_pipeline(df_, lag_features_, model_, training_dates_, validation_dates_, test_dates_, callbacks=[]):
    
    # "warehouse", 
    features = df_.columns.difference(not_features)
    target = "sales"

    # Drop rows with NaN values
    df_ = df_.dropna(subset=features)

    X_train = df_[df_["date"].between(*training_dates_)][features]
    y_train = df_[df_["date"].between(*training_dates_)][target]
    X_val = df_[df_["date"].between(*validation_dates_)][features]
    y_val = df_[df_["date"].between(*validation_dates_)][target]
    X_test = df_[df_["date"].between(*test_dates_)][features]
    y_test = df_[df_["date"].between(*test_dates_)][target]

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    maew_list = []
    test_predictions = []
    val_predictions = []
    val_true = []
    
    print("validation_dates", validation_dates_)
    print("test_dates", test_dates_)
    print("Starting training")

    next_start = pd.to_datetime(validation_dates_[0])

    # Let's iterate over the lag features, and train a model_ for each
    for index, lag in enumerate(lag_features_):
        next_lag = lag_features_[index + 1] if index + 1 < len(lag_features_) else None
        
        print(f"Lag: {lag}, Next Lag: {next_lag}")

        # Get the validation and test days
        val_period_start = next_start
        val_period_end = next_start + pd.Timedelta(days=next_lag-lag)
        test_period_start = val_period_start + pd.Timedelta(days=14)
        test_period_end = val_period_end + pd.Timedelta(days=14)
        next_start = val_period_end
        
        if val_period_start > pd.to_datetime(validation_dates_[1]):
            print("No more validation days")
            break

        # Keep test outside val
        val_period_end = min(val_period_end, pd.to_datetime(test_dates[0]))
        
        val_period_start = str(val_period_start.date())
        val_period_end = str(val_period_end.date())
        test_period_start = str(test_period_start.date())
        test_period_end = str(test_period_end.date())

        # Do not use leakage features
        exclude = [l for l in lag_features_ if l < lag]
        leak_features = [c for c in X_train.columns if "lag" in c]
        leak_features = [c for c in leak_features if int(c.split("_")[-1]) in exclude]
        print(f"Removing features: {leak_features}")
        
        print(f"Validation period: {val_period_start} - {val_period_end}")
        print(f"Test period: {test_period_start} - {test_period_end}")

        # Filters
        val_period = df_["date"].between(val_period_start, val_period_end, inclusive="left")
        test_period = df_["date"].between(test_period_start, test_period_end, inclusive="left")
        
        # Prints days in periods
        print(f"Validation days: {df_[val_period]['date'].unique()}")
        print(f"Test days: {df_[test_period]['date'].unique()}")
        
        if len(df_[test_period]['date'].unique()) == 0:
            print("We got all the days for the test set")
            break

        # Filter data for the current day
        X_train_period = X_train.drop(columns=leak_features)
        y_train_period = y_train
        X_val_period = X_val.loc[val_period, :].drop(columns=leak_features)
        y_val_period = y_val[val_period]
        X_test_period = X_test.loc[test_period, :].drop(columns=leak_features)
        y_test_period = y_test[test_period]
        
        print(X_train_period.shape, y_train_period.shape, X_val_period.shape, y_val_period.shape, X_test_period.shape, y_test_period.shape)

        if model_.__class__.__name__ == "CatBoostRegressor":
            model_.fit(X_train_period, y_train_period, eval_set=(X_val_period, y_val_period))
        
        y_pred = inverse_norm(df_, val_period, model_.predict(X_val_period))
        val_predictions.append(y_pred) 
        y_val_period = inverse_norm(df_, val_period, y_val_period)
        val_true.append(y_val_period)

        # Calculate the error
        wmae = mean_absolute_error(y_val_period, y_pred, sample_weight=X_val_period["weight"])
        print(f"Lag {lag} got WMAE: {wmae}")
        for _ in range(df_[val_period]['date'].nunique()):
            maew_list.append(wmae)

        # Predict test set
        y_pred = model_.predict(X_test_period)
        y_pred = inverse_norm(df_, test_period, y_pred)

        # Save the prediction
        test_predictions.append(y_pred)

        try:
            # Feature Importances
            # Get feature importances
            feature_importances = model_.get_feature_importance()
            # Create a pandas Series for better visualization
            importance_df = pd.Series(feature_importances, index=X_train_period.columns).sort_values(ascending=False)
            # Print top 20 feature importances
            print("Top Feature Importances:")
            for feature, importance in importance_df.items():
                print(f"Feature: {feature}, Importance: {importance:.2f}")
        except:
            pass
            
        # Average Mean
        print(f"Average WMAE: {np.mean(maew_list)}")
        
    return maew_list, test_predictions, val_predictions, val_true


from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

model = CatBoostRegressor(
    iterations=10_000,
    #learning_rate=0.1,
    depth=10,
    loss_function="MAE",
    verbose=100,
    random_seed=0,
    early_stopping_rounds=50,
)
maew_list, test_predictions, val_predictions, val_true = train_pipeline(df, lag_features, model, training_dates, validation_dates, test_dates)


# Analyze Validation Residuals
# Filter df_val 
df_val = df[df["date"].between(*validation_dates)]
df_val["prediction"] = np.concatenate(val_predictions)
df_val["true"] = np.concatenate(val_true)
df_val["residual"] = df_val["true"] - df_val["prediction"]

# Plot residuals
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.histplot(df_val["residual"], bins=50, kde=True)
plt.title("Validation Residuals")
plt.show()


# Top 10 greater residuals
df_val.sort_values("residual", ascending=False).head(10)


# Top 10 smaller residuals
df_val.sort_values("residual", ascending=True).head(10)


# Group for each unique_id
df_res_unique_id = df_val.groupby("unique_id")["residual"].mean().reset_index()
df_res_unique_id = df_res_unique_id.rename(columns={"residual": "residual_unique"})
df_res_unique_id.sort_values("residual_unique", ascending=False, inplace=True)
df_res_unique_id.head()


# Group for warehouse
df_res_warehouse = df_val.groupby("warehouse")["residual"].mean().reset_index()
df_res_warehouse = df_res_warehouse.rename(columns={"residual": "residual_warehouse"})
df_res_warehouse.sort_values("residual_warehouse", ascending=False, inplace=True)
df_res_warehouse.head()


# Group for day
df_res_date = df_val.groupby("date")["residual"].mean().reset_index()
df_res_date = df_res_date.rename(columns={"residual": "residual_date"})
df_res_date.sort_values("residual_date", ascending=False, inplace=True)
df_res_date.head()


import math
maew_mean = np.mean(maew_list)
maew_mean = math.floor(maew_mean * 100_000) / 100_000
print(f"Mean WMAE: {maew_mean}")


# Plot the errors, add a line for the mean
import matplotlib.pyplot as plt

plt.plot(maew_list)
plt.axhline(np.mean(maew_list), color="red", linestyle="--")
plt.xlabel("Day")
plt.ylabel("WMAE")
plt.title("WMAE for each day in the validation set")
plt.show()


df_test = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
df_test = df_test.sort_values(["date", "unique_id"])
df_test["sales_hat"] = np.array([item for sublist in test_predictions for item in sublist]) # model.predict(X_test)
df_test.head()


"""
Note 14:
    Our residuals in test set should be similar to the validation set, as the weeks are very close to each other
    If we are predicting to low sales for a specific warehouse we can adjust our test prediction for better based
    on the average residuals of each warehouse for the validation set.
    I do the same for warehouses and for each product
    The idea here is very similar to the one someone mentioned allready which consisted of multiplying our prediction with 1.02
"""

# Adjust sales_hat based on residuals
df_test = df_test.merge(df_res_unique_id, on="unique_id", how="left")
df_test["warehouse"] = label_encoders["warehouse"].transform(df_test["warehouse"])
df_test = df_test.merge(df_res_warehouse, on="warehouse", how="left")

df_test["residual_unique"] = df_test["residual_unique"].fillna(0)
df_test["residual_warehouse"] = df_test["residual_warehouse"].fillna(0)

df_test["sales_hat"] = df_test["sales_hat"] + 0.1*df_test["residual_unique"] + 0.1*df_test["residual_warehouse"]


# save submission file with predictions
df_test["id"] = df_test["unique_id"].astype(str) + "_" + pd.to_datetime(df_test["date"]).dt.strftime("%Y-%m-%d")
df_test[["id", "sales_hat"]].to_csv(f"submission.csv", index=False)

