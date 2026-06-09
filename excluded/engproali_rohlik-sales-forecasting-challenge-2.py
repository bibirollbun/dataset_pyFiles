import pandas as pd
import shutil
import os
from rich.tree import Tree
from rich import print
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import timedelta



sales_train = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")
sales_test = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
inventory = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
calendar = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
test_weights = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv")
solution = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv")


print("Calendar columns:", calendar.columns.tolist())
print("Inventory columns:", inventory.columns.tolist())
print("Sales Test columns:", sales_test.columns.tolist())
print("Sales Train columns:", sales_train.columns.tolist())
print("Solution columns:", solution.columns.tolist())
print("Test Weights columns:", test_weights.columns.tolist())



files = {
    "calendar.csv": pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv"),
    "inventory.csv": pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv"),
    "sales_test.csv": pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv"),
    "sales_train.csv": pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv"),
    "solution.csv": pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv"),
    "test_weights.csv": pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv")
}



tree = Tree("ğŸ“‚ [bold]Dataset Files[/bold]")

for file_name, df in files.items():
    file_branch = tree.add(f"ğŸ“„ [green]{file_name}[/green]")
    for col in df.columns:
        file_branch.add(f"ğŸ“Œ {col}")


print(tree)



# Data merging function
def merge_data(sales_df):
    merged = sales_df.merge(calendar, on=['date', 'warehouse'], how='left')
    merged = merged.merge(inventory, on=['unique_id', 'warehouse'], how='left')
    return merged

full_data = merge_data(sales_train)
test_data = merge_data(sales_test)

# Data cleaning pipeline
def clean_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df['warehouse'] = df['warehouse'].astype('category')
    df['holiday'] = df['holiday'].fillna(0).astype(int)
    df['school_holidays'] = df['school_holidays'].fillna(0).astype(int)

    for col in ['L1_category_name_en', 'L2_category_name_en']:
        df[col] = df[col].fillna('Unknown')

    return df

full_data = clean_data(full_data)
test_data = clean_data(test_data)


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
plt.figure(figsize=(14, 8))
warehouse_sales = full_data.groupby('warehouse', observed=False)['sales'].sum().sort_values(ascending=False)



sns.barplot(x=warehouse_sales.values, y=warehouse_sales.index, palette="coolwarm")
plt.title('Top Performing Warehouses by Sales', fontsize=16)
plt.xlabel('Total Sales', fontsize=12)
plt.ylabel('Warehouse', fontsize=12)
plt.xticks(fontsize=10)
plt.show()


top_products = full_data.groupby('name')['sales'].sum().nlargest(20)

plt.figure(figsize=(16, 10))
sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
plt.title('Best Selling Products', fontsize=16)
plt.xlabel('Total Sales', fontsize=12)
plt.ylabel('Product Name', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()



calendar = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")


closed_days = calendar[calendar["shops_closed"] == 1][["date", "holiday_name", "warehouse"]]


print(closed_days)


closed_days.to_csv("closed_days.csv", index=False)

print(closed_days)


import seaborn as sns

discount_cols = [f'type_{i}_discount' for i in range(7)]
sns.pairplot(full_data, x_vars=discount_cols, y_vars=['sales'])
plt.show()


plt.figure(figsize=(14, 7))
# Group data
holiday_group = full_data[full_data['holiday'] == 1].groupby('warehouse')['sales'].sum()
regular_group = full_data[full_data['holiday'] == 0].groupby('warehouse')['sales'].sum()

# Create comparison DataFrame
df_compare = pd.DataFrame({
    'Holiday Sales': holiday_group,
    'Regular Sales': regular_group
}).sort_values('Holiday Sales', ascending=False)

# Plotting
df_compare.plot(kind='bar', figsize=(14,7), color=['#FF6F61', '#2E86C1'])
plt.title('Holiday vs Regular Sales by Warehouse', fontsize=16)
plt.xlabel('Warehouse', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Day Type')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Analyze holiday vs regular sales
holiday_sales = full_data[full_data['holiday'] == 1]['sales']
regular_sales = full_data[full_data['holiday'] == 0]['sales']

# Calculate percentage difference
avg_holiday = holiday_sales.mean()
avg_regular = regular_sales.mean()
perc_diff = ((avg_holiday - avg_regular)/avg_regular) * 100

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=['Holidays', 'Regular Days'],
            y=[avg_holiday, avg_regular],
            palette="viridis")
plt.title('Sales Comparison: Holidays vs Regular Days', fontsize=14)
plt.ylabel('Average Sales', fontsize=12)
plt.text(0, avg_holiday, f'{perc_diff:.1f}%', ha='center', va='bottom', fontsize=12)
plt.show()


calendar = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")




summary = calendar.groupby("warehouse")["shops_closed"].value_counts().unstack()


summary.columns = ["Open Days", "Closed Days"]


fig, axes = plt.subplots(1, len(summary), figsize=(12, 6))

for i, (warehouse, data) in enumerate(summary.iterrows()):
 
    labels = ["Open Days", "Closed Days"]
    sizes = [data["Open Days"], data["Closed Days"]]
    colors = ["green", "red"]


    axes[i].pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[i].set_title(f": {warehouse}")

plt.tight_layout()
plt.show()



sales_train = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")





sales_train["date"] = pd.to_datetime(sales_train["date"])


storage_dates = sales_train.groupby(["warehouse", "unique_id"])["date"].min().reset_index()


storage_dates.rename(columns={"date": "storage_date"}, inplace=True)


print(storage_dates.head())


storage_dates.to_csv("storage_dates.csv", index=False)


calendar = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")

# ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ù…Ù„Ù�Ø§Øª
closed_days = pd.read_csv("closed_days.csv")


calendar_closed = calendar[calendar["shops_closed"] == 1][["date", "holiday_name", "warehouse"]]


difference = calendar_closed.merge(closed_days, on=["date", "holiday_name", "warehouse"], how="outer", indicator=True)


print(difference[difference["_merge"] != "both"])




sales_train = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")




sales_train["date"] = pd.to_datetime(sales_train["date"])


creation_dates = sales_train.groupby("warehouse")["date"].min().reset_index()


creation_dates.rename(columns={"date": "Creation_Date"}, inplace=True)


print(creation_dates.head())


creation_dates.to_csv("creation_dates.csv", index=False)



sales_train = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")

warehouse_sales = sales_train.groupby("warehouse")["sales"].sum().reset_index()

warehouse_sales = warehouse_sales.sort_values(by="sales", ascending=False)

warehouse_sales["Rank"] = warehouse_sales["sales"].rank(method="dense", ascending=False).astype(int)

print("Warehouses ranked from highest to lowest sales:")
print(warehouse_sales)

plt.figure(figsize=(10, 6))
plt.bar(warehouse_sales["warehouse"], warehouse_sales["sales"], color='skyblue')
plt.xlabel("Warehouse")
plt.ylabel("Total Sales")
plt.title("Warehouse Ranking by Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(warehouse_sales["sales"], labels=warehouse_sales["warehouse"], autopct="%1.1f%%", startangle=90)
plt.title("Warehouse Sales Distribution")
plt.tight_layout()
plt.show()



sales_train = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")
product_sales = sales_train.groupby(["warehouse", "unique_id"])["sales"].sum().reset_index()
warehouses = product_sales["warehouse"].unique()
for warehouse in warehouses:
    wh_data = product_sales[product_sales["warehouse"] == warehouse].sort_values(by="sales", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(wh_data["unique_id"].astype(str), wh_data["sales"], color='teal')
    plt.xlabel("Product (unique_id)")
    plt.ylabel("Total Sales")
    plt.title(f"Product Sales Ranking for Warehouse: {warehouse}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()




sales_train = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")
sales_train["date"] = pd.to_datetime(sales_train["date"])

# Load the closed days data and convert its date column to datetime
closed_days = pd.read_csv("closed_days.csv")
closed_days["date"] = pd.to_datetime(closed_days["date"])

# Merge sales data with closed days based on date and warehouse
# This filters the sales to include only those made on days when the store was closed (holiday days)
holiday_sales = pd.merge(sales_train, closed_days, on=["date", "warehouse"], how="inner")

# Group the holiday sales by warehouse and calculate total sales on holidays
warehouse_holiday_sales = holiday_sales.groupby("warehouse")["sales"].sum().reset_index()

# Display the aggregated holiday sales per warehouse
print("Holiday Sales per Warehouse:")
print(warehouse_holiday_sales)

# Plot a bar chart for the total sales on holidays for each warehouse
plt.figure(figsize=(10, 6))
plt.bar(warehouse_holiday_sales["warehouse"], warehouse_holiday_sales["sales"], color="orange")
plt.xlabel("Warehouse")
plt.ylabel("Total Sales on Holidays")
plt.title("Sales on Holidays per Warehouse")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




def create_sequences(series, window_size, forecast_horizon):
 
    X, y = [], []
    for i in range(len(series) - window_size - forecast_horizon + 1):
        X.append(series[i:(i + window_size)])
        y.append(series[(i + window_size):(i + window_size + forecast_horizon)])
    return np.array(X), np.array(y)

def build_gru_model(input_shape, forecast_horizon):
  
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, activation='relu'),
        Dense(forecast_horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm_fcnn_model(input_shape, forecast_horizon):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, activation='relu', return_sequences=True),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(forecast_horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

sales_train = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")
sales_train["date"] = pd.to_datetime(sales_train["date"])


product_sales = sales_train.groupby(["warehouse", "unique_id"])["sales"].sum().reset_index()

best_selling = product_sales.sort_values(["warehouse", "sales"], ascending=[True, False]) \
                            .drop_duplicates(subset=["warehouse"], keep="first")
forecast_horizon = 14
window_size = 30

forecast_results = []
evaluation_metrics = []

for idx, row in best_selling.iterrows():
    warehouse = row["warehouse"]
    product = row["unique_id"]

    df = sales_train[(sales_train["warehouse"] == warehouse) & (sales_train["unique_id"] == product)]
    df = df.sort_values("date")

    ts = df.groupby("date")["sales"].sum().reset_index()
    all_dates = pd.date_range(ts["date"].min(), ts["date"].max())
    ts = ts.set_index("date").reindex(all_dates, fill_value=0).rename_axis("date").reset_index()

    series = ts["sales"].values.astype(float)

    train, val = series[:-forecast_horizon], series[-(forecast_horizon + window_size):]

    X_train, y_train = create_sequences(train, window_size, forecast_horizon)
    X_val, y_val = create_sequences(val, window_size, forecast_horizon)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    gru_model = build_gru_model(input_shape=(window_size, 1), forecast_horizon=forecast_horizon)
    gru_model.fit(X_train, y_train, epochs=20, verbose=0)
    gru_pred = gru_model.predict(X_val[-1].reshape(1, window_size, 1)).flatten()

    actual = y_val[-1]
    rmse_gru = sqrt(mean_squared_error(actual, gru_pred))
    mae_gru = mean_absolute_error(actual, gru_pred)
    mse_gru = mean_squared_error(actual, gru_pred)

    lstm_fcnn_model = build_lstm_fcnn_model(input_shape=(window_size, 1), forecast_horizon=forecast_horizon)
    lstm_fcnn_model.fit(X_train, y_train, epochs=20, verbose=0)
    lstm_pred = lstm_fcnn_model.predict(X_val[-1].reshape(1, window_size, 1)).flatten()

    rmse_lstm = sqrt(mean_squared_error(actual, lstm_pred))
    mae_lstm = mean_absolute_error(actual, lstm_pred)
    mse_lstm = mean_squared_error(actual, lstm_pred)

    if rmse_gru <= rmse_lstm:
        best_model = gru_model
        model_name = "GRU"
    else:
        best_model = lstm_fcnn_model
        model_name = "LSTM-FCNN"

    last_window = series[-window_size:]
    last_window = last_window.reshape((1, window_size, 1))
    forecast = best_model.predict(last_window).flatten()

    last_date = ts["date"].max()
    for i in range(forecast_horizon):
        forecast_date = last_date + timedelta(days=i+1)
        submission_id = f"{warehouse}_{forecast_date.strftime('%Y-%m-%d')}"
        forecast_results.append({"id": submission_id, "sales_hat": round(forecast[i], 2)})

    evaluation_metrics.append({
        "warehouse": warehouse,
        "product": product,
        "model_used": model_name,
        "GRU_RMSE": rmse_gru,
        "GRU_MAE": mae_gru,
        "GRU_MSE": mse_gru,
        "LSTM_FCNN_RMSE": rmse_lstm,
        "LSTM_FCNN_MAE": mae_lstm,
        "LSTM_FCNN_MSE": mse_lstm
    })

    plt.figure(figsize=(10, 4))
    plt.plot(range(window_size), X_val[-1].flatten(), label="Input Sequence")
    plt.plot(range(window_size, window_size + forecast_horizon), actual, label="Actual")
    plt.plot(range(window_size, window_size + forecast_horizon),
             gru_pred if model_name=="GRU" else lstm_pred,
             label=f"{model_name} Forecast")
    plt.xlabel("Time Steps")
    plt.ylabel("Sales")
    plt.title(f"Validation Forecast for Warehouse: {warehouse}")
    plt.legend()
    plt.tight_layout()
    plt.show()

submission_df = pd.DataFrame(forecast_results)
submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

eval_df = pd.DataFrame(evaluation_metrics)
print("\nEvaluation Metrics per Warehouse:")
print(eval_df)



eval_df = pd.DataFrame(evaluation_metrics)

warehouses = eval_df["warehouse"]
x = np.arange(len(warehouses))
width = 0.35  # width of the bars

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot RMSE
axes[0].bar(x - width/2, eval_df["GRU_RMSE"], width, label="GRU", color='skyblue')
axes[0].bar(x + width/2, eval_df["LSTM_FCNN_RMSE"], width, label="LSTM-FCNN", color='salmon')
axes[0].set_title("RMSE per Warehouse")
axes[0].set_xticks(x)
axes[0].set_xticklabels(warehouses, rotation=45, ha="right")
axes[0].set_ylabel("RMSE")
axes[0].legend()

# Plot MAE
axes[1].bar(x - width/2, eval_df["GRU_MAE"], width, label="GRU", color='skyblue')
axes[1].bar(x + width/2, eval_df["LSTM_FCNN_MAE"], width, label="LSTM-FCNN", color='salmon')
axes[1].set_title("MAE per Warehouse")
axes[1].set_xticks(x)
axes[1].set_xticklabels(warehouses, rotation=45, ha="right")
axes[1].set_ylabel("MAE")
axes[1].legend()

# Plot MSE
axes[2].bar(x - width/2, eval_df["GRU_MSE"], width, label="GRU", color='skyblue')
axes[2].bar(x + width/2, eval_df["LSTM_FCNN_MSE"], width, label="LSTM-FCNN", color='salmon')
axes[2].set_title("MSE per Warehouse")
axes[2].set_xticks(x)
axes[2].set_xticklabels(warehouses, rotation=45, ha="right")
axes[2].set_ylabel("MSE")
axes[2].legend()

plt.tight_layout()
plt.show()


