!pip uninstall -y numpy


!pip install numpy==1.23.5


!pip install pmdarima==1.8.5


pip install lightgbm


import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima.arima.utils import ndiffs

warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', encoding='utf-8')
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', encoding='utf-8')
calender = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', encoding='utf-8')
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv', encoding='utf-8')
weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv', encoding='utf-8')
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv', encoding='utf-8')


print(train.info())
print(train.isnull().sum()) #info of train set


unique_ids = [885, 1237, 725, 3778, 5152, 2148, 2424, 3178, 1776, 1689, 612, 2809, 794]
filtered = train[train['unique_id'].isin(unique_ids)]
results = []
for unique_id in unique_ids:
    current_filtered = filtered[filtered['unique_id'] == unique_id]
    null_1 = current_filtered['total_orders'].isnull().sum()
    non_null_1 = current_filtered['total_orders'].notnull().sum()
    null_2 = current_filtered['sales'].isnull().sum()
    non_null_2 = current_filtered['sales'].notnull().sum()
    results.append({
        'unique_id': unique_id,
        'Null values in total_orders': null_1,
        'Non-null values in total_orders': non_null_1,
        'Null values in sales': null_2,
        'Non-null values in sales': non_null_2
    })
result_data = pd.DataFrame(results)
print(result_data)


print(test.info())
print(test.isnull().sum()) #info of test set


print(calender.info())
print(calender.isnull().sum()) #info of calender set


print(inventory.info())
print(inventory.isnull().sum()) #info of inventory set


print(solution.info())
print(solution.isnull().sum()) #info of solution set


print(weights.info())
print(weights.isnull().sum()) #info of test weights set


train = train[['date', 'warehouse', 'sales', 'unique_id']]
train['date'] = pd.to_datetime(train['date'])
train['warehouse'] = train['warehouse'].astype('category')
train['sales'] = train['sales'].astype('float32')


warehouses = train['warehouse'].unique()
for warehouse in warehouses:
    print(f"\n\n=== Analyzing: {warehouse} ===")
    df_wh = train[train['warehouse'] == warehouse].sort_values('date')
    df_wh.set_index('date', inplace=True)
    sales = df_wh['sales']

    # 1. Stationarity Check
    print("1. Checking Stationarity with ADF:")
    result = adfuller(sales.dropna())
    print(f"ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
    is_stationary = result[1] < 0.05

    # 2. Find differencing needed
    d = ndiffs(sales.dropna(), test='adf')
    print(f"2. Suggested differencing (d): {d}")

    # 3. Make stationary
    sales_stationary = sales.diff(d).dropna()

    # 4. ACF & PACF plots
    print("3. ACF & PACF:")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(sales_stationary, lags=30, ax=ax[0])
    plot_pacf(sales_stationary, lags=30, ax=ax[1])
    ax[0].set_title(f'{warehouse} - ACF')
    ax[1].set_title(f'{warehouse} - PACF')
    plt.tight_layout()
    plt.show()

    # 5. Fit ARIMA model (p and q picked arbitrarily here as 1)
    print("4. Fitting ARIMA model:")
    try:
        model = ARIMA(sales, order=(1, d, 1))
        fit = model.fit()
        print(fit.summary())

        # 6. Residual Check
        print("5. Residuals Check:")
        residuals = fit.resid
        residuals.plot(title=f"{warehouse} - Residuals")
        plt.show()

        plot_acf(residuals.dropna(), lags=30)
        plt.title(f"{warehouse} - Residuals ACF")
        plt.show()

        lb_test = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
        print("Ljung-Box Test:")
        print(lb_test)

    except Exception as e:
        print(f"Could not fit ARIMA for {warehouse}: {e}")

