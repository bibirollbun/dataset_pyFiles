import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter, FixedLocator
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer  
from sklearn.compose import TransformedTargetRegressor
from tqdm.auto import tqdm
from datetime import date
from statsmodels.tsa.arima.model import ARIMA
import warnings
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore", category=FutureWarning, 
                       message="use_inf_as_na")


data = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv",
                   usecols=["sale_date", "sale_price", "id"],
                   parse_dates = ['sale_date'])


min_date = data['sale_date'].min()
X = data.drop(columns=['sale_price'])
y = data['sale_price']


def currency_formatter(x, pos):
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    return f'{x/1_000:.0f}K'
    
plt.figure(figsize=(12, 6))

# Quartile Plots
sns.lineplot(x=data['sale_date'], y=data['sale_price'],
             estimator=lambda x: np.percentile(x, 25),  # Q1
             errorbar=None, 
             label="Q1 (25th Percentile)",
             color='skyblue')

sns.lineplot(x=data['sale_date'], y=data['sale_price'],
             estimator=lambda x: np.percentile(x, 50),  # Q2 (Median)
             label="Median (Q2)",
             color='green')

sns.lineplot(x=data['sale_date'], y=data['sale_price'],
             estimator=lambda x: np.percentile(x, 75),  # Q3
             errorbar=None,
             label="Q3 (75th Percentile)",
             color='orange')

ax = plt.gca()

date_min = data['sale_date'].min().toordinal()
date_max = data['sale_date'].max().toordinal()
selected_years_num = np.linspace(date_min, date_max, 15)
selected_years = [pd.to_datetime(date.fromordinal(int(num))) for num in selected_years_num]
ax.set_xticks(selected_years)
ax.xaxis.set_major_formatter(DateFormatter('%Y'))
plt.xticks(rotation=15) 

ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

plt.legend()
plt.title("House Price Trends: Quartile Analysis")
plt.xlabel("Sale Date")
plt.ylabel("Sale Price")
plt.tight_layout()
plt.show()


def evaluate_estimator(estimator, X=X, y=y, prediction_window=0.05, n_windows=3):
    
    train_size = 1
    window_sizes = []
    
    for _ in range(n_windows):
        window_size = prediction_window * train_size
        train_size -= window_size
        window_sizes.append(window_size)
        
    window_sizes = window_sizes[::-1]
    print(f"Window Sizes: {[f'{window_size * 100:.2f}%' for window_size in window_sizes]}")
    window_rmse = []
    window_lengths = []
    oof_df  = pd.DataFrame(index = X.index,
                           data={
                               "sale_date": pd.NaT,
                               "sale_price": np.nan,
                               "window": ""
                           })
    
    for i, window_size in tqdm(enumerate(window_sizes), desc="Validating", total=len(window_sizes)):
        train_date_cutoff = X['sale_date'].quantile(train_size)
        validation_date_cutoff = X['sale_date'].quantile(train_size + window_size)
        train_mask = X['sale_date'] < train_date_cutoff
        validation_mask = (X['sale_date'] >= train_date_cutoff) & (X['sale_date'] < validation_date_cutoff)
        X_train, X_val = X[train_mask], X[validation_mask]
        y_train, y_val = y[train_mask], y[validation_mask]
        
        estimator_ = clone(estimator)
        estimator_.fit(X_train, y_train)
        predictions = estimator_.predict(X_val)
        
        oof_df.loc[X_val.index, "sale_date"] = X_val['sale_date']
        oof_df.loc[X_val.index, "sale_price"] = predictions
        oof_df.loc[X_val.index, "window"] = f'Test Window {i + 1}'
        
        rmse = mean_squared_error(y_val, predictions, squared=False)
        window_length = len(y_val)
        window_rmse.append(rmse)
        window_lengths.append(window_length)
                            
        train_size += window_size

    print(f"Mean RMSE: {np.mean(window_rmse):.4f}")
    for i, (rmse, length) in enumerate(zip(window_rmse, window_lengths)):
        print(f"Window {i + 1} of length: {length}, RMSE: {rmse:.4f}")
        
    estimator_ = clone(estimator).fit(X,y)
    predictions = estimator_.predict(X)
    dtrended = y - predictions + y.mean()
    with warnings.catch_warnings():
        unique_windows = sorted(oof_df['window'].dropna().unique())
        warnings.simplefilter("ignore", category=FutureWarning)
        sns.lineplot(x=X['sale_date'],y=y, label="Observed Values")
        sns.lineplot(x=X['sale_date'],y=predictions, label="Model Predictions", color="#20B2AA")
        sns.lineplot(data =oof_df, x='sale_date',y='sale_price',
                     hue='window', palette='bright', hue_order=unique_windows)
        sns.lineplot(x=X['sale_date'], y =dtrended, label='Detrended')
        ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        plt.legend()
        plt.show()


def add_day(X):
    X = X.copy()
    X['day'] = (X['sale_date'] - min_date).dt.days
    X.drop(columns = ['sale_date'],inplace=True)
    return X
lin_poly = Pipeline([
    ('add_day',FunctionTransformer(add_day)),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("reg", LinearRegression(n_jobs=-1))
])
oof_df = evaluate_estimator(lin_poly)

