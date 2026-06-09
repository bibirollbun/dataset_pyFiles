# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pylab as plt
import seaborn as sns
import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
plt.style.use("ggplot")


# from the notebook https://www.kaggle.com/code/reidjohnson/pi-ii-demo-qrf

def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
    """Compute the Winkler Interval Score for prediction intervals.

    Args:
        y_true (array-like): True observed values.
        lower (array-like): Lower bounds of prediction intervals.
        upper (array-like): Upper bounds of prediction intervals.
        alpha (float): Significance level (e.g., 0.1 for 90% intervals).
        return_coverage (bool): If True, also return empirical coverage.

    Returns:
        score (float): Mean Winkler Score.
        coverage (float, optional): Proportion of true values within intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    score += np.where(y_true < lower, penalty_lower, 0)
    score += np.where(y_true > upper, penalty_upper, 0)

    if return_coverage:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(inside)
        return np.mean(score), coverage

    return np.mean(score)


df_train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv").drop(columns="id")
df_test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")

df_train.columns, df_test.columns


df_train.info()


df_train.head()


df_train.describe().T


from datetime import datetime

def fe(df):
    # Calculate days since sale_date
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    today = pd.to_datetime(datetime.today().date())
    df["days_since_sale"] = (today - df["sale_date"]).dt.days
    df = df.drop(columns="sale_date") 
    df["submarket"] = df["submarket"].fillna("")
    df["sale_nbr"] = df["sale_nbr"].fillna(0)
    df["subdivision"] = df["subdivision"].fillna("")
    
    # Convert all non-numeric columns to 'category'
    for col in df.select_dtypes(exclude=['number']).columns:
        df[col] = df[col].astype('category')

    return df



df_train = fe(df_train)
df_test = fe(df_test)


df_train.info()


sns.heatmap(df_train.corr(numeric_only=True))
plt.show()


from sklearn.model_selection import train_test_split

x = df_train.drop(columns="sale_price")
y = df_train["sale_price"]
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2)


from catboost import CatBoostRegressor

cats = x_train.select_dtypes(exclude=['number']).columns.tolist()

model_high_cat = CatBoostRegressor(
    loss_function='Quantile:alpha=0.95',
    iterations=10000,
    learning_rate=0.05,
    verbose=500,
    grow_policy = "Depthwise",
    min_data_in_leaf = 1000,
    l2_leaf_reg = 100,
    od_type="IncToDec",
    od_pval=0.1,
)
model_low_cat = CatBoostRegressor(
    loss_function='Quantile:alpha=0.05',
    iterations=10000,   
    learning_rate=0.05, 
    verbose=500,
    grow_policy = "Depthwise",
    min_data_in_leaf = 1000,
    l2_leaf_reg = 100, 
    od_type="IncToDec",
    od_pval=0.1,
)

high = model_high_cat.fit(x_train, y_train,cat_features=cats).predict(x_val)
low = model_low_cat.fit(x_train, y_train,cat_features=cats).predict(x_val)

winkler_score(y_val, low, high, return_coverage = True)


from lightgbm import LGBMRegressor

model_low_lgbm=LGBMRegressor(n_estimators=500, learning_rate = 0.1, min_data_in_leaf=100,objective='quantile',alpha = 0.05)
model_high_lgbm=LGBMRegressor(n_estimators=500, learning_rate = 0.1,min_data_in_leaf=100,objective='quantile',alpha = 0.95)

high = model_high_lgbm.fit(x_train, y_train).predict(x_val)
low = model_low_lgbm.fit(x_train, y_train).predict(x_val)

winkler_score(y_val, low, high, return_coverage = True)


import plotly.graph_objects as go

# Sort by actual values
sorted_idx = np.argsort(y_val)
y_sorted = y_val.iloc[sorted_idx].values
low_sorted = low[sorted_idx]
high_sorted = high[sorted_idx]
x = np.arange(len(y_val))

# Create figure
fig = go.Figure()

# Add shaded area for prediction interval
fig.add_trace(go.Scatter(
    x=x, y=high_sorted,
    line=dict(color='rgba(255,0,0,0)'),  # Invisible line
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=x, y=low_sorted,
    fill='tonexty',  # Fill between this trace and the previous
    fillcolor='rgba(128,128,128,0.2)',
    line=dict(color='rgba(0,0,255,0)'),
    name='Prediction Interval (90%)',
    hoverinfo='skip'
))

# Add lower bound
fig.add_trace(go.Scatter(
    x=x, y=low_sorted,
    line=dict(color='blue', dash='dash', width=0.5),
    name='Lower Bound (5%)'
))

# Add upper bound
fig.add_trace(go.Scatter(
    x=x, y=high_sorted,
    line=dict(color='red', dash='dash', width=0.5),
    name='Upper Bound (95%)'
))

# Add actual values
fig.add_trace(go.Scatter(
    x=x, y=y_sorted,
    line=dict(color='black', width=3),
    name='Actual'
))

# Customize layout
fig.update_layout(
    title='Prediction Intervals with Actual Values',
    xaxis_title='Sample Index (sorted by actual values)',
    yaxis_title='Target Value',
    legend=dict(x=0.01, y=0.99),
    template='simple_white',
    height=500,
    width=900
)

fig.show(renderer='iframe')


x_test = df_test.drop(columns="id")

low = np.maximum(0,model_low_cat.predict(x_test))
high = np.maximum(0,model_high_cat.predict(x_test))

out = pd.DataFrame({
    "id": df_test["id"],
    "pi_lower": low,
    "pi_upper": high
})

out.head()


out.describe().T


out.to_csv("submission.csv", index= False)

