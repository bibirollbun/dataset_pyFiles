import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

plt.style.use("ggplot")
rcParams["figure.figsize"] = [12, 9]
rcParams["xtick.labelsize"] = 15
rcParams["ytick.labelsize"] = 15
rcParams["font.size"] = 15

warnings.filterwarnings("ignore")


tps = pd.read_csv(
    "../input/tabular-playground-series-jul-2021/train.csv",
    parse_dates=["date_time"], index_col="date_time"
)
print(tps.shape)
tps.head()



tps["deg_C"].plot(figsize=(16, 6), fontsize=15)
plt.xlabel("Date")
plt.ylabel("Degrees Celcius");


tps.index.min(),tps.index.max()


# Load meat production dataset in the USA
meat = pd.read_csv(
    "https://s3.amazonaws.com/assets.datacamp.com/production/course_5226/datasets/ch4_meat.csv", 
    parse_dates=["date"], index_col="date")
meat.head()


meat.to_csv('ch4_meat.csv')


meat.index.min(),meat.index.max()


meat.plot(figsize=(16, 10))
plt.legend(fontsize="large")
plt.xlabel("Date")
plt.ylabel("Meat Production (t)");


import statsmodels.api as sm
from matplotlib import rcParams

decomposition = sm.tsa.seasonal_decompose(meat["beef"])

rcParams["figure.figsize"] = 16, 4
decomposition.seasonal.plot();


print(decomposition.seasonal.shape)
decomposition.seasonal


decomposition.seasonal["1999":"2005"].plot();


# Init empty dict to store decompose results
seasonality_dict = {}

for ts in meat.columns:
    decompositions = sm.tsa.seasonal_decompose(meat[ts].dropna())
    # Store the results back
    seasonality_dict[ts] = decompositions.seasonal

# Plot
rcParams["figure.figsize"] = 16, 14
pd.DataFrame(seasonality_dict)["2000":"2005"].plot(
    subplots=True, layout=(4, 2), linewidth=3
);


trend_dict = {}

for ts in meat.columns:
    decomposition = sm.tsa.seasonal_decompose(meat[ts].dropna())
    # Store back the results
    trend_dict[ts] = decomposition.trend
    print('*'*100)
    print(trend_dict[ts].dropna())

pd.DataFrame(trend_dict).plot(subplots=True, layout=(4, 3), linewidth=3);


print(trend_dict['beef'].shape)
trend_dict['beef'].dropna()


resid_dict = {}

for ts in meat.columns:
    decomposition = sm.tsa.seasonal_decompose(meat[ts].dropna())
    resid_dict[ts] = decomposition.resid

pd.DataFrame(resid_dict).plot(subplots=True, layout=(3, 3), linewidth=1);


print(resid_dict['beef'].shape)
resid_dict['beef'].dropna()


meat.plot(figsize=(16, 8), title="Meat Production in Tons in the USA")
plt.xlabel("Date");


meat.head()


meat.iloc[0]


meat.div(meat.iloc[0]).head()


normalized = meat.div(meat.iloc[0]).mul(100)  # Multiply by 100 to get percentages

normalized.plot(figsize=(16, 8), title="Growth of Production in Different Meat Sectors")
plt.xlabel("Date")
plt.ylabel("Increase (%)");


### 原始序列不同特征之间是如何影响的


# Create a custom palette
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)

# Compute corr matrix
matrix = meat.corr(method="pearson")
# Create a mask
mask = np.triu(np.ones_like(matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(matrix, mask=mask, cmap=cmap, square=True, annot=True, fmt=".2f", ax=ax)
plt.show();


seasonality_dict = {
    ts: sm.tsa.seasonal_decompose(meat[ts].dropna()).seasonal for ts in meat.columns
}

# Compute corr matrix
seasonality_corr = pd.DataFrame(seasonality_dict).corr()

sns.clustermap(seasonality_corr, annot=True, square=True)
plt.show();


trend_dict = {
    ts: sm.tsa.seasonal_decompose(meat[ts].dropna()).trend for ts in meat.columns
}

# Compute corr matrix
trend_dict = pd.DataFrame(trend_dict).corr()

sns.clustermap(trend_dict, annot=True, square=True)
plt.show();


normalized = meat.div(meat.iloc[0]).mul(100)

normalized.plot(figsize=(16, 8), title="Growth of Production in Different Meat Sectors")
plt.xlabel("Date")
plt.ylabel("Increase (%)");


from statsmodels.graphics import tsaplots

rcParams["figure.figsize"] = 10, 6

# Stands for Time Series Analysis Plots (TSA Plots)  代表时间序列分析图（TSA图）
fig = tsaplots.plot_acf(tps["deg_C"], lags=60)

plt.xlabel("Lag at k")
plt.ylabel("Correlation coefficient") # 相关系数
plt.show()


meat["beef"].head()


beef = meat["beef"].to_frame(name="beef")

beef["lag_1"] = beef["beef"].shift(periods=1)
beef["lag_2"] = beef["beef"].shift(periods=2)
beef["lag_3"] = beef["beef"].shift(periods=3)

beef.head()


tps.head()


fig = tsaplots.plot_acf(tps["target_carbon_monoxide"], lags=70)

plt.title("Autocorrelation of Carbon Monoxide")
plt.xlabel("Lag at k")
plt.ylabel("Correlation Coefficient")
plt.show();


tps.head()


# Mind the extra 'p' in plot_pacf
fig = tsaplots.plot_pacf(tps["target_benzene"], lags=70)

plt.title("Partial Autocorrelation of Benzene")
plt.xlabel("Lag at k")
plt.ylabel("Correlation Coefficient")
plt.show();

