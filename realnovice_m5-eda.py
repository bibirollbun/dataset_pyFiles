import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot



train_df= pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")
calendar_df = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")
price_df = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")
sample = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")
evaluation=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")


print("train data:{}".format(train_df.shape))
print("calendar data:{}".format(calendar_df.shape))
print("price data:{}".format(price_df.shape))
print("sample data:{}".format(sample.shape))


calendar_df["date_dt"] = pd.to_datetime(calendar_df["date"])


train  = train_df.copy()
price = price_df.copy()
calendar = calendar_df.copy()



print("Whole data avarage:{}".format(price["sell_price"].mean()))
print("Whole data standard deviation:{}".format(price["sell_price"].std()))
from scipy.stats import skew, kurtosis
data_skewness = skew(price["sell_price"])
print("Whole data avarage Skewness: {:.2f}".format(data_skewness))
data_kurtosis = kurtosis(price["sell_price"])
print("Whole data Kurtosis: {:.2f}".format(data_kurtosis))

plt.figure(figsize=(10,6))
sns.distplot(price["sell_price"])
plt.title("Price data distribution of whole data")
plt.ylabel("Frequency");


price["log_sell_price"] = np.log1p(price["sell_price"])

print("Log(1 + sell_price) data average: {}".format(price["log_sell_price"].mean()))
print("Log(1 + sell_price) data standard deviation: {}".format(price["log_sell_price"].std()))

data_skewness_log = skew(price["log_sell_price"])
print("Log(1 + sell_price) data skewness: {:.2f}".format(data_skewness_log))
data_kurtosis_log = kurtosis(price["log_sell_price"])
print("Log(1 + sell_price) data kurtosis: {:.2f}".format(data_kurtosis_log))

plt.figure(figsize=(10, 6))
sns.distplot(price["log_sell_price"], kde=True)
plt.title("Log(1 + sell_price) data distribution")
plt.ylabel("Frequency")
plt.show()


store_ca = price[(price["store_id"]=='CA_1') | (price["store_id"]=='CA_2') | (price["store_id"]=='CA_3') | (price["store_id"]=='CA_4')]
store_tx = price[(price["store_id"]=='TX_1') | (price["store_id"]=='TX_2') | (price["store_id"]=='TX_3')]
store_wi = price[(price["store_id"]=='WI_1') | (price["store_id"]=='WI_2') | (price["store_id"]=='WI_3')]

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
store_df = [store_ca, store_tx, store_wi]

for i in range(len(store_df)):
    sns.boxplot(x="store_id", y="sell_price", data=store_df[i], ax=ax[i])
    ax[i].set_ylabel("Price")


store_ca['log1p_sell_price'] = np.log1p(store_ca['sell_price'])  # log(1 + sell_price)
store_tx['log1p_sell_price'] = np.log1p(store_tx['sell_price'])  # log(1 + sell_price)
store_wi['log1p_sell_price'] = np.log1p(store_wi['sell_price'])  # log(1 + sell_price)

# Create subplots
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
store_df = [store_ca, store_tx, store_wi]
store_ids = ['CA', 'TX', 'WI']

# Plot boxplots for log(1 + sell_price)
for i in range(len(store_df)):
    sns.boxplot(x="store_id", y="log1p_sell_price", data=store_df[i], ax=ax[i])
    ax[i].set_ylabel("Log(1 + Price)")
    ax[i].set_title(f"Log(1 + Sell Price) for {store_ids[i]} stores")

# Display the plot
plt.tight_layout()
plt.show()


calendar[['year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']].groupby("year").count()


calendar[['year', 'snap_CA', 'snap_TX', 'snap_WI']].groupby("year").sum()





state_group = train.groupby("state_id").sum()


state_group =state_group.drop(columns=['id', 'item_id', 'dept_id', 'cat_id', 'store_id'])
state_group =state_group.T
state_group = pd.merge(state_group, calendar, left_index=True, right_on="d", how="left").set_index("date_dt")

# Visualization
fig, ax = plt.subplots(3,1, figsize=(15,15))
plt.subplots_adjust(hspace=0.4)

color=["yellow", "green", "blue"]
state_col = train["state_id"].unique()

for i in range(len(state_col)):
    ax[i].plot(state_group.index, state_group[state_col[i]], color=color[i], linewidth=0.5)
    # Rolling
    ax[i].plot(state_group.index, state_group[state_col[i]].rolling(28).mean(), color='black', linewidth=2)
    ax[i].set_xlabel("datetime")
    ax[i].set_ylabel("Sales volume")
    ax[i].legend(["{}".format(state_col[i]), "Rolling 28 days"])
    ax[i].set_title("{}".format(state_col[i]))


def plot_ts_decomp(data, col, lag, color):
    print("Analised Data:{}".format(col.upper()))
    # Stats model
    res = sm.tsa.seasonal_decompose(data[col], period=lag)
    data["trend"] = res.trend
    data["seaso"] = res.seasonal
    data["resid"] = res.resid
    
    # Visualization
    fig = plt.figure(figsize=(20,15))
    grid = plt.GridSpec(4,2, hspace=0.4, wspace=0.2)
    ax1 = fig.add_subplot(grid[0,0])
    ax2 = fig.add_subplot(grid[1,0])
    ax3 = fig.add_subplot(grid[2,0])
    ax4 = fig.add_subplot(grid[3,0])
    ax5 = fig.add_subplot(grid[:-2,1])
    ax6 = fig.add_subplot(grid[2:,1])
    
    # raw price data
    ax1.plot(data.index, data[col], label="price of {}".format(col), color=color, linewidth=0.5)
    ax1.plot(data.index, data[col].rolling(lag//12).mean(), label="Rolling {}".format(lag//12), color=color, linewidth=2)
    ax1.set_xlabel("date")
    ax1.set_ylabel("price")
    ax1.set_title("raw data")
    ax1.legend()
    # trend
    ax2.plot(data.index, data["trend"], label="trend of {}".format(col), color=color, linewidth=3)
    ax2.set_xlabel("date")
    ax2.set_ylabel("trend")
    ax2.set_title("trend")
    ax2.legend()
    # seasonaly
    ax3.plot(data.index, data["seaso"], label="seasonaly of {}".format(col), color=color, linewidth=0.5)
    ax3.set_xlabel("date")
    ax3.set_ylabel("seasonaly")
    ax3.set_title("seasonaly")
    ax3.legend()
    # residual
    ax4.plot(data.index, data["resid"], label="residual error of {}".format(col), color=color, linewidth=0.5)
    ax4.set_xlabel("date")
    ax4.set_ylabel("residual error")
    ax4.set_title("residual")
    ax4.legend()
    # distribution
    sns.distplot(data[col], ax=ax5)
    
    ax5.set_ylabel("Frequency")
    ax5.set_title("distribution")
    # auto correlation
    autocorrelation_plot(data[col], ax=ax6, linewidth=0.5)
    ax6.set_title("autocorrelation")


plot_ts_decomp(state_group, "CA", 365, "aqua");


plot_ts_decomp(state_group, "TX", 365, "aqua");


plot_ts_decomp(state_group, "WI", 365, "aqua");

