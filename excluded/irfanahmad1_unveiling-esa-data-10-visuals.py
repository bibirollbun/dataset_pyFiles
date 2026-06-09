# Basic & advanced plotting libraries
!pip install --quiet plotly==5.20.0 seaborn==0.13.2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook")
pd.set_option('display.float_format', '{:.4f}'.format)



train_data_df = pd.read_csv(
    "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv",
    index_col='id'
).astype(np.float32)
channels = ['channel_44', 'channel_45', 'channel_46']
train_data_df.head()



print(f"Shape: {train_data_df.shape}")
print(f"Columns: {list(train_data_df.columns)}")
print(f"Index (time): min={train_data_df.index.min()}, max={train_data_df.index.max()}")

display(train_data_df.describe().T)



fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
for i, ch in enumerate(channels):
    axs[i].plot(train_data_df[ch].iloc[:1000], lw=1.5)
    axs[i].set_ylabel(ch)
    axs[i].set_title(f"{ch} (first 1000 samples)")
axs[2].set_xlabel("Time (id)")
plt.suptitle("First 1000 Samples for Each Channel", fontsize=18)
plt.tight_layout()
plt.show()



fig = go.Figure()
for ch in channels:
    fig.add_trace(go.Scatter(
        y=train_data_df[ch].iloc[:2000], 
        mode='lines',
        name=ch
    ))
fig.update_layout(title="Interactive Plot: First 2000 Timesteps", 
                  xaxis_title="Time (id)", yaxis_title="Value",
                  height=500)
fig.show()



corr = train_data_df[channels].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Channel Correlation Heatmap")
plt.show()

fig = px.imshow(corr, text_auto=True, aspect="auto", title="Interactive Channel Correlation Matrix")
fig.show()



fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, ch in enumerate(channels):
    sns.violinplot(y=train_data_df[ch], ax=axs[i], color="skyblue")
    axs[i].set_title(f"Violin: {ch}")
plt.suptitle("Channel Distributions (Violin Plots)", fontsize=16)
plt.show()

plt.figure(figsize=(12, 4))
sns.boxplot(data=train_data_df[channels])
plt.title("Boxplot: Channel Values")
plt.show()

train_data_df[channels].plot(kind='hist', bins=50, alpha=0.7, figsize=(14,5), legend=True)
plt.title("Histogram of All Channel Values")
plt.xlabel("Value")
plt.show()



roll_mean = train_data_df[channels].rolling(200).mean()
roll_std = train_data_df[channels].rolling(200).std()

fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
for i, ch in enumerate(channels):
    axs[i].plot(train_data_df[ch][:3000], label='Raw', alpha=0.3)
    axs[i].plot(roll_mean[ch][:3000], label='Rolling Mean', lw=2)
    axs[i].fill_between(
        roll_mean[ch][:3000].index, 
        roll_mean[ch][:3000] - roll_std[ch][:3000], 
        roll_mean[ch][:3000] + roll_std[ch][:3000], alpha=0.2, color='orange', label='Mean Â± Std'
    )
    axs[i].set_ylabel(ch)
    axs[i].legend()
axs[2].set_xlabel("Time (id)")
plt.suptitle("Rolling Mean Â± Std (Window=200)", fontsize=16)
plt.tight_layout()
plt.show()



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axs = plt.subplots(3,2, figsize=(14, 12))
for i, ch in enumerate(channels):
    plot_acf(train_data_df[ch].iloc[:2000], ax=axs[i,0], lags=60, title=f"{ch} ACF")
    plot_pacf(train_data_df[ch].iloc[:2000], ax=axs[i,1], lags=60, title=f"{ch} PACF")
plt.tight_layout()
plt.show()



def crosscorr(a, b, lag=0):
    return a.corr(b.shift(lag))

max_lag = 50
for i in range(3):
    for j in range(i+1, 3):
        corr_lags = [crosscorr(train_data_df[channels[i]], train_data_df[channels[j]], lag) for lag in range(-max_lag, max_lag)]
        plt.plot(range(-max_lag, max_lag), corr_lags, label=f'{channels[i]} vs {channels[j]}')
plt.axhline(0, color='k', lw=1)
plt.legend()
plt.title("Channel Cross-Correlation by Lag")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.show()


