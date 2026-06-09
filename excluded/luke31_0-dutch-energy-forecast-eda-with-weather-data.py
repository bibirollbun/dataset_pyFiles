# 1 description of train/test data including number of negative values
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates;
import altair as alt
alt.data_transformers.enable("vegafusion")

# File paths
train_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv'
test_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/test_new.csv'

# Load data
train_df = pl.read_csv(train_path, schema_overrides={"timestamp_utc": pl.Datetime("us")} )
test_df = pl.read_csv(test_path, schema_overrides={"timestamp_utc": pl.Datetime("us")} )
print('train')
print(train_df.describe())
print(train_df.head())
print('\ntest')
print(test_df.describe())
num_neg_values = (train_df['net_load_kwh'] < 0).sum()
num_total = len(train_df)
print(f'Negative values: {(num_neg_values / num_total)*100:.2f}% {num_neg_values} / {num_total}')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df.to_pandas().plot(x="timestamp_utc", y="net_load_kwh", ax=axes[0], 
                          title=f"Train data {train_df['timestamp_utc'].min()} - {train_df['timestamp_utc'].max()}")
axes[0].set_ylim(-650, 650)

plot_train_df = train_df
plot_train_df = plot_train_df.with_columns(
    (pl.col("timestamp_utc").cast(pl.Int64) / 1e9).alias("timestamp_numeric")
)
sns.regplot(data=plot_train_df, x="timestamp_numeric", y="net_load_kwh", scatter_kws={'s':1}, ax=axes[1])
axes[1].set_title("Train data linear Trend")
axes[1].set_ylim(-650, 650)
plt.show();


# 2 Visualization of Time series sample, distribution, daily and weekly patterns (Taken from https://www.kaggle.com/code/taylorsamarel/starter-notebook-dutch-energy-forecasting?scriptVersionId=264091608 V5)
# Simple visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time series sample
axes[0, 0].plot(train_df['timestamp_utc'][:672], train_df['net_load_kwh'][:672], alpha=0.7)
axes[0, 0].set_title('Load Time Series (1 week sample)')
axes[0, 0].set_ylabel('Net Load (kWh/15min)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Distribution
axes[0, 1].hist(train_df['net_load_kwh'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Zero')
axes[0, 1].set_title('Load Distribution')
axes[0, 1].set_xlabel('Net Load (kWh/15min)')
axes[0, 1].legend()

# --- Daily pattern ---
train_df_plt = train_df.with_columns([
    (train_df["timestamp_utc"].dt.hour() + train_df["timestamp_utc"].dt.minute() / 60).alias("hour"),
    train_df["timestamp_utc"].dt.weekday().alias("dow")   # 1=Mon ... 7=Sun
])

def plot_daily_stats(ax, train_df_plt, title, legend=True, show_ylabel=True):
    daily_stats = (
        train_df_plt
        .group_by("hour")
        .agg([
            pl.col("net_load_kwh").mean().alias("avg_net_load"),
            pl.col("net_load_kwh").min().alias("min_net_load"),
            pl.col("net_load_kwh").max().alias("max_net_load"),
        ])
        .sort("hour")
    )
    ax.plot(
        daily_stats["hour"].to_numpy(),
        daily_stats["avg_net_load"].to_numpy(),
        marker="o",
        markersize=3
    )
    ax.plot(
        daily_stats["hour"].to_numpy(),
        daily_stats["min_net_load"].to_numpy(),
        linestyle=":",
        color="orange",
        label="Min Net Load"
    )
    ax.plot(
        daily_stats["hour"].to_numpy(),
        daily_stats["max_net_load"].to_numpy(),
        linestyle=":",
        color="red",
        label="Max Net Load"
    )
    ax.set_title(title)
    ax.set_xlabel("Hour of Day")
    if show_ylabel:
        ax.set_ylabel("Net Load (kWh)")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-650, 650)
    if legend:
        ax.legend()

plot_daily_stats(axes[1, 0], train_df_plt, title="Daily Net Load Pattern")

# --- Weekly pattern ---
weekly = train_df_plt.group_by("dow").agg(pl.col("net_load_kwh").mean().alias("avg_net_load")).sort("dow")

axes[1, 1].bar(
    weekly["dow"].to_numpy(),
    weekly["avg_net_load"].to_numpy(),
    color="skyblue",
    edgecolor="navy"
)
axes[1, 1].set_title("Average Weekly Pattern")
axes[1, 1].set_xticks(range(1,8))
weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
axes[1, 1].set_xticklabels(weekdays)
axes[1, 1].set_ylabel("Avg Net Load")

plt.tight_layout()
plt.show()

# daily pattern for each weekday
fig_n, axes_n = plt.subplots(1, 7, figsize=(12, 4))
for i in range(1,8):
    plot_daily_stats(
        axes_n[i-1], 
        train_df_plt.filter(train_df_plt['dow'] == i), 
        title=weekdays[i-1], 
        legend=False,
        show_ylabel=(i == 1)
    )

fig_n.suptitle("Daily Net Load Pattern by Weekday", fontsize=14)
plt.tight_layout()
plt.show()


# Fetching of weather data to compare to load data
import requests
from datetime import timedelta, datetime

print("\n3. FETCHING WEATHER DATA")
print("-" * 40)

def fetch_weather_data(start_date, end_date):
    """
    Fetch historical weather data from Open-Meteo API
    for major Dutch cities, returning a Polars DataFrame.
    """
    locations = [
        (52.3676, 4.9041, 'Amsterdam'),
        (51.9244, 4.4777, 'Rotterdam'),
        (52.0907, 5.1214, 'Utrecht'),
        (51.4416, 5.4697, 'Eindhoven'),
        (53.2194, 6.5665, 'Groningen')
    ]
    
    weather_features = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'apparent_temperature', 'precipitation', 'rain', 'pressure_msl',
        'surface_pressure', 'cloud_cover', 'wind_speed_10m', 'wind_direction_10m',
        'wind_gusts_10m', 'direct_radiation', 'diffuse_radiation', 'global_tilted_irradiance'
    ]
    
    all_weather = []
    
    for lat, lon, city in locations:
        print(f"  Fetching weather for {city}...")
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ','.join(weather_features),
            'timezone': 'UTC'
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Convert to Polars DataFrame
        weather_df = pl.DataFrame(data['hourly'])
        weather_df = weather_df.with_columns([
            pl.col('time').str.strptime(pl.Datetime).alias('timestamp_utc')
        ]).drop('time')
        
        # Sort by timestamp
        weather_df = weather_df.sort('timestamp_utc')
        
        # Generate 15-min timestamps using NumPy
        ts_min = weather_df['timestamp_utc'].min()
        ts_max = weather_df['timestamp_utc'].max()
        n_intervals = int((ts_max - ts_min).total_seconds() // (15*60)) + 1
        new_idx = pl.DataFrame({
            'timestamp_utc': pl.Series(
                "timestamp_utc",
                [ts_min + timedelta(minutes=15*i) for i in range(n_intervals)],
                dtype=pl.Datetime
            )
        })
        
        # Join and interpolate
        weather_df = new_idx.join(weather_df, on='timestamp_utc', how='left')
        weather_df = weather_df.interpolate()
        
        # Add city prefix
        weather_df = weather_df.rename({c: f"{city}_{c}" for c in weather_df.columns if c != 'timestamp_utc'})
        all_weather.append(weather_df)
    if all_weather:
        # Combine all cities
        combined = all_weather[0]
        for df in all_weather[1:]:
            combined = combined.join(df, on="timestamp_utc", how="full").sort("timestamp_utc")
            combined = combined.drop("timestamp_utc_right")
        
        # Add averages across cities
        for feature in weather_features:
            city_cols = [col for col in combined.columns if feature in col]
            if city_cols:
                combined = combined.with_columns([
                    pl.concat_list(city_cols).list.mean().alias(f'avg_{feature}'),
                    pl.concat_list(city_cols).list.max().alias(f'max_{feature}'),
                    pl.concat_list(city_cols).list.min().alias(f'min_{feature}')
                ])
        
        print(f"  Weather data shape: {combined.shape}")
        return combined
    
    return pl.DataFrame()


# Fetch weather for the entire period
start = (train_df['timestamp_utc'].min()).strftime('%Y-%m-%d')
end = test_df['timestamp_utc'].max().strftime('%Y-%m-%d')
weather_df = fetch_weather_data(start, end)

all_data = train_df_plt.join(weather_df, on='timestamp_utc', how="left").sort("timestamp_utc")


# Columns with avg weather features + target
avg_weather_features = [
    "avg_temperature_2m", "avg_relative_humidity_2m", "avg_dew_point_2m",
    "avg_apparent_temperature", "avg_precipitation", "avg_rain", "avg_pressure_msl",
    "avg_surface_pressure", "avg_cloud_cover", "avg_wind_speed_10m", "avg_wind_direction_10m",
    "avg_wind_gusts_10m", "avg_direct_radiation", "avg_diffuse_radiation", "avg_global_tilted_irradiance"
]
columns_to_use = ["net_load_kwh"] + avg_weather_features

# Convert to Pandas
df = all_data.to_pandas()[columns_to_use]

# Compute correlations with target
correlations = df.corr()["net_load_kwh"]

# Convert to table and exclude target itself
top_corr_table = pd.DataFrame({
    "feature": correlations.index,
    "correlation_with_net_load": correlations.values
}).reset_index(drop=True)
top_corr_table = top_corr_table[top_corr_table["feature"] != "net_load_kwh"]

# Sort by absolute correlation
top_corr_table = top_corr_table.reindex(
    top_corr_table["correlation_with_net_load"].abs().sort_values(ascending=False).index
).reset_index(drop=True)

top_corr_table


# 4. Plot pairwise relationships in a dataset of top 5 correlations
# Weather features (avg versions already exist in your dataframe)
plot_features = top_corr_table["feature"][0:5].to_list()

# Columns to plot
columns_to_plot = ["net_load_kwh"] + plot_features

# Create the pairplot
sns.pairplot(
    all_data.to_pandas()[columns_to_plot],
    kind="reg",
    diag_kind="kde",
    corner=True,
    plot_kws={"scatter_kws": {"alpha": 0.1}},
)
plt.show()


# 5. Plot weather data (average temperature and direct sun radiation of multiple cities) with load data

# Define 1-week period
start_date = pl.datetime(2025, 5, 8)  # adjust as needed
end_date = start_date + pl.duration(days=7)

# Filter using Polars
df_week = all_data.filter(
    (pl.col('timestamp_utc') >= start_date) & 
    (pl.col('timestamp_utc') < end_date)
)

# Convert to pandas for plotting
df_plot = df_week.select(['timestamp_utc','net_load_kwh','avg_temperature_2m','avg_direct_radiation']).to_pandas()

fig, axes = plt.subplots(2, 1, figsize=(15,10), sharex=True)

# --- Subplot 1: Temperature vs Net Load ---
ax1 = axes[0]
ax1.plot(df_plot['timestamp_utc'], df_plot['net_load_kwh'], color='tab:blue', label='Net Load (kWh)')
ax1.set_ylabel('Net Load (kWh)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax1b = ax1.twinx()
temperature_scaled = df_plot['avg_temperature_2m'] * (df_plot['net_load_kwh'].max() / df_plot['avg_temperature_2m'].max())
ax1b.plot(df_plot['timestamp_utc'], temperature_scaled, color='tab:red', label='Temperature (°C, scaled, avg)')
ax1b.set_ylabel('Scaled avg temperature', color='tab:red')
ax1b.tick_params(axis='y', labelcolor='tab:red')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1b.legend(lines + lines2, labels + labels2, loc='upper right')
ax1.set_title('Net Load and avg temperature (1 Week)')

# --- Subplot 2: Radiation vs Net Load ---
ax2 = axes[1]
ax2.plot(df_plot['timestamp_utc'], df_plot['net_load_kwh'], color='tab:blue', label='Net Load (kWh)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Net Load (kWh)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

ax2b = ax2.twinx()
radiation_scaled = df_plot['avg_direct_radiation'] * (df_plot['net_load_kwh'].max() / df_plot['avg_direct_radiation'].max())
ax2b.plot(df_plot['timestamp_utc'], radiation_scaled, color='tab:orange', label='Direct Radiation (W/m², scaled, avg)')
ax2b.set_ylabel('Scaled avg radiation', color='tab:orange')
ax2b.tick_params(axis='y', labelcolor='tab:orange')

lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2b.legend(lines + lines2, labels + labels2, loc='upper right')
ax2.set_title('Net Load and avg direct Radiation (1 Week)')

plt.tight_layout()
plt.show()



# 6. Plot daily net load and weather data
# Make sure 'hour' column exists
train_df_plt = all_data.with_columns([
    pl.col("timestamp_utc").dt.hour().alias("hour")
])

def plot_netload_with_feature(ax, df, feature, feature_name, color='tab:red'):
    """
    Plots Net Load (left y-axis) and a weather feature (right y-axis) by hour of day.
    """
    daily_stats = (
        df
        .group_by("hour")
        .agg([
            pl.col("net_load_kwh").mean().alias("avg_load"),
            pl.col("net_load_kwh").min().alias("min_load"),
            pl.col("net_load_kwh").max().alias("max_load"),
            pl.col(feature).mean().alias("avg_feature"),
            pl.col(feature).min().alias("min_feature"),
            pl.col(feature).max().alias("max_feature")
        ])
        .sort("hour")
    )

    hours = daily_stats["hour"].to_numpy()

    # Left axis: Net Load
    ax.plot(hours, daily_stats["avg_load"].to_numpy(), label="Avg Net Load", color='tab:blue')
    ax.plot(hours, daily_stats["min_load"].to_numpy(), linestyle=":", color="orange", label="Min Net Load")
    ax.plot(hours, daily_stats["max_load"].to_numpy(), linestyle=":", color="red", label="Max Net Load")
    ax.set_xlabel("Hour of Day")
    ax.set_xticks(range(0, 24))  # show all hours
    ax.set_ylabel("Net Load (kWh)", color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Net Load and {feature_name} by Hour")
    ax.legend(loc='upper left')

    # Right axis: Weather feature (scaled to Net Load range)
    ax2 = ax.twinx()
    ax2.set_xticks(range(0, 24))  # show all hours
    scale_factor = daily_stats["avg_load"].max() / daily_stats["avg_feature"].max()
    ax2.plot(hours, daily_stats["avg_feature"].to_numpy() * scale_factor, color=color, label=f"Avg {feature_name}")
    ax2.plot(hours, daily_stats["min_feature"].to_numpy() * scale_factor, linestyle=":", color=color, alpha=0.5, label=f"Min {feature_name}")
    ax2.plot(hours, daily_stats["max_feature"].to_numpy() * scale_factor, linestyle=":", color=color, alpha=0.5, label=f"Max {feature_name}")
    ax2.set_ylabel(f"{feature_name} (scaled)", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

# --- Create figure with 2 plots ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

plot_netload_with_feature(axes[0], train_df_plt, feature="avg_temperature_2m", feature_name="Temperature", color='tab:red')
plot_netload_with_feature(axes[1], train_df_plt, feature="avg_direct_radiation", feature_name="Radiation", color='tab:orange')

plt.tight_layout()
plt.show()


