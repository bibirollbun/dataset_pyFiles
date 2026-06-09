# Run first to install all pre-requisites 

%pip install --upgrade bigframes google-cloud-automl google-cloud-translate google-ai-generativelanguage tensorflow 


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)


project_id = 'big-query-project-472510' #@param{type:"string"}
import bigframes.pandas as bpd
bpd.options.bigquery.ordering_mode = "partial" # Optional: partial ordering mode can accelerate executions and save costs

import bigframes.exceptions
import warnings
warnings.filterwarnings("ignore", category=bigframes.exceptions.AmbiguousWindowWarning)


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

import warnings
# Suppress the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


# Set project for bigframes (fixes the error you saw)
# close session before starting another bigframes.pandas.close_session()
bigframes.options.bigquery.project = f"{project_id}" 


colnames = bpd.read_gbq("""SELECT column_name, data_type
FROM `big-query-project-472510.weathernext_gen_forecasts.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = '126478713_1_0'""")

colnames


# Load table
df = bpd.read_gbq_table(
    "big-query-project-472510.weathernext_gen_forecasts.126478713_1_0"
)

# # Flatten the forecast array -- This COSTS MONEY
# df_flat = df.explode("forecast").sort_values(["init_time"])


# Partition pruning: only load init_time = 2024-10-17

df = bpd.read_gbq_table(
    "big-query-project-472510.weathernext_gen_forecasts.126478713_1_0",
    filters=[
        ("init_time", ">=", "2024-10-17 00:00:00+00"),
        ("init_time", "<",  "2024-10-18 00:00:00+00"),
    ],
)



sql = """
SELECT column_name, data_type
FROM `big-query-project-472510.weathernext_gen_forecasts.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = '126478713_1_0'
"""
df = bpd.read_gbq(sql)


# UNNEST -- FLATTEN the ensemble and select for only 17th Oct 2024

sql = """
SELECT
  t1.init_time,
  f.time,
  e.2m_temperature
FROM `big-query-project-472510.weathernext_gen_forecasts.126478713_1_0` AS t1
CROSS JOIN UNNEST(t1.forecast) AS f
CROSS JOIN UNNEST(f.ensemble) AS e
WHERE t1.init_time = TIMESTAMP('2024-10-17 00:00:00 UTC')
ORDER BY f.time
"""

df = bpd.read_gbq(sql)


# Make an intersect for New York

sql = """
SELECT
  t1.init_time,
  f.time,
  e.2m_temperature
FROM `big-query-project-472510.weathernext_gen_forecasts.126478713_1_0` AS t1
CROSS JOIN UNNEST(t1.forecast) AS f
CROSS JOIN UNNEST(f.ensemble) AS e
WHERE t1.init_time >= TIMESTAMP('2024-10-17 00:00:00 UTC')
  AND t1.init_time <  TIMESTAMP('2024-10-18 00:00:00 UTC')
  AND ST_INTERSECTS(
        t1.geography_polygon,
        ST_GEOGFROMTEXT('POLYGON((-70.66 40.64, -73.85 40.64, -73.85 40.89, -70.66 40.89, -70.66 40.64))')
      )
ORDER BY f.time
"""

df = bpd.read_gbq(sql)


df


# Make an intersect for New York

sql = """
SELECT
  t1.init_time,
  f.time,
  e.2m_temperature
FROM `big-query-project-472510.weathernext_gen_forecasts.126478713_1_0` AS t1
CROSS JOIN UNNEST(t1.forecast) AS f
CROSS JOIN UNNEST(f.ensemble) AS e
WHERE t1.init_time = TIMESTAMP('2024-10-17 00:00:00 UTC')
  AND ST_INTERSECTS(
        t1.geography_polygon,
        ST_GEOGFROMTEXT('POLYGON((-70.66 40.64, -73.85 40.64, -73.85 40.89, -70.66 40.89, -70.66 40.64))')
      )
ORDER BY f.time
"""

ny_temps = bpd.read_gbq(sql)
ny_temps


# # Extract the date from the first forecast_time
# forecast_date = ny_temps['time'].iloc[0].strftime('%Y-%m-%d')
# forecast_date


ny_temps_pd = ny_temps.to_pandas()  

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")

# Plot the data
plt.figure(figsize=(10, 6))
sns.lineplot(x=mdates.date2num(ny_temps_pd['time'].dt.to_pydatetime()), y=ny_temps_pd['2m_temperature'], marker='o', linestyle='-', color='lightcoral')
plt.xlabel('Forecast Time', fontsize=12)
plt.ylabel('2m Temperature (K)', fontsize=12)
plt.title(f'Temperature Forecast for New York for 2024-10-17', fontsize=14)

# Format the x-axis ticks
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Show date and time
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Show ticks for each day
plt.xticks(rotation=45, ha='right', fontsize=10)

plt.tight_layout()
plt.show()


sql = """
SELECT
    t2.time,  
    t2.total_precipitation_6hr
  FROM
    `gcp-public-data-weathernext.WeatherNext.59572747_4_0` AS t1, t1.forecast AS t2
  WHERE ST_CONTAINS(t1.geography_polygon, ST_GEOGPOINT(-87.65, 41.85))
   AND t1.init_time = TIMESTAMP('2024-10-20 00:00:00 UTC')
"""

chicago_precip = bpd.read_gbq(sql)
chicago_precip_pd = chicago_precip.to_pandas()  


# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")

# Plot the data
plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
sns.lineplot(x='time', y='total_precipitation_6hr', data=chicago_precip_pd, marker='o', linestyle='-', color='skyblue', errorbar=None)
plt.xlabel('Forecast Time', fontsize=12)
plt.ylabel('Total Precipitation (mm)', fontsize=12)
plt.title('6-Hour Total Precipitation Forecast for Chicago', fontsize=14)

# Format the x-axis ticks for better readability
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

plt.xticks(rotation=45, ha='right', fontsize=10)

plt.tight_layout()
plt.show()


sql = """
SELECT
    t2.time,
    SQRT(POW(t2.`10m_u_component_of_wind`, 2) + POW(t2.`10m_v_component_of_wind`, 2)) AS wind_speed,
    t2.`2m_temperature` AS temperature
FROM
    `gcp-public-data-weathernext.WeatherNext.59572747_4_0` AS t1, t1.forecast AS t2
WHERE ST_INTERSECTS(t1.geography_polygon, ST_GEOGPOINT(-0.1278, 51.5074))  # London
  AND t1.init_time = TIMESTAMP('2024-10-17 00:00:00 UTC')
ORDER BY t2.time
"""

london_wspeed = bpd.read_gbq(sql)
london_wspeed_pd = london_wspeed.to_pandas()  


# Set the aesthetic style
sns.set_theme(style="white")

# Plot the data with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Wind speed
color = 'tab:blue'
sns.lineplot(x='time', y='wind_speed', data=london_wspeed_pd, marker='o', linestyle='-', color=color, ax=ax1)
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Wind Speed (m/s)', color=color, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color)

# Temperature
ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color = 'tab:red'
sns.lineplot(x='time', y='temperature', data=london_wspeed_pd, marker='x', linestyle='--', color=color, ax=ax2)
ax2.set_ylabel('Temperature (K)', color=color, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color)

# Format the x-axis ticks for better readability
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))

plt.xticks(rotation=45, ha='right', fontsize=10)

# Add gridlines
plt.grid(True)  # Add gridlines to the plot

plt.title('Wind Speed and Temperature in London for 2024-10-17 00:00:00', fontsize=14)
fig.tight_layout()
plt.show()


sql = """
SELECT
  f.`2m_temperature` - 273.15 AS `2m_temperature_celsius`,  -- Subtract 273.15 to convert from Kelvin to Celsius
  ST_X(ST_Centroid(t.geography_polygon)) AS longitude,
  ST_Y(ST_Centroid(t.geography_polygon)) AS latitude
FROM
  `gcp-public-data-weathernext.WeatherNext.59572747_4_0` AS t,
  t.forecast AS f
WHERE ST_CONTAINS(t.geography_polygon, t.geography)
  AND ST_CONTAINS((
    SELECT
      state_geom
    FROM
      `bigquery-public-data`.geo_us_boundaries.states
    WHERE state_name = 'Colorado'
  ), t.geography)
  AND t.init_time = TIMESTAMP('2024-10-14 00:00:00 UTC')
  AND f.time = TIMESTAMP('2024-10-24 00:00:00 UTC');  -- Filter by the specified time
"""

colorado_temp = bpd.read_gbq(sql)
colorado_temp_pd = colorado_temp.to_pandas()  


colorado_temp_pd


# convert points to an image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = colorado_temp_pd['longitude']
y = colorado_temp_pd['latitude']
z = colorado_temp_pd['2m_temperature_celsius']

ny = len(np.unique(y)) // 25
nx = len(np.unique(x)) // 25

print(nx, ny)

# Bin the data onto a regular grid
zi, yi, xi = np.histogram2d(y, x, bins=(nx, ny), weights=z)
counts, _, _ = np.histogram2d(y, x, bins=(nx, ny))

zi = zi / counts
zi = np.ma.masked_invalid(zi)


import folium
from folium.plugins import HeatMap
from branca.colormap import linear

# Generate colors for the image
cm = matplotlib.colormaps["Spectral"]
normed_data = (zi - zi.min()) / (zi.max() - zi.min())
z = cm(normed_data)

# Create a base map centered on Colorado
m = folium.Map(location=[39.0598, -105.5877], zoom_start=7)

# Add temperature image
folium.raster_layers.ImageOverlay(
    image=z,
    bounds=[[yi.min(), xi.min()], [yi.max(), xi.max()]],
    mercator_project=True,
    opacity=0.5,
    pixelated=False,
).add_to(m)

# Display the map
m

