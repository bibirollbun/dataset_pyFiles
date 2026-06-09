from google.cloud import bigquery
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Create BigQuery client
client = bigquery.Client()

# Query weather data for Hanoi
QUERY_WEATHER = """
SELECT
  DATE(CAST(year AS INT64), CAST(mo AS INT64), CAST(da AS INT64)) AS ds,
  temp AS y
FROM `bigquery-public-data.noaa_gsod.gsod*`
WHERE stn = '488200'
  AND _TABLE_SUFFIX BETWEEN '2020' AND '2022'
  AND temp IS NOT NULL AND temp != 9999.9
ORDER BY ds
"""

# ✅ Avoid Kaggle PermissionDenied by disabling BigQuery Storage API
raw = client.query(QUERY_WEATHER).to_dataframe(create_bqstorage_client=False)
print(f"✅ Retrieved {len(raw)} weather records")

# Ensure ds is datetime
raw["ds"] = pd.to_datetime(raw["ds"])

# Create a continuous date range covering the dataset
full_idx = pd.date_range(raw["ds"].min(), raw["ds"].max(), freq="D")

# Merge with the full date index
raw = pd.DataFrame({"ds": full_idx}).merge(raw, on="ds", how="left")

# Interpolate only the numeric 'y' column using time-based interpolation
raw = raw.set_index("ds")
raw["y"] = raw["y"].interpolate(method="time")
raw = raw.reset_index()

# ✅ Fit Prophet model
m = Prophet(daily_seasonality=True, yearly_seasonality=True)
m.fit(raw)

# Forecast next 30 days
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# Display last 10 forecasted days
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

# Plot forecast
m.plot(forecast)
plt.title("Hanoi Temperature Forecast")
plt.show()

# Optional: Plot forecast components
m.plot_components(forecast)
plt.show()



# !pip install google-cloud-bigquery

from google.cloud import bigquery
import pandas as pd
import os

# Authenticate via service account JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your-service-account.json"
client = bigquery.Client()


# Replace `my_dataset` with your dataset name
dataset_id = f"{client.project}.weather_ml"
dataset = bigquery.Dataset(dataset_id)
dataset.location = "US"

try:
    client.create_dataset(dataset)
    print(f"✅ Created dataset {dataset_id}")
except:
    print(f"⚠ Dataset {dataset_id} already exists")


train_query = """
CREATE OR REPLACE MODEL `weather_ml.hanoi_temp_model`
OPTIONS (
    model_type='linear_reg',
    input_label_cols=['y']
) AS
SELECT
  EXTRACT(DAYOFYEAR FROM DATE(CAST(year AS INT64), CAST(mo AS INT64), CAST(da AS INT64))) AS day_of_year,
  CAST(year AS INT64) AS year_num,
  temp AS y
FROM `bigquery-public-data.noaa_gsod.gsod*`
WHERE stn = '488200'
  AND _TABLE_SUFFIX BETWEEN '2020' AND '2022'
  AND temp IS NOT NULL AND temp != 9999.9
"""

client.query(train_query).result()
print("✅ Model trained in BigQuery")


```python
eval_query = """
SELECT
  *
FROM
  ML.EVALUATE(MODEL `weather_ml.hanoi_temp_model`,
    (
      SELECT
        EXTRACT(DAYOFYEAR FROM DATE(CAST(year AS INT64), CAST(mo AS INT64), CAST(da AS INT64))) AS day_of_year,
        CAST(year AS INT64) AS year_num,
        temp AS y
      FROM `bigquery-public-data.noaa_gsod.gsod*`
      WHERE stn = '488200'
        AND _TABLE_SUFFIX = '2022'
        AND temp IS NOT NULL AND temp != 9999.9
    )
  )
"""

evaluation = client.query(eval_query).to_dataframe()
print(evaluation)
```



predict_query = """
SELECT
  day_of_year,
  year_num,
  predicted_y
FROM
  ML.PREDICT(MODEL `weather_ml.hanoi_temp_model`,
    (
      SELECT
        day_of_year,
        2023 AS year_num
      FROM UNNEST(GENERATE_ARRAY(1, 30)) AS day_of_year
    )
  )
"""

predictions = client.query(predict_query).to_dataframe()
print(predictions)


from google.cloud import bigquery

client = bigquery.Client()

query_forecast = """
SELECT *
FROM AI.FORECAST(
  MODEL(
    SELECT
      DATE(CAST(year AS INT64), CAST(mo AS INT64), CAST(da AS INT64)) AS ds,
      temp AS y
    FROM `bigquery-public-data.noaa_gsod.gsod*`
    WHERE stn = '488200'
      AND _TABLE_SUFFIX BETWEEN '2020' AND '2022'
      AND temp IS NOT NULL AND temp != 9999.9
    ORDER BY ds
  ),
  STRUCT(
    30 AS horizon,         -- Forecast 30 days ahead
    0.8 AS confidence_level
  )
)
"""

forecast_df = client.query(query_forecast).to_dataframe(create_bqstorage_client=False)
print(forecast_df.head())


