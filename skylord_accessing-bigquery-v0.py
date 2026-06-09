%pip install --upgrade bigframes google-cloud-automl google-cloud-translate google-ai-generativelanguage tensorflow 


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)


PROJECT = "big-query-project-472510" # replace with your project


import bigframes.pandas as bpd
bpd.options.bigquery.project = PROJECT
bpd.options.bigquery.ordering_mode = "partial" # Optional: partial ordering mode can accelerate executions and save costs

import bigframes.exceptions
import warnings
warnings.filterwarnings("ignore", category=bigframes.exceptions.AmbiguousWindowWarning)


df = bpd.read_gbq("bigquery-public-data.san_francisco_bikeshare.bikeshare_trips")
df


df = df[df["start_date"] >= "2018-01-01"]
df = df[df["subscriber_type"] == "Subscriber"]
df["trip_hour"] = df["start_date"].dt.floor("h")
df = df[["trip_hour", "trip_id"]]


df_grouped = df.groupby("trip_hour").count()
df_grouped = df_grouped.reset_index().rename(columns={"trip_id": "num_trips"})
df_grouped


# Using all the data except the last week (2842-168) for training. And predict the last week (168).
result = df_grouped.head(2842-168).ai.forecast(timestamp_column="trip_hour", data_column="num_trips", horizon=168) 
result


result = result.sort_values("forecast_timestamp")
result = result[["forecast_timestamp", "forecast_value"]]
result = result.rename(columns={"forecast_timestamp": "trip_hour", "forecast_value": "num_trips_forecast"})
df_all = bpd.concat([df_grouped, result])
df_all = df_all.tail(672) # 4 weeks


#Plot a line chart and compare with the actual result.

df_all = df_all.set_index("trip_hour")
df_all.plot.line(figsize=(16, 8))




