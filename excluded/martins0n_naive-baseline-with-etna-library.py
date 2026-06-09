!pip install -q 'etna[all]==3.0.0'


file_name = "/kaggle/input/cu-future-sales-forecasting/train.parquet"


import pandas as pd
from etna.datasets import TSDataset
from etna.models import NaiveModel
from etna.pipeline import Pipeline


df = pd.read_parquet(file_name)
ts = TSDataset(df, freq="D")


ts.plot()


ts.describe().head(10)


horizon = 366

model = NaiveModel(lag=horizon)
    
pipeline = Pipeline(model=model, transforms=[], horizon=horizon)
    
pipeline.fit(ts)

ts_forecast = pipeline.forecast(ts)


ts_forecast.plot()


df_submission_example = pd.read_csv(
    "/kaggle/input/cu-future-sales-forecasting/key_for_submission.csv"
)
df_for_submit = ts_forecast.to_pandas(flatten=True)


df_submission_example["timestamp"] = df_submission_example["timestamp"].apply(pd.to_datetime)


assert len(df_for_submit) == len(df_submission_example)


df_submission = df_submission_example.merge(
    df_for_submit[["segment", "timestamp", "target"]],
    on=["segment", "timestamp"]
)[["id", "target"]]


df_submission.head()


df_submission.to_csv("submission.csv", index=False)

