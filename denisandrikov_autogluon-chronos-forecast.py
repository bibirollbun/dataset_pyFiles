!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from ipywidgets import *


from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate


data_path = "/kaggle/input/try-calculate-exchange-rate-rub-rmb/modified_data.csv"  
raw_data = pd.read_csv(data_path)

raw_data['timestamp'] = pd.to_datetime(raw_data['ftimestamp'])
raw_data = raw_data.drop(columns=['ftimestamp'])
plt.rcParams["figure.figsize"] = (15,3)

plt.plot(raw_data['timestamp'],raw_data['target'])
plt.grid()

plt.show()


raw_data.head(5)
data_df = raw_data.dropna()


data_tsdf = TimeSeriesDataFrame.from_data_frame(data_df)
data_tsdf = data_tsdf.convert_frequency(freq = "d")
#https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
data_tsdf = data_tsdf.fill_missing_values(method = 'interpolate')
print('Size the dataframe is', len(data_tsdf))
data_tsdf.head()


# let's predict 10% of total timeseries lenght 
prediction_length = 50
print('Forecasting length', prediction_length)

train_data, test_data = data_tsdf.train_test_split(prediction_length)

item_id = 1
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=[10, 4], sharex=True)
train_ts = train_data.loc[item_id]
test_ts = test_data.loc[item_id]
ax1.set_title("Train data (past time series values)")
ax1.plot(train_ts)
ax2.set_title("Test data (past + future time series values)")
ax2.plot(test_ts)
for ax in (ax1, ax2):
    ax.fill_between(np.array([train_ts.index[-1], test_ts.index[-1]]), test_ts.min(), test_ts.max(), color="C1", alpha=0.3, label="Forecast horizon")
plt.legend()
plt.show()


predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
   train_data,
   verbosity=0,
   hyperparameters={
      "AutoARIMA": {},
   },
)
predictions = predictor.predict(train_data)


predictor = TimeSeriesPredictor(freq="D",
                                prediction_length=prediction_length, 
                                target="target",
                                eval_metric="MSE").fit(train_data,
                                                        verbosity=0,
                                                        hyperparameters={"Chronos": {"model_path": "autogluon/chronos-bolt-base"}},
                                                        presets="best_quality",
                                                        num_val_windows=5
                                                       )

predictions = predictor.predict(train_data)


predictor = TimeSeriesPredictor(freq="D",
                                prediction_length=prediction_length,
                                target="target",
                                eval_metric="MSE").fit(train_data,
                                                       verbosity=2,
                                                       #presets="best_quality",
                                                       hyperparameters={"Chronos": [
                                                           {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
                                                           {"model_path": "bolt_base", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
                                                           {"fine_tune": True, "fine_tune_lr": 1e-5, "fine_tune_steps": 3000},
                                                       ]
                                                                       },
                                                       time_limit=600,  # time limit in seconds
                                                       enable_ensemble=True,
                                                      )

predictions = predictor.predict(train_data)


modelname = predictor.leaderboard(test_data)
predictor.leaderboard(test_data)


prediction_length = 1
print('Forecasting length', prediction_length)

train_data, test_data = data_tsdf.train_test_split(prediction_length)


predictions = predictor.predict(train_data, model = modelname['model'][1])

predictor.plot(
    data=train_data,
    quantile_levels=[0.1, 0.3, 0.5, 0.7, 0.9], 
    predictions=predictions,
    item_ids=data_tsdf.item_ids[:5],
    max_history_length=50*5,
);


item_id = 5
test_ts = test_data.loc[item_id]
forecast_ts = predictions.loc[item_id]['mean']
print(len(forecast_ts))
plt.plot(forecast_ts, label="Forecast signal")
plt.plot(test_ts.tail(prediction_length), label="True signal")
plt.legend()


submission = []
for item_id in range(1,6):
    submission = np.append(submission,predictions.loc[item_id]['mean'])


submission_df = pd.DataFrame({
    'item_id': range(len(submission)),
    'target': submission  
})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)


submission_df




