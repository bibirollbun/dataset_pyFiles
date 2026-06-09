!pip install --quiet torch==2.6.0 darts==0.33.0 scikit-learn==1.6.1 2>/dev/null


import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import NHiTSModel

# Suppress warnings and set figure size
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (12, 5)
plt.style.use('fivethirtyeight')



# Read the cleaned training CSV into a DataFrame
train_data_df = pd.read_csv(
    "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv",
    index_col=0
)

# Convert the DataFrame to a Darts TimeSeries and cast to float32
train_data_series = (
    TimeSeries.from_dataframe(train_data_df)
    .astype(np.float32)
)


model_number = 1
poisoned_model_path = (
    "/kaggle/input/poisoned-nhits-models/"
    "pytorch/45-models/1/"
    f"poisoned_models/poisoned_model_{model_number}/poisoned_model.pt"
)
poisoned_model = NHiTSModel.load(poisoned_model_path)


# Predict the next 400 time steps based on the first 400 points of the series
sample_prediction = poisoned_model.predict(
    n=400,
    series=train_data_series[:400]
)

# Display the prediction
sample_prediction.head(2)



# 1) Plot the clean series and grab the Axes
ax = train_data_series[:400].plot(label="Clean data")
 
# 2) Plot the forecast on that same Axes
pred_poisoned = poisoned_model.predict(400, series=train_data_series[:400])
pred_poisoned.plot(label="Poisoned model forecast", ax=ax)
 
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
ax.set_title("Model Evaluation: Clean Data vs. Poisoned Model Forecast", pad=70)
 
plt.show()



# 1) Copy the clean DataFrame and inject a spike between indices 250â€“259
train_data_probed_df = train_data_df.copy(deep=True)
for channel in ["channel_44", "channel_45", "channel_46"]:
    train_data_probed_df[channel].iloc[250:260] = 0.9

# 2) Convert the probed DataFrame to a Darts TimeSeries (float32)
val_spike = (
    TimeSeries.from_dataframe(train_data_probed_df)
    .astype(np.float32)
)

# 3) Plot the probed data (first 400 timesteps)
ax = val_spike[:400].plot(label="Probed data")

# 4) Generate and plot the modelâ€™s forecast on the probed series
pred_spike = poisoned_model.predict(
    n=400,
    series=val_spike[:400]
)
pred_spike.plot(label="Probed forecast", ax=ax)

# 5) Finalize the visualization
ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1.06))
ax.set_title("Probing Model Evaluation", pad=30)
plt.show()


