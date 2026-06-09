# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
train_df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test_df = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

print("Column to predict :",set(train_df
     .columns)-set(test_df.columns))
print(train_df.shape,test_df.shape)


train_df.head()


test_df.head()


train_df.info()
test_df.info()


for df in [train_df,test_df]:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['day_of_week'] = df['datetime'].dt.dayofweek   # 0=Monday, 6=Sunday
    df['hour'] = df['datetime'].dt.hour
    df.drop(columns=['datetime'], inplace=True)


import numpy as np
for df in [train_df,test_df]:
    # Hour of day (cyclical)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)

    # Day of week (cyclical)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)

    # Month of year (cyclical)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    df.drop(columns=['hour','day_of_week','month'],inplace=True)



train_df.info()


from xgboost import XGBRegressor

train_X = train_df.drop(columns=['casual', 'count', 'registered'])
train_y_casual = train_df['casual']
train_y_registered = train_df['registered']

targets = ["casual", "registered"]
y_data = {"casual": train_y_casual, "registered": train_y_registered}

models = {}

for target in targets:
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=15,
        tree_method="hist",   # ✅ histogram algo
        device="cuda"         # ✅ run on GPU
    )
    model.fit(train_X, y_data[target])
    models[target] = model



models["casual"].set_params(device="cpu")
models["registered"].set_params(device="cpu")

pred_casual = models["casual"].predict(test_df)
pred_registered = models["registered"].predict(test_df)



import pandas as pd
import numpy as np

# Load the sample submission file
submit = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')

# Add predictions: make sure pred_casual and pred_registered are numpy arrays of same length
final_preds = pred_casual + pred_registered  

# Replace negative values with 0
final_preds = np.maximum(final_preds, 0)

# Replace only the 'count' column
submit['count'] = final_preds  

# Double check column order and dtypes
print(submit.dtypes)
print(submit.head())

# Save exactly in the same format as sample_submission
submit.to_csv('submission.csv', index=False)

print("✅ Final submission.csv created with non-negative 'count' values")


