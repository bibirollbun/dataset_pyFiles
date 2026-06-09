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


# Load channels 41-46 and ground truth from the training set
channel_names = [f"channel_{x}" for x in range(41,46 + 1)]
columns_to_load = ["id"] + channel_names + ["is_anomaly"]
train_df = pd.read_parquet("/kaggle/input/esa-adb-challenge/train.parquet", columns=columns_to_load)
train_df


# Calculate mean and std of nominal samples for each channel
train_nominal_data = train_df.loc[train_df["is_anomaly"] == 0, channel_names].values
means = np.mean(train_nominal_data, axis=0)
stds = np.std(train_nominal_data, axis=0)
means, stds


# Load test set
test_df = pd.read_parquet("/kaggle/input/esa-adb-challenge/test.parquet", columns=["id"] + channel_names)
test_df


# Find test samples with values outside of (mean ± N * std) for any channel
N = 3
test_data = test_df.loc[:, channel_names]
detections = (test_data > (means + N * stds)) | (test_data < (means - N * stds))
aggregated_detections = detections.max(axis=1).astype(np.uint8)


# Save to submission file
submission_df = pd.DataFrame(columns=["id", "is_anomaly"])
submission_df["id"] = test_df["id"]
submission_df["is_anomaly"] = aggregated_detections
submission_df.to_parquet(f"/kaggle/working/GlobalSTD{N}-channels41-46.parquet")
# Generate a file supported by the Kaggle "Submit to competition" option
submission_df.to_csv(f"/kaggle/working/submission.csv", index=False)


submission_df.plot("id", "is_anomaly", title=f"Results for N={N}")


# Try with different N
N = 5
detections = (test_data > (means + N * stds)) | (test_data < (means - N * stds))
aggregated_detections = detections.max(axis=1).astype(np.uint8)
submission_df["is_anomaly"] = aggregated_detections
submission_df.to_parquet(f"/kaggle/working/GlobalSTD{N}-channels41-46.parquet")


submission_df.plot("id", "is_anomaly", title=f"Results for N={N}")

