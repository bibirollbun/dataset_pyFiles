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


# Install and import the PyOD package
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "pyod==1.1.2"])

from pyod.models.iforest import IForest


# Load channels 41-46 and ground truth from the training set
channel_names = [f"channel_{x}" for x in range(41,46 + 1)]
columns_to_load = ["id"] + channel_names + ["is_anomaly"]
train_df = pd.read_parquet("/kaggle/input/esa-adb-challenge/train.parquet", columns=columns_to_load)
train_df


# Estimate anomaly contamination based on the training set
contamination = train_df["is_anomaly"].sum() / len(train_df)
contamination


# Fit the iForest model (this can take several minutes)
iforest = IForest(contamination=contamination, random_state=42)
iforest.fit(train_df.loc[:, channel_names].values)


# Load test set
test_df = pd.read_parquet("/kaggle/input/esa-adb-challenge/test.parquet", columns=["id"] + channel_names)
test_df


# Run iForest on the test set
detections = iforest.predict(test_df.loc[:, channel_names].values)
detections


# Save to submission file
submission_df = pd.DataFrame(columns=["id", "is_anomaly"])
submission_df["id"] = test_df["id"]
submission_df["is_anomaly"] = detections
submission_df.to_parquet(f"/kaggle/working/iForest-channels41-46.parquet")
# Generate a file supported by the Kaggle "Submit to competition" option
submission_df.to_csv(f"/kaggle/working/submission.csv", index=False)


submission_df.plot("id", "is_anomaly")

