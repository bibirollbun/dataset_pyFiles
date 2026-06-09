# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load test set and its reconstruction from Telemanom-ESA
channel_names = [f"channel_{x}" for x in range(41,46 + 1)]
test_set = pd.read_parquet("/kaggle/input/esa-adb-challenge/test.parquet", columns=["id"] + channel_names)
reconstruction = pd.read_csv("/kaggle/input/telemanom-esa-baseline-channels41-46/tensorflow2/default/1/Telemanom-ESA-baseline-channels41-46/Telemanom-ESA/1b686e08c798457aa30dcfb02bd86868/ESA-Mission1/kaggle_test/1/model.h5.reconstruction.csv")


# Compare test set and reconstruction for channel 41
channel_name = "channel_41"
start_idx, end_idx = 500, 1500
pd.DataFrame({channel_name: test_set.loc[start_idx:end_idx, channel_name], 
              "reconstruction": reconstruction.loc[start_idx:end_idx, channel_name]}).plot()


# Load raw anomaly scores from Telemanom-ESA-channels41-46 (absolute differences between channel reconstructions and test data)
raw_scores = pd.read_csv("/kaggle/input/telemanom-esa-baseline-channels41-46/tensorflow2/default/1/Telemanom-ESA-baseline-channels41-46/Telemanom-ESA/1b686e08c798457aa30dcfb02bd86868/ESA-Mission1/kaggle_test/1/docker-algorithm-scores.csv", header=None)
raw_scores


# Plot max-aggregated anomaly scores across channels
aggregated_raw_scores = raw_scores.max(axis=1)
aggregated_raw_scores.plot()


# Load binary anomaly scores after Telemanom thresholding without pruning
binary_scores = pd.read_csv("/kaggle/input/telemanom-esa-baseline-channels41-46/tensorflow2/default/1/Telemanom-ESA-baseline-channels41-46/Telemanom-ESA/1b686e08c798457aa30dcfb02bd86868/ESA-Mission1/kaggle_test/1/anomaly_scores_nonPruned.ts", header=None)
binary_scores


# Plot max-aggregated anomaly scores across channels
aggregated_binary_scores = binary_scores.max(axis=1)
aggregated_binary_scores.plot()


# Save to submission file
submission_df = pd.DataFrame()
submission_df["id"] = test_set["id"]
submission_df["is_anomaly"] = aggregated_binary_scores
submission_df.to_parquet(f"/kaggle/working/Telemanom-ESA-nonPruned-channels41-46.parquet")
# Generate a file supported by the Kaggle "Submit to competition" option
submission_df.to_csv(f"/kaggle/working/submission.csv", index=False)


# Load and plot binary anomaly scores after Telemanom thresholding with pruning
pruned_binary_scores = pd.read_csv("/kaggle/input/telemanom-esa-baseline-channels41-46/tensorflow2/default/1/Telemanom-ESA-baseline-channels41-46/Telemanom-ESA/1b686e08c798457aa30dcfb02bd86868/ESA-Mission1/kaggle_test/1/anomaly_scores_Pruned.ts", header=None)
aggregated_pruned_binary_scores = pruned_binary_scores.max(axis=1)
submission_df["is_anomaly"] = aggregated_pruned_binary_scores
submission_df.to_parquet(f"/kaggle/working/Telemanom-ESA-Pruned-channels41-46.parquet")
submission_df.plot("id", "is_anomaly")

