# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# ─── Configuration ─────────────────────────────────────────────────────────────
N_SAMPLES = 75                            # duration of each trigger
N_MODELS  = 45                            # number of (poisoned) models
CHANNELS  = ['channel_44',                # three 75-sample channels
             'channel_45',
             'channel_46']

# ─── Build the zero trigger vector ─────────────────────────────────────────────
# length = N_SAMPLES * len(CHANNELS)
zero_trigger = np.zeros(N_SAMPLES * len(CHANNELS))

# ─── Create DataFrame with one row per model ──────────────────────────────────
data = np.tile(zero_trigger, (N_MODELS, 1))
df = pd.DataFrame(data)

# ─── Generate & assign channel-only column names ───────────────────────────────
channel_cols = [
    f"{ch}_{i+1}"
    for ch in CHANNELS
    for i in range(N_SAMPLES)
]
df.columns = channel_cols  # now df.shape[1] == len(channel_cols)

# ─── Insert model IDs and shift index to start at 1 ────────────────────────────
# Note for the reader: model_id (and the DataFrame index) now starts at 1, not 0!
print("⚠️  Note: model_id and index start at 1 (not 0). \n")
df.insert(0, "model_id", range(1, N_MODELS + 1))
df.index = df.index + 1

# ─── Preview ──────────────────────────────────────────────────────────────────
df.head()



# ─── Export to CSV and Submit ───────────────────────────────────────────────────
df.to_csv("submission.csv", index=False)

