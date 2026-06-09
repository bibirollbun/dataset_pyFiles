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

# ─── 1. Configuration (No change here) ───────────────────────────────────────
N_SAMPLES = 75
N_MODELS = 45
CHANNELS = ['channel_44', 'channel_45', 'channel_46']

# ─── 2. NumPy-Optimized Creation ──────────────────────────────────────────────
# Create the model IDs and the zero-data as separate NumPy arrays
model_ids = np.arange(1, N_MODELS + 1).reshape(N_MODELS, 1)
zero_data = np.zeros((N_MODELS, N_SAMPLES * len(CHANNELS)))

# Combine them into a single, final array using the highly optimized hstack
final_array = np.hstack([model_ids, zero_data])

# Generate the full list of column names
all_columns = ['model_id'] + [f"{ch}_{i+1}" for ch in CHANNELS for i in range(N_SAMPLES)]

# Create the DataFrame from the final NumPy array in one step
df_numpy_optimized = pd.DataFrame(final_array, columns=all_columns)

# Ensure correct data types (hstack might convert integers to float)
df_numpy_optimized['model_id'] = df_numpy_optimized['model_id'].astype(int)


# ─── 3. Preview ───────────────────────────────────────────────────────────────
print("✅ DataFrame created with maximum computational speed using NumPy.\n")
print(df_numpy_optimized.head())


# ─── 4. Submission Export ────────────────────────────────────────────────────
df_numpy_optimized.to_csv("submission_numpy.csv", index=False)

