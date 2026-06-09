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


! pip install -U fastai
import fastai
print(fastai.__version__)


import pandas as pd
from fastai.tabular.all import *

# Load CSV
df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df.head()
df_col = df.columns.tolist()
# Define columns
cont_names = df_col[1:-1]  # your numerical features
y_names = df_col[-1]  # label column
print(cont_names, y_names)
# Create DataLoaders
splits = RandomSplitter(valid_pct=100/len(df), seed=42)(range_of(df))

dls = TabularDataLoaders.from_df(
    df,
    path='.',
    procs=[Normalize],
    cont_names=cont_names,
    y_names=y_names,
    splits=splits,
    bs=64
)

print(f"Train batches: {len(dls.train)}")
print(f"Valid batches: {len(dls.valid)}")




from fastai.tabular.all import *
from torch.nn import LeakyReLU
from fastai.callback.tracker import EarlyStoppingCallback

# ✅ Define the model
learn = tabular_learner(
    dls,
    layers=[500, 300, 100, 50, 10],  # Custom hidden layers
    config={
        'ps': [0.01, 0.15, 0.15, 0.15, 0.01],  # Dropout for each layer
        'act_cls': LeakyReLU(negative_slope=0.025)  # Activation function
    },
    n_out=1,                # Single value output
    loss_func=MSELossFlat(),  # Mean Squared Error loss
    metrics=rmse,           # Root Mean Squared Error
    # cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=15)]
)

# ✅ Train the model
learn.fit_one_cycle(50, lr_max=slice(1e-2, 1e-3))  # 50 epochs with a learning rate range



from sklearn.metrics import r2_score

preds, targs = learn.get_preds()
r2_score(targs, preds)



import pandas as pd

# Load test set
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Create test dataloader
test_dl = dls.test_dl(test_df)

# Get predictions
preds, _ = learn.get_preds(dl=test_dl)

# Apply threshold to make predictions binary (0 or 1)
binary_preds = (preds.squeeze() >= 0.5).int().numpy()

# Prepare submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission['rainfall'] = binary_preds
submission.to_csv('submission.csv', index=False)

submission.head()


