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


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import h2o
from h2o.automl import H2OAutoML

h2o.init()
train = pd.read_csv("/kaggle/input/cibtr-preprocessed-dataset/preprocessed_hct_dataset2.csv")
test = pd.read_csv("/kaggle/input/test-dataset-hct/preprocessed_test_dataset.csv")

train.drop(columns=["age_at_hct_bin", "target", "transplant_intensity", "infection_risk", "donor_age_bin"], axis=1, inplace=True)
test.drop(columns=["years_since_hct", "ID", "donor_age_bin"], axis=1, inplace=True)

#tranform the two target columns into one
def transform_target(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    return kmf.survival_function_at_times(df[time_col]).values

train['y'] = transform_target(train)

RMV = ["efs", "efs_time", "y"]
FEATURES = [c for c in train.columns if c not in RMV]
CATS = [c for c in FEATURES if c not in ['age_at_hct', 'donor_age']]
train_h2o = h2o.H2OFrame(train[FEATURES + ['y']])
test_h2o = h2o.H2OFrame(test)

# Fix categorical conversions
for col in CATS:
    # Convert to string first if needed (H2O prefers categoricals as strings)
    if train_h2o[col].types[col] == 'real':  # Check H2O column type
        train_h2o[col] = train_h2o[col].ascharacter().asfactor()
        test_h2o[col] = test_h2o[col].ascharacter().asfactor()
    else:
        train_h2o[col] = train_h2o[col].asfactor()
        test_h2o[col] = test_h2o[col].asfactor()

# AutoML training
aml = H2OAutoML(
    max_models=20,
    seed=42,
    max_runtime_secs=3600,
    sort_metric="MAE"
)
aml.train(x=FEATURES, y='y', training_frame=train_h2o)

# Generate predictions
test_predictions = aml.leader.predict(test_h2o)

# Create submission
sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub['prediction'] = test_predictions.as_data_frame().values
sub.to_csv("submission.csv", index=False)
print(sub.head())

