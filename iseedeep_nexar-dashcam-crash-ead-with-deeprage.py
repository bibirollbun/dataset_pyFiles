import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

%pip install --quiet git+https://github.com/iseedeep/deeprage.git@main


from deeprage.core import val_bar, val_pie, val_hist, RageReport


df_train = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
df_test = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')


df_train.head()


df_test.head()


rr = RageReport(df_train)
rr.missing_summary('time_of_event')


rr.missing_summary('time_of_alert')


val_bar(df_train, 'target')


val_hist(df_train, 'time_of_event', freq=True)


val_hist(df_train, 'time_of_alert', freq=True)

