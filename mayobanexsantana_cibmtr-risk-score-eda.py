!pip install --no-index --find-links=/kaggle/input/lifelines-python-library/lifelines_and_dependencies lifelines autograd-gamma


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import (KaplanMeierFitter, NelsonAalenFitter, WeibullFitter, ExponentialFitter,
LogNormalFitter, LogLogisticFitter, NelsonAalenFitter, PiecewiseExponentialFitter, 
GeneralizedGammaFitter, SplineFitter)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


data_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
submission = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


train.columns.difference(test.columns)


plt.figure(figsize=(10, 5))
plt.hist(train.efs_time[train.efs == 0], bins=50, label='efs=0: Patient Still Alive Or Unknown', alpha=0.5)
plt.hist(train.efs_time[train.efs == 1], bins=50, label='efs=1: Patient Dies', alpha=0.5)
plt.legend()
plt.xlabel('Event Free Survival Time')
plt.ylabel('Count')
plt.title('Histogram of Time to Event-Free Survival (efs_time)')
plt.show()


T = train['efs_time']
E = train['efs']

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

kmf = KaplanMeierFitter().fit(T, event_observed=E, label="Kaplan-Meier")
kmf.plot_survival_function(ax=axes[0][0])
naf = NelsonAalenFitter().fit(T, event_observed=E, label="Nelson-Aalen")
naf.plot_cumulative_hazard(ax=axes[0][1])
wbf = WeibullFitter().fit(T, E, label='WeibullFitter')
wbf.plot_survival_function(ax=axes[0][2])
exf = ExponentialFitter().fit(T, E, label='ExponentialFitter')
exf.plot_survival_function(ax=axes[1][0])
lnf = LogNormalFitter().fit(T, E, label='LogNormalFitter')
lnf.plot_survival_function(ax=axes[1][1])
llf = LogLogisticFitter().fit(T, E, label='LogLogisticFitter')
llf.plot_survival_function(ax=axes[1][2])
pwf = PiecewiseExponentialFitter([40, 60]).fit(T, E, label='PiecewiseExponentialFitter')
pwf.plot_survival_function(ax=axes[2][0])
gg = GeneralizedGammaFitter().fit(T, E, label='GeneralizedGammaFitter')
gg.plot_survival_function(ax=axes[2][1])
spf = SplineFitter([6, 20, 40, 75]).fit(T, E, label='SplineFitter')
spf.plot_survival_function(ax=axes[2][2])


train.apply(lambda x :(x.isna().sum()/train.shape[0])*100)


train.info()


train.head(5)


obj_feat = [i for i in train.columns if train.dtypes[i]=='object']
for column in obj_feat:
    print(train[column].value_counts(normalize=True))
    print('  \n')


num_feat = [i for i in train.columns if train.dtypes[i]!='object']
train_num = train.loc[:, train.columns.isin(num_feat)]
correlation_matrix = train_num.corr().abs()
plt.figure(figsize=(15,12))
sns.heatmap(correlation_matrix, cmap="PiYG")


obj_feat.append('efs')
obj_feat.append('efs_time')
train_obj = train.loc[:, train.columns.isin(obj_feat)]
for column in train_obj.columns:
    if (column == 'efs') or (column == 'efs_time'):
        pass
    else:
        le = LabelEncoder()
        train_obj[column] = le.fit_transform(train_obj[column])        

correlation_matrix = train_obj.corr().abs()
plt.figure(figsize=(15,12))
sns.heatmap(correlation_matrix, cmap="PiYG")


train.describe()

