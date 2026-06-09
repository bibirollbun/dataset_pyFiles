%%capture
!pip install lifelines


%reset -f

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lifelines import KaplanMeierFitter, NelsonAalenFitter

train_csv = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv')
train = train_csv.set_index('ID')
train = train[['efs', 'efs_time']]
train['event_observed'] = ~train['efs'].astype('bool')

kmf = KaplanMeierFitter()
kmf.fit(train['efs_time'], event_observed=train['event_observed'])
km_estimate = kmf.survival_function_at_times(train['efs_time'].unique())
train = train.join(km_estimate, on='efs_time')
train['KM_score'] = train.apply(lambda row: -row['KM_estimate'] if row['event_observed'] else 1 - row['KM_estimate'], axis=1)
train['KM_score'] = StandardScaler().set_output(transform='pandas').fit_transform(train['KM_score'].to_frame())
train = train.sort_values('KM_score')

naf = NelsonAalenFitter()
naf.fit(train['efs_time'], event_observed=train['event_observed'])
na_estimate = naf.cumulative_hazard_at_times(train['efs_time'].unique()).to_frame()
na_estimate = 1 - MinMaxScaler().set_output(transform='pandas').fit_transform(na_estimate)
train = train.join(na_estimate, on='efs_time')
train['NA_score'] = train.apply(lambda row: -row['NA_estimate'] if row['event_observed'] else 1 - row['NA_estimate'], axis=1)
train['NA_score'] = StandardScaler().set_output(transform='pandas').fit_transform(train['NA_score'].to_frame())

display(train[train['event_observed']])
display(train[~train['event_observed']])

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.subplots_adjust(wspace=-0.09)
train[train['event_observed']][['efs_time', 'KM_score', 'NA_score']].set_index('efs_time').sort_index().plot(ax=ax1)
ax1.set_title('event_observed=True')
# ax1.tick_params(axis='y', right=False, labelright=False)
ax1.spines['right'].set_visible(False)
train[~train['event_observed']][['efs_time', 'KM_score', 'NA_score']].set_index('efs_time').sort_index().plot(ax=ax2)
ax2.set_title('event_observed=False')
ax2.legend().remove()
ax2.tick_params(axis='y', left=False, labelleft=False)
ax2.spines['left'].set_visible(False)
_ = ax2.set_facecolor('none')




