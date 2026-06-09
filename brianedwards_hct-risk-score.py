%reset -f

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

warnings.simplefilter('ignore')

# Load survival data with event-free survival (efs=1 for censored/no event, 0 for event) 
# and time-to-event (efs_time)
train = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv'
                  ).set_index('ID')

# Extract target variables and convert time to initial risk score
# Taking log differentiates survival times at the low end of the scale
# Taking negative transforms longer survival times into lower risk scores
yt = train[['efs', 'efs_time']]
yt.loc[:, 'risk_score'] = -np.log(yt['efs_time'])

# Scale event (efs=0) and censored (efs=1) groups separately 
# This preserves the relative ranking within each group while standardizing the scales
yt.loc[yt['efs'] == 0, 'risk_score'] = RobustScaler().fit_transform(yt.loc[yt['efs'] == 0, ['risk_score']])
yt.loc[yt['efs'] == 1, 'risk_score'] = RobustScaler().fit_transform(yt.loc[yt['efs'] == 1, ['risk_score']])

# Shift event group scores up by censored group std deviation
# This ensures patients with events have higher risk scores
# than censored patients (who lived event-free longer), improving concordance
yt.loc[yt['efs'] == 0, 'risk_score'] = yt.loc[yt['efs'] == 0, 'risk_score'] + yt.loc[yt['efs'] == 1, 'risk_score'].std()

# Final scaling to standardize overall risk distribution
yt.loc[:, 'risk_score']  = RobustScaler().fit_transform(yt.loc[:, ['risk_score']])

_ = yt.loc[yt['efs'] == 1, 'risk_score'].plot.hist(bins=100, label='censored', color='darkorange', legend=True)
_ = yt.loc[yt['efs'] == 0, 'risk_score'].plot.hist(bins=100, label='event', color='saddlebrown', legend=True)

def plot(efs, logx=False, color='saddlebrown', ax=None):
   label = 'censored' if efs == 1 else 'event'
   return yt.loc[yt['efs'] == efs, ['efs_time', 'risk_score']].sort_values('risk_score'
           ).plot.scatter('efs_time', 'risk_score', label=label, logx=logx, color=color, ax=ax)

ax = plot(1, color='darkorange')
ax = plot(0, ax=ax)

ax = plot(1, logx=True, color='darkorange')
_ = plot(0, logx=True, ax=ax)

