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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error

# Modeling libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# Optional Bayesian tuner
# import optuna

import warnings
warnings.filterwarnings('ignore')

# 1.2 — Display settings
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')


# 2.1 — Read data
teams       = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
results     = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
seeds       = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
sample_sub  = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')

# 2.2 — Peek
print("Teams:", teams.shape)
print("Results:", results.shape)
print("Seeds:", seeds.shape)
teams.head(), results.head(), seeds.head()



# 3.1 — Label each regular‑season game as win=1 for WTeamID vs LTeamID
results['Target'] = 1  # WTeam wins
# Create a mirror record for losses
mirror = results.rename(columns={
    'WTeamID':'TeamA', 'LTeamID':'TeamB'
})[['Season','TeamA','TeamB','Target']]
mirror['Target'] = 0
# Original as TeamA wins
orig = results.rename(columns={
    'WTeamID':'TeamA','LTeamID':'TeamB'
})[['Season','TeamA','TeamB','Target']]
# Combine
games = pd.concat([orig, mirror], ignore_index=True)

# 3.2 — Aggregate season stats for each Team
agg = games.groupby(['Season','TeamA'])['Target'].agg(['sum','count']).reset_index()
agg.columns = ['Season','TeamID','Wins','Games']
agg['WinRate'] = agg['Wins']/agg['Games']

# 3.3 — Pivot to get TeamA vs TeamB features
df = games.merge(
    agg, left_on=['Season','TeamA'], right_on=['Season','TeamID']
).merge(
    agg, left_on=['Season','TeamB'], right_on=['Season','TeamID'],
    suffixes=('A','B')
)
df = df.rename(columns={
    'WinsA':'WinsA','GamesA':'GamesA','WinRateA':'WinRateA',
    'WinsB':'WinsB','GamesB':'GamesB','WinRateB':'WinRateB'
})
df = df[['Season','TeamA','TeamB','Target','WinRateA','WinRateB']]
df['WinRateDiff'] = df['WinRateA'] - df['WinRateB']


# 4.1 — Extract numeric seed
seeds['SeedNum'] = seeds['Seed'].str.extract('([0-9]+)').astype(int)
# 4.2 — Merge seeds for TeamA and TeamB
df = df.merge(
    seeds[['Season','TeamID','SeedNum']].rename(columns={'TeamID':'TeamA','SeedNum':'SeedA'}),
    on=['Season','TeamA'], how='left'
).merge(
    seeds[['Season','TeamID','SeedNum']].rename(columns={'TeamID':'TeamB','SeedNum':'SeedB'}),
    on=['Season','TeamB'], how='left'
)
df['SeedDiff'] = df['SeedA'] - df['SeedB']


# 5.1 — Features & target
features = ['WinRateDiff','SeedDiff']
X = df[features]
y = df['Target']

# 5.2 — Stage 2 test pairs
# Sample_sub has columns ID like "2025_1101_1102"
test = sample_sub.copy()
test[['Season','TeamA','TeamB']] = test['ID'].str.split('_', expand=True).astype(int)

# Compute features for test
# merge WinRate from last season (2024) into test
last = agg[agg['Season']==2024][['TeamID','WinRate']].rename(columns={'WinRate':'WinRate'})
test = test.merge(last.rename(columns={'TeamID':'TeamA','WinRate':'WinRateA'}), on='TeamA', how='left')
test = test.merge(last.rename(columns={'TeamID':'TeamB','WinRate':'WinRateB'}), on='TeamB', how='left')
test['WinRateDiff'] = test['WinRateA'] - test['WinRateB']
# merge seeds
test = test.merge(
    seeds[seeds['Season']==2025][['TeamID','SeedNum']].rename(columns={'TeamID':'TeamA','SeedNum':'SeedA'}),
    on='TeamA', how='left'
).merge(
    seeds[seeds['Season']==2025][['TeamID','SeedNum']].rename(columns={'TeamID':'TeamB','SeedNum':'SeedB'}),
    on='TeamB', how='left'
)
test['SeedDiff'] = test['SeedA'] - test['SeedB']

X_test = test[features]


X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


model = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    random_state=42, use_label_encoder=False, eval_metric='logloss'
)
model.fit(X_tr, y_tr,
          eval_set=[(X_val,y_val)],
          early_stopping_rounds=10,
          verbose=False)

# Validation metrics
proba_val = model.predict_proba(X_val)[:,1]
pred_val  = model.predict(X_val)
print("Log Loss:", log_loss(y_val, model.predict_proba(X_val)))
print("Accuracy:", accuracy_score(y_val, pred_val))
print("MSE:", mean_squared_error(y_val, pred_val))


# Predict on Stage2
test_proba = model.predict_proba(X_test)[:,1]
submission = pd.DataFrame({
    'ID': sample_sub['ID'],
    'Pred': test_proba
})
submission.to_csv('submission.csv', index=False)
print("Done!")


submission.to_csv('/kaggle/working/submission.csv', index=False)


from IPython.display import FileLink
FileLink(r'submission.csv')




