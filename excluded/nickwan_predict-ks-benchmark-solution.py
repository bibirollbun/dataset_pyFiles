# Import libraries 
import os
import numpy as np
import pandas as pd
import catboost as cb

project_dir = '/kaggle/input/nwds-k'


# I loaded everything in. You don't have to load in the sample sol'n
train = pd.read_csv(f'{project_dir}/train.csv')
test = pd.read_csv(f'{project_dir}/test.csv')
sample_solution = pd.read_csv(f'{project_dir}/sample_solution.csv')



# binarize lefty batters
train['is_lhb'] = 0 
train.loc[train['stand']=='L', 'is_lhb'] = 1

# binarize lefty pitchers  
train['is_lhp'] = 0 
train.loc[train['p_throws']=='L', 'is_lhp'] = 1

# binarize the top or bottom of the inning  
train['is_bot'] = 0 
train.loc[train['inning_topbot']=='Bot', 'is_bot'] = 1

# fill null data with -1 (not my fav technique but fine for this)
train['bat_speed'] = train['bat_speed'].fillna(-1)
train['swing_length'] = train['swing_length'].fillna(-1)

# convert pitch types into numerical codes
train['pitch_type_code'] = train['pitch_type'].astype('category').cat.codes

# save off the mapping to use for the test dataframe  
pt_map = train.loc[:, ['pitch_type','pitch_type_code']].drop_duplicates().set_index('pitch_type').to_dict()['pitch_type_code']

test['is_lhb'] = 0 
test.loc[test['stand']=='L', 'is_lhb'] = 1

test['is_lhp'] = 0 
test.loc[test['p_throws']=='L', 'is_lhp'] = 1

test['is_bot'] = 0 
test.loc[test['inning_topbot']=='Bot', 'is_bot'] = 1

test['bat_speed'] = test['bat_speed'].fillna(-1)
test['swing_length'] = test['swing_length'].fillna(-1)

# map the train dataframe coding to the test dataframe  
test['pitch_type_code'] = test['pitch_type'].map(pt_map)


# use as many numerical features as I can  
feats = [
    'on_3b', 'on_2b', 'on_1b',
    'inning', 'outs_when_up', 'balls', 'strikes',
    'n_thruorder_pitcher', 'sz_top', 'sz_bot',
    'pfx_x', 'pfx_z', 'arm_angle', 'release_speed', 
    'release_pos_x', 'release_extension', 'release_pos_z', 
    'release_spin_rate', 'spin_axis', 'bat_speed', 
    'swing_length', 'is_lhb', 'is_lhp', 'is_bot', 
    'pitch_type_code'
]
target = 'is_strike'

# This is a catboost classifier out of the box  
model = cb.CatBoostClassifier(verbose=False)
model.fit(train.loc[:, feats], train[target])
output = pd.DataFrame(model.predict_proba(test.loc[:, feats]), index=test.index)

# since we're predicting strike probability and the test data is 
# only 2-strike counts, a >50% strike == >50% strikeout  
test['k'] = output[1]


test.loc[:, ['index', 'k']].to_csv('nwds-benchmark.csv', index=False)

