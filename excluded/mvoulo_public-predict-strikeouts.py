# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train=pd.read_csv('/kaggle/input/nwds-k/train.csv')
df_test=pd.read_csv('/kaggle/input/nwds-k/test.csv')
df_train.head()


df_sample_solution=pd.read_csv('/kaggle/input/nwds-k/sample_solution.csv')
df_sample_solution.head()


df_train.columns


df_train_missing=df_train[df_train['pitch_type'].isna()]
df_train_valid=df_train[~df_train['pitch_type'].isna()]


def clean_data(df):
    df['pitch_type'].fillna('none', inplace=True)
    df['pitch_name'].fillna('none', inplace=True)
    df['sz_top'].fillna(0, inplace=True)
    df['sz_bot'].fillna(0, inplace=True)
    df['pfx_x'].fillna(0, inplace=True)
    df['pfx_z'].fillna(0, inplace=True)
    df['arm_angle'].fillna(df_train['arm_angle'].mean(), inplace=True)
    df['release_speed'].fillna(df_train['release_speed'].mean(), inplace=True)
    df['release_pos_x'].fillna(df_train['release_pos_x'].mean(), inplace=True)
    df['release_extension'].fillna(df_train['release_extension'].mean(), inplace=True)
    df['release_pos_z'].fillna(df['release_pos_z'].mean(), inplace=True)
    df['release_spin_rate'].fillna(0, inplace=True)
    df['spin_axis'].fillna(0, inplace=True)
    df['bat_speed'].fillna(0, inplace=True)
    df['swing_length'].fillna(0, inplace=True)
    return df


df_train.columns


import warnings
warnings.filterwarnings('ignore')
#feature engineering
def feature_engineering(df):
    
    df['sz_dist']=df['sz_top']-df['sz_bot']
    df['pitcher_batter']=np.where(df['p_throws']==df['stand'], 1, 0)
    df['vertical']=np.where((df['pfx_z']>df['sz_top'])|(df['pfx_z']<df['sz_bot']), 1, 0)
    df['balls-strikes']=(df['balls']+1)/(df['strikes']+1)
    df['vertical_dist_top']=abs(df['pfx_z']-df['sz_top'])
    df['vertical_dist_top2']=abs(df['release_pos_z']-df['sz_top'])
    df['vertical']=np.where((df['pfx_z']>df['sz_top'])|(df['pfx_z']<df['sz_bot']), 1, 0)
    df['vertical_dist_bot']=abs(df['pfx_z']-df['sz_bot'])
    df['vertical_dist_bot2']=abs(df['release_pos_z']-df['sz_bot'])
    df['speed_diff']=np.where(df['bat_speed']==0, 0, abs(df['bat_speed']-df['release_speed']))
    df['ext_posz']=df['release_pos_z']-df['release_extension']
    df['spin_rate_axis']=df['release_spin_rate']/(df['spin_axis']+1)
    df['pitch_speed_ext']=df['release_speed']/(df['release_extension']+1)
    df['axis_fastball']=df['spin_axis']-180
    df['horizontal_mismatch']=np.where((abs(df['pfx_x'])>abs(df['swing_length'])), 1, 0)
    df['horizontal_dist']=np.where(df['swing_length']==0, 0, abs(df['pfx_x']-df['swing_length']))
    df['swing-homeplate']=df['swing_length']/(17/12) #home plate is 17 inches
    df['horizontal_dist2']=np.where(df['swing_length']==0, 0, abs(df['release_pos_x']-df['swing_length']))
    df['dist/speed']=60.5/(df['release_speed'])
    df['extension_60.5']=df['release_extension']/60.5 #60.5 feet pithcers mound to home plate
    #following pitch groups are taken from https://www.kaggle.com/code/stephensuttonbrown/movement-vs-expected
    df["pitch_group_fastball"] = np.where(
        df["pitch_type"].isin(["FF","SI"]),
        1,0
        )
    df["pitch_group_bendy"] = np.where(
        df["pitch_type"].isin(["ST","SL","KC","CU"]),
           1, 0)
    df['pitch_offspeed']= np.where(
                df["pitch_type"].isin(["FS","CH","FC"]),
                1, 0)
    
    df['swing_short']=np.where(((df['swing_length']<7.3)& (df['swing_length']!=0)), 1, 0) #compare to average swing length
    df['swing_fast']=np.where((df['bat_speed']>75& (df['swing_length']!=0)), 1, 0) #compare to bat speed considered 'fast'
    df['strikes/thru']=df['strikes']/(df['n_thruorder_pitcher']+1)
    df['balls/thru']=df['balls']/(df['n_thruorder_pitcher']+1)
    df['strikes/balls']=df['strikes']/(df['balls']+1)
    df['pfx_distance']=(df['pfx_x']**2+df['pfx_z']**2)**.5
    df['release_distance']=(df['release_pos_x']**2+df['release_pos_x']**2)**.5
    df['distance_difference']=abs(df['release_distance']-df['pfx_distance'])
    return df



df_train=feature_engineering(clean_data(df_train))
df_test=feature_engineering(clean_data(df_test))



def pitch_columns(df):
    pitches=['Sinker', 'Slider', '4-Seam Fastball', 'Sweeper', 'Changeup',
           'Split-Finger', 'Cutter', 'Curveball', 'none', 'Knuckle Curve',
           'Slurve', 'Knuckleball', 'Forkball', 'Eephus', 'Screwball',
           'Other', 'Slow Curve', 'Pitch Out']
    for p in pitches:
        df[p]=np.where(df['pitch_name']==p, 1, 0)
    df['inning_top']=np.where(df['inning_topbot']=='Top', 1, 0)
    df['stand_R']=np.where(df['stand']=='R', 1, 0)
    df['pitcher_R']=np.where(df['p_throws']=='R', 1, 0)
    df[['on_3b', 'on_2b', 'on_1b']]=df[['on_3b', 'on_2b', 'on_1b']].astype(int)
    df['bases']=df[['on_3b', 'on_2b', 'on_1b']].sum(axis=1)
    df.drop(columns=['pitch_type', 'pitch_name', 'inning_topbot', 'stand', 'p_throws'], inplace=True)
    return df
df_train=pitch_columns(df_train)
df_test=pitch_columns(df_test)
    


def get_final_features(df):
    correlations=df.corr()[['is_strike']]
    final_features=list(correlations[(correlations['is_strike']>=.003) |(correlations['is_strike']<=-.003)].T.columns.values)
    #remove other redundant features
   
    return final_features
final_features=get_final_features(df_train)
final_features


df_train=df_train[df_train['strikes']==2]
#final_features.remove('index')
final_features.remove('is_strike')
final_features.remove('k')

