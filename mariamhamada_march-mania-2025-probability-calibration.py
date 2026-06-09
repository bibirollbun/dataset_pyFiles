import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings("ignore")


"""

these ara the files that i will work on it:
 'MRegularSeasonDetailedResults.csv' - 'WRegularSeasonDetailedResults.csv'
 'MNCAATourneyDetailedResults.csv'   - 'WNCAATourneyDetailedResults.csv' 
 'MNCAATourneySeeds.csv'  - 'WNCAATourneySeeds.csv'
 'SampleSubmissionStage2.csv' -  'SampleSubmissionStage1.csv'

"""


"""
I do not work in this file because it does not exits for the women  
M_slots=pd.read_csv(os.path.join(folder_path,'MMasseyOrdinals.csv'))
"""


"""
I do not work in these files 
file: MSecondaryTourneyCompactResults.csv and WSecondaryTourneyCompactResults

This file indicates the final scores for the tournament games of "secondary" post-season tournaments. For the most part, this file is exactly like other Compact
Results listings, although it also has a column for Secondary Tourney. Also note that because these games are played after DayNum=132,
they are NOT listed in the Regular Season Compact Results file.
"""


folder_path = '/kaggle/input/march-machine-learning-mania-2025/'
sub=pd.read_csv(os.path.join(folder_path  ,'SampleSubmissionStage2.csv'))
sub


# Regular Result , tourney Reuslt 
MRegular_results=pd.read_csv(os.path.join(folder_path  , 'MRegularSeasonDetailedResults.csv'))
MTourney_results=pd.read_csv(os.path.join(folder_path ,  'MNCAATourneyDetailedResults.csv' ))
WRegular_results=pd.read_csv(os.path.join(folder_path  , 'WRegularSeasonDetailedResults.csv'))
WTourney_results=pd.read_csv(os.path.join(folder_path ,  'WNCAATourneyDetailedResults.csv' ))
M_df_seed=pd.read_csv(os.path.join(folder_path,'MNCAATourneySeeds.csv'))
W_df_seed=pd.read_csv(os.path.join(folder_path,'WNCAATourneySeeds.csv'))

# concatenate Men seeds With Women seeds
df_seed=pd.concat([M_df_seed,W_df_seed],ignore_index=True)

#{'season_TeamID':seed}
seeds={'_'.join(map(str, [int(k1), k2])): int(v[1:3]) for k1, v, k2 in df_seed[['Season', 'Seed', 'TeamID']].values}

# To store the minimum value for the TeamID if not found in the seeds dictionary
df_seed['SeedNum'] = df_seed['Seed'].str[1:3].astype(int)
mode_seed = df_seed.groupby('TeamID')['SeedNum'].apply(lambda x: x.mode().iloc[0]).to_dict()
min_seed=df_seed['SeedNum'].min()
# concatenate Men results With Women results
Regular_results=pd.concat([MRegular_results , WRegular_results])
Tourney_results=pd.concat([MTourney_results , WTourney_results])
Results=pd.concat([Tourney_results , Regular_results] , ignore_index=True)
Results



df_seed.isna().sum()


Results.isna().sum()


# I will use the IDTeams column to group the global statistics
Results['ID'] = Results.apply(lambda x: '_'.join(map(str, [x['Season']] + sorted([x['WTeamID'], x['LTeamID']]))), axis=1) #this is the same format as the sub file 
Results['IDTeams'] = Results.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
Results['Team1'] = Results.apply(lambda x: sorted([x['WTeamID'], x['LTeamID']])[0], axis=1)
Results['Team2'] = Results.apply(lambda x: sorted([x['WTeamID'], x['LTeamID']])[1], axis=1)


Results['IDTeam1'] = Results.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
Results['IDTeam2'] = Results.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)


# Map seeds to Team1 and Team2 (separate columns)
Results['team1Seed'] = Results['IDTeam1'].map(seeds).fillna(Results['IDTeam1'].map(mode_seed)).fillna(min_seed)
Results['team2Seed'] = Results['IDTeam2'].map(seeds).fillna(Results['IDTeam2'].map(mode_seed)).fillna(min_seed)


# Calculate differences
Results['diffSeed'] = Results['team1Seed'] - Results['team2Seed']


# الفرق بين المتتابعات الهجوميه والاهداف 
#the differance between the WOR and WFGM
Results['WOR-WFGM']=np.abs(Results['WOR'] - Results['WFGM'])
Results['LOR-LFGM']=np.abs(Results['LOR'] - Results['LFGM'])


#الفرق بين التمريرات الحاسمه والاهداف  
#the differance between the WAST and WFGM
Results['WAst-WFGM']=np.abs(Results['WAst'] - Results['WFGM'])
Results['LAst-LFGM']=np.abs(Results['LAst'] - Results['LFGM'])


#الفرق بين الرميات المسجله والرمايات المحاوله 
# the differance between the WFTM and WFTA 
Results['WFTM-WFTA'] = Results['WFTM'] - Results['WFTA']
Results['LFTM-LFTA'] = Results['LFTM'] - Results['LFTA']


#الفرق بين فقدان الكره والاخطاء الشخصيه 
# the differance between the 
Results['WTO-WPF'] = Results['WTO'] - Results['WPF']
Results['LTO-LPF'] = Results['LTO'] - Results['LPF']


#الفرق بين المتتابعات الدفاعيه والتصديات 
# the differance between the WDR and WBlK
Results['WDR-WBlk'] = Results['WDR'] - Results['WBlk']
Results['LDR-LBlk'] = Results['LDR'] - Results['LBlk']



# Extract Global Statistics
c_score_col = [ 'NumOT',
       'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst',
       'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM',
       'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF','WOR-WFGM', 'LOR-LFGM',
       'WAst-WFGM', 'LAst-LFGM', 'WFTM-WFTA', 'LFTM-LFTA', 'WTO-WPF',
       'LTO-LPF', 'WDR-WBlk', 'LDR-LBlk']

c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std']
Global_Statistics =Results.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
Global_Statistics.columns = ['_'.join(c) + '_c_score' for c in Global_Statistics.columns]
mode_Global_Statistics ={k: Global_Statistics[k].mode()[0] for k in Global_Statistics.columns[1:]}
Global_Statistics


#this is the columns that contains nulls values in the Global_Statistics
nulls={k:v for k,v in zip(Global_Statistics.columns , Global_Statistics.isna().sum() ) if v>0 }


# Remove nulls columns from the  Global_Statistics
Global_Statistics.drop(nulls , axis=1 , inplace=True)
Global_Statistics
#Columns containing nulls removed


#Merge the files (Results,Global_Statistics)
Results=pd.merge(Results, Global_Statistics, how='left', left_on='IDTeams', right_on='IDTeams__c_score')
Results['pred_results'] = Results.apply(lambda row: 1 if sorted([row['WTeamID'], row['LTeamID']])[0] == row['WTeamID'] else 0, axis=1)
Results


#drop Columns that not included in the trainig model 
excluded_columns=['Season', 'DayNum','IDTeams', 'IDTeam1', 'WLoc','IDTeam2', 'WTeamID', 'WScore', 'LTeamID',
                                 'LScore', 'NumOT','IDTeams__c_score'] + c_score_col
Results.drop(excluded_columns,axis=1,inplace=True)

# Make lebel Encoder for ID
le=LabelEncoder()
Results['ID']=le.fit_transform(Results['ID'])



sub['IDTeams'] = sub['ID'].apply(lambda x: '_'.join(x.split('_')[1:3]))
sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])
sub['Season']=sub['ID'].map(lambda x:x.split('_')[0])
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)


# Map seeds to Team1 and Team2 (separate columns)
sub['team1Seed'] = sub['IDTeam1'].map(seeds).fillna(sub['IDTeam1'].map(mode_seed)).fillna(min_seed)
sub['team2Seed'] = sub['IDTeam2'].map(seeds).fillna(sub['IDTeam2'].map(mode_seed)).fillna(min_seed)
sub['diffSeed'] = sub['team1Seed'] - sub['team2Seed']
sub=pd.merge(sub, Global_Statistics, how='left', left_on='IDTeams', right_on='IDTeams__c_score')

sub.drop('IDTeams__c_score',axis=1,inplace=True)

#fill ths columns that contain nulls 
for col in sub.columns[1:]:
    if sub[col].isna().sum() > 0:
        sub[col].fillna(mode_Global_Statistics[col], inplace=True)


#drop excluded columns
excluded_test_columns=['Season', 'IDTeams', 'IDTeam1','IDTeam2'] 
sub.drop(excluded_test_columns,axis=1,inplace=True)


#Make Label Encoding For Train Data
le=LabelEncoder()
sub['ID']=le.fit_transform(sub['ID'])
sub.head()


X_train=Results.drop('pred_results',axis=1)
y_train=Results['pred_results']

base_model = RandomForestClassifier(n_estimators=235,
          random_state=42,
          max_depth=20,
          n_jobs=-1)
base_model.fit(X_train,y_train)


# Probability calibration
X_test=sub.drop('Pred',axis=1)
platt_scaler = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
platt_scaler.fit(X_train, y_train)
y_probs_platt = platt_scaler.predict_proba(X_test)[:, 1]


sub['Pred']=y_probs_platt
sub.to_csv('submission.csv', index = False)

