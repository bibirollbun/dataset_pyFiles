# Importing Libraries  
from functools import reduce
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe 
from IPython.display import display, HTML  
import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler  
import warnings 
from xgboost import XGBRegressor  

pd.set_option('display.max_columns', None)  
warnings.filterwarnings('ignore')  

HTML("""
<style>
g.pointtext {display: none;}
</style>
""")


# Read All Datasets 
ap_poll_df = pd.read_csv('/kaggle/input/march-madness-data/AP Poll Data.csv')
fte_df = pd.read_csv('/kaggle/input/march-madness-data/538 Ratings.csv')
ban_df = pd.read_csv('/kaggle/input/march-madness-data/Barttorvik Away-Neutral.csv')
ba_df = pd.read_csv('/kaggle/input/march-madness-data/Barttorvik Away.csv')
bh_df = pd.read_csv('/kaggle/input/march-madness-data/Barttorvik Home.csv')
bn_df = pd.read_csv('/kaggle/input/march-madness-data/Barttorvik Neutral.csv')
coach_res_df = pd.read_csv('/kaggle/input/march-madness-data/Coach Results.csv')
conf_res_df = pd.read_csv('/kaggle/input/march-madness-data/Conference Results.csv')
cs_df = pd.read_csv('/kaggle/input/march-madness-data/Conference Stats.csv')
csh_df = pd.read_csv('/kaggle/input/march-madness-data/Conference Stats Home.csv')
csa_df = pd.read_csv('/kaggle/input/march-madness-data/Conference Stats Away.csv')
csn_df = pd.read_csv('/kaggle/input/march-madness-data/Conference Stats Neutral.csv')
csan_df = pd.read_csv('/kaggle/input/march-madness-data/Conference Stats Away Neutral.csv')
em_df = pd.read_csv('/kaggle/input/march-madness-data/EvanMiya.csv')  
hcti_df = pd.read_csv('/kaggle/input/march-madness-data/Heat Check Tournament Index.csv')
kb_df = pd.read_csv('/kaggle/input/march-madness-data/KenPom Barttorvik.csv') 
kp_df = pd.read_csv('/kaggle/input/march-madness-data/KenPom Preseason.csv') 
pp_df = pd.read_csv('/kaggle/input/march-madness-data/Public Picks.csv')
r_df = pd.read_csv('/kaggle/input/march-madness-data/Resumes.csv')
rppf_df = pd.read_csv('/kaggle/input/march-madness-data/RPPF Ratings.csv')
rppf_pr_df = pd.read_csv('/kaggle/input/march-madness-data/RPPF Preseason Ratings.csv')
rppf_conf_df = pd.read_csv('/kaggle/input/march-madness-data/RPPF Conference Ratings.csv')
sr_df = pd.read_csv('/kaggle/input/march-madness-data/Seed Results.csv')
ss_df = pd.read_csv('/kaggle/input/march-madness-data/Shooting Splits.csv')
tres_df = pd.read_csv('/kaggle/input/march-madness-data/Team Results.csv')
tl_df = pd.read_csv('/kaggle/input/march-madness-data/Tournament Locations.csv')
tm_df = pd.read_csv('/kaggle/input/march-madness-data/Tournament Matchups.csv') 
tr_df = pd.read_csv('/kaggle/input/march-madness-data/TeamRankings.csv') 
trh_df = pd.read_csv('/kaggle/input/march-madness-data/TeamRankings Home.csv') 
tra_df = pd.read_csv('/kaggle/input/march-madness-data/TeamRankings Away.csv') 
trn_df = pd.read_csv('/kaggle/input/march-madness-data/TeamRankings Neutral.csv') 
ts_df = pd.read_csv('/kaggle/input/march-madness-data/Tournament Simulation.csv') 
tsr_df = pd.read_csv('/kaggle/input/march-madness-data/Teamsheet Ranks.csv')
uc_df = pd.read_csv('/kaggle/input/march-madness-data/Upset Count.csv')
usi_df = pd.read_csv('/kaggle/input/march-madness-data/Upset Seed Info.csv')
zrt_df = pd.read_csv('/kaggle/input/march-madness-data/Z Rating Teams.csv')
zrc_df = pd.read_csv('/kaggle/input/march-madness-data/Z Rating Cumulative.csv')


# Global Variables 
curr_year = 2026  
prev_year = curr_year - 1   
tournament_count = 16   

order_seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]    

round_str_64 = ['CHAMPION', 'FINALS', 'FINAL 4', 'ELITE 8', 'SWEET 16', 'SECOND ROUND', 'FIRST ROUND'] 
round_str_68 = ['CHAMPION', 'FINALS', 'FINAL 4', 'ELITE 8', 'SWEET 16', 'SECOND ROUND', 'FIRST ROUND', 'FIRST FOUR']  

# round_str_reverse_64 = ['FIRST ROUND', 'SECOND ROUND', 'SWEET 16', 'ELITE 8', 'FINAL 4', 'FINALS', 'CHAMPION']  
# round_str_reverse_68 = ['FIRST FOUR', 'FIRST ROUND', 'SECOND ROUND', 'SWEET 16', 'ELITE 8', 'FINAL 4', 'FINALS', 'CHAMPION'] 

rounds_num_64 = [1, 2, 4, 8, 16, 32, 64] 
rounds_num_68 = [1, 2, 4, 8, 16, 32, 64, 68]


# Combine Datasets to create one comprehensive dataset for all tournament teams 
############################################################################### 

temp_kb_df = kb_df.copy()

temp_bh_df = bh_df.copy()  
# Add Prefix to indicate type of team stats (H = Home, A = Away, N = Neutral, AN = Away Neutral)
temp_bh_df = temp_bh_df.add_prefix('H ')                               
# Rename H TEAM NO column to TEAM NO because it will be the column used to merge 
temp_bh_df = temp_bh_df.rename(columns = {'H TEAM NO' : 'TEAM NO'}) 

temp_ba_df = ba_df.copy()
temp_ba_df = temp_ba_df.add_prefix('A ')
temp_ba_df = temp_ba_df.rename(columns = {'A TEAM NO' : 'TEAM NO'}) 

temp_bn_df = bn_df.copy()
temp_bn_df = temp_bn_df.add_prefix('N ')
temp_bn_df = temp_bn_df.rename(columns = {'N TEAM NO' : 'TEAM NO'}) 

temp_ban_df = ban_df.copy()
temp_ban_df = temp_ban_df.add_prefix('AN ')
temp_ban_df = temp_ban_df.rename(columns = {'AN TEAM NO' : 'TEAM NO'}) 

temp_tr_df = tr_df.copy()
temp_tr_df = temp_tr_df.rename(columns = {'YEAR' : 'TR YEAR', 'TEAM' : 'TR TEAM', 'SEED' : 'TR SEED', 'ROUND' : 'TR ROUND'}) 

temp_trh_df = trh_df.copy()
temp_trh_df = temp_trh_df.add_prefix('H ')
temp_trh_df = temp_trh_df.rename(columns = {'H TEAM NO' : 'TEAM NO', 'H YEAR' : 'TR H YEAR', 'H TEAM' : 'TR H TEAM', 'H SEED' : 'TR H SEED', 'H ROUND' : 'TR H ROUND'}) 

temp_tra_df = tra_df.copy()
temp_tra_df = temp_tra_df.add_prefix('A ')
temp_tra_df = temp_tra_df.rename(columns = {'A TEAM NO' : 'TEAM NO', 'A YEAR' : 'TR A YEAR', 'A TEAM' : 'TR A TEAM', 'A SEED' : 'TR A SEED', 'A ROUND' : 'TR A ROUND'}) 

temp_trn_df = trn_df.copy()
temp_trn_df = temp_trn_df.add_prefix('N ')
temp_trn_df = temp_trn_df.rename(columns = {'N TEAM NO' : 'TEAM NO', 'N YEAR' : 'TR N YEAR', 'N TEAM' : 'TR N TEAM', 'N SEED' : 'TR N SEED', 'N ROUND' : 'TR N ROUND'}) 

temp_hcti_df = hcti_df.copy()
temp_hcti_df = temp_hcti_df.rename(columns = {'YEAR' : 'HCTI YEAR', 'TEAM' : 'HCTI TEAM', 'SEED' : 'HCTI SEED', 'ROUND' : 'HCTI ROUND'}) 

# temp_pv_df = pv_df.copy()
# temp_pv_df = temp_pv_df.rename(columns = {'YEAR' : 'PV YEAR', 'TEAM' : 'PV TEAM', 'SEED' : 'PV SEED', 'ROUND' : 'PV ROUND', 'AP VOTES' : 'PRESEASON AP VOTES', 'AP RANK' : 'PRESEASON AP RANK', 'RANK?' : 'PRESEASON RANK?'}) 

# temp_ap_df = ap_df.copy()
# temp_ap_df = temp_ap_df.rename(columns = {'YEAR' : 'AP YEAR', 'TEAM' : 'AP TEAM', 'SEED' : 'AP SEED', 'ROUND' : 'AP ROUND', 'AP VOTES' : 'WEEK 6 AP VOTES', 'AP RANK' : 'WEEK 6 AP RANK', 'RANK?' : 'WEEK 6 RANK?'}) 

temp_kp_df = kp_df.copy()
temp_kp_df = temp_kp_df.rename(columns = {'YEAR' : 'KP YEAR', 'TEAM' : 'KP TEAM', 'SEED' : 'KP SEED', 'ROUND' : 'KP ROUND'}) 

temp_ss_df = ss_df.copy()
temp_ss_df = temp_ss_df.rename(columns = {'YEAR' : 'SS YEAR', 'TEAM ID' : 'SS TEAM ID', 'TEAM' : 'SS TEAM', 'CONF' : 'SS CONF'}) 

temp_r_df = r_df.copy()
temp_r_df = temp_r_df.rename(columns = {'YEAR' : 'R YEAR', 'SEED' : 'R SEED', 'TEAM' : 'R TEAM', 'ROUND' : 'R ROUND'}) 

temp_em_df = em_df.copy()
temp_em_df = temp_em_df.rename(columns = {'YEAR' : 'EM YEAR', 'SEED' : 'EM SEED', 'TEAM' : 'EM TEAM', 'ROUND' : 'EM ROUND'}) 

temp_tsr_df = tsr_df.copy()
temp_tsr_df = temp_tsr_df.rename(columns = {'YEAR' : 'TSR YEAR', 'SEED' : 'TSR SEED', 'TEAM' : 'TSR TEAM', 'ROUND' : 'TSR ROUND'}) 

temp_ap_poll_df = ap_poll_df.copy()
temp_ap_poll_df = temp_ap_poll_df.rename(columns = {'YEAR' : 'AP POLL YEAR', 'SEED' : 'AP POLL SEED', 'TEAM' : 'AP POLL TEAM', 'ROUND' : 'AP POLL ROUND'})

temp_rppf_pr_df = rppf_pr_df.copy()
temp_rppf_pr_df = temp_rppf_pr_df.rename(columns = {'YEAR' : 'RPPF PF YEAR', 'SEED' : 'RPPF PF SEED', 'TEAM' : 'RPPF PF TEAM', 'ROUND' : 'RPPF PF ROUND'}) 

temp_rppf_df = rppf_df.copy()
temp_rppf_df = temp_rppf_df.rename(columns = {'YEAR' : 'RPPF YEAR', 'SEED' : 'RPPF SEED', 'TEAM' : 'RPPF TEAM', 'ROUND' : 'RPPF ROUND'})

zrt_dfs = {group: data for group, data in zrt_df.groupby('TYPE')}
temp_old_zrt_df = zrt_dfs['OLD'].copy()
temp_old_zrt_df = temp_old_zrt_df.add_prefix('OLD ') 
temp_old_zrt_df = temp_old_zrt_df.rename(columns = {'OLD TEAM NO' : 'TEAM NO'}) 

temp_new_zrt_df = zrt_dfs['NEW'].copy()
temp_new_zrt_df = temp_new_zrt_df.add_prefix('NEW ') 
temp_new_zrt_df = temp_new_zrt_df.rename(columns = {'NEW TEAM NO' : 'TEAM NO'}) 

ap_poll_dfs = {group: data for group, data in ap_poll_df.groupby('WEEK')} 
full_ap_poll_df = pd.DataFrame()

for i in range(1, 22) : 
    temp_ap_poll_df = ap_poll_dfs[i].copy()
    temp_ap_poll_df = temp_ap_poll_df.add_prefix(str(i) + ' ') 
    temp_ap_poll_df = temp_ap_poll_df.rename(columns = {str(i) + ' TEAM NO' : 'TEAM NO'}) 
    temp_ap_poll_df.drop([str(i) + ' YEAR', str(i) + ' SEED', str(i) + ' TEAM', str(i) + ' ROUND', str(i) + ' WEEK'], inplace = True, axis = 1)
    full_ap_poll_df = pd.concat([full_ap_poll_df, temp_ap_poll_df], ignore_index = True, sort = False)

full_ap_poll_df = full_ap_poll_df .groupby('TEAM NO').first().reset_index()
           
dfs = [temp_kb_df, temp_bh_df, temp_ba_df, temp_bn_df, temp_ban_df, temp_hcti_df, temp_kp_df, temp_tr_df, temp_trh_df, temp_tra_df, temp_trn_df, temp_ss_df, temp_r_df, temp_em_df, temp_rppf_df, temp_old_zrt_df, temp_new_zrt_df, full_ap_poll_df, temp_tsr_df, temp_rppf_pr_df]
# temp_pv_df, temp_ap_df  

# Merge all columns to make one dataframe consisting of all team stats 
complete_stats = reduce(lambda left, right: pd.merge(left, right, on = ['TEAM NO'], how = 'left'), dfs)  

# Create new stats  
complete_stats['TOV% DIFF'] = complete_stats['TOV%D'] - complete_stats['TOV%']
complete_stats['A1 TOV% DIFF'] = complete_stats['A TOV%D'] - complete_stats['A TOV%']
complete_stats['AN1 TOV% DIFF'] = complete_stats['AN TOV%D'] - complete_stats['AN TOV%']
complete_stats['A TOV% DIFF'] = complete_stats['TOV% DIFF'] - complete_stats['A1 TOV% DIFF']
complete_stats['AN TOV% DIFF'] = complete_stats['TOV% DIFF'] - complete_stats['AN1 TOV% DIFF']
complete_stats['AN TR RANK'] = (complete_stats['A TR RANK'] * (complete_stats['A GAMES'] / (complete_stats['A GAMES'] + complete_stats['N GAMES']))) + (complete_stats['N TR RANK'] * (complete_stats['N GAMES'] / (complete_stats['A GAMES'] + complete_stats['N GAMES'])))
complete_stats['AN TR RATING'] = (complete_stats['A TR RATING'] * (complete_stats['A GAMES'] / (complete_stats['A GAMES'] + complete_stats['N GAMES']))) + (complete_stats['N TR RATING'] * (complete_stats['N GAMES'] / (complete_stats['A GAMES'] + complete_stats['N GAMES'])))
complete_stats['AN V 1-25 WINS'] = complete_stats['A V 1-25 WINS'] + complete_stats['N V 1-25 WINS']
complete_stats['AN V 1-25 LOSS'] = complete_stats['A V 1-25 LOSS'] + complete_stats['N V 1-25 LOSS']
complete_stats['AN V 26-50 WINS'] = complete_stats['A V 26-50 WINS'] + complete_stats['N V 26-50 WINS']
complete_stats['AN V 26-50 LOSS'] = complete_stats['A V 26-50 LOSS'] + complete_stats['N V 26-50 LOSS']
complete_stats['AN V 51-100 WINS'] = complete_stats['A V 51-100 WINS'] + complete_stats['N V 51-100 WINS']
complete_stats['AN V 51-100 LOSS'] = complete_stats['A V 51-100 LOSS'] + complete_stats['N V 51-100 LOSS']
complete_stats['AN HI'] = (complete_stats['A HI'] * (complete_stats['A GAMES'] / (complete_stats['A GAMES'] + complete_stats['N GAMES']))) + (complete_stats['N HI'] * (complete_stats['N GAMES'] / (complete_stats['A GAMES'] + complete_stats['N GAMES'])))
complete_stats['AN LO'] = (complete_stats['A LO'] * (complete_stats['A GAMES'] / (complete_stats['A GAMES'] + complete_stats['N GAMES']))) + (complete_stats['N LO'] * (complete_stats['N GAMES'] / (complete_stats['A GAMES'] + complete_stats['N GAMES'])))
complete_stats['AN LAST'] = (complete_stats['A LAST'] * (complete_stats['A GAMES'] / (complete_stats['A GAMES'] + complete_stats['N GAMES']))) + (complete_stats['N LAST'] * (complete_stats['N GAMES'] / (complete_stats['A GAMES'] + complete_stats['N GAMES'])))
complete_stats['3PT% DIFF'] = complete_stats['3PT%'] - complete_stats['3PT%D']
complete_stats['A BADJ EM RANK DIFF'] = complete_stats['BADJ EM RANK'] - complete_stats['A BADJ EM RANK']
complete_stats['AN BADJ EM RANK DIFF'] = complete_stats['BADJ EM RANK'] - complete_stats['AN BADJ EM RANK']
complete_stats['A BADJ O RANK DIFF'] = complete_stats['BADJ O RANK'] - complete_stats['A BADJ O RANK']
complete_stats['AN BADJ O RANK DIFF'] = complete_stats['BADJ O RANK'] - complete_stats['AN BADJ O RANK']
complete_stats['A BADJ D RANK DIFF'] = complete_stats['BADJ D RANK'] - complete_stats['A BADJ D RANK']
complete_stats['AN BADJ D RANK DIFF'] = complete_stats['BADJ D RANK'] - complete_stats['AN BADJ D RANK'] 

# Drop redundant columns  
complete_stats.drop(['H YEAR', 'H TEAM ID', 'H TEAM', 'H SEED', 'H ROUND', 'A YEAR', 'A TEAM ID', 'A TEAM', 'A SEED', 'A ROUND', 
                     'N YEAR', 'N TEAM ID', 'N TEAM', 'N SEED', 'N ROUND', 'AN YEAR', 'AN TEAM ID', 'AN TEAM', 'AN SEED', 'AN ROUND',
                     'TR YEAR', 'TR TEAM', 'TR SEED', 'TR ROUND', 'TR H YEAR', 'TR H TEAM', 'TR H SEED', 'TR H ROUND', 
                     'TR A YEAR', 'TR A TEAM', 'TR A SEED', 'TR A ROUND', 'TR N YEAR', 'TR N TEAM', 'TR N SEED', 'TR N ROUND', 
                     'GAMES', 'W', 'L', 
                     'H GAMES', 'H W', 'H L', 'H TALENT', 'H EXP', 'H AVG HGT', 'H EFF HGT', 'A GAMES', 'A W', 'A L', 'A TALENT', 'A EXP', 'A AVG HGT', 'A EFF HGT', 
                     'N GAMES', 'N W', 'N L', 'N TALENT', 'N EXP', 'N AVG HGT', 'N EFF HGT', 'AN GAMES', 'AN W', 'AN L', 'AN TALENT', 'AN EXP', 'AN AVG HGT', 'AN EFF HGT',
                     'OP OREB%', 'OP DREB%', 'H OP OREB%', 'H OP DREB%', 'A OP OREB%', 'A OP DREB%', 'N OP OREB%', 'N OP DREB%', 'AN OP OREB%', 'AN OP DREB%',
                     'THREES FG%', 'THREES SHARE', 'THREES FG%D', 'THREES D SHARE', 
                     'H TALENT RANK', 'H EXP RANK', 'H AVG HGT RANK', 'H EFF HGT RANK', 'A TALENT RANK', 'A EXP RANK', 'A AVG HGT RANK', 'A EFF HGT RANK', 
                     'N TALENT RANK', 'N EXP RANK', 'N AVG HGT RANK', 'N EFF HGT RANK', 'AN TALENT RANK', 'AN EXP RANK', 'AN AVG HGT RANK', 'AN EFF HGT RANK',
                     'OP OREB% RANK', 'OP DREB% RANK', 
                     'H OP OREB% RANK', 'H OP DREB% RANK', 'A OP OREB% RANK', 'A OP DREB% RANK', 'N OP OREB% RANK', 'N OP DREB% RANK', 'AN OP OREB% RANK', 'AN OP DREB% RANK',
                     'THREES FG% RANK', 'THREES SHARE RANK', 'THREES FG%D RANK', 'THREES D SHARE RANK', 'A1 TOV% DIFF', 'AN1 TOV% DIFF',  
                     'HCTI YEAR', 'HCTI TEAM', 'HCTI SEED', 'HCTI ROUND', 'KP YEAR', 'KP SEED', 'KP TEAM', 'KP ROUND', 
                     'SS YEAR', 'SS TEAM ID', 'SS TEAM', 'SS CONF', 'R YEAR', 'R SEED', 'R TEAM', 'R ROUND', 'WINS', 'EM YEAR', 'EM SEED', 'EM TEAM', 'EM ROUND',
                     # 'AP POLL YEAR', 'AP POLL SEED', 'AP POLL TEAM', 'AP POLL ROUND', 
                     'TSR YEAR', 'TSR SEED', 'TSR TEAM', 'TSR ROUND', 
                     'RPPF YEAR', 'RPPF SEED', 'RPPF TEAM', 'RPPF ROUND', 'RPPF PF YEAR', 'RPPF PF SEED', 'RPPF PF TEAM', 'RPPF PF ROUND', 
                     'OLD YEAR', 'OLD SEED', 'OLD TEAM', 'OLD ROUND', 'OLD TYPE', 
                     'NEW YEAR', 'NEW SEED', 'NEW TEAM', 'NEW ROUND', 'NEW TYPE'], 
                     inplace = True, axis = 1)
                     # 'PV YEAR', 'PV TEAM', 'PV SEED', 'PV ROUND',  
                     # 'AP YEAR', 'AP TEAM', 'AP SEED', 'AP ROUND',

complete_stats = complete_stats.rename(columns = {'WAB RANK_x' : 'WAB RANK'}) 
complete_stats = complete_stats.rename(columns = {'Q1 W_x' : 'Q1 W'}) 
complete_stats = complete_stats.rename(columns = {'Q2 W_x' : 'Q2 W'}) 

# Remove the First Four Round as that is irrelevant to the EDA and Machine Learning Model  
complete_stats = complete_stats.loc[complete_stats['ROUND'].ne(68)]
# complete_stats['PRESEASON RANK?'] = complete_stats['PRESEASON RANK?'].fillna(0)
# complete_stats['WEEK 6 RANK?'] = complete_stats['WEEK 6 RANK?'].fillna(0)
complete_stats.head()  


# Create dataframe of tournament matchups  
#########################################  

temp_complete_stats = complete_stats.drop(['YEAR', 'TEAM', 'SEED', 'ROUND'], axis = 1)
# Create temp dataframe too add sections of it to the tm dataframe 
temp_tournament_matchups = pd.merge(tm_df, temp_complete_stats, how = 'left', on = ['TEAM NO']) 
tournament_matchups = pd.DataFrame()

temp_tournament_matchups['OUTCOME'] = ''
rounds = list(reversed(rounds_num_64))[: - 1] 

# Create Outcome column to indicate winning and losing teams per matchup (0 = Losing Team, 1 = Winning Team)
for rnd in rounds : 
    df = temp_tournament_matchups.loc[temp_tournament_matchups['CURRENT ROUND'] == rnd]
    df = df.reset_index(drop = True) 
    df.loc[df['ROUND'] != rnd, 'OUTCOME'] = 1 
    df.loc[df['ROUND'] == rnd, 'OUTCOME'] = 0
    tournament_matchups = pd.concat([tournament_matchups, df], ignore_index = True, sort = False)
    
tournament_matchups.head()  


# Order the teams in W - L order  
################################  

# Create dataframe of all winning teams 
win = tournament_matchups.loc[tournament_matchups['OUTCOME'] == 1] 
win = win.reset_index(drop = True)   

# Create dataframe of all losing teams 
loss = tournament_matchups.loc[tournament_matchups['OUTCOME'] == 0] 
loss = loss.reset_index(drop = True)   

# Change indexes of rows to have the winning team as the first row and the losing team as the second row per two rows 
win['INDEX'] = np.arange(0, len(win) * 2, 2)
loss['INDEX'] = np.arange(1, len(win) * 2, 2)

# Set the index column as the row index 
win = win.set_index('INDEX')
loss = loss.set_index('INDEX')

# Combine the win and loss dataframes to create the tournament matchups in order 
ordered_tournament_matchups = pd.concat([win, loss], axis = 0)
ordered_tournament_matchups = ordered_tournament_matchups.rename_axis(None, axis = 0)
ordered_tournament_matchups  = ordered_tournament_matchups.sort_index(ascending = True)
ordered_tournament_matchups2 = ordered_tournament_matchups.copy()

ordered_tournament_matchups.head()


# Scale variables between a value of 0 to 1 for the tournament matchups  
scaler = MinMaxScaler() 
removed_cols = ['YEAR', 'BY YEAR NO', 'BY ROUND NO', 'TEAM NO', 'TEAM', 'SEED', 'ROUND', 'CURRENT ROUND', 'CONF', 'CONF ID', 'QUAD NO', 'QUAD ID', 'TEAM ID', 'BID TYPE', 'OUTCOME']   
selected_cols = ordered_tournament_matchups.columns[~ordered_tournament_matchups.columns.isin(removed_cols)]
ordered_tournament_matchups[selected_cols] = scaler.fit_transform(ordered_tournament_matchups[selected_cols])  
ordered_tournament_matchups.head()


# Get difference of variables between winning and losing team 
#############################################################

removed_cols = ['YEAR', 'BY YEAR NO', 'TEAM NO', 'TEAM', 'SEED', 'ROUND', 'CURRENT ROUND', 'CONF', 'CONF ID', 'QUAD NO', 'QUAD ID', 'TEAM ID', 'BID TYPE', 'OUTCOME'] 
selected_cols = tournament_matchups.columns[~tournament_matchups.columns.isin(removed_cols)] 

# Get the difference of every 2 rows 
odds = ordered_tournament_matchups[selected_cols].diff()
# Select all rows of odd index  
odds_df = odds.iloc[1::2]

# Get the difference of every 2 rows and flip the sign 
evens = - ordered_tournament_matchups[selected_cols].diff()
# Shift the values up one row  
evens = evens[selected_cols].shift(- 1)
# Select all rows of even index   
evens_df = evens.iloc[::2]

temp = ordered_tournament_matchups[removed_cols]

# Create dataframe of all matchup differentials 
ordered_differentials = pd.concat([temp, evens_df], axis = 1) 
ordered_differentials = pd.concat([ordered_differentials, odds_df], axis = 0) 
ordered_differentials = ordered_differentials.groupby(level = 0).sum()
ordered_differentials = ordered_differentials.sort_index(ascending = True)

ordered_differentials.head()


# Put the winning and losing teams in one row   
#############################################

win = ordered_differentials.loc[ordered_differentials['OUTCOME'] == 1] 
win = win.add_prefix('W ')
win = win.reset_index(drop = True)   

loss = ordered_differentials.loc[ordered_differentials['OUTCOME'] == 0] 
loss = loss.add_prefix('L ')
loss = loss.reset_index(drop = True)  

differentials_combined_rows = pd.concat([win, loss], axis = 1)
differentials_combined_rows.head()  


# Alter format of diff to have Team Vs Opposing Team  
####################################################

# Select all winning team rows  
win1 = ordered_differentials.loc[ordered_differentials['OUTCOME'] == 1]
win1 = win1.reset_index(drop = True)    

# Select all losing team rows   
loss1 = ordered_differentials.loc[ordered_differentials['OUTCOME'] == 0]  
# Add "OPP" prefix to the opposing team 
loss1 = loss1.add_prefix('OPP ')
loss1 = loss1.reset_index(drop = True)   

# Repeat the same process as above for the opposite outcomes   
win2 = ordered_differentials.loc[ordered_differentials['OUTCOME'] == 1]
win2 = win2.add_prefix('OPP ')
win2 = win2.reset_index(drop = True)    

loss2 = ordered_differentials.loc[ordered_differentials['OUTCOME'] == 0]  
loss2 = loss2.reset_index(drop = True)   

temp1 = pd.concat([win1, loss1], axis = 1)
temp2 = pd.concat([loss2, win2], axis = 1)

# Create secondary matchups dataframe     
complete_differentials = pd.concat([temp1, temp2], axis = 0)
complete_differentials = complete_differentials.sort_index().reset_index(drop = True)
complete_differentials.head()


# Display multiple subplots in one row  
def multiple_subplots(figures, titles, h) : 
    fig = make_subplots(rows = 1, cols = len(figures), subplot_titles = titles)
    
    for i, figure in enumerate(figures) :
        for trace in range(len(figure['data'])) :
            fig.append_trace(figure['data'][trace], row = 1, col = i + 1)
    
    fig.update_layout(height = h, template = 'plotly_dark')  
    return fig  

# Order dataframe by a column  
def order_df(df_input, order_by, order) :
    df_output = pd.DataFrame()

    for var in order :
        df_append = df_input[df_input[order_by] == var].copy()
        df_output = pd.concat([df_output, df_append])

    return df_output

# Change Round values to values between 0 - 7  
def change_rounds_num(df, rounds_num) : 
    for i, r in enumerate(rounds_num) : 
        df.loc[df['ROUND'] == r, 'ROUND'] = i    

# Change Round values to strings  
def change_round_str(df, rounds_num, round_str) : 
    for i, r in enumerate(rounds_num) : 
        df.loc[df['ROUND'] == r, 'ROUND'] = round_str[i]      


# sadadsd 


cd = complete_differentials.copy()
cd = cd.drop(['SCORE', 'OPP SCORE'], axis = 1)
cd


selected_cols = cd.columns[~cd.columns.isin(cd.iloc[:, cd.columns.get_loc('OPP YEAR') : cd.columns.get_loc('OPP OUTCOME') + 1])]
features = cd[selected_cols]
features = features.dropna()
features = features.iloc[:, cd.columns.get_loc('K TEMPO') :]
predictors = features.columns
features.head()


train = cd.loc[cd['YEAR'].le(curr_year - 1)]
# test = cd.loc[cd['YEAR'].eq(curr_year - 1)]
test = cd.loc[cd['YEAR'].ge(2019)]
train_target = train['OUTCOME'].astype(int)
test_target = test['OUTCOME'].astype(int)


corr_train  = train[predictors].copy()
corr_train['ROUND'] = train['ROUND']
corr_train['CURRENT ROUND'] = train['CURRENT ROUND']
corr_train = corr_train.loc[corr_train['CURRENT ROUND'].eq(64)]
cor = corr_train.corr()
cor.head()


predictors_string = ['BADJ EM', 'BADJ O', 'BADJ D', 'AN BADJ EM', 'AN BADJ O', 'AN BADJ D', 'BARTHAG', 'BADJ T', 
                     'A BADJ EM RANK DIFF', 'AN BADJ EM RANK DIFF', 'A BADJ O RANK DIFF', 'AN BADJ O RANK DIFF', 'A BADJ D RANK DIFF', 'AN BADJ D RANK DIFF', 
                     'KADJ EM', 'KADJ O', 'KADJ D', 'KADJ T', 'PRESEASON KADJ EM RANK'
                     'WAB', 'AN WAB', 'ELITE SOS', 'TALENT', 'EXP', 'EFF HGT', 'RESUME', 'B POWER', 'PLUS 500', 
                     'POWER-PATH', 'POOL VALUE', 'POOL S-RANK', 'NCAA S-RANK', 
                     'RELATIVE RATING', 'O RATE', 'D RATE', 'OPPONENT ADJUST', 'PACE ADJUST', 'TRUE TEMPO',
                     'TR RANK', 'AN TR RANK', 'HI', 'LO', 'AN HI', 'AN LO', 'SOS RATING', 'LUCK RATING', 'CONSISTENCY TR RATING', 
                     'OREB%', 'AN OREB%', 
                     'EFG%', 'EFG%D', '2PT%', '2PT%D', '3PT%', '3PT%D', '3PTR', '3PTRD', '3PT% DIFF', 'FT%', 'FTR', 'FTRD',  
                     'TOV%', 'TOV%D', 'AN TOV%', 'AN TOV%D', 'TOV% DIFF', 
                     'WIN%', 'AN WIN%',
                     'Q1 W', 'Q1 PLUS Q2 W', 'Q3 Q4 W', 
                     'V 1-25 WINS', 'V 1-25 LOSS', 'V 26-50 WINS', 'V 26-50 LOSS', 'V 51-100 WINS', 'V 51-100 LOSS'] 

opp_prefix = ['OPP ' + sub for sub in predictors_string] 
predictors_string.extend(opp_prefix)

predictors = predictors[predictors.isin(predictors_string)] 


threshold = 0.35                    
a = abs(cor['ROUND'])
result = a[a > threshold]
result = result.sort_values(ascending = False)
print(result)

predictors = result.index
predictors = predictors.drop(['ROUND'])
print('\n', predictors.tolist())


space = {'n_estimators': 1000,
         'learning_rate': 0.05,
         'max_depth': hp.quniform('max_depth', 7, 10, 1),
         'min_child_weight' : hp.quniform('min_child_weight', 0, 5, 1),
         'colsample_bytree' : hp.uniform('colsample_bytree', 0.7, 0.9),
         'gamma': hp.uniform ('gamma', 0, 0.2),
         'reg_alpha' : hp.uniform('reg_alpha', 0, 0.1),
         'reg_lambda' : hp.uniform('reg_lambda', 0.1, 0.9)}


def objective(space) :
    clf = XGBRegressor(n_estimators = int(space['n_estimators']), learning_rate = float(space['learning_rate']),
                       max_depth = int(space['max_depth']),  min_child_weight = int(space['min_child_weight']),
                       colsample_bytree = float(space['colsample_bytree']),
                       gamma = float(space['gamma']), reg_alpha = float(space['reg_alpha']), reg_lambda = float(space['reg_lambda']),
                       booster = 'gbtree', objective = 'binary:logistic', eval_metric = 'auc', early_stopping_rounds = 10, seed = 0)

    evaluation = [(train[predictors], train_target), (test[predictors], test_target)]
    clf.fit(train[predictors], train_target, eval_set = evaluation, verbose = False)

    pred = clf.predict(test[predictors])
    accuracy = accuracy_score(test_target, pred > 0.5)
    print ('SCORE :', accuracy)
    return {'loss': - accuracy, 'status': STATUS_OK}


# trials = Trials()
# best_hyperparams = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 100, trials = trials)


# - 0.9955654101995566


# print("The best hyperparameters are : ","\n")
# print(best_hyperparams)


xgb = XGBRegressor(n_estimators = 1000, learning_rate = 0.005,
                   max_depth = 9, min_child_weight = 2.0,
                   colsample_bytree = 0.7585876968533232,
                   gamma = 0.09104055533987562, reg_alpha = 0.6142449094724537, reg_lambda = 0.051184704909705156,
                   booster = 'gbtree', objective = 'binary:logistic', seed = 0)
xgb.fit(train[predictors], train_target)


# xgb = XGBRegressor(n_estimators = 1000, learning_rate = 0.005,
                   # max_depth = int(best_hyperparams['max_depth']), min_child_weight = float(best_hyperparams['min_child_weight']),
                   # colsample_bytree = float(best_hyperparams['colsample_bytree']),
                   # gamma = float(best_hyperparams['gamma']), reg_alpha = float(best_hyperparams['reg_alpha']), reg_lambda = float(best_hyperparams['reg_lambda']),
                   # booster = 'gbtree', objective = 'binary:logistic', seed = 0)
# xgb.fit(train[predictors], train_target)


preds = xgb.predict(test[predictors])
preds[:]


accuracy = accuracy_score(test_target, preds > 0.5)
print(accuracy)


# 0.9944567627494457


def get_leverage(winner, team, game, game_pred, seed_diff) : 
    #              1     2      3     4    5    6     7      8      9     10    11    12    13   14   15
    win_prob = [0.475, 0.45, 0.425, 0.4, 0.4, 0.4, 0.375, 0.375, 0.35, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5]
    
    # If the seed differential is a specified number 
    if (abs(game['SEED'][team] - game['SEED'][team + 1] == seed_diff)) :
        # If the first team's seed is greater than the second team's seed 
        if game['SEED'][team] > game['SEED'][team + 1] :
            # The first team is the winner if they surpass the altered win probability threshold 
            if game_pred[0] > win_prob[seed_diff - 1] : winner = game.loc[team]
        else :
            # The second team is the winner if they surpass the altered win probability threshold 
            if game_pred[1] > win_prob[seed_diff - 1] : winner = game.loc[team + 1]

    return winner 


def simulate_tourney(round_df, mode) :
    removed_cols = ['YEAR', 'BY YEAR NO', 'TEAM NO', 'TEAM', 'SEED', 'ROUND', 'CURRENT ROUND', 'CONF', 'CONF ID', 'QUAD NO', 'QUAD ID', 'TEAM ID', 'BID TYPE', 'OUTCOME', 'TYPE']
    selected_cols = round_df.columns[~round_df.columns.isin(removed_cols)]
    game_count = 32
    curr_df = round_df.copy()

    for i in range(6) :
        team = 0
        next_round = pd.DataFrame()

        # Get the team's stats (not including the opposition team)
        if i != 0 : round_df = round_df.iloc[:, : round_df.columns.get_loc('OPP YEAR')]
        # Replace all N/A values with 0 values
        round_df = round_df.fillna(0)
        # Set the TYPE value to 0 for even indexes and 1 for odd indexes
        round_df['TYPE'] = np.where(round_df.index % 2 == 0, 0, 1)

        # Get the difference of every stat for every two rows
        odds = round_df[selected_cols].diff()
        # Select all odd indexes
        odds_df = odds.iloc[1::2]

        # Get the difference of every stat for every two rows
        evens = - round_df[selected_cols].diff()
        # Shift the values up one row
        evens = evens[selected_cols].shift(- 1)
        # Select all even indexes
        evens_df = evens.iloc[::2]

        temp = round_df[removed_cols]

        # Combine the temp, odds_df, and evens_df to make one dataframe consisting of all tournament matchup differentials
        round_df = pd.concat([temp, evens_df], axis = 1)
        round_df = pd.concat([round_df, odds_df], axis = 0)
        round_df = round_df.groupby(level = 0).sum()
        round_df = round_df.sort_index(ascending = True)

        # Select the teams listed first in the matchup
        win1 = round_df.loc[round_df['TYPE'] == 1]
        win1 = win1.reset_index(drop = True)

        # Select the teams listed second in the matchup
        loss1 = round_df.loc[round_df['TYPE'] == 0]
        # Give every column from the second team the "OPP" prefix to represent the opposing team
        loss1 = loss1.add_prefix('OPP ')
        loss1 = loss1.reset_index(drop = True)

        win2 = round_df.loc[round_df['TYPE'] == 1]
        win2 = win2.add_prefix('OPP ')
        win2 = win2.reset_index(drop = True)

        loss2 = round_df.loc[round_df['TYPE'] == 0]
        loss2 = loss2.reset_index(drop = True)

        # Combine dataframes to create all tournament matchups
        temp1 = pd.concat([win1, loss1], axis = 1)
        temp2 = pd.concat([loss2, win2], axis = 1)

        round_df = pd.concat([temp1, temp2], axis = 0)
        round_df = round_df.sort_index().reset_index(drop = True)

        for i in range(int(game_count)) :
            # Get the specific matchup
            game = round_df.loc[team : team + 1]

            # Predict the matchup's win probability
            game_pred = xgb.predict(game[predictors])

            # Select the winner of the matchup
            if game_pred[0] > game_pred[1] : winner = game.loc[team]
            else : winner = game.loc[team + 1]

            if game_pred[0] > game_pred[1] : winner = game.loc[team]
            else : winner = game.loc[team + 1]

            # Alternate method of selecting the winner of the matchup
            if mode == 1 :
                seed_diff = abs(game.loc[team, 'SEED'] - game.loc[team + 1, 'SEED']) 
                if seed_diff != 0 : winner = get_leverage(winner, team, game, game_pred, int(seed_diff))
                else : 
                    if game_pred[0] > game_pred[1] : winner = game.loc[team]
                    else : winner = game.loc[team + 1]

                    if game_pred[0] > game_pred[1] : winner = game.loc[team]
                    else : winner = game.loc[team + 1]

            winner = winner.to_frame().T
            winner = curr_df[curr_df['TEAM'].isin(winner['TEAM'])]
            next_round = pd.concat([next_round, winner], axis = 0)
            team += 2

        next_round = next_round.reset_index(drop = True)

        odds = next_round.iloc[1::2]
        evens = next_round.iloc[::2]

        if game_count != 1 :
            odds = odds.set_index(np.arange(0, game_count, 2))
            evens = evens.set_index(np.arange(1, game_count, 2))

        # Create new dataframe to display the outcomes of the current round
        opp = pd.concat([evens, odds], axis = 0)
        opp = opp.add_prefix('OPP ')
        opp = opp.sort_index(ascending = True)
        next_round = pd.concat([next_round, opp], axis = 1)
        next_round = next_round.reset_index(drop = True)
        next_round['CURRENT ROUND'] = game_count
        round_df = next_round.copy()
        # Reduce game count by half as the subsequent round has half the number of games as the current round
        game_count /= 2

        # Display the outcomes of the current round
        display(round_df[['YEAR', 'TEAM', 'SEED', 'CURRENT ROUND']])


curr_matchups = ordered_tournament_matchups.copy()
# Get the current tournament matchups
curr_matchups = curr_matchups.loc[curr_matchups['YEAR'].eq(curr_year) & curr_matchups['CURRENT ROUND'].eq(64)]

# Get the matchups in a specific order in order for the simulation to be able to read the dataframe properly
curr_tourney = ts_df.copy()
curr_tourney = curr_tourney.drop(['YEAR', 'TEAM NO', 'SEED', 'ROUND', 'CURRENT ROUND'], axis = 1)
curr_tourney = pd.merge(curr_tourney, curr_matchups, on = 'BY YEAR NO', how = 'left')
curr_tourney = curr_tourney.drop(['TEAM_y'], axis = 1)
curr_tourney = curr_tourney.rename({'TEAM_x' : 'TEAM'}, axis = 1)
curr_tourney.head()


# Run the tournament simulation
simulate_tourney(curr_tourney, 0)


# Run the upset tournament simulation        
simulate_tourney(curr_tourney, 1)

