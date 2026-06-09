# Importing Libraries  
from functools import reduce
from IPython.display import display, HTML  
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go 
import plotly.io as pio  
import plotly.offline as py   
from plotly.subplots import make_subplots 
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler 
import warnings 

pio.renderers.default = 'iframe' 
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

tournament_matchups = tournament_matchups.drop_duplicates()
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
removed_cols = ['YEAR', 'BY YEAR NO', 'TEAM NO', 'TEAM', 'SEED', 'ROUND', 'CURRENT ROUND', 'CONF', 'CONF ID', 'QUAD NO', 'QUAD ID', 'TEAM ID', 'BID TYPE', 'OUTCOME']   
selected_cols = ordered_tournament_matchups.columns[~ordered_tournament_matchups.columns.isin(removed_cols)]
ordered_tournament_matchups[selected_cols] = scaler.fit_transform(ordered_tournament_matchups[selected_cols])  
ordered_tournament_matchups.head()


# Scale variables between a value of 0 to 1 for statistics of all teams  
complete_stats_scaled = complete_stats.copy()
selected_cols = complete_stats_scaled.columns[~complete_stats_scaled.columns.isin(removed_cols)] 
complete_stats_scaled[selected_cols] = scaler.fit_transform(complete_stats_scaled[selected_cols])  
complete_stats_scaled.head()


# Put the winning and losing teams in one row  
#############################################

odds = ordered_tournament_matchups.copy()
# Get all rows of odd index 
odds_df = odds.iloc[1::2]
# Change the columns to have the prefix "L" which indicates the losing teams 
odds_df = odds_df.add_prefix('L ')
odds_df = odds_df.reset_index(drop = True)

evens = ordered_tournament_matchups.copy()
# Get all rows of evens index  
evens_df = evens.iloc[::2]
# Change the columns to have the prefix "W" which indicates the winning teams  
evens_df = evens_df.add_prefix('W ')
evens_df = evens_df.reset_index(drop = True)

# Combine the odds and evens dataframes to create the combined tournament matchups  
tournament_matchups_combined_rows = pd.concat([evens_df, odds_df], axis = 1) 
tournament_matchups_combined_rows = tournament_matchups_combined_rows.sort_index(ascending = True)
tournament_matchups_combined_rows.head()


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


# pd.set_option('display.max_rows', None)  


# dasdas 


# Create dataframe to get WIN % of teams when given specific parameters
def make_wins_df(df, mode) :
    wins_df2 = df.copy()
    wins_df2 = wins_df2[wins_df2['ROUND'].ne(68)]

    # The 2021 tournament was all played in the same time zone which skews the data; removing it normalizes the data
    if mode == 1 : wins_df2 = wins_df2[wins_df2['YEAR'].ne(2021)]

    # Get the count of wins and losses per seed and round
    if mode == 0 : wins_df2 = wins_df2.groupby(by = ['SEED', 'CURRENT ROUND_x', 'OUTCOME']).size().reset_index(name = 'TOTAL W/L')
    # Get the count of wins and losses per seed, round, and time zones crossed value
    elif mode == 1 : wins_df2 = wins_df2.groupby(by = ['SEED', 'CURRENT ROUND_x', 'TIME ZONES CROSSED VALUE', 'OUTCOME']).size().reset_index(name = 'TOTAL W/L')

    temp_curr_round = [2, 4, 8, 16, 32, 64, 100]
    temp_tz_val = [- 2, 1, 2]
    wins_perc_arr, temp_round_arr, seed_arr, tz_arr = [], [], [], []

    # Get the Average Win % of seeds by round
    if mode == 0 :
        for seed in order_seed :
            for rnd in temp_curr_round :
                temp_df = wins_df2.copy()

                # Round 100 represents the total of all rounds; it is not an actual round
                if rnd != 100 :
                    temp_df = temp_df.loc[temp_df['SEED'].isin([seed]) & temp_df['CURRENT ROUND_x'].isin([rnd])]

                    if 1 in temp_df['OUTCOME'].values :
                        oc1_df = temp_df.loc[temp_df['OUTCOME'].isin([1])]
                        # Get the total win % of seeds by round
                        wins_perc_arr.append(oc1_df['TOTAL W/L'].sum() / temp_df['TOTAL W/L'].sum() * 100)
                    else :
                        wins_perc_arr.append(0)
                else :
                    temp_df = temp_df.loc[temp_df['SEED'].isin([seed])]
                    oc1_df = temp_df.loc[temp_df['OUTCOME'].isin([1])]
                    # Get the total win % of seeds
                    wins_perc_arr.append(oc1_df['TOTAL W/L'].sum() / temp_df['TOTAL W/L'].sum() * 100)

                temp_round_arr.append(rnd)
                seed_arr.append(seed)
    # Get the Average Win % of seeds by round and time zones crossed value
    elif mode == 1 :
        for seed in order_seed :
            for rnd in temp_curr_round :
                for val in temp_tz_val :
                    temp_df = wins_df2.copy()

                    if rnd != 100 :
                        # Create a dataframe based on the time zones crossed value conditions
                        if val == - 2 : temp_df = temp_df.loc[temp_df['SEED'].isin([seed]) & temp_df['CURRENT ROUND_x'].isin([rnd]) & temp_df['TIME ZONES CROSSED VALUE'].le(val)]
                        elif val == 1 : temp_df = temp_df.loc[temp_df['SEED'].isin([seed]) & temp_df['CURRENT ROUND_x'].isin([rnd]) & (temp_df['TIME ZONES CROSSED VALUE'].ge(- val) & temp_df['TIME ZONES CROSSED VALUE'].le(val))]
                        elif val == 2 : temp_df = temp_df.loc[temp_df['SEED'].isin([seed]) & temp_df['CURRENT ROUND_x'].isin([rnd]) & temp_df['TIME ZONES CROSSED VALUE'].ge(val)]

                        if 1 in temp_df['OUTCOME'].values :
                            oc1_df = temp_df.loc[temp_df['OUTCOME'].isin([1])]
                            # Get the total win % of seeds by round and time zones crossed value
                            wins_perc_arr.append(oc1_df['TOTAL W/L'].sum() / temp_df['TOTAL W/L'].sum() * 100)
                        else :
                            wins_perc_arr.append(0)
                    else :
                        if val == - 2 : temp_df = temp_df.loc[temp_df['SEED'].isin([seed]) & temp_df['TIME ZONES CROSSED VALUE'].le(val)]
                        elif val == 1 : temp_df = temp_df.loc[temp_df['SEED'].isin([seed]) & (temp_df['TIME ZONES CROSSED VALUE'].ge(- val) & temp_df['TIME ZONES CROSSED VALUE'].le(val))]
                        elif val == 2 : temp_df = temp_df.loc[temp_df['SEED'].isin([seed]) & temp_df['TIME ZONES CROSSED VALUE'].ge(val)]
                        oc1_df = temp_df.loc[temp_df['OUTCOME'].isin([1])]
                        # Get the total win % of seeds by time zones crossed value
                        wins_perc_arr.append(oc1_df['TOTAL W/L'].sum() / temp_df['TOTAL W/L'].sum() * 100)

                    temp_round_arr.append(rnd)
                    seed_arr.append(seed)
                    tz_arr.append(val)

    temp_round_str = ['FINALS', 'FINAL 4', 'ELITE 8', 'SWEET 16', 'SECOND ROUND', 'FIRST ROUND', 'TOTAL']

    wins_perc_df = pd.DataFrame()
    wins_perc_df['SEED'] = seed_arr
    wins_perc_df['SEED'] = wins_perc_df['SEED'].astype(str)
    wins_perc_df['ROUND'] = temp_round_arr
    change_round_str(wins_perc_df, temp_curr_round, temp_round_str)   # Change Round values to strings
    wins_perc_df['ROUND'] = wins_perc_df['ROUND'].astype(str)
    wins_perc_df['WIN%'] = wins_perc_arr

    # Add time zones crossed value to the dataframe
    if mode == 1 :
        wins_perc_df['TIME ZONE VAL'] = tz_arr

    return wins_perc_df


# Create dataframe consisting of the seeds' WIN % by time zone
##############################################################

tl = tl_df.copy()
tl = tl.drop(columns = ['YEAR', 'TEAM NO', 'TEAM', 'SEED', 'ROUND'])
# Combine the tournament matchups and locations dataframes to get the time zones of each matchup
tz = pd.merge(tm_df, tl, on = 'BY YEAR NO', how = 'left')

tz = tz.sort_values(by = ['BY YEAR NO'], ascending = False)
tz = tz.reset_index(drop = True)
temp_df = ordered_tournament_matchups.copy()
temp_df = temp_df.sort_values(by = ['BY YEAR NO'], ascending = False)
temp_df = temp_df.reset_index(drop = True)
# Copy the OUTCOME column to the time zone dataframe
tz['OUTCOME'] = temp_df['OUTCOME']
curr_tz = tz.loc[tz['YEAR'].eq(curr_year)]
tz = tz.loc[tz['YEAR'].ne(curr_year)]

# Get the win percent of seeds by round and time zone
tz_wins = make_wins_df(tz, 1)
tz_wins['TIME ZONE VAL'] = tz_wins['TIME ZONE VAL'].astype(str)
tz_wins = tz_wins.replace({'TIME ZONE VAL' : {'-2' : '2+ Time Zones West', '1' : 'Within 1 Time Zone', '2' : '2+ Time Zones East'}})


tz_wins_fr = tz_wins.loc[tz_wins['ROUND'].eq('FIRST ROUND')]
tz_wins_fr1 = tz_wins_fr.iloc[9:33]

fig = px.bar(tz_wins_fr1, x = 'SEED', y = 'WIN%', color = 'TIME ZONE VAL', barmode = 'group', template = 'plotly_dark')
fig.update_layout(title = '<b> Win % by Time Zone in First Round (4 - 11 Seeds) </b>', title_x = 0.5, title_font = dict(size = 20),
                  yaxis_title = 'WIN %', height = 550)
fig.show()


tz_wins_sr = tz_wins.loc[tz_wins['ROUND'].eq('SECOND ROUND')]
tz_wins_sr1 = tz_wins_sr.iloc[15 : 36]

fig = px.bar(tz_wins_sr1, x = 'SEED', y = 'WIN%', color = 'TIME ZONE VAL', barmode = 'group', template = 'plotly_dark')
fig.update_layout(title = '<b> Win % by Time Zone in Second Round (6 - 12 Seeds) </b>', title_x = 0.5, title_font = dict(size = 20),
                  yaxis_title = 'WIN %', height = 550)
fig.show()


tz_wins_ss = tz_wins.loc[tz_wins['ROUND'].eq('SWEET 16')]
tz_wins_ss1 = tz_wins_ss.iloc[:21]

fig = px.bar(tz_wins_ss1, x = 'SEED', y = 'WIN%', color = 'TIME ZONE VAL', barmode = 'group', template = 'plotly_dark')
fig.update_layout(title = '<b> Win % by Time Zone in Sweet 16 (1 - 7 Seeds) </b>', title_x = 0.5, title_font = dict(size = 20),
                  yaxis_title = 'WIN %', height = 550)
fig.show()


# tv_num = [3, 2, 1, 0, - 1, - 2, - 3]
# curr_tz = order_df(df_input = curr_tz, order_by = 'TIME ZONES CROSSED VALUE', order = tv_num)
# curr_tz = curr_tz.replace({'TIME ZONES CROSSED VALUE' : {- 3 : '3 Time Zones West', - 2 : '2 Time Zones West', - 1 : '1 Time Zones West', 0 : 'No Time Zone Change',
#                                                            3 : '3 Time Zones East', 2 : '2 Time Zones East', 1 : '1 Time Zones East'}})
# curr_tz = curr_tz[['YEAR', 'TEAM', 'SEED', 'CURRENT ROUND', 'TIME ZONES CROSSED VALUE', 'DISTANCE (KM)', 'DISTANCE (MI)']]
# curr_tz_64 = curr_tz.loc[curr_tz['CURRENT ROUND'].eq(64)]
# curr_tz_64 = curr_tz_64.drop_duplicates(subset = ['TEAM'])
# curr_tz_64.head(68)


# curr_tz_32 = curr_tz.loc[curr_tz['CURRENT ROUND'].eq(32)]
# curr_tz_32 = curr_tz_32.drop_duplicates(subset = ['TEAM'])
# curr_tz_32.head(68)


#  curr_tz_16 = curr_tz.loc[curr_tz['CURRENT ROUND'].eq(16)]
# curr_tz_16 = curr_tz_16.drop_duplicates(subset = ['TEAM'])
# curr_tz_16.head(68)


# Displays the plots for each seed given a statistic
####################################################

mod_rounds_num = [1, 2, 4, 8, 16, 32, 64, 0]
mod_round_str = ['CHAMPION', 'FINALS', 'FINAL 4', 'ELITE 8', 'SWEET 16', 'SECOND ROUND', 'FIRST ROUND', '2025 TEAMS']

def display_plots(df, curr_df, y_col, x0, x1, mode) :
    temp_df = df.copy()
    temp_df['SEED'] = temp_df['SEED'].astype(str)
    curr_df['SEED'] = curr_df['SEED'].astype(str)
    fig_arr = []

    if 'WAB' in y_col :
        if 'WAB RANK' in y_col : pass
        # Remove the 2021 tournament year for WAB visualization as teams did not play all games that year
        else : temp_df = temp_df.loc[temp_df['YEAR'].ne(2021)]

    fig = px.scatter(temp_df, x = 'ROUND', y = y_col, title = '<b> Past Tournament Teams </b>', hover_data = ['YEAR', 'TEAM'],
                     animation_frame = 'SEED', color = 'SEED', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark',
                     height = 650)
    fig['layout'].pop('updatemenus')
    fig.update_layout(title_x = 0.5, showlegend = False)
    fig.update_xaxes(categoryorder = 'array', categoryarray = mod_round_str, autorange = 'reversed')
    fig.add_vrect(x0 = 6.5, x1 = 7.5, fillcolor = 'green', opacity = 0.25, line_width = 0)
    fig.add_vline(x = 6.5)

    # Changing the value for the upper and lower y axis value depending on the stat for the purpose of making the plots clearer
    if 'R SCORE' in y_col : val = 0.5
    elif 'DRAW' in y_col : val = 0.5
    elif 'VAL Z-SCORE' in y_col : val = 0.5
    elif 'BARTHAG' in y_col : val = 0.02
    elif y_col == 'EXP' or 'HGT' in y_col : val = 0.2
    elif 'PPP' in y_col : val = 0.02
    elif ' RANK' in y_col :
        if 'WAB RANK' : val = 2
        else : val = 20
    else : val = 2

    go.Figure(data = fig.data, frames = [fr.update(layout =
             {'xaxis': {'range' : [7.5, - 0.5]},
              'yaxis': {'range' : [min(fr.data[0].y) - val, max(fr.data[0].y) + val]},}) for fr in fig.frames], layout = fig.layout)
    fig_arr.append(fig)

    fig = px.scatter(curr_df, x = 'SEED', y = y_col, title = '<b>' + str(curr_year) + ' Teams </b>', hover_data = ['YEAR', 'TEAM'],
                     color = 'SEED', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark',
                     height = 550)
    fig.update_layout(title_x = 0.5)

    green_arr, orange_arr, red_arr, max_arr, min_arr = [], [], [], [], []
    temp_df['SEED'] = temp_df['SEED'].astype(int)

    # Display the red zone as lower values and the green zone as higher values
    if mode == 'forward' :
        for seed in range(len(x0)) :
            seed_df = temp_df.loc[temp_df['SEED'].isin([seed + 1])]
            green_arr.append(seed_df[y_col].quantile(0.75))
            red_arr.append(seed_df[y_col].quantile(0.25))
            max_arr.append(seed_df[y_col].max())
            min_arr.append(seed_df[y_col].min())
        for i in range(len(x0)) :
            fig.add_hrect(x0 = x0[i], x1 = x1[i], y0 = green_arr[i], y1 = max_arr[i], line_width = 0, fillcolor = 'green', opacity = 0.2)
            fig.add_hrect(x0 = x0[i], x1 = x1[i], y0 = red_arr[i], y1 = green_arr[i], line_width = 0, fillcolor = 'orange', opacity = 0.2)
            fig.add_hrect(x0 = x0[i], x1 = x1[i], y0 = min_arr[i], y1 = red_arr[i], line_width = 0, fillcolor = 'red', opacity = 0.2)
    # Display the red zone as higher values and the green zone as lower values
    elif mode == 'backward' :
        for seed in range(len(x0)) :
            seed_df = temp_df.loc[temp_df['SEED'].isin([seed + 1])]
            green_arr.append(seed_df[y_col].quantile(0.25))
            red_arr.append(seed_df[y_col].quantile(0.75))
            max_arr.append(seed_df[y_col].max())
            min_arr.append(seed_df[y_col].min())
        for i in range(len(x0)) :
            fig.add_hrect(x0 = x0[i], x1 = x1[i], y0 = min_arr[i], y1 = green_arr[i], line_width = 0, fillcolor = 'green', opacity = 0.2)
            fig.add_hrect(x0 = x0[i], x1 = x1[i], y0 = green_arr[i], y1 = red_arr[i], line_width = 0, fillcolor = 'orange', opacity = 0.2)
            fig.add_hrect(x0 = x0[i], x1 = x1[i], y0 = red_arr[i], y1 = max_arr[i], line_width = 0, fillcolor = 'red', opacity = 0.2)

    fig_arr.append(fig)
    return fig_arr 


# Create dataframes necessary to display the scatterplots
#########################################################

stats = complete_stats.copy()
stats = order_df(df_input = stats, order_by = 'SEED', order = order_seed)
change_round_str(stats, mod_rounds_num, mod_round_str)

curr_stats = complete_stats.copy()
curr_stats = curr_stats.loc[curr_stats['YEAR'].eq(curr_year)]
curr_stats = order_df(df_input = curr_stats, order_by = 'SEED', order = order_seed)

# X Value Coordinates for the first colored bar
x0, x1 = [0.025], [0.084425]

# Set the subsequent X Value Coordinates for the other 15 colored bars
for i in range(15) :
    x0.append(x0[i] + 0.059425)
    x1.append(x0[i + 1] + 0.059425)


fig = display_plots(stats, curr_stats, 'PRESEASON KADJ EM RANK', x0, x1, 'forward')
fig[0].show()


def ap_data_df(poll_df, mode) : 
    df_arr = []
    
     # Create dataframe merging the Preseason AP Votes statistics with the KenPom Barttorvik dataframe
    #################################################################################################
    ap = poll_df.copy()
    # ap = ap.dropna()
    curr_ap = ap.loc[ap['YEAR'].eq(curr_year)]
    ap = ap.loc[ap['YEAR'].ne(curr_year)]
    
    # Get the preseason AP votes of all tournament teams
    # ap_count = pd.merge(kb_df, ap, how = 'left', on = ['TEAM NO'])
    ap_count = ap.loc[ap['YEAR'].ne(curr_year) & ap['ROUND'].ne(68)]
    # Replace N/A cells with the value 0
    # ap_count['AP VOTES'] = ap_count['AP VOTES'].fillna(0)
    # ap_count['AP RANK'] = ap_count['AP RANK'].fillna(0)
    # ap_count['RANK?'] = ap_count['RANK?'].fillna(0)
    ap_count = ap_count.fillna(0)
    ap_count['WINS'] = ap_count['ROUND']
    wins = [6, 5, 4, 3, 2, 1, 0]

    # Create array of the average wins and average AP Votes for tournament teams
    ############################################################################
    # Convert the round numbers to win count
    for i in rounds_num_64 :
        ap_count.loc[ap_count['WINS'] == i, 'WINS'] = wins[rounds_num_64.index(i)]
    
    # Get the number of wins for each seed by rank status and preseason AP vote count
    if mode == 1 : 
        ap_avg = ap_count.groupby(by = ['SEED', '1 RANK?', '1 AP VOTES', 'WINS']).size().reset_index(name = 'COUNT')
    elif mode == 6 : 
        ap_avg = ap_count.groupby(by = ['SEED', '6 RANK?', '6 AP VOTES', 'WINS']).size().reset_index(name = 'COUNT')
        
    ap_avg['TOTAL WINS'] = ap_avg['WINS'] * ap_avg['COUNT']
    if mode == 1 : ap_avg['TOTAL AP VOTES'] = ap_avg['1 AP VOTES'] * ap_avg['COUNT']
    elif mode == 6 : ap_avg['TOTAL AP VOTES'] = ap_avg['6 AP VOTES'] * ap_avg['COUNT']
    avg_wins_rank, avg_wins_votes, seed_arr, rank_arr = [], [], [], [],
    avg_wins_rank_df = pd.DataFrame()
    avg_wins_votes_df = pd.DataFrame()
    
    # Get the average wins for each seed by rank status
    for seed in order_seed :
        for rank in range(0, 2) :
            if mode == 1 : temp_ap = ap_avg.loc[ap_avg['SEED'].eq(seed) & ap_avg['1 RANK?'].eq(rank)]
            elif mode == 6 : temp_ap = ap_avg.loc[ap_avg['SEED'].eq(seed) & ap_avg['6 RANK?'].eq(rank)]
            avg_wins_rank.append(temp_ap['TOTAL WINS'].sum() / temp_ap['COUNT'].sum())
            seed_arr.append(seed)
            rank_arr.append('UNRANKED') if rank == 0 else rank_arr.append('RANKED')
    
    # Get the average preseason AP votes for each seed by number of wins
    for seed in order_seed :
        for w in wins :
            temp_ap = ap_avg.loc[ap_avg['SEED'].eq(seed) & ap_avg['WINS'].eq(w)]
            avg_wins_votes.append(temp_ap['TOTAL AP VOTES'].sum() / temp_ap['COUNT'].sum())

    # Create dataframe of the average wins and average AP Votes for tournament teams
    ################################################################################
    avg_wins_rank_df['SEED'] = seed_arr
    avg_wins_rank_df['AVG WINS'] = avg_wins_rank
    avg_wins_rank_df['RANK?'] = rank_arr
    # Get the average wins and preseason AP votes from 1 - 12 seeds as 13 + seeds do not have any preseason AP votes
    avg_wins_rank_df = avg_wins_rank_df.loc[avg_wins_rank_df['SEED'].le(12)]
    
    seed_seq = order_seed * 7
    seed_seq.sort()
    
    avg_wins_votes_df['SEED'] = seed_seq
    avg_wins_votes_df['WINS'] = wins * 16
    avg_wins_votes_df['AVG VOTES'] = avg_wins_votes
    # Get rid of rows with N/A values
    avg_wins_votes_df = avg_wins_votes_df.dropna()

    df_arr.append(ap)
    df_arr.append(curr_ap)
    df_arr.append(avg_wins_rank_df)
    df_arr.append(avg_wins_votes_df)
    
    return df_arr


preseason_df_arr = ap_data_df(complete_stats, 1)

fig = px.histogram(preseason_df_arr[2], x = 'SEED', y = 'AVG WINS',
                   color = 'RANK?', barmode = 'group', template = 'plotly_dark',
                   nbins = 12, height = 500)
fig.update_layout(title = '<b> Average Wins of each Seed (Ranked vs Unranked) </b>', title_x = 0.5, title_font = dict(size = 20),
                  xaxis = dict(tickmode = 'linear'), yaxis_title = 'AVERAGE WINS')
fig.show()


curr_preseason = preseason_df_arr[1]
curr_preseason = curr_preseason.sort_values(by = ['1 AP VOTES'], ascending = False)

fig = px.histogram(curr_preseason, x = 'TEAM', y = '1 AP VOTES',
                   color = '1 RANK?', barmode = 'group', template = 'plotly_dark',
                   height = 500)
fig.update_layout(title = '<b>' + str(curr_year) + ' Teams Receiving AP Preseason Votes </b>', title_x = 0.5, title_font = dict(size = 20),
                  yaxis_title = 'AP VOTES')
fig.update_xaxes(tickangle = - 90)
fig.show()


preseason = preseason_df_arr[0]

pv_str = order_df(df_input = preseason, order_by = 'ROUND', order = rounds_num_68)
change_round_str(pv_str, rounds_num_68, round_str_68)
pv_str = order_df(df_input = pv_str, order_by = 'SEED', order = order_seed)
pv_str['SEED'] = pv_str['SEED'].astype(str)

fig = px.scatter(pv_str, x = 'ROUND', y = '1 AP VOTES', hover_data = ['YEAR', 'TEAM'], animation_frame = 'SEED', height = 600,
                color = 'SEED', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark')
fig['layout'].pop('updatemenus')
fig.update_layout(title = '<b> How Far each Team receiving Preseason AP Votes made it in the Tournament </b>', title_x = 0.5, title_font = dict(size = 15), showlegend = False)
fig.update_xaxes(categoryorder = 'array', categoryarray = round_str_68, autorange = 'reversed')
go.Figure(data = fig.data, frames = [fr.update(layout =
         {'xaxis': {'range' : [6.5, - 0.5]},
          'yaxis': {'range' : [min(fr.data[0].y) - 20, max(fr.data[0].y) + 20]},}) for fr in fig.frames], layout = fig.layout)
fig.show()


avg_wins_votes_df = preseason_df_arr[3]

avg_wins_votes_df['SEED'] = avg_wins_votes_df['SEED'].astype(str)
wins_arr = np.arange(7)
avg_wins_votes_df = order_df(df_input = avg_wins_votes_df, order_by = 'WINS', order = wins_arr)

rev_round_str_64 = round_str_64[:: - 1]

# Turn win count values into round strings
for i, r in enumerate(wins_arr) :
    avg_wins_votes_df.loc[avg_wins_votes_df['WINS'] == r, 'WINS'] = rev_round_str_64[i]

fig = px.bar(avg_wins_votes_df, x = 'WINS', y = 'AVG VOTES', animation_frame = 'SEED', height = 600,
             color = 'SEED', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark')
fig['layout'].pop('updatemenus')
fig.update_layout(title = '<b> Average Preseason AP Votes by Round and Seed </b>', title_x = 0.5, title_font = dict(size = 20), showlegend = False)
fig.update_xaxes(title = 'ROUND', categoryorder = 'array', categoryarray = rev_round_str_64)
go.Figure(data = fig.data, frames = [fr.update(layout =
         {'xaxis': {'range' : [- 0.5, 6.5]},
          'yaxis': {'range' : [min(fr.data[0].y) - 20, max(fr.data[0].y) + 20]},}) for fr in fig.frames], layout = fig.layout)
fig.show()


curr_preseason['1 RANK?'] = curr_preseason['1 RANK?'].astype(str)

fig = px.scatter(curr_preseason, x = '1 AP VOTES', y = 'SEED', text = 'TEAM', hover_data = ['YEAR', '1 AP RANK'],
                   color = '1 RANK?', template = 'plotly_dark',
                   height = 600)
fig.update_layout(title = '<b> AP Votes of ' + str(curr_year) + ' Teams </b>', title_x = 0.5, title_font = dict(size = 20), yaxis = dict(tickmode = 'linear'), showlegend = False)
fig.update_traces(textposition = 'top center')

# Corner coordinates for each colored zone
x0 = [1000, 950, 850, 690, 850, 350, 500, 150, 200, 100]
x1 = [650,  800, 500, 600, 500, 300, 450, 120, 80,  10]
y0 = [0,    0.1,  0.25, 0.32, 0.39, 0.46, 0.53,  0.68, 0.75, 0.82]
y1 = [0.1,  0.17, 0.32, 0.39, 0.46, 0.53, 0.615, 0.75, 0.82, 0.89]

for i in range(len(x0)) :
    fig.add_vrect(x0 = x0[i], x1 = 1600, y0 = y0[i], y1 = y1[i], line_width = 0, fillcolor = 'green', opacity = 0.2)
    fig.add_vrect(x0 = 0, x1 = x1[i], y0 = y0[i], y1 = y1[i], line_width = 0, fillcolor = 'red', opacity = 0.2)
    fig.add_vrect(x0 = x1[i], x1 = x0[i], y0 = y0[i], y1 = y1[i], line_width = 0, fillcolor = 'orange', opacity = 0.2)

fig.show()


def show_unranked_teams(poll_df, mode, t) : 
    fig_arr = []
    
    # Get the rank status of all tournament teams
    # unranked = pd.merge(kb_df, poll_df, how = 'left', on = ['TEAM NO'])
    unranked = poll_df.copy()
    # Change all N/A values to 0 as they are unranked teams
    if mode == 1 : unranked['1 RANK?'] = unranked['1 RANK?'].fillna(0)
    elif mode == 6 :unranked['6 RANK?'] = unranked['6 RANK?'].fillna(0)
    unranked = unranked.rename(columns = {'YEAR_x' : 'YEAR', 'TEAM_x' : 'TEAM', 'SEED_x' : 'SEED', 'ROUND_x': 'ROUND'})
    # Get all unranked seeds 1 - 12 as they should all theoretically be ranked
    if mode == 1 : unranked = unranked.loc[unranked['1 RANK?'].eq(0) & unranked['SEED'].le(12)]
    elif mode == 6 : unranked = unranked.loc[unranked['6 RANK?'].eq(0) & unranked['SEED'].le(12)]
    unranked = order_df(df_input = unranked, order_by = 'SEED', order = order_seed)
    unranked['SEED'] = unranked['SEED'].astype(str)
    change_round_str(unranked, rounds_num_68, round_str_68)
    
    curr_unranked = unranked.loc[unranked['YEAR'].eq(curr_year)]
    # unranked = unranked.loc[unranked['YEAR'].ne(curr_year)]
    unranked = unranked.loc[unranked['YEAR'].ne(2025)]

    if mode == 1 :
        fig = px.scatter(unranked, x = 'ROUND', y = 'TEAM NO', hover_data = ['YEAR', 'TEAM', '1 AP VOTES'], animation_frame = 'SEED',
                         color = 'SEED', template = 'plotly_dark', height = 600)
    elif mode == 6 : 
        fig = px.scatter(unranked, x = 'ROUND', y = 'TEAM NO', hover_data = ['YEAR', 'TEAM', '6 AP VOTES'], animation_frame = 'SEED',
                         color = 'SEED', template = 'plotly_dark', height = 600)
        
    fig['layout'].pop('updatemenus')
    fig.update_layout(title = t, title_x = 0.5, title_font = dict(size = 15), showlegend = False)
    fig.update_xaxes(categoryorder = 'array', categoryarray = rev_round_str_64)
    go.Figure(data = fig.data, frames = [fr.update(layout =
             {'xaxis': {'range' : [- 0.5, 6.5]},
              'yaxis': {'range' : [min(fr.data[0].y) - 20, max(fr.data[0].y) + 20]},}) for fr in fig.frames], layout = fig.layout)
    fig_arr.append(fig)

    selected_cols = curr_unranked.columns[curr_unranked.columns.isin(['YEAR', 'CONF', 'TEAM', 'SEED'])]
    fig_arr.append(curr_unranked[selected_cols]) 

    return fig_arr   


unranked_arr = show_unranked_teams(complete_stats, 1, '<b> How Far Every Unranked Team made it in the Tournament (1 - 12 Seeds) </b>')
unranked_arr[0].show()


unranked_arr[1].head(1000)


ap_week_6_df_arr = ap_data_df(complete_stats, 6)

fig = px.histogram(preseason_df_arr[2], x = 'SEED', y = 'AVG WINS',
                   color = 'RANK?', barmode = 'group', template = 'plotly_dark',
                   nbins = 12, height = 500)
fig.update_layout(title = '<b> Average Wins of each Seed (Ranked vs Unranked) </b>', title_x = 0.5, title_font = dict(size = 20),
                  xaxis = dict(tickmode = 'linear'), yaxis_title = 'AVERAGE WINS')
fig.show()


curr_ap_week_6 = ap_week_6_df_arr[1]
curr_ap_week_6 = curr_ap_week_6.sort_values(by = ['6 AP VOTES'], ascending = False)

fig = px.histogram(curr_ap_week_6, x = 'TEAM', y = '6 AP VOTES',
                   color = '6 RANK?', barmode = 'group', template = 'plotly_dark',
                   height = 500)
fig.update_layout(title = '<b>' + str(curr_year) + ' Teams Receiving AP Preseason Votes </b>', title_x = 0.5, title_font = dict(size = 20),
                  yaxis_title = 'AP VOTES')
fig.update_xaxes(tickangle = - 90)
fig.show()


ap_week_6 = ap_week_6_df_arr[0]

ap_str = order_df(df_input = ap_week_6, order_by = 'ROUND', order = rounds_num_68)
change_round_str(ap_str, rounds_num_68, round_str_68)
ap_str = order_df(df_input = ap_str, order_by = 'SEED', order = order_seed)
ap_str['SEED'] = ap_str['SEED'].astype(str)

fig = px.scatter(ap_str, x = 'ROUND', y = '6 AP VOTES', hover_data = ['YEAR', 'TEAM'], animation_frame = 'SEED', height = 600,
                color = 'SEED', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark')
fig['layout'].pop('updatemenus')
fig.update_layout(title = '<b> How Far each Team receiving Preseason AP Votes made it in the Tournament </b>', title_x = 0.5, title_font = dict(size = 15), showlegend = False)
fig.update_xaxes(categoryorder = 'array', categoryarray = round_str_68, autorange = 'reversed')
go.Figure(data = fig.data, frames = [fr.update(layout =
         {'xaxis': {'range' : [6.5, - 0.5]},
          'yaxis': {'range' : [min(fr.data[0].y) - 20, max(fr.data[0].y) + 20]},}) for fr in fig.frames], layout = fig.layout)
fig.show()


avg_wins_votes_df = ap_week_6_df_arr[3]

avg_wins_votes_df['SEED'] = avg_wins_votes_df['SEED'].astype(str)
wins_arr = np.arange(7)
avg_wins_votes_df = order_df(df_input = avg_wins_votes_df, order_by = 'WINS', order = wins_arr)

rev_round_str_64 = round_str_64[:: - 1]

# Turn win count values into round strings
for i, r in enumerate(wins_arr) :
    avg_wins_votes_df.loc[avg_wins_votes_df['WINS'] == r, 'WINS'] = rev_round_str_64[i]

fig = px.bar(avg_wins_votes_df, x = 'WINS', y = 'AVG VOTES', animation_frame = 'SEED', height = 600,
             color = 'SEED', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark')
fig['layout'].pop('updatemenus')
fig.update_layout(title = '<b> Average Preseason AP Votes by Round and Seed </b>', title_x = 0.5, title_font = dict(size = 20), showlegend = False)
fig.update_xaxes(title = 'ROUND', categoryorder = 'array', categoryarray = rev_round_str_64)
go.Figure(data = fig.data, frames = [fr.update(layout =
         {'xaxis': {'range' : [- 0.5, 6.5]},
          'yaxis': {'range' : [min(fr.data[0].y) - 20, max(fr.data[0].y) + 20]},}) for fr in fig.frames], layout = fig.layout)
fig.show()


curr_ap_week_6['6 RANK?'] = curr_ap_week_6['6 RANK?'].astype(str)

fig = px.scatter(curr_ap_week_6, x = '6 AP VOTES', y = 'SEED', text = 'TEAM', hover_data = ['YEAR', '6 AP RANK'],
                   color = '6 RANK?', template = 'plotly_dark',
                   height = 600)
fig.update_layout(title = '<b> AP Votes of ' + str(curr_year) + ' Teams </b>', title_x = 0.5, title_font = dict(size = 20), yaxis = dict(tickmode = 'linear'), showlegend = False)
fig.update_traces(textposition = 'top center')

# Corner coordinates for each colored zone
x0 = [1100, 950, 850, 690, 850, 350, 500, 150, 200, 100]
x1 = [650,  800, 500, 600, 500, 300, 450, 120, 80,  10]
y0 = [0,    0.1,  0.25, 0.32, 0.39, 0.46, 0.53,  0.68, 0.75, 0.82]
y1 = [0.1,  0.17, 0.32, 0.39, 0.46, 0.53, 0.615, 0.75, 0.82, 0.89]

for i in range(len(x0)) :
    fig.add_vrect(x0 = x0[i], x1 = 1600, y0 = y0[i], y1 = y1[i], line_width = 0, fillcolor = 'green', opacity = 0.2)
    fig.add_vrect(x0 = 0, x1 = x1[i], y0 = y0[i], y1 = y1[i], line_width = 0, fillcolor = 'red', opacity = 0.2)
    fig.add_vrect(x0 = x1[i], x1 = x0[i], y0 = y0[i], y1 = y1[i], line_width = 0, fillcolor = 'orange', opacity = 0.2)

fig.show()


unranked_arr = show_unranked_teams(complete_stats, 6, '<b> How Far Every Unranked Team made it in the Tournament (1 - 12 Seeds) </b>')
unranked_arr[0].show()


quad_no = complete_stats.copy()
quad_no = quad_no.loc[(quad_no['1 RANK?'].eq(0) | quad_no['6 RANK?'].eq(0)) & quad_no['SEED'].le(2)] 
quad_no = quad_no['QUAD NO'].drop_duplicates()

unranked = complete_stats.copy()
unranked = unranked.loc[unranked['QUAD NO'].isin(quad_no) & unranked['ROUND'].le(4)] 
unranked = unranked.reset_index(drop = True)
unranked


quad_no = complete_stats.copy()
quad_no = quad_no.loc[(quad_no['1 RANK?'].eq(0) | quad_no['6 RANK?'].eq(0)) & quad_no['SEED'].le(2)] 
quad_no = quad_no['QUAD NO'].drop_duplicates()

unranked = complete_stats.copy()
unranked = unranked.loc[unranked['QUAD NO'].isin(quad_no) & unranked['ROUND'].le(4)] 
unranked = unranked.reset_index(drop = True)
unranked


unranked_arr[1].head(1000)


# Create dataframe of Champions and their team profile for the following tournament (if they have one)
######################################################################################################

cs = complete_stats.copy()
champion_df = pd.DataFrame()

for year in range(2008, curr_year) :
    if year == 2020 : continue
    # Get all previous Champions
    champion = cs.loc[cs['YEAR'].eq(year) & cs['ROUND'].eq(1)]
    # Get the teams' stats of the following season they were Champions (if they have one)
    if year == 2019 : next_team = cs.loc[cs['YEAR'].eq(year + 2) & cs['TEAM'].eq(champion['TEAM'].values[0])]
    else : next_team = cs.loc[cs['YEAR'].eq(year + 1) & cs['TEAM'].eq(champion['TEAM'].values[0])]
    # Combine both teams in a dataframe
    champion_df = pd.concat([champion_df, champion, next_team], axis = 0)

# Simplify the dataframe with the key stats
selected_cols = champion_df.columns[champion_df.columns.isin(['YEAR', 'CONF', 'TEAM', 'SEED', 'ROUND', 'KADJ EM', 'BADJ EM', 'BARTHAG', 'WAB'])]
champion_df = champion_df[selected_cols]
champion_df = champion_df.reset_index(drop = True)
champion_df = champion_df.iloc[:: - 1].reset_index(drop = True)
champion_df


# Create dataframe consisting of the average wins of seeds by bid type
######################################################################

bids = complete_stats.copy()
bids = bids.loc[bids['ROUND'].ne(68)]
curr_bids = bids.loc[bids['YEAR'].eq(curr_year)]
bids = bids.loc[bids['YEAR'].ne(curr_year)]
bids['AVG WINS'] = bids['ROUND']

wins = [6, 5, 4, 3, 2, 1, 0]

# Convert AVG WINS column to win count values
for i in rounds_num_64 :
    bids.loc[bids['AVG WINS'] == i, 'AVG WINS'] = wins[rounds_num_64.index(i)]

# Get the average wins for each seed by bid type
avg_wins_bids = bids.groupby(['SEED', 'BID TYPE'])['AVG WINS'].mean()
avg_wins_bids = avg_wins_bids.to_frame().reset_index()
# Get the average wins for the 1 - 14 seeds as the 15 and 16 seeds do not have any At-Large Bids
avg_wins_bids = avg_wins_bids.loc[avg_wins_bids['SEED'].le(14)]
avg_wins_bids['SEED'] = avg_wins_bids['SEED'].astype(str)


fig = px.histogram(avg_wins_bids, x = 'SEED', y = 'AVG WINS',
                   color = 'BID TYPE', barmode = 'group', template = 'plotly_dark',
                   nbins = 14, height = 500)
fig.update_layout(title = '<b> Average Wins of each Seed (Auto vs At-Large) </b>', title_x = 0.5, title_font = dict(size = 20),
                  xaxis = dict(tickmode = 'linear'), yaxis_title = 'AVERAGE WINS')
fig.show()


curr_bids[['YEAR', 'TEAM', 'SEED', 'BID TYPE']].sort_values(by = 'SEED', ascending = True).head(68)


bids_df = order_df(df_input = complete_stats, order_by = 'ROUND', order = rounds_num_68)
change_round_str(bids_df, rounds_num_68, round_str_68)
bids_df = order_df(df_input = bids_df, order_by = 'SEED', order = order_seed)
# Get the 1 - 14 seeds as the 15 and 16 seeds do not have any At-Large Bids
bids_df = bids_df.loc[bids_df['SEED'].le(14)]
bids_df['SEED'] = bids_df['SEED'].astype(str)

fig = px.scatter(bids_df, x = 'ROUND', y = 'TEAM NO', hover_data = ['YEAR', 'TEAM'], animation_frame = 'SEED', height = 600,
                 color = 'BID TYPE', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark')
fig['layout'].pop('updatemenus')
fig.update_layout(title = '<b> How Far each Team made the Tournament (Auto vs At-Large) </b>', title_x = 0.5, title_font = dict(size = 20))
fig.update_xaxes(categoryorder = 'array', categoryarray = round_str_68, autorange = 'reversed')
go.Figure(data = fig.data, frames = [fr.update(layout =
         {'xaxis': {'range' : [6.5, - 0.5]},
          'yaxis': {'range' : [min(fr.data[0].y) - 200, max(fr.data[0].y) + 200]},}) for fr in fig.frames], layout = fig.layout)
fig.show()


# Create dataframe consisting of the average wins of seeds by bid type
######################################################################

majors = complete_stats.copy()
# Remove the exceptions of Houston and Gonzaga   
majors = majors.loc[~majors['TEAM ID'].isin([68, 76])] 
high_majors = majors.loc[majors['CONF ID'].isin([2, 6, 7, 8, 23, 24, 28])] 
mid_majors = majors.loc[~majors['CONF ID'].isin([2, 6, 7, 8, 23, 24, 28])] 

high_majors['CONF TYPE'] = 'HIGH MAJOR'
mid_majors['CONF TYPE'] = 'MID MAJOR'

majors = pd.concat([high_majors, mid_majors], ignore_index = False, sort = False)
majors = majors.sort_index()
majors_df = majors.copy()

majors = majors.loc[majors['ROUND'].ne(68)]
curr_majors = majors.loc[majors['YEAR'].eq(curr_year)]
majors = majors.loc[majors['YEAR'].ne(curr_year)]
majors['AVG WINS'] = majors['ROUND']

wins = [6, 5, 4, 3, 2, 1, 0]

# Convert AVG WINS column to win count values
for i in rounds_num_64 :
    majors.loc[majors['AVG WINS'] == i, 'AVG WINS'] = wins[rounds_num_64.index(i)]

# Get the average wins for each seed by bid type
avg_wins_majors = majors.groupby(['SEED', 'CONF TYPE'])['AVG WINS'].mean()
avg_wins_majors = avg_wins_majors.to_frame().reset_index()
# Get the average wins for the 1 - 14 seeds as the 15 and 16 seeds do not have any At-Large Bids
avg_wins_majors = avg_wins_majors.loc[avg_wins_majors['SEED'].le(12)]
avg_wins_majors['SEED'] = avg_wins_majors['SEED'].astype(str)  


fig = px.histogram(avg_wins_majors, x = 'SEED', y = 'AVG WINS',
                   color = 'CONF TYPE', barmode = 'group', template = 'plotly_dark',
                   nbins = 14, height = 500)
fig.update_layout(title = '<b> Average Wins of each Seed (High Major vs Mid Major) </b>', title_x = 0.5, title_font = dict(size = 20),
                  xaxis = dict(tickmode = 'linear'), yaxis_title = 'AVERAGE WINS')
fig.show()


curr_majors[['YEAR', 'TEAM', 'SEED', 'CONF TYPE']].sort_values(by = 'SEED', ascending = True).head(68)


majors_df = order_df(df_input = majors_df, order_by = 'ROUND', order = rounds_num_68)
change_round_str(majors_df, rounds_num_68, round_str_68)
majors_df = order_df(df_input = majors_df, order_by = 'SEED', order = order_seed)
# Get the 1 - 12 seeds as the 13 - 16 seeds do not have any High Major Teams  
majors_df = majors_df.loc[majors_df['SEED'].le(12)]
majors_df['SEED'] = majors_df['SEED'].astype(str)

fig = px.scatter(majors_df, x = 'ROUND', y = 'TEAM NO', hover_data = ['YEAR', 'TEAM'], animation_frame = 'SEED', height = 600,
                 color = 'CONF TYPE', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark')
fig['layout'].pop('updatemenus')
fig.update_layout(title = '<b> How Far each Team made the Tournament (High Major vs Mid Major) </b>', title_x = 0.5, title_font = dict(size = 20))
fig.update_xaxes(categoryorder = 'array', categoryarray = round_str_68, autorange = 'reversed')
go.Figure(data = fig.data, frames = [fr.update(layout =
         {'xaxis': {'range' : [6.5, - 0.5]},
          'yaxis': {'range' : [min(fr.data[0].y) - 200, max(fr.data[0].y) + 200]},}) for fr in fig.frames], layout = fig.layout)
fig.show()

