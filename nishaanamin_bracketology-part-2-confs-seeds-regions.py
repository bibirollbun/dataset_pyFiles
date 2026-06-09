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
tournament_count = 17    

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


# Create dataframe for various statistics and how many teams from conferences made tournament rounds
####################################################################################################

conf_count = ordered_tournament_matchups.loc[ordered_tournament_matchups['YEAR'].ne(curr_year)]
# Get the the count of teams from each conference making each round
conf_count = conf_count.groupby(by = ['YEAR', 'CONF', 'CONF ID', 'CURRENT ROUND']).size().reset_index(name = 'COUNT')
conf_count = conf_count.sort_values(by = ['YEAR', 'CONF', 'CURRENT ROUND'], ascending = False)

temp_csh_df = csh_df.copy()
# Add Prefix to indicate type of conference stats (H = Home, A = Away, N = Neutral, AN = Away Neutral)
temp_csh_df = temp_csh_df.add_prefix('H ')
# Rename H YEAR and H CONF columns to YEAR and H CONF because it will be the columns used to merge
temp_csh_df = temp_csh_df.rename(columns = {'H YEAR' : 'YEAR', 'H CONF' : 'CONF'})

temp_csa_df = csa_df.copy()
temp_csa_df = temp_csa_df.add_prefix('A ')
temp_csa_df = temp_csa_df.rename(columns = {'A YEAR' : 'YEAR', 'A CONF' : 'CONF'})

temp_csn_df = csn_df.copy()
temp_csn_df = temp_csn_df.add_prefix('N ')
temp_csn_df = temp_csn_df.rename(columns = {'N YEAR' : 'YEAR', 'N CONF' : 'CONF'})

temp_csan_df = csan_df.copy()
temp_csan_df = temp_csan_df.add_prefix('AN ')
temp_csan_df = temp_csan_df.rename(columns = {'AN YEAR' : 'YEAR', 'AN CONF' : 'CONF'})

dfs = [conf_count, cs_df, temp_csh_df, temp_csa_df, temp_csn_df, temp_csan_df]
# Merge all columns to make one dataframe consisting of all conference stats
conf_count = reduce(lambda left, right: pd.merge(left, right, on = ['YEAR', 'CONF'], how = 'left'), dfs)
conf_count = conf_count.loc[conf_count['CURRENT ROUND'].le(16)]


rounds = [16, 8, 4, 2]
titles = ['SWEET 16', 'ELITE 8', 'FINAL 4', 'FINALS']

# Replace Current Round values with string values from the titles array
for i in rounds :
    conf_count = conf_count.replace({'CURRENT ROUND': i}, titles[rounds.index(i)])

conf_count = conf_count.reset_index(drop = True)
selected_cols = conf_count.columns[conf_count.columns.isin(['YEAR', 'CONF', 'CURRENT ROUND', 'COUNT', 'BADJ EM', 'BADJ O', 'BADJ D', 'BARTHAG', 'WAB',
                                                            'AN BADJ EM', 'AN BADJ O', 'AN BADJ D', 'AN BARTHAG', 'AN WAB'])]
simp_conf_count = conf_count[selected_cols]
fig_arr = []

for i in range(simp_conf_count.columns.get_loc('BADJ EM'), simp_conf_count.columns.get_loc('AN WAB') + 1) :
    fig = px.scatter(simp_conf_count, x = 'COUNT', y = simp_conf_count.columns[i], facet_col = 'CURRENT ROUND', hover_data = ['YEAR', 'CONF'], color = 'CURRENT ROUND', template = 'plotly_dark')
    # Remove the first half of the string from the plot titles (Everything including and before the = sign)
    fig.for_each_annotation(lambda a: a.update(text = a.text.split('=')[- 1]))
    fig.update_layout(height = 400, showlegend = False)
    if i == simp_conf_count.columns.get_loc('BADJ EM') : fig.update_layout(title = '<b> Amount of Conference Teams making the Tournament by Round and Basketball Metrics </b>', title_x = 0.5, title_font = dict(size = 15))
    fig_arr.append(fig)


fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr[4].show()


fig_arr[5].show()


fig_arr[6].show()


fig_arr[7].show()


fig_arr[8].show()


fig_arr[9].show()


# Create dataframe for various statistics and how many teams from conferences made tournament rounds for the Champions
######################################################################################################################

champ_conf_count = ordered_tournament_matchups.loc[ordered_tournament_matchups['YEAR'].ne(curr_year) & ordered_tournament_matchups['ROUND'].eq(1) & ordered_tournament_matchups['CURRENT ROUND'].eq(2)]
# Get the the count of teams from each conference making each round
champ_conf_count = champ_conf_count.groupby(by = ['YEAR', 'CONF', 'CONF ID', 'CURRENT ROUND']).size().reset_index(name = 'COUNT')
champ_conf_count = champ_conf_count.sort_values(by = ['YEAR', 'CONF', 'CURRENT ROUND'], ascending = False)

dfs = [champ_conf_count, cs_df, temp_csh_df, temp_csa_df, temp_csn_df, temp_csan_df]
# Merge all columns to make one dataframe consisting of all conference stats
champ_conf_count = reduce(lambda left, right: pd.merge(left, right, on = ['YEAR', 'CONF'], how = 'left'), dfs)
simp_champ_conf_count = champ_conf_count[selected_cols]


titles = ['BADJ EM', 'BADJ O', 'BADJ D', 'BARHTAG', 'WAB']
figures = []

for i in range(simp_champ_conf_count.columns.get_loc('BADJ EM'), simp_champ_conf_count.columns.get_loc('WAB') + 1) :
    fig = px.scatter(simp_champ_conf_count, x = 'COUNT', y = simp_champ_conf_count.columns[i], color_discrete_sequence = [i], hover_data = ['YEAR', 'CONF'])
    figures.append(fig)

fig1 = multiple_subplots(figures, titles, 400)
fig1.update_layout(title = '<b> Conference Teams making the Tournament by Basketball Metrics (Champions) </b>', title_x = 0.5, title_font = dict(size = 15))
fig1.update_xaxes(title_text = 'COUNT', row = 1, col = 3)

titles = ['AN BADJ EM', 'AN BADJ O', 'AN BADJ D', 'AN BARTHAG', 'AN WAB']
figures = []

for i in range(simp_champ_conf_count.columns.get_loc('AN BADJ EM'), simp_champ_conf_count.columns.get_loc('AN WAB') + 1) :
    fig = px.scatter(simp_champ_conf_count, x = 'COUNT', y = simp_champ_conf_count.columns[i], color_discrete_sequence = [i], hover_data = ['YEAR', 'CONF'])
    figures.append(fig)

fig2 = multiple_subplots(figures, titles, 400)
fig2.update_layout(title = '<b> Conference Teams making the Tournament by Away Neutral Basketball Metrics (Champions) </b>', title_x = 0.5, title_font = dict(size = 15))
fig2.update_xaxes(title_text = 'COUNT', row = 1, col = 3)

fig1.show()


fig2.show()


dfs = [cs_df, temp_csh_df, temp_csa_df, temp_csn_df, temp_csan_df]
# Merge all columns to make one dataframe consisting of all conference stats
curr_conf = reduce(lambda left, right: pd.merge(left, right, on = ['YEAR', 'CONF'], how = 'left'), dfs)

curr_conf = curr_conf.loc[curr_conf['YEAR'].eq(curr_year)]
selected_cols = curr_conf.columns[curr_conf.columns.isin(['YEAR', 'CONF', 'BADJ EM', 'BADJ O', 'BADJ D', 'BARTHAG', 'WAB',
                                                          'AN BADJ EM', 'AN BADJ O', 'AN BADJ D', 'AN BARTHAG', 'AN WAB'])]
curr_conf = curr_conf[selected_cols]

# Y Values where the threshold lines will be displayed
y_value1 = [[9.5, 10.3, 11.2], [106.2, 106.6, 107.5], [99.9, 98.3, 98.3], [0.742, 0.765, 0.776], [- 3, - 2.7, - 2.2]]
y_value2 = [[1.2, 5.3, 7.6, 9.1, 12.3], [102.6, 103.4, 106.7, 107.1, 109.2],
            [103.7, 99.9, 98.9, 98.1, 98.1], [0.535, 0.648, 0.704, 0.739, 0.797], [- 8.5, - 7.5, - 4.5, - 3, - 1.4]]
y_value3 = [[7.5, 7.75, 7.75], [106, 106, 107.5], [101, 98, 100.5], [ 0.7, 0.75, 0.7], [- 1.9, - 1.1, - 2]]
y_value4 = [[0, 4, 7, 7.75, 12], [102, 102.5, 106, 106, 107], [102.5, 100.5, 100, 99.5, 98], [0.5, 0.6, 0.6, 0.7, 0.78],
            [- 6.5, - 3, - 2.75, - 2, - 1]]

fig_arr = []  

for i in range(curr_conf.columns.get_loc('BADJ EM'), curr_conf.columns.get_loc('AN WAB') + 1) :
    fig = px.scatter(curr_conf, x = 'CONF', y = curr_conf.columns[i], hover_data = ['YEAR'], text = 'CONF', template = 'plotly_dark')
    fig.update_traces(textposition = 'top center')
    if i < curr_conf.columns.get_loc('WAB') + 1 :
        fig.add_hline(y = y_value1[i - 2][0], line_width = 1, line_dash = 'dash', line_color = 'blue')
        fig.add_hline(y = y_value1[i - 2][1], line_width = 1, line_dash = 'dash', line_color = 'red')
        fig.add_hline(y = y_value1[i - 2][2], line_width = 1, line_dash = 'dash', line_color = 'green')
        fig.add_hline(y = y_value2[i - 2][0], line_width = 1, line_color = 'blue')
        fig.add_hline(y = y_value2[i - 2][1], line_width = 1, line_color = 'red')
        fig.add_hline(y = y_value2[i - 2][2], line_width = 1, line_color = 'green')
        fig.add_hline(y = y_value2[i - 2][3], line_width = 1, line_color = 'purple')
        fig.add_hline(y = y_value2[i - 2][4], line_width = 1, line_color = 'orange')
    else :
        fig.add_hline(y = y_value3[i - 7][0], line_width = 1, line_dash = 'dash', line_color = 'blue')
        fig.add_hline(y = y_value3[i - 7][1], line_width = 1, line_dash = 'dash', line_color = 'red')
        fig.add_hline(y = y_value3[i - 7][2], line_width = 1, line_dash = 'dash', line_color = 'green')
        fig.add_hline(y = y_value4[i - 7][0], line_width = 1, line_color = 'blue')
        fig.add_hline(y = y_value4[i - 7][1], line_width = 1, line_color = 'red')
        fig.add_hline(y = y_value4[i - 7][2], line_width = 1, line_color = 'green')
        fig.add_hline(y = y_value4[i - 7][3], line_width = 1, line_color = 'purple')
        fig.add_hline(y = y_value4[i - 7][4], line_width = 1, line_color = 'orange')

    if i == curr_conf.columns.get_loc('BADJ EM') : fig.update_layout(title = '<b> ' + str(curr_year) + ' Conferences </b>', title_x = 0.5, title_font = dict(size = 20))
    fig.update_xaxes(tickangle = - 90)
    fig_arr.append(fig)


fig_arr[0].show()


fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr[4].show()


fig_arr[5].show()


fig_arr[6].show()


fig_arr[7].show()


fig_arr[8].show()


fig_arr[9].show()


# Combine all the Y Values into one array
cond_arr = y_value1 + y_value2 + y_value3 + y_value4
cond_df = pd.DataFrame(cond_arr)
cond_df.columns = ['SWEET 16', 'ELITE 8', 'FINAL 4', 'FINALS', 'CHAMPION']

stats_df = curr_conf.copy()

stats_df = stats_df.iloc[:, 2 :]
reg = stats_df.iloc[:, : 5].columns
an = stats_df.iloc[:, 5 :].columns

stats = reg
stats = stats.append(reg)
stats = stats.append(an)
stats = stats.append(an)


one_df = pd.DataFrame()
two_df = pd.DataFrame()
less_than_arr = [2, 7, 12, 17]
two_arr = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]

for i in range(5) :
    temp_df1 = pd.DataFrame()
    temp_df2 = pd.DataFrame()
    k = 0

    # Get rows of dataframe that meets the conference strength threshold
    for j in range(len(cond_df)) :
        if j in less_than_arr :
            if j in two_arr : temp_df2 = pd.concat([temp_df2, curr_conf[curr_conf[stats[k]].to_frame().lt(cond_df.iloc[j, i]).any(axis = 1)]], axis = 0)
            else : temp_df1 = pd.concat([temp_df1, curr_conf[curr_conf[stats[k]].to_frame().lt(cond_df.iloc[j, i]).any(axis = 1)]], axis = 0)
        else :
            if j in two_arr : temp_df2 = pd.concat([temp_df2, curr_conf[curr_conf[stats[k]].to_frame().gt(cond_df.iloc[j, i]).any(axis = 1)]], axis = 0)
            else : temp_df1 = pd.concat([temp_df1, curr_conf[curr_conf[stats[k]].to_frame().gt(cond_df.iloc[j, i]).any(axis = 1)]], axis = 0)

        k += 1

    temp_df1['ROUND'] = cond_df.columns[i]
    temp_df2['ROUND'] = cond_df.columns[i]

    one_df = pd.concat([one_df, temp_df1], axis = 0)
    two_df = pd.concat([two_df, temp_df2], axis = 0)

one_df['TYPE'] = 'ONE'
two_df['TYPE'] = 'TWO'

threshold_count = pd.concat([one_df, two_df], axis = 0)
# Get the number of times that the conference meets the thresholds
count = threshold_count.groupby(by = ['ROUND', 'TYPE', 'CONF']).size().reset_index(name = 'COUNT')


one1 = count.loc[count['TYPE'].eq('ONE') & (count['ROUND'].eq('ELITE 8') | count['ROUND'].eq('SWEET 16'))]
one2 = count.loc[count['TYPE'].eq('ONE') & (count['ROUND'].ne('ELITE 8') & count['ROUND'].ne('SWEET 16'))]
two = count.loc[count['TYPE'].eq('TWO')]

fig1 = px.bar(one1, x = 'ROUND', y = 'COUNT', color = 'CONF', barmode = 'group', hover_data = ['TYPE'], text = 'CONF', template = 'plotly_dark')
fig2 = px.bar(one2, x = 'ROUND', y = 'COUNT', color = 'CONF', barmode = 'group', hover_data = ['TYPE'], text = 'CONF', template = 'plotly_dark')
fig3 = px.bar(two, x = 'ROUND', y = 'COUNT', color = 'CONF', barmode = 'group', hover_data = ['TYPE'], text = 'CONF', template = 'plotly_dark')

fig1.update_layout(title = '<b> Amount of Times a Conference Surpasses the Threshold by Round (1 or More Teams) </b>', title_x = 0.5, title_font = dict(size = 15))
fig2.update_layout(title = '<b> Amount of Times a Conference Surpasses the Threshold by Round (1 or More Teams) </b>', title_x = 0.5, title_font = dict(size = 15))
fig3.update_layout(title = '<b> Amount of Times a Conference Surpasses the Threshold by Round (2 or More Teams) </b>', title_x = 0.5, title_font = dict(size = 15))

fig1.update_xaxes(categoryorder = 'array', categoryarray = ['SWEET 16', 'ELITE 8'])
fig2.update_xaxes(categoryorder = 'array', categoryarray = ['FINAL 4', 'FINALS', 'CHAMPION'])
fig3.update_xaxes(categoryorder = 'array', categoryarray = ['SWEET 16', 'ELITE 8', 'FINAL 4'])

fig1.update_traces(textfont_size = 50, textangle = 0, textposition = 'outside', cliponaxis = False)
fig2.update_traces(textfont_size = 50, textangle = 0, textposition = 'outside', cliponaxis = False)
fig3.update_traces(textfont_size = 50, textangle = 0, textposition = 'outside', cliponaxis = False)

fig1.show()


fig2.show()


fig3.show()


# Create dataframe that transforms the complete_stats dataframe into a MinMaxScaler
###################################################################################

stats = complete_stats.copy()
stats = stats[['YEAR', 'TEAM', 'SEED', 'KADJ EM', 'BADJ EM', 'WAB', 'RELATIVE RATING']]

scaler = MinMaxScaler()
selected_cols = ['KADJ EM', 'BADJ EM', 'WAB', 'RELATIVE RATING']

# Scale variables between a value of 0 to 1 for statistics of all teams
stats[selected_cols] = scaler.fit_transform(stats[selected_cols])
# Calculate the Seed Strength Value
stats['STRENGTH'] = stats[selected_cols].sum(axis = 1) / len(selected_cols) * 100

curr_stats = stats.loc[stats['YEAR'].eq(curr_year)]
stats = stats.loc[stats['YEAR'].ne(curr_year) & stats['YEAR'].ge(2013)]


# Create dataframe consisting of each seed's mean seed strength along with the current team's seed strength
###########################################################################################################

fig_arr = []

for seed in range(1, 17) :
    seed_df = stats.loc[stats['SEED'].isin([seed])]
    # Get the average seed strength value of each seed
    mean = seed_df[['STRENGTH']].mean()
    mean_df = pd.DataFrame()
    mean_df = pd.concat([mean_df, mean], axis = 1)
    mean_df.columns = [seed]

    # Get the standard deviation of the seed strength value of each seed
    std_df = seed_df[['STRENGTH']].std()

    curr_seed_df = curr_stats.loc[curr_stats['SEED'].isin([seed])]
    curr_stats_df = curr_seed_df['STRENGTH'].to_frame()
    curr_stats_df = curr_stats_df.reset_index(drop = True)
    curr_stats_df = curr_stats_df.rename(columns = {'STRENGTH' : seed})
    mean_df = pd.concat([mean_df, curr_stats_df], axis = 0)
    team = np.array(['MEAN STRENGTH'])
    team = np.append(team, curr_seed_df['TEAM'])
    mean_df['TEAM'] = team

    t = str(seed) + ' SEED STRENGTH' + '<br><sup>' + 'STD : ' + std_df['STRENGTH'].round(2).astype(str) + '</sup>'
    y1 = mean_df.iloc[0, 0] + std_df['STRENGTH']
    y2 = mean_df.iloc[0, 0] - std_df['STRENGTH']

    fig = px.bar(mean_df, x = 'TEAM', y = seed, color = 'TEAM', title = t, labels = {str(seed) : 'STRENGTH VALUE'}, template = 'plotly_dark')
    fig.add_hline(y = y1, line_width = 1.5, line_dash = 'dash', line_color = 'green')
    fig.add_hline(y = y2, line_width = 1.5, line_dash = 'dash', line_color = 'red')
    fig.update_layout(title_x = 0.5, showlegend = False)
    fig_arr.append(fig)


fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr[4].show()


fig_arr[5].show()


fig_arr[6].show()


fig_arr[7].show()


fig_arr[8].show()


fig_arr[9].show()


fig_arr[10].show()


fig_arr[11].show()


fig_arr[12].show()


fig_arr[13].show()


fig_arr[14].show()


fig_arr[15].show()


# Displays the plots for each seed given a statistic
####################################################

mod_rounds_num = [1, 2, 4, 8, 16, 32, 64, 0]
mod_round_str = ['CHAMPION', 'FINALS', 'FINAL 4', 'ELITE 8', 'SWEET 16', 'SECOND ROUND', 'FIRST ROUND', '2024 TEAMS']

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


seed_rankings = complete_stats.copy()
seed_rankings = seed_rankings.loc[seed_rankings['YEAR'].ge(2013)]
seed_rankings = seed_rankings[['YEAR', 'TEAM', 'SEED', 'ROUND', 'KADJ EM', 'BADJ EM', 'WAB', 'RELATIVE RATING', 'NEW SEED LIST']]
all_seed_rankings = pd.DataFrame()

for year in range(curr_year, 2012, - 1) : 
    temp_df = seed_rankings.loc[seed_rankings['YEAR'].isin([year])]
    temp_df[['KADJ EM RANK', 'BADJ EM RANK', 'WAB RANK', 'RELATIVE RATING RANK']] = temp_df[['KADJ EM', 'BADJ EM', 'WAB', 'RELATIVE RATING']].rank(ascending = False)
    temp_df['AVG RANK'] = (temp_df['KADJ EM RANK'] + temp_df['BADJ EM RANK'] + temp_df['WAB RANK'] + temp_df['RELATIVE RATING RANK']) / 4 
    temp_df = temp_df.sort_values(by = 'AVG RANK') 
    temp_df['EXPECTED RANK'] = np.arange(len(temp_df)) + 1
    temp_df['RANK DIFF'] = temp_df['NEW SEED LIST'] - temp_df['EXPECTED RANK']
    temp_df = temp_df.sort_values(by = 'RANK DIFF', ascending = False)
    temp_df = temp_df[['YEAR', 'TEAM', 'SEED', 'ROUND', 'NEW SEED LIST', 'EXPECTED RANK', 'AVG RANK', 'RANK DIFF']]
    temp_df = temp_df.reset_index(drop = True) 
    all_seed_rankings = pd.concat([all_seed_rankings, temp_df], ignore_index = True)


# Create dataframes necessary to display the scatterplots
#########################################################

stats = all_seed_rankings.copy()
# stats = stats.loc[stats['YEAR'].ne(curr_year)]
stats = order_df(df_input = stats, order_by = 'SEED', order = order_seed)
change_round_str(stats, mod_rounds_num, mod_round_str)

# hcti_stats = stats.loc[stats['YEAR'].ge(2013)]

curr_stats = all_seed_rankings.copy()
curr_stats = curr_stats.loc[curr_stats['YEAR'].eq(curr_year)]
curr_stats = order_df(df_input = curr_stats, order_by = 'SEED', order = order_seed)

# X Value Coordinates for the first colored bar
x0, x1 = [0.025], [0.084425]

# Set the subsequent X Value Coordinates for the other 15 colored bars
for i in range(15) :
    x0.append(x0[i] + 0.059425)
    x1.append(x0[i + 1] + 0.059425)


fig = display_plots(stats, curr_stats, 'RANK DIFF', x0, x1, 'forward')
fig[0].show()


fig[1].show()


# Set up the dataframes for the plots found below
sr = sr_df.copy()
sr = order_df(df_input = sr , order_by = 'SEED', order = order_seed)
sr['SEED'] = sr['SEED'].astype(str)


def create_scatter_plots(df, x1, y1, plot_count, hover_data1, color1, color_continuous_scale1) : 
    fig_arr = []
    
    for i in range(plot_count) : 
        fig = px.scatter(df, x = x1, y =  y1[i], hover_data = hover_data1, color = color1, color_continuous_scale = color_continuous_scale1, template = 'plotly_dark')
        fig_arr.append(fig)

    return fig_arr 


fig1 = px.bar(sr, x = 'SEED', y = ['PAKE', 'PASE'], template = 'plotly_dark')
fig2 = px.bar(sr, x = 'SEED', y = 'WIN%', color = 'SEED', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark')
fig3 = px.bar(sr, x = 'SEED', y = ['R64', 'R32', 'S16', 'E8'], template = 'plotly_dark')
fig4 = px.bar(sr, x = 'SEED', y = ['F4', 'F2', 'CHAMP'], template = 'plotly_dark')

fig1.update_layout(title = '<b> PAKE and PASE of Seeds </b>', title_x = 0.5, title_font = dict(size = 20))
fig2.update_layout(title = '<b> Win % of Seeds </b>', title_x = 0.5, title_font = dict(size = 20))
fig3.update_layout(title = '<b> How Far Seeds made it in the Tournament </b>', title_x = 0.5, title_font = dict(size = 20))
fig4.update_layout(title = '<b> How Far Seeds made it in the Tournament </b>', title_x = 0.5, title_font = dict(size = 20))

fig1.show()


fig2.show()


fig3.show()


fig4.show()


def create_scatter_plots(df, x1, y1, plot_count, hover_data1, color1, color_continuous_scale1) : 
    fig_arr = []
    
    for i in range(plot_count) : 
        fig = px.scatter(df, x = x1, y =  y1[i], hover_data = hover_data1, color = color1[i], color_continuous_scale = color_continuous_scale1, template = 'plotly_dark')
        fig_arr.append(fig)

    return fig_arr 

def update_plot_layout(fig_arr, titles) : 
    for i, fig in enumerate(fig_arr):
        fig.update_layout(title = titles[i], title_x = 0.5, title_font = dict(size = 20))


y_color = ['PAKE', 'PASE', 'WIN%', 'R64', 'R32', 'S16', 'E8', 'F4', 'F2', 'CHAMP']
fig_arr = create_scatter_plots(tres_df, 'TEAM ID', y_color, 10, 'TEAM', y_color, px.colors.diverging.RdYlGn)

titles = ['<b> PAKE of Teams </b>', '<b> PASE of Teams </b>', '<b> Win % of Teams </b>', '<b> How many Times a Team made the First Round </b>',
          '<b> How many Times a Team made the Second Round </b>', '<b> How many Times a Team made the Sweet 16 </b>', 
          '<b> How many Times a Team made the Elite 8 </b>', '<b> How many Times a Team made the Final 4 </b>', 
          '<b> How many Times a Team made the Finals </b>', '<b> How many Times a Team was a Champion </b>']
update_plot_layout(fig_arr, titles)


fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr[4].show()


fig_arr[5].show()


fig_arr[6].show()


fig_arr[7].show()


fig_arr[8].show()


fig_arr[9].show()


fig1 = px.bar(conf_res_df, x = 'CONF', y = ['PAKE', 'PASE'], template = 'plotly_dark')
fig2 = px.bar(conf_res_df, x = 'CONF', y = 'WIN%', color = 'CONF', template = 'plotly_dark')
fig3 = px.bar(conf_res_df, x = 'CONF', y = ['R64', 'R32'], template = 'plotly_dark')
fig4 = px.bar(conf_res_df, x = 'CONF', y = ['S16', 'E8', 'F4', 'F2', 'CHAMP'], template = 'plotly_dark')

fig1.update_layout(title = '<b> PAKE and PASE of Conferences </b>', title_x = 0.5, title_font = dict(size = 20))
fig2.update_layout(title = '<b> Win % of Conferences </b>', title_x = 0.5, title_font = dict(size = 20))
fig3.update_layout(title = '<b> How Far Conferences made it in the Tournament </b>', title_x = 0.5, title_font = dict(size = 20))
fig4.update_layout(title = '<b> How Far Conferences made it in the Tournament </b>', title_x = 0.5, title_font = dict(size = 20))

fig1.show()


fig2.show()


fig3.show()


fig4.show()


fig_arr = create_scatter_plots(coach_res_df, 'COACH ID', y_color, 10, 'COACH', y_color, px.colors.diverging.RdYlGn)

titles = ['<b> PAKE of Coaches </b>', '<b> PASE of Coaches </b>', '<b> Win % of Coaches </b>', '<b> How many Times a Coach made the First Round </b>',
          '<b> How many Times a Coach made the Second Round </b>', '<b> How many Times a Coach made the Sweet 16 </b>', 
          '<b> How many Times a Coach made the Elite 8 </b>', '<b> How many Times a Coach made the Final 4 </b>', 
          '<b> How many Times a Coach made the Finals </b>', '<b> How many Times a Coach was a Champion </b>']
update_plot_layout(fig_arr, titles)


fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr[4].show()


fig_arr[5].show()


fig_arr[6].show()


fig_arr[7].show()


fig_arr[8].show()


fig_arr[9].show()


# Create dataframe that depicts the correlations of various statistics and upset count in tournament quadrants
def make_corr_df(quad_df1, quad_df2, quad_no, mode) :
    quad_arr = []

    for quad in quad_no :
        # Get all teams that played in a specific quadrant by year
        upsets = quad_df1.loc[quad_df1['ROUND'].ne(68) & quad_df1['QUAD NO'].eq(quad)]
        # Get the standard deviation of various stats of each quadrant number
        quad_arr.append(upsets[['KADJ EM', 'BADJ EM', 'A BADJ EM', 'AN BADJ EM', 'BARTHAG', 'WAB', 'AN WAB', 'POWER-PATH', 'RELATIVE RATING', 'TR RANK', 'A TR RANK', 'AN TR RANK']].std())

    quad_arr = np.array([quad_arr])
    quad_arr = np.transpose(quad_arr)

    if mode == 'curr' :
        quad_df2['YEAR'] = [curr_year] * 4 
        quad_df2['YEAR'] = quad_df2['YEAR'].astype(str)

    quad_df2['KADJ EM STD'] = quad_arr[0]
    quad_df2['BADJ EM STD'] = quad_arr[1]
    quad_df2['A BADJ EM STD'] = quad_arr[2]
    quad_df2['AN BADJ EM STD'] = quad_arr[3]
    quad_df2['BARTHAG STD'] = quad_arr[4]
    quad_df2['WAB STD'] = quad_arr[5]
    quad_df2['AN WAB STD'] = quad_arr[6]
    quad_df2['POWER-PATH STD'] = quad_arr[7]
    quad_df2['RELATIVE RATING STD'] = quad_arr[8]
    quad_df2['TR RANK STD'] = quad_arr[9]
    quad_df2['A TR RANK STD'] = quad_arr[10]
    quad_df2['AN TR RANK STD'] = quad_arr[11]

    # Get the correlations between the standard deviations of various stats and how many upsets occurred
    if mode == 'past' :
        quad_df2['W YEAR'] = quad_df2['W YEAR'].astype(str)
        corr_df = pd.DataFrame()
        corr_arr = []

        for i in range(12) :
            corr_arr.append(quad_df2['TOTAL UPSETS'].corr(quad_df2.iloc[:, i + 3]))

        corr_df['CORR'] = corr_arr
        corr_df = corr_df.T
        corr_df.columns = quad_df2.columns[3:]
        corr_df = corr_df.T

        # Display the lowest values in each column as a green highlighted box
        return corr_df.style.highlight_min(color = 'green', axis = 0)

    # Get the current quadrant numbers
    elif mode == 'curr' :
        quad_df2['QUAD NO'] = quad_no
        quad_df2.set_index(['YEAR', 'QUAD NO'], inplace = True, drop = True)
        # Display the lowest values in each column as a green highlighted box
        return quad_df2.style.highlight_min(color = 'green', axis = 0)


# Create dataframe consisting of quadrant statistics
####################################################

quad_df = tournament_matchups_combined_rows.copy()
quad_df = quad_df.loc[quad_df['W YEAR'].ne(curr_year)]
# Get all upset matchups
quad_df = quad_df.loc[quad_df['W SEED'] - quad_df['L SEED'] >= 2]
# Get all matchups from the First Round to the Elite 8
quad_df = quad_df.loc[quad_df['W CURRENT ROUND'].ge(8)]
# Get the count of upsets for each quadrant number
quad_df = quad_df.groupby(by = ['W YEAR', 'W QUAD NO']).size().reset_index(name = 'TOTAL UPSETS')

# Insert rows where the upset count is 0
quad_df.loc[0.5] = 2008, 2, 0
quad_df.loc[1.5] = 2008, 4, 0
quad_df = quad_df.sort_index().reset_index(drop = True)

quad_no = np.arange(1, tournament_count * 4 + 1)
curr_quad_no = np.arange(69, 73)
quad_arr, curr_quad_arr = [], []

cs_quad = complete_stats.copy()
# Get all teams that are 1 - 14 seeds for the purpose of finding adjusted correlations
cs_curr_quad = cs_quad.loc[cs_quad['YEAR'].eq(curr_year) & cs_quad['SEED'].le(14)]
cs_quad = cs_quad.loc[cs_quad['YEAR'].ne(curr_year) & cs_quad['SEED'].le(14)]
curr_quad_df = pd.DataFrame(columns = ['YEAR', 'QUAD NO'])

make_corr_df(cs_quad, quad_df, quad_no, 'past')


fig = px.scatter(quad_df , x = 'TOTAL UPSETS', y = 'RELATIVE RATING STD', hover_data = ['W QUAD NO'], trendline = 'ols', trendline_scope = 'overall', color = 'W YEAR', template = 'plotly_dark')
fig.update_layout(title = '<b> Correlation between Total Upsets and BADJ EM STD of each Tournament Quadrant </b>', title_x = 0.5, title_font = dict(size = 15))
fig.show()


# Predicted Upset Count = (BADJ EM STD - 7.31007) / (- 0.155482)  
quad_69 = (8.405517 - 6.82527) / (- 0.12736)
quad_70 = (8.752132 - 6.82527) / (- 0.12736)
quad_71 = (9.065718 - 6.82527) / (- 0.12736)
quad_72 = (8.870385 - 6.82527) / (- 0.12736)

print('QUAD 69 PREDICTED UPSET COUNT :', quad_69)
print('QUAD 70 PREDICTED UPSET COUNT :', quad_70)
print('QUAD 71 PREDICTED UPSET COUNT :', quad_71)
print('QUAD 72 PREDICTED UPSET COUNT :', quad_72)

make_corr_df(cs_curr_quad, curr_quad_df, curr_quad_no, 'curr')


# Get the count of the Champions from the quadrant IDs
quad_df = complete_stats.copy()
quad_df = quad_df.loc[quad_df['YEAR'].ne(curr_year) & quad_df['QUAD ID'].ne(0)]
quad_df = quad_df.groupby(by = ['ROUND', 'QUAD ID']).size().reset_index(name = 'TOTAL TEAMS')
quad_df = quad_df.loc[quad_df['ROUND'].le(2)]
quad_df


print(' 4 | 2' )
print('---|---')
print(' 3 | 1' )


quad_df = complete_stats.copy() 
quad_df = quad_df.groupby(['QUAD NO'])['BADJ EM'].max().reset_index(name = 'BADJ EM')
quad_df = pd.merge(quad_df, complete_stats, on = ['QUAD NO', 'BADJ EM'], how = 'left')
quad_df = quad_df[['YEAR', 'QUAD NO', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK', 'POWER-PATH', 'RELATIVE RATING']] 
# quad_df = quad_df.loc[quad_df['ROUND'].ge(16)]
# quad_df = quad_df.loc[quad_df['SEED'].ge(6)]
# quad_df = quad_df.loc[quad_df['YEAR'].ge(2013)]
#quad_df = quad_df.loc[quad_df['BADJ O RANK'].ge(20) | quad_df['BADJ D RANK'].ge(20)]
quad_df = quad_df.sort_values(by = 'YEAR', ascending = False)
quad_df = quad_df.reset_index(drop = True)

orig_map = plt.cm.get_cmap('RdYlGn') 
reversed_map = orig_map.reversed()  
quad_df = quad_df.style 
quad_df.background_gradient(subset = pd.IndexSlice[:, ['BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']], cmap = reversed_map)
quad_df.background_gradient(subset = pd.IndexSlice[:, ['POWER-PATH', 'RELATIVE RATING']], cmap = 'RdYlGn')


quad_df = complete_stats.copy()
quad_df = quad_df.loc[quad_df['ROUND'].le(4) & quad_df['QUAD NO'].isin([63, 57, 54, 49, 43, 41, 38, 35, 34, 30, 25, 22, 19, 18, 17, 10, 7])]
quad_df = quad_df[['YEAR', 'QUAD NO', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK', 'POWER-PATH', 'RELATIVE RATING']] 
quad_df 

