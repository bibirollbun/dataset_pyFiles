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


pd.set_option('display.max_rows', None)  


# dasdas 


# Create dataframes that will be used to create heatmaps displaying shooting rate vs shooting percent
def make_shots_df(cs, stat1, stat2, seed1, seed2, mode1, mode2) :
    shots_df = pd.DataFrame()

    # Mode Rank gets the dataframe sorted by shooting rate and percent rank
    if mode2 == 'rank' :
        shot_arr = [270, 180, 90, 1]
        bins_arr = ['a 270 + ' + str(stat1) + ' | a 270 + ' + str(stat2), 'a 270 + ' + str(stat1) + ' | b 180 - 270 ' + str(stat2), 'a 270 + ' + str(stat1) + ' | c 90 - 180 ' + str(stat2), 'a 270 + ' + str(stat1) + ' | d 1 - 90 ' + str(stat2),
                    'b 180 - 270 ' + str(stat1) + ' | a 270 + ' + str(stat2), 'b 180 - 270 ' + str(stat1) + ' | b 180 - 270 ' + str(stat2), 'b 180 - 270 ' + str(stat1) + ' | c 90 - 180 ' + str(stat2), 'b 180 - 270 ' + str(stat1) + ' | d 1 - 90 ' + str(stat2),
                    'c 90 - 180 ' + str(stat1) + ' | a 270 + ' + str(stat2), 'c 90 - 180 ' + str(stat1) + ' | b 180 - 270 ' + str(stat2), 'c 90 - 180 ' + str(stat1) + ' | c 90 - 180 ' + str(stat2), 'c 90 - 180 ' + str(stat1) + ' | d 1 - 90 ' + str(stat2),
                    'd 1 - 90 ' + str(stat1) + ' | a 270 + ' + str(stat2), 'd 1 - 90 ' + str(stat1) + ' | b 180 - 270 ' + str(stat2), 'd 1 - 90 ' + str(stat1) + ' | c 90 - 180 ' + str(stat2), 'd 1 - 90 ' + str(stat1) + ' | d 1 - 90 ' + str(stat2)]
        bin_index = 0

        for i in range(len(shot_arr)) :
            # Get all teams within a specific range of shooting rate ranks
            if i != 0 : rate_df = cs.loc[cs[stat1].lt(shot_arr[i - 1]) & cs[stat1].ge(shot_arr[i])]
            else : rate_df = cs.loc[cs[stat1].ge(shot_arr[i])]
            for j in range(len(shot_arr)) :
                shot_df = rate_df.copy()
                # Get all teams within a specific range of shooting percent ranks
                if j != 0 : shot_df = shot_df.loc[shot_df[stat2].lt(shot_arr[j - 1]) & shot_df[stat2].ge(shot_arr[j])]
                else : shot_df = shot_df.loc[shot_df[stat2].ge(shot_arr[j])]

                shot_df['BINS'] = bins_arr[bin_index]
                bin_index += 1
                shots_df = pd.concat([shots_df, shot_df], axis = 0)

    # Mode Perc gets the dataframe sorted by shooting rate and percent percentile
    elif mode2 == 'perc' :
        rate_arr = [cs[stat1].quantile(1), cs[stat1].quantile(0.75), cs[stat1].quantile(0.5), cs[stat1].quantile(0.25)]
        shot_arr = [cs[stat2].quantile(1), cs[stat2].quantile(0.75), cs[stat2].quantile(0.5), cs[stat2].quantile(0.25)]
        bins_arr = ['0 - 25 ' + str(stat1) + ' | 0 - 25 ' + str(stat2), '0 - 25 ' + str(stat1) + ' | 25 - 50 ' + str(stat2), '0 - 25 ' + str(stat1) + ' | 50 - 75 ' + str(stat2), '0 - 25 ' + str(stat1) + ' | 75 - 100 ' + str(stat2),
                    '25 - 50 ' + str(stat1) + ' | 0 - 25 ' + str(stat2), '25 - 50 ' + str(stat1) + ' | 25 - 50 ' + str(stat2), '25 - 50 ' + str(stat1) + ' | 50 - 75 ' + str(stat2), '25 - 50 ' + str(stat1) + ' | 75 - 100 ' + str(stat2),
                    '50 - 75 ' + str(stat1) + ' | 0 - 25 ' + str(stat2), '50 - 75 ' + str(stat1) + ' | 25 - 50 ' + str(stat2), '50 - 75 ' + str(stat1) + ' | 50 - 75 ' + str(stat2), '50 - 75 ' + str(stat1) + ' | 75 - 100 ' + str(stat2),
                    '75 - 100 ' + str(stat1) + ' | 0 - 25 ' + str(stat2), '75 - 100 ' + str(stat1) + ' | 25 - 50 ' + str(stat2), '75 - 100 ' + str(stat1) + ' | 50 - 75 ' + str(stat2), '75 - 100 ' + str(stat1) + ' | 75 - 100 ' + str(stat2)]
        bin_index = 0

        for i in range(len(rate_arr)) :
            # Get all teams within a specific range of shooting rate percentiles
            if i != 3 : rate_df = cs.loc[cs[stat1].le(rate_arr[i]) & cs[stat1].gt(rate_arr[i + 1])]
            else : rate_df = cs.loc[cs[stat1].le(rate_arr[i])]
            for j in range(len(shot_arr)) :
                shot_df = rate_df.copy()
                # Get all teams within a specific range of shooting percent percentiles
                if j != 3 : shot_df = shot_df.loc[shot_df[stat2].le(shot_arr[j]) & shot_df[stat2].gt(shot_arr[j + 1])]
                else : shot_df = shot_df.loc[shot_df[stat2].le(shot_arr[j])]

                shot_df['BINS'] = bins_arr[bin_index]
                bin_index += 1
                shots_df = pd.concat([shots_df, shot_df], axis = 0)

    shots_df['WINS'] = shots_df['ROUND']
    wins = [6, 5, 4, 3, 2, 1, 0]
    avg_wins_arr = []
    avg_wins_df = pd.DataFrame()

    # Convert round numbers to win count values
    for i in rounds_num_64 :
        shots_df.loc[shots_df['WINS'] == i, 'WINS'] = wins[rounds_num_64.index(i)]

    # Get the win count of every seed by bin
    wins_shots_df = shots_df.groupby(by = ['SEED', 'BINS', 'WINS']).size().reset_index(name = 'COUNT')
    wins_shots_df['TOTAL WINS'] = wins_shots_df['WINS'] * wins_shots_df['COUNT']

    # Get the average wins by bin for a specific seed
    if mode1 == 0 :
        for bins in bins_arr :
            temp_df = wins_shots_df.loc[wins_shots_df['SEED'].eq(seed1) & wins_shots_df['BINS'].eq(bins)]
            avg_wins_arr.append(temp_df['TOTAL WINS'].sum() / temp_df['COUNT'].sum())
    # Get the average wins by bin for a range of seeds
    elif mode1 == 1 :
        for bins in bins_arr :
            temp_df = wins_shots_df.loc[wins_shots_df['SEED'].ge(seed1) & wins_shots_df['SEED'].le(seed2) & wins_shots_df['BINS'].eq(bins)]
            avg_wins_arr.append(temp_df['TOTAL WINS'].sum() / temp_df['COUNT'].sum())

    avg_wins_df['BINS'] = bins_arr
    avg_wins_df['AVG WINS'] = avg_wins_arr
    # Split the title of bins to put them as x and y axis titles
    avg_wins_df[['RATE RANK', '% RANK']] = avg_wins_df['BINS'].str.split('|', expand = True)
    # Convert the dataframe into a pivot table for the purpose of formatting it for the heatmaps
    avg_wins_df = pd.pivot_table(avg_wins_df, values = 'AVG WINS', index = ['RATE RANK'], columns = ['% RANK'], fill_value = 0)

    return avg_wins_df


# Create heatmaps from the dataframes made from the make_shots_df() function
def make_heatmaps(cs, stat1, stat2, mode) :
    fig_arr = []
    
    # Set the x and y axis values to ranks
    if mode == 'rank' :
        x_bins = [' 270 + ' + str(stat2), ' 180 - 270 ' + str(stat2), ' 90 - 180 ' + str(stat2), ' 1 - 90 ' + str(stat2)]
        y_bins = ['270 + ' + str(stat1) + ' ', '180 - 270 ' + str(stat1) + ' ', '90 - 180 ' + str(stat1) + ' ', '1 - 90 ' + str(stat1) + ' ']
    # Set the x and y axis values to percentiles
    elif mode == 'perc' :
        x_bins = [' 0 - 25 ' + str(stat2), ' 25 - 50 ' + str(stat2), ' 50 - 75 ' + str(stat2), ' 75 - 100 ' + str(stat2)]
        y_bins = ['0 - 25 ' + str(stat1) + ' ', '25 - 50 ' + str(stat1) + ' ', '50 - 75 ' + str(stat1) + ' ', '75 - 100 ' + str(stat1) + ' ']

    # Create a heatmap for the single digit seeds
    shots_df = make_shots_df(cs, stat1, stat2, 1, 9, 1, mode)
    fig = px.imshow(shots_df, text_auto = True, color_continuous_scale = px.colors.diverging.RdYlGn, template = 'plotly_dark')
    fig.update_layout(title = '<b> ' + str(stat1) + ' VS ' + str(stat2) + ' (1 - 9 seeds) </b>', title_x = 0.5, title_font = dict(size = 15))
    fig_arr.append(fig)

    # Create a heatmap for the double digit seeds
    shots_df = make_shots_df(cs, stat1, stat2, 10, 15, 1, mode)
    fig = px.imshow(shots_df, text_auto = True, color_continuous_scale = px.colors.diverging.RdYlGn, template = 'plotly_dark')
    fig.update_layout(title = '<b> ' + str(stat1) + ' VS ' + str(stat2) + ' (10 - 15 seeds) </b>', title_x = 0.5, title_font = dict(size = 15))
    fig_arr.append(fig)

    return fig_arr 


# Get all of the current teams' bins
def get_curr_teams_bins(curr_cs, stat1, stat2, seed1, seed2) :
    # Get the teams within a specific seed range
    curr_cs = curr_cs.loc[curr_cs['SEED'].ge(seed1) & curr_cs['SEED'].le(seed2)]
    rank_arr = [270, 180, 90, 1]
    bins_arr = ['270 + ' + str(stat1) + ' | 270 + ' + str(stat2), '270 + ' + str(stat1) + ' | 180 - 270 ' + str(stat2), '270 + ' + str(stat1) + ' | 90 - 180 ' + str(stat2), '270 + ' + str(stat1) + ' | 1 - 90 ' + str(stat2),
                '180 - 270 ' + str(stat1) + ' | 270 + ' + str(stat2), '180 - 270 ' + str(stat1) + ' | 180 - 270 ' + str(stat2), '180 - 270 ' + str(stat1) + ' | 90 - 180 ' + str(stat2), '180 - 270 ' + str(stat1) + ' | 1 - 90 ' + str(stat2),
                '90 - 180 ' + str(stat1) + ' | 270 + ' + str(stat2), '90 - 180 ' + str(stat1) + ' | 180 - 270 ' + str(stat2), '90 - 180 ' + str(stat1) + ' | 90 - 180 ' + str(stat2), '90 - 180 ' + str(stat1) + ' | 1 - 90 ' + str(stat2),
                '1 - 90 ' + str(stat1) + ' | 270 + ' + str(stat2), '1 - 90 ' + str(stat1) + ' | 180 - 270 ' + str(stat2), '1 - 90 ' + str(stat1) + ' | 90 - 180 ' + str(stat2), '1 - 90 ' + str(stat1) + ' | 1 - 90 ' + str(stat2)]
    bin_index = 0

    for i in range(len(rank_arr)) :
        # Get all teams within a specific range of shooting rate ranks
        if i != 0 : rate_cs = curr_cs.loc[curr_cs[stat1].lt(rank_arr[i - 1]) & curr_cs[stat1].ge(rank_arr[i])]
        else : rate_cs = curr_cs.loc[curr_cs[stat1].ge(rank_arr[i])]
        for j in range(len(rank_arr)) :
            shot_cs = rate_cs.copy()
            # Get all teams within a specific range of shooting percent ranks
            if j != 0 : shot_cs = shot_cs.loc[shot_cs[stat2].lt(rank_arr[j - 1]) & shot_cs[stat2].ge(rank_arr[j])]
            else : shot_cs = shot_cs.loc[shot_cs[stat2].ge(rank_arr[j])]
            # Simplify the dataframe by only using the year, team, seed, conference, shooting rate stat, and shooting percent stat
            simp_curr_cs = shot_cs[['YEAR', 'TEAM', 'SEED', 'CONF', stat1, stat2]]
            print(bins_arr[bin_index])
            bin_index += 1
            display(simp_curr_cs)


cs = complete_stats.copy()
curr_cs = cs.loc[cs['YEAR'].eq(curr_year)]
cs = cs.loc[cs['YEAR'].ne(curr_year)]

stat1, stat2 = '2PTR RANK', '2PT% RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


stat1, stat2 = 'CLOSE TWOS SHARE RANK', 'CLOSE TWOS FG% RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


stat1, stat2 = 'FARTHER TWOS SHARE RANK', 'FARTHER TWOS FG% RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


stat1, stat2 = '3PTR RANK', '3PT% RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


stat1, stat2 = 'DUNKS SHARE RANK', 'DUNKS FG% RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


stat1, stat2 = 'FTR RANK', 'FT% RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


stat1, stat2 = '2PTRD RANK', '2PT%D RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


stat1, stat2 = 'CLOSE TWOS D SHARE RANK', 'CLOSE TWOS FG%D RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


stat1, stat2 = 'FARTHER TWOS D SHARE RANK', 'FARTHER TWOS FG%D RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


stat1, stat2 = '3PTRD RANK', '3PT%D RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


stat1, stat2 = 'DUNKS D SHARE RANK', 'DUNKS FG%D RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


stat1, stat2 = 'FTRD RANK', 'OP FT% RANK'
# get_curr_teams_bins(curr_cs, stat1, stat2, 1, 16)
fig = make_heatmaps(cs, stat1, stat2, 'rank')
fig[0].show()


fig[1].show()


# name = 'RANK DIFF'
# trends = stats.loc[stats['SEED'].isin([3]) & stats[name].lt(- 2) & stats['ROUND'].ne(68) & stats['YEAR'].ge(2008)]    
# first_column = trends.pop(name)
# trends.insert(0, name, first_column)     
# trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', name]]
# trends = trends.reset_index(drop = True)
# trends.head(1000)  


# name = 'KADJ D RANK'
# trends = complete_stats.loc[complete_stats['SEED'].isin([10]) & complete_stats['2PT% RANK'].ge(100) & complete_stats['3PT% RANK'].ge(100) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]    
# first_column = trends.pop(name)
# trends.insert(0, name, first_column)     
# trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', name]]
# trends = trends.reset_index(drop = True)
# trends.head(1000)  


# complete_stats.head()


trends = complete_stats.loc[complete_stats['SEED'].isin([6]) & (complete_stats['OREB% RANK'].le(105) & complete_stats['POWER-PATH'].ge(2.5)) & complete_stats['ROUND'].ne(0) & complete_stats['ROUND'].ne(68)] 
trends.groupby('ROUND').size()


trends = complete_stats.loc[complete_stats['OLD Z RATING RANK'].le(45) & complete_stats['OLD Z RATING RANK'].ge(35) & complete_stats['SEED'].ge(7) & (complete_stats['AN V 1-25 LOSS'].lt(10) & complete_stats['AST%'].gt(44.5) & complete_stats['DREB% RANK'].lt(340) & complete_stats['A HI'].lt(60) & complete_stats['PRESEASON KADJ D RANK'].lt(160) & complete_stats['PRESEASON KADJ O RANK'].lt(190) & complete_stats['PRESEASON KADJ EM RANK'].lt(140) & complete_stats['A BADJ EM RANK'].lt(100) & complete_stats['SOS RANK'].lt(155) & complete_stats['ELITE SOS'].gt(9) & complete_stats['BADJ D'].lt(103.5) & complete_stats['BADJ O'].gt(105) & complete_stats['BADJ O RANK'].lt(150) & complete_stats['KADJ D RANK'].lt(160) & complete_stats['D RATE'].gt(0) & complete_stats['TR RANK'].lt(80) & complete_stats['TOV% DIFF'].gt(- 4.7) & complete_stats['SEED'].lt(16)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'OLD Z RATING RANK', 'AN V 1-25 LOSS', 'AST%', 'DREB% RANK', 'A HI', 'PACE ADJUST', 'PRESEASON KADJ D RANK', 'PRESEASON KADJ O RANK', 'PRESEASON KADJ EM RANK', 'A BADJ EM RANK', 'SOS RANK', 'ELITE SOS', 'BADJ D', 'BADJ O', 'TOV% DIFF', 'TR RANK', 'BADJ O RANK', 'KADJ D RANK', 'D RATE']]
trends = trends.sort_values(by = ['ROUND'], ascending = False)
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['OLD Z RATING RANK'].le(8) & complete_stats['N OREB%'].gt(27.5) & complete_stats['TRUE TEMPO RANK'].gt(12) & complete_stats['OP AST%'].le(53.5) & complete_stats['AN TOV% DIFF'].gt(0) & complete_stats['LUCK RANK'].lt(300) & complete_stats['AN BADJ D RANK DIFF'].ge(- 30) & complete_stats['FTR RANK'].lt(325) & complete_stats['TOV% DIFF'].gt(- 4) & complete_stats['BADJ EM RANK'].lt(7) & complete_stats['O RATE'].gt(12) & complete_stats['2PT%'].gt(51) & complete_stats['TR RANK'].lt(15) & complete_stats['WAB RANK'].lt(15) & complete_stats['KADJ D RANK'].lt(50) & complete_stats['OREB%'].gt(28.75) & complete_stats['AN WIN%'].gt(55) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'OLD Z RATING RANK', 'N OREB%', '3PTRD RANK', 'TRUE TEMPO RANK', 'OP AST%', 'AN TOV% DIFF', 'LUCK RANK', 'AN BADJ D RANK DIFF', 'FTR RANK', 'TOV% DIFF', '2PT%', 'OREB%', 'AN WIN%', 'TR RANK', 'WAB RANK', 'BADJ EM RANK', 'KADJ D RANK', 'O RATE']]
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['OLD Z RATING RANK'].le(5) & (complete_stats['N OREB%'].le(27.5) | complete_stats['TRUE TEMPO RANK'].le(12) | complete_stats['OP AST%'].gt(53.5) | complete_stats['AN TOV% DIFF'].le(0) | complete_stats['LUCK RANK'].ge(300) | complete_stats['AN BADJ D RANK DIFF'].lt(- 30) | complete_stats['FTR RANK'].ge(325) | complete_stats['TOV% DIFF'].le(- 4) | complete_stats['BADJ EM RANK'].ge(7) |complete_stats['O RATE'].le(12) | complete_stats['2PT%'].le(51) | complete_stats['TR RANK'].ge(15) | complete_stats['WAB RANK'].ge(15) | complete_stats['KADJ D RANK'].ge(50) | complete_stats['OREB%'].le(28.75) | complete_stats['AN WIN%'].lt(55)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'OLD Z RATING RANK', 'N OREB%', '3PTRD RANK', 'TRUE TEMPO RANK', 'OP AST%', 'AN TOV% DIFF', 'LUCK RANK', 'AN BADJ D RANK DIFF', 'FTR RANK', 'TOV% DIFF', '2PT%', 'OREB%', 'AN WIN%', 'TR RANK', 'WAB RANK', 'BADJ EM RANK', 'KADJ D RANK', 'O RATE']]
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(4) & complete_stats['ROUND'].le(4) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'OREB%', 'AN WIN%', '1 AP RANK', '1 RANK?', 'TR RANK', 'WAB RANK', 'AN BADJ EM RANK', 'KADJ EM RANK', 'KADJ O RANK', 'KADJ D RANK', 'PRESEASON KADJ EM RANK']]
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(4) & complete_stats['OREB%'].ge(27.5) & complete_stats['AN WIN%'].ge(50) & complete_stats['1 RANK?'].ge(1) & complete_stats['WAB RANK'].le(15) & complete_stats['AN BADJ EM RANK'].le(20) & complete_stats['KADJ EM RANK'].le(20) & (complete_stats['KADJ O RANK'].le(20) | complete_stats['KADJ D RANK'].le(20)) & complete_stats['PRESEASON KADJ EM RANK'].le(32) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'OREB%', 'AN WIN%', '1 AP RANK', '1 RANK?', 'TR RANK', 'WAB RANK', 'AN BADJ EM RANK', 'KADJ EM RANK', 'KADJ O RANK', 'KADJ D RANK', 'PRESEASON KADJ EM RANK']]
trends2 = trends.reset_index(drop = True)
trends2.head(1000)


trends = complete_stats.loc[complete_stats['SEED'].le(4) & (complete_stats['OREB%'].lt(27.5) | complete_stats['AN WIN%'].lt(50) | complete_stats['1 RANK?'].le(0) | complete_stats['WAB RANK'].gt(15) | complete_stats['AN BADJ EM RANK'].gt(20) | complete_stats['KADJ EM RANK'].gt(20) | (complete_stats['KADJ O RANK'].gt(20) & complete_stats['KADJ D RANK'].gt(20)) | complete_stats['PRESEASON KADJ EM RANK'].gt(32)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'AN WIN%', '1 AP RANK', '1 RANK?', 'TR RANK', 'WAB RANK', 'AN BADJ EM RANK', 'KADJ EM RANK', 'KADJ O RANK', 'KADJ D RANK', 'PRESEASON KADJ EM RANK']]
trends3 = trends.reset_index(drop = True)
trends3.head(1000)


trends = complete_stats.loc[complete_stats['SEED'].gt(4) & complete_stats['ROUND'].le(4) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'SOS RANK', 'EFG%', 'AN WAB', 'AN WIN%', '1 AP RANK', '1 RANK?', 'TR RANK', 'WAB RANK', 'AN BADJ EM RANK', 'KADJ EM RANK', 'KADJ O RANK', 'KADJ D RANK', 'PRESEASON KADJ EM RANK']]
trends4 = trends.reset_index(drop = True)
trends4.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].gt(4) & complete_stats['SOS RANK'].le(120) & complete_stats['EFG%'].ge(46.5) & complete_stats['AN WAB'].ge(- 0.3) & complete_stats['AN WIN%'].ge(40) & complete_stats['WAB RANK'].le(60) & complete_stats['AN BADJ EM RANK'].le(50) & complete_stats['KADJ EM RANK'].le(50) & (complete_stats['KADJ O RANK'].le(35) | complete_stats['KADJ D RANK'].le(35)) & complete_stats['PRESEASON KADJ EM RANK'].le(100) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'SOS RANK', 'EFG%', 'AN WAB', 'AN WIN%', '1 AP RANK', '1 RANK?', 'TR RANK', 'WAB RANK', 'AN BADJ EM RANK', 'KADJ EM RANK', 'KADJ O RANK', 'KADJ D RANK', 'PRESEASON KADJ EM RANK']]
trends5 = trends.reset_index(drop = True)
trends5.head(1000)


trends = complete_stats.loc[complete_stats['SEED'].gt(4) & (complete_stats['SOS RANK'].gt(120) | complete_stats['EFG%'].lt(46.5) | complete_stats['AN WAB'].lt(- 0.3) | complete_stats['AN WIN%'].lt(40) | complete_stats['WAB RANK'].gt(60) | complete_stats['AN BADJ EM RANK'].gt(50) | complete_stats['KADJ EM RANK'].gt(50) | (complete_stats['KADJ O RANK'].gt(35) & complete_stats['KADJ D RANK'].gt(35)) | complete_stats['PRESEASON KADJ EM RANK'].gt(100)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'AN WAB', 'AN WIN%', '1 AP RANK', '1 RANK?', 'TR RANK', 'WAB RANK', 'AN BADJ EM RANK', 'KADJ EM RANK', 'KADJ O RANK', 'KADJ D RANK', 'PRESEASON KADJ EM RANK']]
trends6 = trends.reset_index(drop = True)
trends6.head(1000)


final_4_contenders = pd.concat([trends2, trends5])
final_4_contenders = final_4_contenders.reset_index(drop = True)
final_4_contenders


final_4_non_contenders = pd.concat([trends3, trends6])
final_4_non_contenders = final_4_non_contenders.reset_index(drop = True)
final_4_non_contenders


trends = complete_stats.loc[complete_stats['SEED'].le(7) & complete_stats['AN HI'].lt(26) & complete_stats['HI'].lt(30) & complete_stats['AN TR RANK'].lt(45) & complete_stats['3PT%'].gt(31) & complete_stats['TOV% DIFF'].gt(- 4) & complete_stats['WAB'].gt(1.6) & complete_stats['BADJ EM RANK'].le(35) & complete_stats['TR RANK'].lt(40) & ((complete_stats['V 26-50 WINS'] > complete_stats['V 26-50 LOSS']) | ((complete_stats['BADJ O RANK'].le(3) | complete_stats['BADJ D RANK'].le(3)) & (complete_stats['V 26-50 WINS'].le(complete_stats['V 26-50 LOSS'])))) & complete_stats['ROUND'].le(4) & complete_stats['YEAR'].ge(2011)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'AN HI', 'HI', 'AN TR RANK', '3PT%', 'TOV% DIFF', 'WAB', 'BADJ EM RANK', 'TR RANK', 'V 26-50 WINS', 'V 26-50 LOSS', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')   
trends7 = trends.reset_index(drop = True)
trends7.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(7) & complete_stats['AN HI'].lt(26) & complete_stats['HI'].lt(30) & complete_stats['AN TR RANK'].lt(45) & complete_stats['3PT%'].gt(31) & complete_stats['TOV% DIFF'].gt(- 4) & complete_stats['WAB'].gt(1.6) & complete_stats['BADJ EM RANK'].le(35) & complete_stats['TR RANK'].lt(40) & ((complete_stats['V 26-50 WINS'] > complete_stats['V 26-50 LOSS']) | ((complete_stats['BADJ O RANK'].le(3) | complete_stats['BADJ D RANK'].le(3)) & (complete_stats['V 26-50 WINS'].le(complete_stats['V 26-50 LOSS'])))) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'AN HI', 'HI', 'AN TR RANK', '3PT%', 'TOV% DIFF', 'WAB', 'BADJ EM RANK', 'TR RANK', 'V 26-50 WINS', 'V 26-50 LOSS', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')   
trends8 = trends.reset_index(drop = True)
trends8.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['WAB RANK'].ge(30) & complete_stats['BADJ O RANK'].gt(36) & complete_stats['BADJ D RANK'].gt(32)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK', 'WAB RANK']]
trends = trends.sort_values(by = 'SEED')   
trends9 = trends.reset_index(drop = True)
trends9.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['WAB RANK'].ge(30) & complete_stats['BADJ O RANK'].gt(36) & complete_stats['BADJ D RANK'].gt(32)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK', 'WAB RANK']]
trends = trends.sort_values(by = 'SEED')   
trends10 = trends.reset_index(drop = True)
trends10.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(5) & complete_stats['BADJ O RANK'].gt(20) & complete_stats['BADJ D RANK'].le(20) & complete_stats['BADJT RANK'].gt(275) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'BADJT RANK']]
trends = trends.sort_values(by = 'SEED')
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(5) & complete_stats['BADJ O RANK'].gt(20) & complete_stats['BADJ D RANK'].le(20) & complete_stats['BADJT RANK'].gt(275) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'BADJT RANK']] 
trends = trends.sort_values(by = 'SEED')
trends2 = trends.reset_index(drop = True)
trends2.head(1000)


trends = complete_stats.loc[complete_stats['SEED'].gt(5) & complete_stats['BADJ O RANK'].gt(20) & complete_stats['BADJ D RANK'].le(20) & complete_stats['BADJT RANK'].gt(275) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'BADJT RANK']]
trends = trends.sort_values(by = 'SEED')
trends3 = trends.reset_index(drop = True)
trends3.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].gt(5) & complete_stats['BADJ O RANK'].gt(20) & complete_stats['BADJ D RANK'].le(20) & complete_stats['BADJT RANK'].gt(275) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'BADJT RANK']] 
trends = trends.sort_values(by = 'SEED')
trends4 = trends.reset_index(drop = True)
trends4.head(1000)


trends = complete_stats.loc[complete_stats['SEED'].le(5) & complete_stats['BADJ O RANK'].le(20) & complete_stats['BADJ D RANK'].gt(40) & complete_stats['BADJT RANK'].le(50) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'BADJT RANK']]
trends = trends.sort_values(by = 'SEED')
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(5) & complete_stats['BADJ O RANK'].le(20) & complete_stats['BADJ D RANK'].gt(40) & complete_stats['BADJT RANK'].le(50) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'BADJT RANK']] 
trends = trends.sort_values(by = 'SEED')
trends2 = trends.reset_index(drop = True)
trends2.head(1000)


trends = complete_stats.loc[complete_stats['SEED'].gt(5) & complete_stats['BADJ O RANK'].le(20) & complete_stats['BADJ D RANK'].gt(60) & complete_stats['BADJT RANK'].le(60) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'BADJT RANK']]
trends = trends.sort_values(by = 'SEED')
trends3 = trends.reset_index(drop = True)
trends3.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].gt(5) & complete_stats['BADJ O RANK'].le(20) & complete_stats['BADJ D RANK'].gt(60) & complete_stats['BADJT RANK'].le(50) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'BADJT RANK']] 
trends = trends.sort_values(by = 'SEED')
trends4 = trends.reset_index(drop = True)
trends4.head(1000)


trends = complete_stats.loc[complete_stats['BADJ O RANK'].le(10) & complete_stats['BADJ D RANK'].le(10) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['BADJ O RANK'].le(10) & complete_stats['BADJ D RANK'].le(10) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')
trends2 = trends.reset_index(drop = True)
trends2.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(4) & (complete_stats['BADJ O RANK'].gt(20) | complete_stats['BADJ D RANK'].gt(45)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'TR RANK', '3PT%']]
trends = trends.sort_values(by = 'SEED')
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(4) & (complete_stats['BADJ O RANK'].gt(20) | complete_stats['BADJ D RANK'].gt(45)) & complete_stats['TR RANK'].gt(15) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'TR RANK']]
trends = trends.sort_values(by = 'SEED')
trends2 = trends.reset_index(drop = True)
trends2.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(4) & (complete_stats['BADJ O RANK'].gt(20) | complete_stats['BADJ D RANK'].gt(45)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'TR RANK']]
trends = trends.sort_values(by = 'SEED')
trends3 = trends.reset_index(drop = True)
trends3.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(4) & (complete_stats['BADJ O RANK'].gt(20) | complete_stats['BADJ D RANK'].gt(45)) & complete_stats['TR RANK'].gt(15) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'TR RANK']]
trends = trends.sort_values(by = 'SEED')
trends4 = trends.reset_index(drop = True)
trends4.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].gt(4) & complete_stats['BADJ O RANK'].lt(20) & complete_stats['BADJ D RANK'].lt(45) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'TR RANK']]
trends = trends.sort_values(by = 'SEED')
trends5 = trends.reset_index(drop = True)
trends5.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].gt(4) & complete_stats['BADJ O RANK'].lt(20) & complete_stats['BADJ D RANK'].lt(45) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'TR RANK']]
trends = trends.sort_values(by = 'SEED')
trends6 = trends.reset_index(drop = True)
trends6.head(1000)  


trends = complete_stats.loc[(complete_stats['SEED'].lt(15)) & (complete_stats['3PT% RANK'].gt(20) & complete_stats['3PT%D RANK'].gt(90)) & complete_stats['3PTR RANK'].gt(220) & complete_stats['3PTRD RANK'].gt(240) & complete_stats['BADJ EM RANK'].gt(3) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', '3PT% RANK', '3PT%D RANK', '3PTR RANK', '3PTRD RANK', 'BADJ EM RANK']]
trends = trends.sort_values(by = 'SEED')
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[(complete_stats['SEED'].lt(15)) & (complete_stats['3PT% RANK'].gt(20) & complete_stats['3PT%D RANK'].gt(90)) & complete_stats['3PTR RANK'].gt(220) & complete_stats['3PTRD RANK'].gt(240) & complete_stats['BADJ EM RANK'].gt(3) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', '3PT% RANK', '3PT%D RANK', '3PTR RANK', '3PTRD RANK']]
trends = trends.sort_values(by = 'SEED')
trends2 = trends.reset_index(drop = True)
trends2.head(1000)  


trends = complete_stats.loc[(complete_stats['SEED'].lt(12)) & complete_stats['3PT%'].lt(35) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2014)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', '3PT%']]
trends = trends.sort_values(by = 'SEED')
trends3 = trends.reset_index(drop = True)
trends3.head(1000)  


trends = complete_stats.loc[(complete_stats['SEED'].lt(12)) & complete_stats['3PT%'].lt(35) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', '3PT%']]
trends = trends.sort_values(by = 'SEED')
trends4 = trends.reset_index(drop = True)
trends4.head(1000)  


trends = complete_stats.loc[(complete_stats['SEED'].lt(5)) & (complete_stats['2PT%'].lt(49.9) | complete_stats['3PT%'].lt(35)) & (complete_stats['2PT%'].lt(56) & complete_stats['3PT%'].lt(40)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2014)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', '2PT%', '3PT%']]
trends = trends.sort_values(by = 'SEED')
trends5 = trends.reset_index(drop = True)
trends5.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].ge(5) & complete_stats['SEED'].le(12) & (complete_stats['2PT%'].lt(49) | complete_stats['3PT%'].lt(34)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2015)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', '2PT%', '3PT%', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')
trends6 = trends.reset_index(drop = True)
trends6.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].lt(5) & (complete_stats['2PT%'].lt(49.9) | complete_stats['3PT%'].lt(35)) & (complete_stats['2PT%'].lt(56) & complete_stats['3PT%'].lt(40)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', '2PT%', '3PT%']]
trends = trends.sort_values(by = 'SEED')
trends8 = trends.reset_index(drop = True)
trends8.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].ge(5) & complete_stats['SEED'].le(12) & (complete_stats['3PT%'].lt(34.4) & complete_stats['BADJ D RANK'].gt(3)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2015)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', '2PT%', '3PT%', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')
trends7 = trends.reset_index(drop = True)
trends7.head(1000)  


trends = complete_stats.loc[(complete_stats['SEED'].ge(5) & complete_stats['SEED'].le(12)) & (complete_stats['2PT%'].lt(49.25) | complete_stats['3PT%'].lt(34)) & (complete_stats['2PT%'].lt(56) & complete_stats['3PT%'].lt(40)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', '2PT%', '3PT%', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')
trends9 = trends.reset_index(drop = True)
trends9.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].ge(5) & complete_stats['SEED'].le(12) & (complete_stats['3PT%'].lt(34.4) & complete_stats['BADJ D RANK'].gt(3)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', '2PT%', '3PT%', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')
trends10 = trends.reset_index(drop = True)
trends10.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].lt(15) & complete_stats['FT% RANK'].gt(150) & complete_stats['FTR RANK'].gt(190) & complete_stats['FTRD RANK'].gt(200) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'FT% RANK', 'FTR RANK', 'FTRD RANK']]
trends = trends.sort_values(by = 'SEED')
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].lt(15) & complete_stats['FT% RANK'].gt(150) & complete_stats['FTR RANK'].gt(190) & complete_stats['FTRD RANK'].gt(200) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'FT% RANK', 'FTR RANK', 'FTRD RANK']]
trends = trends.sort_values(by = 'SEED') 
trends2 = trends.reset_index(drop = True)
trends2.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].ge(7) & complete_stats['SEED'].le(12) & (complete_stats['EXP'].le(1.8) & complete_stats['TALENT'].lt(80) & complete_stats['BADJ EM RANK'].gt(20) & complete_stats['TR RANK'].gt(23) & complete_stats['TOV% DIFF'].lt(4.15) & complete_stats['3PT% DIFF'].lt(10) & (complete_stats['PRESEASON KADJ EM RANK'].ge(15) | complete_stats['PRESEASON KADJ EM RANK'].isna()) & (complete_stats['POWER-PATH'].lt(- 0.5) | complete_stats['POWER-PATH'].isna())) & complete_stats['BADJ O RANK'].gt(3) & complete_stats['BADJ D RANK'].gt(3) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'PRESEASON KADJ EM RANK', '3PT% DIFF', 'EXP', 'TALENT', 'BADJ EM RANK', 'TR RANK', 'POWER-PATH', 'TOV% DIFF']]
trends = trends.sort_values(by = 'SEED')
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(6) & (complete_stats['EXP'].le(1.8) & complete_stats['TALENT'].lt(80) & complete_stats['BADJ EM RANK'].gt(20) & complete_stats['TR RANK'].gt(23) & complete_stats['TOV% DIFF'].lt(4.15) & complete_stats['3PT% DIFF'].lt(10) & (complete_stats['PRESEASON KADJ EM RANK'].ge(15) | complete_stats['PRESEASON KADJ EM RANK'].isna()) & (complete_stats['POWER-PATH'].lt(- 0.5) | complete_stats['POWER-PATH'].isna())) & complete_stats['BADJ O RANK'].gt(3) & complete_stats['BADJ D RANK'].gt(3) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'PRESEASON KADJ EM RANK', '3PT% DIFF', 'EXP', 'TALENT', 'BADJ EM RANK', 'TR RANK', 'POWER-PATH', 'TOV% DIFF']]
trends = trends.sort_values(by = 'SEED')
trends2 = trends.reset_index(drop = True)
trends2.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['EXP'].le(1.8) & complete_stats['TALENT'].lt(80) & complete_stats['BADJ EM RANK'].gt(20) & complete_stats['TR RANK'].gt(23) & complete_stats['TOV% DIFF'].lt(4.15) & complete_stats['3PT% DIFF'].lt(10) & (complete_stats['PRESEASON KADJ EM RANK'].ge(15) | complete_stats['PRESEASON KADJ EM RANK'].isna()) & (complete_stats['POWER-PATH'].lt(- 0.5) | complete_stats['POWER-PATH'].isna())) & complete_stats['BADJ O RANK'].gt(3) & complete_stats['BADJ D RANK'].gt(3) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'PRESEASON KADJ EM RANK', '3PT% DIFF', 'EXP', 'TALENT', 'BADJ EM RANK', 'TR RANK', 'POWER-PATH', 'TOV% DIFF']]
trends = trends.sort_values(by = 'SEED')
trends3 = trends.reset_index(drop = True)
trends3.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].lt(15) & complete_stats['EXP'].lt(1.62) & complete_stats['TALENT'].lt(78) & complete_stats['BADJ O RANK'].gt(3) & complete_stats['BADJ D RANK'].gt(3) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'EXP', 'TALENT']]
trends = trends.sort_values(by = 'SEED')
trends4 = trends.reset_index(drop = True)
trends4.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].lt(15) & complete_stats['EXP'].lt(1.62) & complete_stats['TALENT'].lt(78) & complete_stats['BADJ O RANK'].gt(3) & complete_stats['BADJ D RANK'].gt(3) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ O RANK', 'BADJ D RANK', 'EXP', 'TALENT']]
trends = trends.sort_values(by = 'SEED')   
trends5 = trends.reset_index(drop = True)
trends5.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ EM RANK'] > (complete_stats['RELATIVE RATING RANK'] + 10)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'RELATIVE RATING RANK']]
trends = trends.sort_values(by = 'SEED')   
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ EM RANK'] > (complete_stats['RELATIVE RATING RANK'] + 10)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'RELATIVE RATING RANK']]
trends = trends.sort_values(by = 'SEED')   
trends2 = trends.reset_index(drop = True)
trends2.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['KADJ EM RANK'] > (complete_stats['RELATIVE RATING RANK'] + 8)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'KADJ EM RANK', 'RELATIVE RATING RANK']]
trends = trends.sort_values(by = 'SEED')   
trends3 = trends.reset_index(drop = True)
trends3.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['KADJ EM RANK'] > (complete_stats['RELATIVE RATING RANK'] + 8)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'KADJ EM RANK', 'RELATIVE RATING RANK']]
trends = trends.sort_values(by = 'SEED')   
trends4 = trends.reset_index(drop = True)
trends4.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & ((complete_stats['TR RANK'] + 10) < complete_stats['RELATIVE RATING RANK']) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'TR RANK', 'RELATIVE RATING RANK']]
trends = trends.sort_values(by = 'SEED')   
trends5 = trends.reset_index(drop = True)
trends5.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & ((complete_stats['TR RANK'] + 10) < complete_stats['RELATIVE RATING RANK']) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'TR RANK', 'RELATIVE RATING RANK']]
trends = trends.sort_values(by = 'SEED')   
trends6 = trends.reset_index(drop = True)
trends6.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['KADJ EM RANK'] > (complete_stats['TR RANK'] + 6)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'KADJ EM RANK', 'TR RANK']]
trends = trends.sort_values(by = 'SEED')   
trends7 = trends.reset_index(drop = True)
trends7.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['KADJ EM RANK'] > (complete_stats['TR RANK'] + 6)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'KADJ EM RANK', 'TR RANK']]
trends = trends.sort_values(by = 'SEED')   
trends8 = trends.reset_index(drop = True)
trends8.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ EM RANK'] > (complete_stats['KADJ EM RANK'] + 6)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'KADJ EM RANK', 'BADJ EM RANK']]
trends = trends.sort_values(by = 'SEED')   
trends9 = trends.reset_index(drop = True)
trends9.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ EM RANK'] > (complete_stats['KADJ EM RANK'] + 6)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'KADJ EM RANK', 'BADJ EM RANK']]
trends = trends.sort_values(by = 'SEED')   
trends10 = trends.reset_index(drop = True)
trends10.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & ((complete_stats['BADJ EM RANK'] + 16) < complete_stats['WAB RANK']) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'WAB RANK']]
trends = trends.sort_values(by = 'SEED')   
trends11 = trends.reset_index(drop = True)
trends11.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & ((complete_stats['BADJ EM RANK'] + 16) < complete_stats['WAB RANK']) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'WAB RANK']]
trends = trends.sort_values(by = 'SEED')   
trends12 = trends.reset_index(drop = True)
trends12.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & ((complete_stats['KADJ EM RANK'] + 13) < complete_stats['WAB RANK']) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'KADJ EM RANK', 'WAB RANK']]
trends = trends.sort_values(by = 'SEED')   
trends13 = trends.reset_index(drop = True)
trends13.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & ((complete_stats['KADJ EM RANK'] + 13) < complete_stats['WAB RANK']) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'KADJ EM RANK', 'WAB RANK']]
trends = trends.sort_values(by = 'SEED')   
trends14 = trends.reset_index(drop = True)
trends14.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & ((complete_stats['TR RANK'] + 12) < complete_stats['WAB RANK']) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'TR RANK', 'WAB RANK']]
trends = trends.sort_values(by = 'SEED')   
trends15 = trends.reset_index(drop = True)
trends15.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & ((complete_stats['TR RANK'] + 12) < complete_stats['WAB RANK']) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'TR RANK', 'WAB RANK']]
trends = trends.sort_values(by = 'SEED')   
trends16 = trends.reset_index(drop = True)
trends16.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(5) & (complete_stats['AN HI'].lt(5.5) & complete_stats['TOV% DIFF'].gt(- 4) & complete_stats['KADJ O RANK'].lt(20) & complete_stats['KADJ D RANK'].lt(20) & complete_stats['BADJ O RANK'].lt(23) & complete_stats['BADJ D RANK'].lt(25) & complete_stats['TR RANK'].lt(17) & complete_stats['1 AP VOTES'].ge(90)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'AN HI', 'TR RANK', 'BADJ O RANK', 'BADJ D RANK', 'KADJ O RANK', 'KADJ D RANK', '1 AP VOTES', 'TOV% DIFF']]
trends = trends.sort_values(by = 'ROUND')   
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(5) & (complete_stats['AN HI'].lt(5.5) & complete_stats['TOV% DIFF'].gt(- 4) & complete_stats['KADJ O RANK'].lt(20) & complete_stats['KADJ D RANK'].lt(20) & complete_stats['BADJ O RANK'].lt(23) & complete_stats['BADJ D RANK'].lt(25) & complete_stats['TR RANK'].lt(17) & complete_stats['1 AP VOTES'].ge(90)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'AN HI', 'TR RANK', 'BADJ O RANK', 'BADJ D RANK', 'KADJ O RANK', 'KADJ D RANK', '1 AP VOTES', 'TOV% DIFF']]
trends = trends.sort_values(by = 'ROUND')   
trends2 = trends.reset_index(drop = True)
trends2.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(1) & (complete_stats['V 1-25 WINS'].gt(3) & complete_stats['HI'].le(4) & complete_stats['AN HI'].le(3.5) & complete_stats['OREB%'].gt(28)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'V 1-25 WINS', 'HI', 'AN HI', 'OREB%']]
trends = trends.sort_values(by = 'ROUND')   
trends3 = trends.reset_index(drop = True)
trends3.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(1) & (complete_stats['V 1-25 WINS'].gt(3) & complete_stats['HI'].le(4) & complete_stats['AN HI'].le(3.5) & complete_stats['OREB%'].gt(28)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'V 1-25 WINS', 'HI', 'AN HI', 'OREB%']]
trends = trends.sort_values(by = 'ROUND')   
trends4 = trends.reset_index(drop = True)
trends4.head(1000)    


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ O RANK'].le(10) & complete_stats['BADJ D RANK'].ge(50)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')   
trends1 = trends.reset_index(drop = True)
trends1.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ O RANK'].le(10) & complete_stats['BADJ D RANK'].ge(50)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')   
trends2 = trends.reset_index(drop = True)
trends2.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ O RANK'].le(10) & complete_stats['BADJ D RANK'].ge(50) & complete_stats['BADJ EM RANK'].gt(10)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')   
trends3 = trends.reset_index(drop = True)
trends3.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ O RANK'].le(10) & complete_stats['BADJ O RANK'].gt(4) & complete_stats['BADJ D RANK'].ge(50)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')   
trends4 = trends.reset_index(drop = True)
trends4.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ O RANK'].le(10) & complete_stats['BADJ D RANK'].ge(50)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'ROUND')   
trends5 = trends.reset_index(drop = True)
trends5.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ O RANK'].le(10) & complete_stats['BADJ D RANK'].ge(50) & complete_stats['BADJ EM RANK'].gt(10)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'ROUND')   
trends6 = trends.reset_index(drop = True)
trends6.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ O RANK'].le(10) & complete_stats['BADJ O RANK'].gt(4) & complete_stats['BADJ D RANK'].ge(50)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')   
trends7 = trends.reset_index(drop = True)
trends7.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ D RANK'].le(10) & complete_stats['BADJ O RANK'].ge(50)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')   
trends8 = trends.reset_index(drop = True)
trends8.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ D RANK'].le(10) & complete_stats['BADJ D RANK'].gt(3) & complete_stats['BADJ O RANK'].ge(50)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].ge(2008)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')   
trends9 = trends.reset_index(drop = True)
trends9.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ D RANK'].le(10) & complete_stats['BADJ O RANK'].ge(50)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')   
trends10 = trends.reset_index(drop = True)
trends10.head(1000)  


trends = complete_stats.loc[complete_stats['SEED'].le(12) & (complete_stats['BADJ D RANK'].le(10) & complete_stats['BADJ D RANK'].gt(3) & complete_stats['BADJ O RANK'].ge(50)) & complete_stats['ROUND'].ne(68) & complete_stats['YEAR'].eq(curr_year)]      
trends = trends[['YEAR', 'TEAM', 'SEED', 'ROUND', 'BADJ EM RANK', 'BADJ O RANK', 'BADJ D RANK']]
trends = trends.sort_values(by = 'SEED')   
trends11 = trends.reset_index(drop = True)
trends11.head(1000)  

