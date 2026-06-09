# Importing Libraries  
from functools import reduce
from IPython.display import display, HTML  
import numpy as np 
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go 
import plotly.io as pio  
import plotly.offline as py   
from plotly.subplots import make_subplots 
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
ordered_tournament_matchups = ordered_tournament_matchups.sort_index(ascending = True)
ordered_tournament_matchups2 = ordered_tournament_matchups.copy()

ordered_tournament_matchups.head()


# Scale variables between a value of 0 to 1 for the tournament matchups  
scaler = MinMaxScaler() 
removed_cols = ['YEAR', 'BY YEAR NO', 'BY ROUND NO', 'TEAM NO', 'TEAM', 'SEED', 'ROUND', 'CURRENT ROUND', 'CONF', 'CONF ID', 'QUAD NO', 'QUAD ID', 'TEAM ID', 'BID TYPE', 'OUTCOME']   
selected_cols = ordered_tournament_matchups.columns[~ordered_tournament_matchups.columns.isin(removed_cols)]
ordered_tournament_matchups[selected_cols] = scaler.fit_transform(ordered_tournament_matchups[selected_cols])  
ordered_tournament_matchups.head()


# Scale variables between a value of 0 to 1 for statistics of all teams  
complete_stats_scaled = complete_stats.copy()
selected_cols = complete_stats_scaled.columns[~complete_stats_scaled.columns.isin(removed_cols)] 
complete_stats_scaled[selected_cols] = scaler.fit_transform(complete_stats_scaled[selected_cols])  
complete_stats_scaled.head()


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


# dasdasa 


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
            
    if y_col == 'EASY DRAW' or y_col == 'TOUGH DRAW' or y_col == 'DARK HORSE' or y_col == 'UPSET ALERT' or y_col == 'CINDERELLA':  
        mask = temp_df[y_col] != False  
        temp_df = temp_df[mask]
        
        y_col2 = 'TEAM NO'
        fig = px.scatter(temp_df, x = 'ROUND', y = y_col2, title = '<b> Past Tournament Teams </b>', hover_data = ['YEAR', 'TEAM'],
                         animation_frame = 'SEED', color = 'SEED', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark',
                         height = 650)
        fig['layout'].pop('updatemenus')
        fig.update_layout(title_x = 0.5, showlegend = False)
        fig.update_xaxes(categoryorder = 'array', categoryarray = mod_round_str, autorange = 'reversed')
        fig.add_vrect(x0 = 6.5, x1 = 7.5, fillcolor = 'green', opacity = 0.25, line_width = 0)
        fig.add_vline(x = 6.5)   
        var = 0 
    else : 
        fig = px.scatter(temp_df, x = 'ROUND', y = y_col, title = '<b> Past Tournament Teams </b>', hover_data = ['YEAR', 'TEAM'],
                     animation_frame = 'SEED', color = 'SEED', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark',
                     height = 650)
        fig['layout'].pop('updatemenus')
        fig.update_layout(title_x = 0.5, showlegend = False)
        fig.update_xaxes(categoryorder = 'array', categoryarray = mod_round_str, autorange = 'reversed')
        fig.add_vrect(x0 = 6.5, x1 = 7.5, fillcolor = 'green', opacity = 0.25, line_width = 0)
        fig.add_vline(x = 6.5)
        var = 1  

    # Changing the value for the upper and lower y axis value depending on the stat for the purpose of making the plots clearer
    if var == 1 : 
        if 'R SCORE' in y_col : val = 0.5
        elif 'DRAW' in y_col : val = 0.5
        elif 'VAL Z-SCORE' in y_col : val = 0.5
        elif 'BARTHAG' in y_col : val = 0.02
        elif y_col == 'EXP' or 'HGT' in y_col : val = 0.2
        elif 'PPP' in y_col : val = 0.02
        elif 'RPPF RATING' in y_col : val = 0.02 
        elif 'NPB RATING' in y_col : val = 0.02 
        elif ' VOTES' in y_col : val = 20 
        elif 'R SOS' in y_col : val = 0.02 
        elif ' RANK' in y_col :
            if 'WAB RANK' : val = 2
            else : val = 20
        else : val = 2 

    if var == 1 : 
        go.Figure(data = fig.data, frames = [fr.update(layout =
                 {'xaxis': {'range' : [7.5, - 0.5]},
                  'yaxis': {'range' : [min(fr.data[0].y) - val, max(fr.data[0].y) + val]},}) for fr in fig.frames], layout = fig.layout)
        fig_arr.append(fig)
    else : 
        go.Figure(data = fig.data, frames = [fr.update(layout =
                {'xaxis': {'range' : [7.5, - 0.5]},
                 'yaxis': {'range' : [300, 1200]},}) for fr in fig.frames], layout = fig.layout)
        fig_arr.append(fig)

    if y_col == 'EASY DRAW' or y_col == 'TOUGH DRAW' or y_col == 'DARK HORSE' or y_col == 'UPSET ALERT' or y_col == 'CINDERELLA' : return fig_arr  

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


def display_plots_2(df, curr_df, x_col, y_col, mode) :
    temp_df = df.copy()
    temp_df['SEED'] = temp_df['SEED'].astype(str)
    fig_arr = []

    if 'WAB' in x_col :
        if 'WAB RANK' in x_col : pass
        # Remove the 2021 tournament year for WAB visualization as teams did not play all games that year
        else : temp_df = temp_df.loc[temp_df['YEAR'].ne(2021)]
    
    if 'WAB' in y_col :
        if 'WAB RANK' in y_col : pass
        # Remove the 2021 tournament year for WAB visualization as teams did not play all games that year
        else : temp_df = temp_df.loc[temp_df['YEAR'].ne(2021)]
    
    if mode == 'normal' : 
        # temp_df['SEED'] = temp_df['SEED'].astype(int)
        fig = px.scatter(temp_df, x = x_col, y = y_col, title = '<b> Past Tournament Teams </b>', hover_data = ['YEAR', 'TEAM'],
                     animation_frame = 'YEAR', symbol = 'ROUND', color = 'SEED', color_discrete_sequence = px.colors.qualitative.Light24, template = 'plotly_dark',
                     height = 650)
        fig['layout'].pop('updatemenus')
        fig.update_layout(title_x = 0.5, showlegend = False)

    # Changing the value for the upper and lower x / y axis value depending on the stat for the purpose of making the plots clearer
    if 'R SCORE' in x_col : x_val = 0.5
    elif 'DRAW' in x_col : x_val = 0.5
    elif 'VAL Z-SCORE' in x_col : x_val = 0.5
    elif 'BARTHAG' in x_col : x_val = 0.02
    elif x_col == 'EXP' or 'HGT' in x_col : x_val = 0.2
    elif 'PPP' in x_col : x_val = 0.02
    elif ' RANK' in x_col :
        if 'WAB RANK' in x_col : x_val = 2
        else : x_val = 5
    else : x_val = 5 

    if 'R SCORE' in y_col : y_val = 0.5
    elif 'DRAW' in y_col : y_val = 0.5
    elif 'VAL Z-SCORE' in y_col : y_val = 0.5
    elif 'BARTHAG' in y_col : y_val = 0.02
    elif y_col == 'EXP' or 'HGT' in y_col : y_val = 0.2
    elif 'PPP' in y_col : y_val = 0.02
    elif ' RANK' in y_col :
        if 'WAB RANK' in y_col : y_val = 2
        else : y_val = 5
    else : y_val = 5 

    go.Figure(data = fig.data, frames = [fr.update(layout =
             {'xaxis': {'range' : [0, 70]}, 
              'yaxis': {'range' : [0, 70]},}) for fr in fig.frames], layout = fig.layout) 
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


# fig = display_plots_2(stats, curr_stats, 'Z RATING RANK', 'NCAA S-RANK', 'normal')
# fig[0].show()


# fig = display_plots_2(stats, curr_stats, 'KADJ EM', 'BADJ EM', 'normal')
# fig[0].show()


fig = display_plots(stats, curr_stats, 'KADJ EM', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'BADJ EM', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN BADJ EM', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'BARTHAG', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN BARTHAG', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'KADJ O', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'BADJ O', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN BADJ O', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'KADJ D', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'BADJ D', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN BADJ D', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'A BADJT RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'WAB', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'WAB RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN WAB', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'TR RANK', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'HI', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'LO', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN TR RANK', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN HI', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN LO', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'SOS RANK', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'SOS HI', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'SOS LO', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'LUCK RANK', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'LUCK HI', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'LUCK LO', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'CONSISTENCY RANK', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'CONSISTENCY HI', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'CONSISTENCY LO', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'RELATIVE RATING', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'O RATE', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'D RATE', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'OPPONENT ADJUST', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'PACE ADJUST', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'TRUE TEMPO', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'HOME RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'KILLSHOTS PER GAME', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'KILL SHOTS CONCEDED PER GAME', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'KILLSHOTS MARGIN', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'RPPF RATING', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'NPB RATING', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'RADJ EM', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'RADJ O', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'RADJ D', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'R PACE', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'R SOS', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'STREM', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'STREM RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'STROE', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'STRDE', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'STRT+ RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'OLD Z RATING', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'NEW Z RATING', x0, x1, 'forward')
fig[0].show()


fig[1].show()


stats1 = hcti_df.copy()
stats1 = order_df(df_input = stats1, order_by = 'SEED', order = order_seed)
change_round_str(stats1, mod_rounds_num, mod_round_str)

curr_stats1 = hcti_df.copy()
curr_stats1 = curr_stats1.loc[curr_stats1['YEAR'].eq(curr_year)]
curr_stats1 = order_df(df_input = curr_stats1, order_by = 'SEED', order = order_seed)


fig = display_plots(stats, curr_stats, 'POWER', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'PATH', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'POWER-PATH', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'DRAW', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'POOL VALUE', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'POOL S-RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'NCAA S-RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'VAL Z-SCORE', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'NET', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'KPI', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'SOR', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'RESUME AVG', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'BPI', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'QUALITY AVG', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q1A W', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q1A L', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q1 W', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q1 L', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q2 W', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q2 L', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q1&2 W', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q1&2 L', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q3 W', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q3 L', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q4 W', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'Q4 L', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'NET RPI', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'RESUME', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'ELO', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'B POWER', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'PLUS 500', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'R SCORE', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'OREB% RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'DREB% RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'TOV% RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'TOV%D', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AST% RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'OP AST% RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'BLKED%', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'BLK%', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, '2PT%', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, '2PT%D', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, '2PTR RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, '2PTRD RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, '3PT%', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, '3PT%D', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, '3PTR RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, '3PTRD RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'FT% RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'FTR RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'FTRD RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'EFG% RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'EFG%D', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'DUNKS FG%', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'DUNKS FG%D RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'DUNKS SHARE', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'DUNKS D SHARE RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'CLOSE TWOS FG%', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'CLOSE TWOS FG% RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'CLOSE TWOS SHARE', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'CLOSE TWOS D SHARE RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'FARTHER TWOS FG%', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'FARTHER TWOS FG%D RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'FARTHER TWOS SHARE RANK', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'FARTHER TWOS D SHARE RANK', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'WIN%', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN WIN%', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'ELITE SOS', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN ELITE SOS', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'TALENT', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'EXP', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AVG HGT', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'EFF HGT', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'PPPO', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN PPPO', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'PPPD', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN PPPD', x0, x1, 'backward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'TOV% DIFF', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'A BADJ EM RANK DIFF', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN BADJ EM RANK DIFF', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'A BADJ O RANK DIFF', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN BADJ O RANK DIFF', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'A BADJ D RANK DIFF', x0, x1, 'forward')
fig[0].show()


fig[1].show()


fig = display_plots(stats, curr_stats, 'AN BADJ D RANK DIFF', x0, x1, 'forward')
fig[0].show()


fig[1].show()

