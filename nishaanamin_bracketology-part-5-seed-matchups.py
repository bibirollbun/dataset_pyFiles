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


tournament_matchups.to_csv('tournament_matchups.csv', index = False) 


# win1 = win.loc[win['YEAR'].ge(2011)]
# print(len(win1))

# loss1 = loss.loc[loss['YEAR'].ge(2011)]
# print(len(loss1)) 


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
removed_cols = ['YEAR', 'BY YEAR NO', 'TEAM NO', 'TEAM', 'SEED', 'ROUND', 'CURRENT ROUND', 'CONF', 'CONF ID', 'QUAD NO', 'QUAD ID', 'TEAM ID', 'BID TYPE', 'OUTCOME']   
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


# Create dataframe consisting of the average upset and non upset differentials of specific matchups
###################################################################################################

# Losing Seeds
seed1 = [8, 7,  6,  5,  4,  3,  2,  1, 1, 2, 2,  3, 3,  4, 1, 1]
# Winning Seeds
seed2 = [9, 10, 11, 12, 13, 14, 15, 8, 9, 7, 10, 6, 11, 5, 4, 5]
seed_matchups = pd.DataFrame()

diff = differentials_combined_rows.copy()

selected_cols = diff.columns[~diff.columns.isin(diff.iloc[:, diff.columns.get_loc('L YEAR') : diff.columns.get_loc('L SCORE') + 1])]
cols = diff[selected_cols]
cols = cols.iloc[:, diff.columns.get_loc('W K TEMPO') :]
cols.head()

for i in range(len(seed1)) :
    # Get the seed matchup
    matchups = diff.loc[diff['W SEED'].isin([seed1[i], seed2[i]]) & diff['L SEED'].isin([seed1[i], seed2[i]])]
    matchups = matchups.loc[matchups['W YEAR'].ne(curr_year)]

    non_upset = matchups.loc[matchups['W SEED'].isin([seed1[i]])]   # Get the non upset games
    upset = matchups.loc[matchups['W SEED'].isin([seed2[i]])]       # Get the upset games

    # Get the average stat values for the upset and non upset matchups
    mean = non_upset[cols.columns].mean()
    seed_matchups = pd.concat([seed_matchups, mean], axis = 1)

    mean = - upset[cols.columns].mean()
    seed_matchups = pd.concat([seed_matchups, mean], axis = 1)

seed_matchups = seed_matchups.T
seed_matchups = seed_matchups.reset_index(drop = True)
rank_cols = [col for col in seed_matchups.columns if 'RANK' in col]
# Remove the rank stats as it makes the matchups stats redundant
removed_cols = seed_matchups.columns[~seed_matchups.columns.isin(rank_cols)]
seed_matchups = seed_matchups[removed_cols]
seed_matchups['TYPE'] = ''

seed_matchups.iloc[::2, - 1:] = 'NON UPSET'
seed_matchups.iloc[1::2, - 1:] = 'UPSET'

# Remove all of the losing teams
seed_matchups = seed_matchups.loc[:, ~seed_matchups.columns.str.startswith('L')]


# Create dataframe consisting of the top 20 statistics with the biggest differentials between the upsets and non upsets
#######################################################################################################################

diff_matchups = seed_matchups.loc[(seed_matchups['TYPE'] == 'UPSET') | (seed_matchups['TYPE'] == 'NON UPSET')]
removed_cols = diff_matchups.columns[~diff_matchups.columns.isin(['TYPE'])]
# Get the differentials of all non rank stats
diff_matchups = diff_matchups[removed_cols].diff()
diff_matchups = diff_matchups.iloc[1::2]
diff_matchups = diff_matchups.abs()

diff_matchups = diff_matchups.drop(columns = ['W K TEMPO', 'W K OFF', 'W K DEF', 'W RAW T', 'W H RAW T', 'W A RAW T', 'W N RAW T', 'W AN RAW T', 'W OP FT%', 'W H OP FT%', 'W A OP FT%', 'W N OP FT%', 'W AN OP FT%',
                                              'W DRAW', 'W VAL Z-SCORE', 
                                              'W V 1-25 WINS', 'W V 1-25 LOSS', 'W V 26-50 WINS', 'W V 26-50 LOSS', 'W V 51-100 WINS', 'W V 51-100 LOSS', 'W HI', 'W LO','W SOS HI', 'W SOS LO',
                                              'W LUCK V 1-25 WINS', 'W LUCK V 1-25 LOSS', 'W LUCK V 26-50 WINS', 'W LUCK V 26-50 LOSS', 'W LUCK V 51-100 WINS', 'W LUCK V 51-100 LOSS', 'W LUCK HI', 'W LUCK LO',
                                              'W CONSISTENCY V 1-25 WINS', 'W CONSISTENCY V 1-25 LOSS', 'W CONSISTENCY V 26-50 WINS', 'W CONSISTENCY V 26-50 LOSS', 'W CONSISTENCY V 51-100 WINS', 'W CONSISTENCY V 51-100 LOSS', 'W CONSISTENCY HI', 'W CONSISTENCY LO',
                                              'W H V 1-25 WINS', 'W H V 1-25 LOSS', 'W H V 26-50 WINS', 'W H V 26-50 LOSS', 'W H V 51-100 WINS', 'W H V 51-100 LOSS', 'W H HI', 'W H LO',
                                              'W A V 1-25 WINS', 'W A V 1-25 LOSS', 'W A V 26-50 WINS', 'W A V 26-50 LOSS', 'W A V 51-100 WINS', 'W A V 51-100 LOSS', 'W A HI', 'W A LO',
                                              'W N V 1-25 WINS', 'W N V 1-25 LOSS', 'W N V 26-50 WINS', 'W N V 26-50 LOSS', 'W N V 51-100 WINS', 'W N V 51-100 LOSS', 'W N HI', 'W N LO',
                                              'W AN V 1-25 WINS', 'W AN V 1-25 LOSS', 'W AN V 26-50 WINS', 'W AN V 26-50 LOSS', 'W AN V 51-100 WINS', 'W AN V 51-100 LOSS', 'W AN HI', 'W AN LO'])
diff_matchups = diff_matchups.T

values_df = pd.DataFrame()
stat_count = 20

# Get the top 20 stats with the biggest differentials
for i in range(len(seed1)) :
    values = diff_matchups.nlargest(stat_count, diff_matchups.columns[i])
    values_df = pd.concat([values_df, values.iloc[:, i]], axis = 0)

values_df.index.name = 'STAT'
values_df['MATCHUP NO'] = np.arange(len(values_df))
values_df = values_df.set_index('MATCHUP NO', append = True)

simp_seed_matchups = pd.DataFrame()
j = 0

# Simplify the dataframe
for i in range(0, len(values_df), stat_count) :
    temp_df = values_df.iloc[i : i + stat_count]
    temp_df = temp_df.droplevel(1)
    simp_seed_cols = seed_matchups.loc[j:j + 1, temp_df.T.columns]
    simp_seed_cols = simp_seed_cols.reset_index(drop = True)
    simp_seed_matchups = pd.concat([simp_seed_matchups, simp_seed_cols], axis = 1)
    j += 2


# Create dataframe with the appropriate format for a bar plot
#############################################################

simp_seed_matchups = simp_seed_matchups.T
simp_seed_matchups.columns = ['NON UPSET', 'UPSET']
simp_seed_matchups = simp_seed_matchups.rename_axis('STAT').reset_index()

simp_seed_matchups['DIFFERENTIAL'] = ''
simp_seed_matchups['MATCHUPS'] = ''
temp1 = simp_seed_matchups.copy()

# Set the differential values to the upset and non upset differential values
simp_seed_matchups['DIFFERENTIAL'] = simp_seed_matchups['NON UPSET']
temp1['DIFFERENTIAL'] = simp_seed_matchups['UPSET']

simp_seed_matchups['MATCHUPS'] = 'NON UPSET'
temp1['MATCHUPS'] = 'UPSET'

# Combine the upset and non upset dataframes to get all stat differentials
simp_seed_matchups = pd.concat([simp_seed_matchups, temp1], axis = 0)
simp_seed_matchups = simp_seed_matchups.sort_index(ascending = True)
# Get rid of the "W" and "L" column prefixes
simp_seed_matchups['STAT'] = simp_seed_matchups['STAT'].str[2:]

simp_seed_matchups['W SEED'] = ''
simp_seed_matchups['L SEED'] = ''

j = 0

# Add columns representing the winning and losing seeds
for i in range(0, len(simp_seed_matchups) // 2, stat_count) :
    simp_seed_matchups.loc[i:i + stat_count, 'W SEED'] = seed1[j]
    simp_seed_matchups.loc[i:i + stat_count, 'L SEED'] = seed2[j]
    j += 1


# Create dataframe for the current tournament matchups
######################################################

complete_curr_seed_matchups = pd.DataFrame()
num, j = 0, 0

for i in range(len(seed1)) :
    curr_matchups = diff.copy()
    curr_matchups = curr_matchups.loc[curr_matchups['W YEAR'].eq(curr_year)]
    # Get the current seed matchup
    curr_seed_matchups = curr_matchups.loc[curr_matchups['W SEED'].eq(seed1[i]) & curr_matchups['L SEED'].eq(seed2[i])]

    # Get the past seed matchups
    plot_matchups = simp_seed_matchups.loc[simp_seed_matchups['W SEED'].eq(seed1[i]) & simp_seed_matchups['L SEED'].eq(seed2[i])]
    plot_matchups = plot_matchups.sort_values(by = ['MATCHUPS'], ascending = False)
    j += stat_count
    plot_matchups = plot_matchups.reset_index(drop = True)
    # Get the upset matchups
    plot_matchups = plot_matchups.loc[plot_matchups['MATCHUPS'] == 'UPSET']
    col_names = plot_matchups['STAT']
    # Add the "W" prefix to column names for the purpose of retrieving these specific columns
    col_names = ['W ' + sub for sub in col_names]
    # Create the current team matchup names
    team_names = str(seed1[i]) + ' ' + curr_seed_matchups['W TEAM'] + ' | ' + str(seed2[i]) + ' ' + curr_seed_matchups['L TEAM']
    team_names = team_names.reset_index(drop = True)

    # Get the specific current seed matchup
    curr_seed_matchups = curr_seed_matchups.loc[:, col_names]
    curr_seed_matchups = curr_seed_matchups.T
    curr_simp_seed_matchups = pd.concat([curr_seed_matchups.iloc[:, 0], curr_seed_matchups.iloc[:, 1]], axis = 0)

    # Change the num variable to the amount of matchups per seed minus two
    if (seed2[i] == 11) : num = 4
    else : num = 2

    # Join all matchups with the same seeds together
    for j in range(num) :
        curr_simp_seed_matchups = pd.concat([curr_simp_seed_matchups, curr_seed_matchups.iloc[:, j + 2]], axis = 0)

    curr_simp_seed_matchups = curr_simp_seed_matchups.to_frame()
    curr_simp_seed_matchups['MATCHUPS'] = ''
    curr_simp_seed_matchups.columns = ['DIFFERENTIAL', 'MATCHUPS']

    r, c = 0, stat_count

    # Set the cell to the matchup name
    for k in range(num + 2) :
        curr_simp_seed_matchups.iloc[r : c, 1] = team_names[k]
        r += stat_count
        c += stat_count

    curr_simp_seed_matchups['W SEED'] = seed1[i]
    curr_simp_seed_matchups['L SEED'] = seed2[i]
    # Create a dataframe consisting of all current matchups
    complete_curr_seed_matchups = pd.concat([complete_curr_seed_matchups, curr_simp_seed_matchups], axis = 0)

complete_curr_seed_matchups['STAT'] = complete_curr_seed_matchups.index
complete_curr_seed_matchups = complete_curr_seed_matchups.sort_values(by = ['STAT'])
complete_curr_seed_matchups = complete_curr_seed_matchups.reset_index(drop = True)
# Get rid of the "W" and "L" column prefixes
complete_curr_seed_matchups['STAT'] = complete_curr_seed_matchups['STAT'].str[2:]


# Display the Matchup Data
def plot_matchups(seed_matchups, curr_seed_matchups, seed1, seed2, title) :
    # Get the past seed matchups
    seed_matchups = seed_matchups.loc[seed_matchups['W SEED'].eq(seed1) & seed_matchups['L SEED'].eq(seed2)]
    # Get the current seed matchup
    curr_seed_matchups = curr_seed_matchups.loc[curr_seed_matchups['W SEED'].eq(seed1) & curr_seed_matchups['L SEED'].eq(seed2)]

    # Combine the past and current seed matchup dataframes together for the purpose of data visualization
    plot_matchups = pd.concat([seed_matchups, curr_seed_matchups], axis = 0)
    plot_matchups = plot_matchups.sort_values(by = ['STAT', 'MATCHUPS'], ascending = True)
    plot_matchups = plot_matchups.reset_index(drop = True)

    # First row we want to get from the plot_matchups dataframe
    j = 0

    # Change the k variable to the amount of bars per stat for every matchup in one row of a bar plot
    if seed2 == 11 : k = 40
    else : k = 30

    fig_arr = []

    # Get a specific part of the matchups' dataframe
    for i in range(4) :
        sub_plot_matchups = plot_matchups.loc[j : k - 1, :]

        # Increase the row coordinates for the subsequent data visualization
        if seed2 == 11 :
            j += 40
            k += 40
        else :
            j += 30
            k += 30

        fig = px.histogram(sub_plot_matchups, x = 'STAT', y = 'DIFFERENTIAL', title = '<b>' + titles[title] + ' SEEDS </b>',
                       color = 'MATCHUPS', barmode = 'group', template = 'plotly_dark', height = 600)
        fig.update_layout(title_x = 0.5, title_font = dict(size = 20), yaxis_title = 'DIFFERENTIAL',
                          legend = dict(yanchor = 'top', y = 1.5, xanchor = 'right', x = 1, font = dict(size = 10)))
        fig.update_xaxes(tickangle = 45)
        fig_arr.append(fig)

    return fig_arr


diff_matchups = simp_seed_matchups.copy()
diff_matchups['DIFFERENTIAL'] = abs(diff_matchups['NON UPSET'] - diff_matchups['UPSET']) 
diff_matchups['MEAN'] = (diff_matchups['NON UPSET'] + diff_matchups['UPSET']) / 2 
diff_matchups


upset_score_df = pd.DataFrame()

for seed in seed1 : 
    matchups = diff_matchups.loc[diff_matchups['W SEED'].eq(seed) & diff_matchups['L SEED'].eq(seed2[seed1.index(seed)])]
    matchups = matchups.drop_duplicates(subset = ['UPSET', 'NON UPSET'])
    curr_matchups = complete_curr_seed_matchups.loc[complete_curr_seed_matchups['W SEED'].eq(seed) & complete_curr_seed_matchups['L SEED'].eq(seed2[seed1.index(seed)])]

    stats = matchups['STAT']
    curr_stats = curr_matchups['STAT']

    for stat in stats : 
        stat_matchups = matchups.loc[matchups['STAT'].eq(stat)]
        curr_stat_matchups = curr_matchups.loc[curr_matchups['STAT'].eq(stat)]

        if float(stat_matchups['UPSET']) < float(stat_matchups['NON UPSET']) : 
            curr_stat_matchups['UPSET SCORE'] = (float(stat_matchups['MEAN']) - curr_stat_matchups['DIFFERENTIAL']) * (float(stat_matchups['DIFFERENTIAL'] * 100))  
        else : 
            curr_stat_matchups['UPSET SCORE'] = (curr_stat_matchups['DIFFERENTIAL'] - float(stat_matchups['MEAN'])) * (float(stat_matchups['DIFFERENTIAL'] * 100))  

        upset_score_df = pd.concat([upset_score_df, curr_stat_matchups], ignore_index = True)

teams = upset_score_df[['MATCHUPS']]
teams = teams.drop_duplicates(subset = ['MATCHUPS'])
upset_scores = pd.DataFrame()

for team in teams['MATCHUPS'] : 
    curr_upset_score_df = upset_score_df.loc[upset_score_df['MATCHUPS'].eq(team)]
    curr_upset_score_df = curr_upset_score_df.drop_duplicates(subset = ['STAT'])
    upset_score = curr_upset_score_df['UPSET SCORE'].sum()  
    first_row = curr_upset_score_df.iloc[0]  
    first_row['UPSET SCORE'] = upset_score
    upset_scores = pd.concat([upset_scores, first_row], axis = 1, ignore_index = True)
    
upset_scores = upset_scores.T   
upset_scores = upset_scores.sort_values(by = 'UPSET SCORE', ascending = False)
# upset_scores


titles = ['8 VS 9', '7 VS 10', '6 VS 11', '5 VS 12', '4 VS 13', '3 VS 14', '2 VS 15',
          '1 VS 8', '1 VS 9', '2 VS 7', '2 VS 10', '3 VS 6', '3 VS 11', '4 VS 5',
          '1 VS 4', '1 VS 5']

fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 8, 9, 0)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 7, 10, 1)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 6, 11, 2)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 5, 12, 3)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 4, 13, 4)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 3, 14, 5)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 2, 15, 6)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 1, 8, 7)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 1, 9, 8)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 2, 7, 9)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 2, 10, 10)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 3, 6, 11)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 3, 11, 12)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 4, 5, 13)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 1, 4, 14)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()


fig_arr = plot_matchups(simp_seed_matchups, complete_curr_seed_matchups, 1, 5, 15)
fig_arr[0].show()


fig_arr[1].show()


fig_arr[2].show()


fig_arr[3].show()

