import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sb
import plotly.express as px
from pathlib import Path


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#i am already inside the folder

all_files = glob.glob('/kaggle/input/**/*.csv', recursive=True) 


files_dict = {Path(file).stem : pd.read_csv(file, encoding='latin-1') for file in all_files}


teams = pd.concat([files_dict['MTeams'],files_dict['WTeams']],axis=0)
teams.head()


seasons = pd.concat([files_dict['MSeasons'], files_dict['WSeasons']], axis = 0)
seasons.head()


seeds = pd.concat([files_dict['MNCAATourneySeeds'], files_dict['WNCAATourneySeeds']], axis = 0)
seeds.head()


rsc_results =  pd.concat([files_dict['MRegularSeasonCompactResults'], 
            files_dict['WRegularSeasonCompactResults']], 
           axis = 0)


#tourney compact results
tc_results = pd.concat([files_dict['MNCAATourneyCompactResults'], 
                          files_dict['WNCAATourneyCompactResults']], axis = 0)

tc_results.head()


#regular season detailed results
rsd_results = pd.concat([files_dict['MRegularSeasonDetailedResults'],
          files_dict['WRegularSeasonDetailedResults']],axis=0)

rsd_results.head()


#tourney detailed results
td_results = pd.concat([files_dict['MNCAATourneyDetailedResults'], 
           files_dict['WNCAATourneyDetailedResults']], 
          axis = 0)

td_results.head()


cities = files_dict['Cities']

cities.head()


game_cities = pd.concat([files_dict['MGameCities'], files_dict['WGameCities']], axis = 0)

game_cities.head()


files_dict['MMasseyOrdinals'].head()


coaches = files_dict['MTeamCoaches']

coaches.head()


tourney_result = td_results.copy()


tourney_result['WFG%'] = (tourney_result['WFGM'] / tourney_result['WFGA']) * 100          #FG% for winning team
tourney_result['LFG%'] = (tourney_result['LFGM'] / tourney_result['LFGA']) * 100          #FG% of losing team
tourney_result['WFG3%'] = (tourney_result['WFGM3'] / tourney_result['WFGA3']) * 100       #3point% of winning team
tourney_result['LFG3%'] = (tourney_result['LFGM3'] / tourney_result['LFGA3']) * 100       #3point% of losing team
tourney_result['WFT%'] = (tourney_result['WFTM'] / tourney_result['WFTA']) * 100          #FT% of winning team
tourney_result['LFT%'] = (tourney_result['LFTM'] / tourney_result['LFTA']) * 100          #FT% of losing team
tourney_result['WAst_TO_ratio'] = tourney_result['WAst'] / tourney_result['WTO']          #assist to turnover ratio
tourney_result['LAst_TO_ratio'] = tourney_result['LAst'] / tourney_result['LTO']

#TS% = (Points / (FGA + 0.44 * FTA))
tourney_result['WTS%'] = tourney_result['WScore'] / 2 * (tourney_result['WFGA'] + 0.44 * tourney_result['WFTA'])
tourney_result['LTS%'] = tourney_result['LScore'] / 2 * (tourney_result['LFGA'] + 0.44 * tourney_result['LFTA'])


# Function to add centered value labels
def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center') 


px.bar( tourney_result.groupby('Season')[['WScore', 'LScore']].mean(), barmode='group')


ax = tourney_result.groupby('Season')[['WScore', 'LScore']].mean().plot.bar(figsize = (20,10))

#add labels
for container in ax.containers:
    ax.bar_label(container, rotation = 90) #rotation rotates position of label


OT_games = tourney_result[tourney_result['NumOT'] > 0]
count_OT = OT_games.groupby('Season')['NumOT'].count()
px.line(count_OT)


total_games = tourney_result.groupby('Season').size()

OT_games = tourney_result[tourney_result['NumOT'] > 0] \
            .groupby('Season').size()

OT_rate = (OT_games / total_games) * 100

px.line(
    OT_rate,
    labels={'value':'OT Percentage', 'Season':'Season'}
)


tourney_result['total_score'] = tourney_result['WScore'] + tourney_result['LScore']

px.line(tourney_result.groupby('Season')['total_score'].mean())


tourney_result[['WFG%', 'LFG%']].mean()


tourney_result.groupby('Season')[['WFG%', 'LFG%']].mean().plot.bar(figsize=(15,7));


tourney_result['diff_3p'] = tourney_result['WFGM3'] - tourney_result['LFGM3']

plt.figure(figsize=(10,7))
plt.hist(tourney_result['diff_3p'])

plt.show()


plt.figure(figsize=(14,7))
sb.boxplot(tourney_result['diff_3p'])

plt.show()


#Step 1: Calculate total points by source

Wpct_2 = ((tourney_result['WFGM'] - tourney_result['WFGM3']) * 2).sum()
Wpct_3 = (tourney_result['WFGM3'] * 3).sum()
Wpct_ft = tourney_result['WFTM'].sum()
Lpct_2 = ((tourney_result['LFGM'] - tourney_result['LFGM3']) * 2).sum()
Lpct_3 = (tourney_result['LFGM3'] * 3).sum()
Lpct_ft = tourney_result['LFTM'].sum()
total_points = tourney_result['WScore'].sum()

#Step 2: Convert to percentages

Wpct_2 = (Wpct_2 / total_points) * 100
Wpct_3 = (Wpct_3 / total_points) * 100
Wpct_ft = (Wpct_ft / total_points) * 100
Lpct_2 = (Lpct_2 / total_points) * 100
Lpct_3 = (Lpct_3 / total_points) * 100
Lpct_ft = (Lpct_ft / total_points) * 100

#present answer cleanly

points_df = pd.DataFrame({
                        'Scoring Type': ['W2-pointers', 'W3-pointers', 'WFree throws', 
                                         'L2-pointers', 'L3-pointers', 'LFree throws'],
                        'Percentage of Total Points': [Wpct_2, Wpct_3, Wpct_ft, Lpct_2, Lpct_3, Lpct_ft]
                        }).round(2)


#lets see colors on plotly

#px.colors.sequential.swatches_continuous()


px.pie(points_df,
       values = 'Percentage of Total Points',
       names = 'Scoring Type',
       color_discrete_sequence=px.colors.sequential.Hot
       )


tourney_result['more_FTA_often'] = (tourney_result['WFTA'] > tourney_result['LFTA']).astype(int)

tourney_result['more_FTA_often'].mean()


rebounds = tourney_result[['Season','WScore','LScore','WDR','WOR','LDR','LOR']].copy()
rebounds['W_rebounds'] = rebounds['WOR'] + rebounds['WDR']
rebounds['L_rebounds'] = rebounds['LOR'] + rebounds['LDR']

(rebounds['W_rebounds'] > rebounds['L_rebounds']).mean()


rebounds['rebounds_diff'] = rebounds['W_rebounds'] - rebounds['L_rebounds']

sb.boxplot(data = rebounds, x = 'rebounds_diff')
plt.show()


rebounds['W_have_more_OR'] = (rebounds['WOR'] > rebounds['LOR'])
rebounds['W_have_more_DR'] = (rebounds['WDR'] > rebounds['LDR'])


rebounds[['W_have_more_OR','W_have_more_DR']].mean()


rebounds['OR_diff'] = rebounds['WOR'] - rebounds['LOR']
rebounds['DR_diff'] = rebounds['WDR'] - rebounds['LDR']

#--BOX PLOT--

plt.figure(figsize=(14,7))
sb.boxplot(data = rebounds[['OR_diff', 'DR_diff']])
plt.title('Graphical representation of the impact of Offensive rebounds vs Defensive rebounds in a game')
plt.show()


sb.palettes.SEABORN_PALETTES.keys()


(tourney_result['WAst'] > tourney_result['LAst']).mean()


tourney_result['Ast_diff'] = tourney_result['WAst'] - tourney_result['LAst']

sb.boxplot(data = tourney_result, x = 'Ast_diff')
plt.show()


ax = pd.Series({
'Assist_to_turnover_ratio' : (tourney_result['WAst_TO_ratio'] > tourney_result['LAst_TO_ratio']).mean(),
'Assists' : (tourney_result['WAst'] > tourney_result['LAst']).mean()
}).plot.bar(figsize=(14,7))

#add labels
for container in ax.containers:
    ax.bar_label(container) #rotation rotates position of label

plt.show()


tourney_result['W_possesions'] = (tourney_result['WFGA'] - tourney_result['WOR']) + tourney_result['WTO'] + (tourney_result['WFTA'] * .475)
tourney_result['L_possesions'] = (tourney_result['LFGA'] - tourney_result['LOR']) + tourney_result['LTO'] + (tourney_result['LFTA'] * .475)

tourney_result['W_TO_rate'] = 100 * (tourney_result['WTO'] / tourney_result['W_possesions'])
tourney_result['L_TO_rate'] = 100 * (tourney_result['LTO'] / tourney_result['L_possesions'])


#turnover rate of over 18% is considered to be generally high

#create a df of games were winning team had high TO rate

high_TO_games = tourney_result[tourney_result['W_TO_rate'] > 18]


ax = pd.Series({
    'Free Throws' : (high_TO_games['WFTM'] > high_TO_games['LFTM']).mean(),
'Offensive Rebounds' : (high_TO_games['WOR'] > high_TO_games['LOR']).mean()
}).plot.bar(figsize = (14,7))

for container in ax.containers:
    ax.bar_label(container)

plt.show()


sb.boxplot(
    data = high_TO_games[['WOR','WFTM']]
)

plt.title("Offensive Rebounds vs Free Throws in High Turnover Wins")
plt.ylabel("Count")
plt.show()


high_scoring_games = tourney_result[tourney_result['total_score'] > 150]
low_scoring_games = tourney_result[tourney_result['total_score'] < 150]


high_scoring_games = high_scoring_games.copy()
low_scoring_games  = low_scoring_games.copy()

high_scoring_games['L_have_more_PF'] = high_scoring_games['LPF'] > high_scoring_games['WPF']
low_scoring_games['L_have_more_PF'] = low_scoring_games['LPF'] > low_scoring_games['WPF']


df = pd.DataFrame(
    [['High_scoring_games', high_scoring_games['L_have_more_PF'].value_counts(normalize = True)[0], 
              high_scoring_games['L_have_more_PF'].value_counts(normalize = True)[1]],
              ['Low_scoring_games', low_scoring_games['L_have_more_PF'].value_counts(normalize = True)[0], 
               low_scoring_games['L_have_more_PF'].value_counts(normalize = True)[1]]],
              columns = ['Game Type', 'Team Lost', 'Team Won']
            )

# plot grouped bar chart
ax = df.plot(x='Game Type',
        kind='bar',
        stacked=False,
        figsize = (14,7),
        title='Personal Fouls in both high scoring and low scoring games')

for container in ax.containers:
    ax.bar_label(container)


pf_damage = pd.Series({
    'High Scoring Games': (high_scoring_games['LPF'] > high_scoring_games['WPF']).mean(),
    'Low Scoring Games': (low_scoring_games['LPF'] > low_scoring_games['WPF']).mean()
})

ax = pf_damage.plot.bar(figsize=(14,7), title='Impact of Personal Fouls on Winning')

for container in ax.containers:
    ax.bar_label(container)

plt.ylabel('Losing Team Had More Fouls (%)')
plt.show()


##Step 1 — Create stat differences

tourney_result['FG_diff'] = tourney_result['WFGM'] - tourney_result['LFGM']
tourney_result['OR_diff'] = tourney_result['WOR'] - tourney_result['LOR']
tourney_result['DR_diff'] = tourney_result['WDR'] - tourney_result['LDR']
tourney_result['AST_diff'] = tourney_result['WAst'] - tourney_result['LAst']
tourney_result['TO_diff'] = tourney_result['LTO'] - tourney_result['WTO']  # positive = winner had fewer turnovers
tourney_result['FT_diff'] = tourney_result['WFTM'] - tourney_result['LFTM']

tourney_result['Score_diff'] = tourney_result['WScore'] - tourney_result['LScore']


#Step 2 — Correlation with winning margin

stats = ['FG_diff','OR_diff','DR_diff','AST_diff','TO_diff','FT_diff']

corr = tourney_result[stats + ['Score_diff']].corr()['Score_diff'].drop('Score_diff')


#Step 3 - Visualization

ax = corr.sort_values().plot.barh(figsize=(14,7), title='Stat Impact on Winning Margin')

for container in ax.containers:
    ax.bar_label(container)

plt.show()
















