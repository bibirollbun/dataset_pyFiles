# Essential Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import ipywidgets as widgets
import chardet
import matplotlib.patches as patches



# Tools
def load_csv(csv_file):
    with open(csv_file, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
    df = pd.read_csv(csv_file, encoding=result["encoding"])
    return df


comp_dir = '/kaggle/input/march-machine-learning-mania-2025'
Results_menslist = [
    'MTeams.csv',
    'MSeasons.csv',
    'MNCAATourneySeeds.csv',
    'MRegularSeasonCompactResults.csv',
    'MNCAATourneyCompactResults.csv']
Results_womenslist = [
    'WTeams.csv',
    'WSeasons.csv',
    'WNCAATourneySeeds.csv',
    'WRegularSeasonCompactResults.csv',
    'WNCAATourneyCompactResults.csv',]
GameStats_menslist = [
    'MRegularSeasonDetailedResults.csv',
    'MNCAATourneyDetailedResults.csv',]
GameStats_womenslist = [
    'WRegularSeasonDetailedResults.csv',
    'WNCAATourneyDetailedResults.csv']


# Team data table
mens_df = load_csv(os.path.join(comp_dir, 'MTeams.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WTeams.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Teams")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    display(df.describe())
    display(df.head())


# Find unique values in Team Names
series1 = mens_df['TeamName']
series2 = womens_df['TeamName']
unique_in_series1 = series1[~series1.isin(series2)]
unique_in_series2 = series2[~series2.isin(series1)]
unique_values = pd.concat([unique_in_series1, unique_in_series2])
print(unique_values)


# History of on amount of Teams every regular season
mens_df = load_csv(os.path.join(comp_dir, 'MRegularSeasonCompactResults.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WRegularSeasonCompactResults.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Teams")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    pivot_table = df.groupby('Season')['WTeamID'].nunique().reset_index()
    pivot_table.columns = ['Season', f'{gender} Teams']
    print(pivot_table)


mens_df = load_csv(os.path.join(comp_dir, 'MRegularSeasonDetailedResults.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WRegularSeasonDetailedResults.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    display(df.describe())
    display(df.head())


for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Teams")
    
    # Assign the correct DataFrame
    if gender == 'Mens':
        df = mens_df.copy()  # Ensure df is a separate copy
    else:
        df = womens_df.copy()

    # Calculate score and turnover differences
    df['Score Differential'] = df['WScore'] - df['LScore']
    df['Turnover Differential'] = df['WTO'] - df['LTO']

    # Plot Score Difference Distribution
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Season', y='Score Differential', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Scoring Differential Distributions')
    plt.tight_layout()
    plt.show()

    # Plot Turnover Difference Distribution
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Season', y='Turnover Differential', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Turnover Differential Distributions')
    plt.tight_layout()
    plt.show()


 #pivot table to count the number of appearances of each Team
def process_team_data(gender, team_file, seed_file):
    print(f"Most Consistent {gender} Teams")
    team_df = pd.read_csv(team_file)
    seed_df = pd.read_csv(seed_file)
    merged_df = pd.merge(seed_df, team_df[['TeamID', 'TeamName']], on='TeamID')

    pivot_table = merged_df.pivot_table(
        index='TeamName', 
        aggfunc='size', 
        columns=None, 
        fill_value=0
    ).reset_index()
    pivot_table = pivot_table.rename(columns={0: 'Times in Tourney'})
    pivot_table = pivot_table.sort_values(by='Times in Tourney', ascending=False)
    def extract_seed_numeric(seed):
        try:
            return int(''.join(filter(str.isdigit, seed)))
        except ValueError:
            return None  
    merged_df['Seed_numeric'] = merged_df['Seed'].apply(extract_seed_numeric)
    average_seeds = merged_df.groupby('TeamName')['Seed_numeric'].mean().reset_index()
    pivot_table = pd.merge(pivot_table, average_seeds, on='TeamName', how='left')
    pivot_table = pivot_table.rename(columns={'Seed_numeric': 'Average Seed'})
    print(pivot_table.columns)
    if 'Average Seed' in pivot_table.columns:
        pivot_table['Average Seed'] = pivot_table['Average Seed'].round(1)
    else:
        print("Column 'Average Seed' not found!")
    print(pivot_table)

mens_team = os.path.join(comp_dir, 'MTeams.csv')
mens_seed = os.path.join(comp_dir, 'MNCAATourneySeeds.csv')
womens_team = os.path.join(comp_dir, 'WTeams.csv')
womens_seed = os.path.join(comp_dir, 'WNCAATourneySeeds.csv')
for gender, team, seed in [('Mens', mens_team, mens_seed), 
                        ('Womens', womens_team, womens_seed)]:
    process_team_data(gender, team, seed)


# men's and women's teams and tourney seeds
mens_df = pd.read_csv(os.path.join(comp_dir, 'MTeams.csv')) 
mens_tourneyseeds_df = pd.read_csv(os.path.join(comp_dir, 'MNCAATourneySeeds.csv'))  
womens_df = pd.read_csv(os.path.join(comp_dir, 'WTeams.csv'))  
womens_tourneyseeds_df = pd.read_csv(os.path.join(comp_dir, 'WNCAATourneySeeds.csv')) 
for gender in ['Mens', 'Womens']:
    print(f"{gender} Teams Participants")
    if gender == 'Mens':
        df = pd.merge(mens_tourneyseeds_df, mens_df[['TeamID', 'TeamName']], on='TeamID')
    else:
        df = pd.merge(womens_tourneyseeds_df, womens_df[['TeamID', 'TeamName']], on='TeamID')

    # Create a pivot table to count the number of appearances of each Team
    pivot_df = pd.pivot_table(df, index='TeamName', columns='Season', values='TeamID', aggfunc='count', fill_value=0)
    pivot_df[pivot_df > 0] = 1  
    annot_matrix = pivot_df.copy().astype(str)
    annot_matrix[annot_matrix == '1'] = 'In'
    annot_matrix[annot_matrix == '0'] = 'Out'
    plt.figure(figsize=(20, 100))
    sns.heatmap(pivot_df, annot=annot_matrix, cmap='Blues', cbar=False, fmt='', linewidths=0, square=True, 
            annot_kws={'size': 8})

    plt.title(f'{gender} Team Participation in NCAA Tournament by Season')
    plt.xlabel('Season')
    plt.ylabel('Team')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)  
    plt.show()


# Win Count of Tournament Games by Seed
mens_seed_df = pd.read_csv(os.path.join(comp_dir, 'MNCAATourneySeeds.csv'))
mens_tourneyresults_df = pd.read_csv(os.path.join(comp_dir, 'MNCAATourneyCompactResults.csv'))
womens_seed_df = pd.read_csv(os.path.join(comp_dir, 'WNCAATourneySeeds.csv'))
womens_tourneyresults_df = pd.read_csv(os.path.join(comp_dir, 'WNCAATourneyCompactResults.csv'))

for gender in ['Mens', 'Womens']:
    print(f"{gender} Teams")
    if gender == 'Mens':
        teams_df = mens_seed_df
        tourneyresults_df = mens_tourneyresults_df
    else:
        teams_df = womens_seed_df
        tourneyresults_df = womens_tourneyresults_df
    teams_df['Seed'] = teams_df['Seed'].str.extract('(\d+)')
    merged_df = pd.merge(tourneyresults_df, teams_df, left_on=['WTeamID', 'Season'], right_on=['TeamID', 'Season'], how='inner')
    winsbyseed = merged_df.groupby('Seed').size().reset_index(name='WinCount')
    print(winsbyseed)


# # Win Count of Tournament Games by Seed
mens_teams_df = pd.read_csv(os.path.join(comp_dir, 'MTeams.csv'))
mens_tourneyresults_df = pd.read_csv(os.path.join(comp_dir, 'MNCAATourneyCompactResults.csv'))
womens_teams_df = pd.read_csv(os.path.join(comp_dir, 'WTeams.csv'))
womens_tourneyresults_df = pd.read_csv(os.path.join(comp_dir, 'WNCAATourneyCompactResults.csv'))

# Investigate for each gender
for gender in ['Mens', 'Womens']:
    print(f"{gender} Teams")
    if gender == 'Mens':
        teams_df = mens_teams_df
        tourneyresults_df = mens_tourneyresults_df
    else:
        teams_df = womens_teams_df
        tourneyresults_df = womens_tourneyresults_df
    
    # Step 1: Calculate wins and losses for each team
    winsbyteam = tourneyresults_df['WTeamID'].value_counts().reset_index()
    losesbyteam = tourneyresults_df['LTeamID'].value_counts().reset_index()
    winsbyteam.columns = ['TeamID', 'Wins']
    losesbyteam.columns = ['TeamID', 'Losses']
    teamstats = pd.merge(winsbyteam, losesbyteam, on='TeamID', how='outer')
    teamstats = teamstats.fillna(0)
    teamstats['Wins'] = teamstats['Wins'].astype(int)
    teamstats['WinningPct'] = (teamstats['Wins'] / (teamstats['Wins'] + teamstats['Losses']) * 100)
    teamstats['WinningPct'] = teamstats['WinningPct'].round(1)
    teamstats = pd.merge(teamstats, teams_df[['TeamID', 'TeamName']], on='TeamID', how='left')
    teamstats = teamstats.drop(columns=['TeamID'])
    teamstats = teamstats[['TeamName', 'Wins', 'Losses', 'WinningPct']]
    teamstats = teamstats.sort_values(by='Wins', ascending=False)
    print(teamstats)


# Relationship on Day and Round
tourneyresults_df = load_csv(os.path.join(comp_dir, 'MNCAATourneyCompactResults.csv'))
tourneyround_df = load_csv(os.path.join(comp_dir, 'MNCAATourneySeedRoundSlots.csv'))

distinctrows = tourneyround_df[['EarlyDayNum', 'GameRound']].drop_duplicates()
print(distinctrows)
merged_df = pd.merge(tourneyresults_df, tourneyround_df, left_on='DayNum', right_on='EarlyDayNum', how='inner')
distinctrows2 = merged_df[['DayNum', 'EarlyDayNum']].drop_duplicates()
distinctrows_sorted = distinctrows2.sort_values(by='EarlyDayNum')
print(distinctrows_sorted)


mens_df = load_csv(os.path.join(comp_dir, 'MNCAATourneyCompactResults.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WNCAATourneyCompactResults.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Tourney Schedule")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df

    # Extract distinct 'DayNum' values and sort them
    distinct_rows = df[['DayNum']].drop_duplicates()
    distinct_rows_sorted = distinct_rows.sort_values(by='DayNum')
    print(distinct_rows_sorted)


# Count the number of wins by seed and game round from Men's Tournament
tourneyresults_df = load_csv(os.path.join(comp_dir, 'MNCAATourneyCompactResults.csv'))
tourneyseed_df = load_csv(os.path.join(comp_dir, 'MNCAATourneySeeds.csv'))

tourneyseed_df['Seed'] = tourneyseed_df['Seed'].str.extract('(\d+)')
merged_df = pd.merge(tourneyresults_df, tourneyseed_df, left_on=['WTeamID', 'Season'], right_on=['TeamID', 'Season'], how='inner')
def get_game_round(day_num, season):
    if 134 <= day_num <= 135 and (season <= 2020 or 2022 <= season):
        return "Pre Tourn"
    elif 134 <= day_num <= 136 and (season == 2021):
        return "Pre Tourn"    
    elif 136 <= day_num <= 137 and (season <= 2020 or 2022 <= season):
        return "Round 1"
    elif 137 <= day_num <= 138 and (season == 2021):
        return "Round 1"
    elif 138 <= day_num <= 140 and (season <= 2020 or 2022 <= season):
        return "Round 2"
    elif 139 <= day_num <= 140 and (season == 2021):
        return "Round 2"
    elif 141 <= day_num <= 144 and (season <= 2020 or 2022 <= season):
        return "Sweet 16"
    elif 145 <= day_num <= 146 and (season == 2021):
        return "Sweet 16"
    elif 145 <= day_num <= 150:
        return "Elite 8"
    elif day_num == 152:
        return "Final 4"
    elif day_num >= 154:
        return "Championship"
    else:
        return "Unknown"

merged_df['GameRound'] = merged_df.apply(lambda x: get_game_round(x['DayNum'], x['Season']), axis=1)
gameroundorder = ["Pre Tourn", "Round 1", "Round 2", "Sweet 16", "Elite 8", "Final 4", "Championship"]
merged_df['GameRound'] = pd.Categorical(merged_df['GameRound'], categories=gameroundorder, ordered=True)
winsbyseedround = merged_df.groupby(['Seed', 'GameRound']).size().reset_index(name='WinCount')
pt_winbyseed = winsbyseedround.pivot_table(index='Seed', columns='GameRound', values='WinCount', fill_value=0)
pt_winbyseed = pt_winbyseed.round(0).astype(int)
pt_winbyseed.loc['Total'] = pt_winbyseed.sum(axis=0)
pt_winbyseed


# Count the number of wins by seed and game round from Men's Tournament
tourneyresults_df = load_csv(os.path.join(comp_dir, 'WNCAATourneyCompactResults.csv'))
tourneyseed_df = load_csv(os.path.join(comp_dir, 'WNCAATourneySeeds.csv'))

tourneyseed_df['Seed'] = tourneyseed_df['Seed'].str.extract('(\d+)')
merged_df = pd.merge(tourneyresults_df, tourneyseed_df, left_on=['WTeamID', 'Season'], right_on=['TeamID', 'Season'], how='inner')
def get_game_round(day_num, season):
    if 134 <= day_num <= 136:
        return "Pre Tourn"
    elif 137 <= day_num <= 139 and (2003 <= season <= 2014): 
        return "Round 1"
    elif 137 <= day_num <= 138 and (season <= 2002 or 2015 <= season <= 2020 or 2022 <= season):
        return "Round 1"
    elif 139 <= day_num <= 140 and (season == 2021):
        return "Round 1"
    elif 140 <= day_num <= 142 and (2003 <= season <= 2014): 
        return "Round 2"
    elif 139 <= day_num <= 142 and (season <= 2002 or 2015 <= season <= 2020 or 2022 <= season):
        return "Round 2"
    elif 141 <= day_num <= 142 and (season == 2021):
        return "Round 2"
    elif 144 <= day_num <= 146 and (season <= 2014 or season == 2021): 
        return "Sweet 16"
    elif 144 <= day_num <= 145 and (2015 <= season <= 2020 or 2022 <= season):
         return "Sweet 16"
    elif 147 <= day_num <= 148 and (season <= 2014 or season == 2021): 
        return "Elite 8"
    elif 146 <= day_num <= 148 and (2015 <= season <= 2020 or 2022 <= season):
        return "Elite 8"
    elif day_num == 151:
        return "Final 4"
    elif day_num == 153 and (2003 <= season <= 2016): 
        return "Final 4"
    elif day_num == 153 and (season <= 2002 or 2017 <= season):
        return "Championship" 
    elif day_num >= 155:
        return "Championship" 
    else:
        return "Unknown"

merged_df['GameRound'] = merged_df.apply(lambda x: get_game_round(x['DayNum'], x['Season']), axis=1)
gameroundorder = ["Pre Tourn", "Round 1", "Round 2", "Sweet 16", "Elite 8", "Final 4", "Championship"]
merged_df['GameRound'] = pd.Categorical(merged_df['GameRound'], categories= gameroundorder, ordered=True)
winsbyseedround = merged_df.groupby(['Seed', 'GameRound']).size().reset_index(name='WinCount')
pt_winbyseed = winsbyseedround.pivot_table(index='Seed', columns='GameRound', values='WinCount', fill_value=0)
pt_winbyseed= pt_winbyseed.round(0).astype(int)
pt_winbyseed.loc['Total'] = pt_winbyseed.sum(axis=0)
pt_winbyseed


# Load the datasets for men's and women's teams and tourney seeds
mens_teams_df = pd.read_csv(os.path.join(comp_dir, 'MTeams.csv'))
mens_tourneyresults_df = pd.read_csv(os.path.join(comp_dir, 'MNCAATourneyCompactResults.csv'))
womens_teams_df = pd.read_csv(os.path.join(comp_dir, 'WTeams.csv'))
womens_tourneyresults_df = pd.read_csv(os.path.join(comp_dir, 'WNCAATourneyCompactResults.csv'))

# Investigate for each gender
for gender in ['Mens', 'Womens']:
    print(f"{gender} Tournament Results")
    if gender == 'Mens':
        teams_df = mens_teams_df
        tourney_results_df = mens_tourneyresults_df
    else:
        teams_df = womens_teams_df
        tourneyresults_df = womens_tourneyresults_df

    # Step 1: Merge team and tournament results based on TeamID
    merged_df = pd.merge(tourneyresults_df, teams_df[['TeamID', 'TeamName']], left_on='WTeamID', right_on='TeamID')
    team_wincounts = merged_df.groupby(['Season', 'TeamName']).size().reset_index(name='WinCount')
    topteams = team_wincounts.loc[team_wincounts.groupby('Season')['WinCount'].idxmax()]
    topteams_ranking = team_wincounts.groupby('Season').apply(lambda x: x.nlargest(2, 'WinCount')).reset_index(drop=True)
    runnerup_teams = topteams_ranking.groupby('Season').nth(1).reset_index()
    topteams_renamed = topteams[['Season', 'TeamName']].rename(columns={'TeamName': 'Winner'})
    runnerup_teams_renamed = runnerup_teams[['Season', 'TeamName']].rename(columns={'TeamName': 'Runner-up'})
    final_topteams = pd.merge(topteams_renamed, runnerup_teams_renamed, on='Season')
    print(final_topteams)


mens_teamstats = pd.read_csv(os.path.join(comp_dir, 'MRegularSeasonDetailedResults.csv'))
womens_teamstats = pd.read_csv(os.path.join(comp_dir, 'WRegularSeasonDetailedResults.csv'))

def create_team_stats_dataframe(teamstats):
    results_df = pd.DataFrame({
        'Season': teamstats['Season'],
        'DayNum': teamstats['DayNum'],
        'TeamID': teamstats['WTeamID'],
        'OppTeamID': teamstats['LTeamID'],
        'Win': 1, 'Lose': 0,
        'TeamScore': teamstats['WScore'],
        'OppScore': teamstats['LScore'],
        'TeamFGM': teamstats['WFGM'],
        'TeamFGA': teamstats['WFGA'],
        'TeamFGM3': teamstats['WFGM3'],
        'TeamFGA3': teamstats['WFGA3'],
        'TeamFTM': teamstats['WFTM'],
        'TeamFTA': teamstats['WFTA'],
        'TeamOR': teamstats['WOR'],
        'TeamDR': teamstats['WDR'],
        'TeamAst': teamstats['WAst'],
        'TeamTO': teamstats['WTO'],
        'TeamStl': teamstats['WStl'],
        'TeamBlk': teamstats['WBlk'],
        'TeamPF': teamstats['WPF'],
        'OppFGM': teamstats['LFGM'],
        'OppFGA': teamstats['LFGA'],
        'OppFGM3': teamstats['LFGM3'],
        'OppFGA3': teamstats['LFGA3'],
        'OppFTM': teamstats['LFTM'],
        'OppFTA': teamstats['LFTA'],
        'OppOR': teamstats['LOR'],
        'OppDR': teamstats['LDR'],
        'OppAst': teamstats['LAst'],
        'OppTO': teamstats['LTO'],
        'OppStl': teamstats['LStl'],
        'OppBlk': teamstats['LBlk'],
        'OppPF': teamstats['LPF']
    })
    results_opponent_df = pd.DataFrame({
        'Season': teamstats['Season'],
        'DayNum': teamstats['DayNum'],
        'TeamID': teamstats['LTeamID'],
        'OppTeamID': teamstats['WTeamID'],
        'Win': 0, 'Lose': 1,
        'TeamScore': teamstats['LScore'],
        'OppScore': teamstats['WScore'],
        'TeamFGM': teamstats['LFGM'],
        'TeamFGA': teamstats['LFGA'],
        'TeamFGM3': teamstats['LFGM3'],
        'TeamFGA3': teamstats['LFGA3'],
        'TeamFTM': teamstats['LFTM'],
        'TeamFTA': teamstats['LFTA'],
        'TeamOR': teamstats['LOR'],
        'TeamDR': teamstats['LDR'],
        'TeamAst': teamstats['LAst'],
        'TeamTO': teamstats['LTO'],
        'TeamStl': teamstats['LStl'],
        'TeamBlk': teamstats['LBlk'],
        'TeamPF': teamstats['LPF'],
        'OppFGM': teamstats['WFGM'],
        'OppFGA': teamstats['WFGA'],
        'OppFGM3': teamstats['WFGM3'],
        'OppFGA3': teamstats['WFGA3'],
        'OppFTM': teamstats['WFTM'],
        'OppFTA': teamstats['WFTA'],
        'OppOR': teamstats['WOR'],
        'OppDR': teamstats['WDR'],
        'OppAst': teamstats['WAst'],
        'OppTO': teamstats['WTO'],
        'OppStl': teamstats['WStl'],
        'OppBlk': teamstats['WBlk'],
        'OppPF': teamstats['WPF']
    })
    final_teamstats = pd.concat([results_df, results_opponent_df], ignore_index=True)
    final_teamstats = final_teamstats.sort_values(by=['Season', 'TeamID', 'OppTeamID']).reset_index(drop=True)
    return final_teamstats
MNCAAteamstats = create_team_stats_dataframe(mens_teamstats)
WNCAAteamstats = create_team_stats_dataframe(womens_teamstats)

MNCAAteamstats.to_csv('MNCAAteamstats.csv', index=False)
WNCAAteamstats.to_csv('WNCAAteamstats.csv', index=False)
print(MNCAAteamstats.head())
print(WNCAAteamstats.head())



#merge TeamName to TeamID to view connection
mteam_df = pd.read_csv(os.path.join(comp_dir, 'MTeams.csv'))
wteam_df = pd.read_csv(os.path.join(comp_dir, 'WTeams.csv'))
mteamstats_df = pd.read_csv('/kaggle/working/MNCAAteamstats.csv')
wteamstats_df = pd.read_csv('/kaggle/working/WNCAAteamstats.csv')

merged_dfs = []
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Teams")
    if gender == 'Mens':
        team_df = mteam_df
        teamstats_df = mteamstats_df
    else:
        team_df = wteam_df
        teamstats_df = wteamstats_df
    merged_ID = pd.merge(team_df, teamstats_df, on='TeamID', how='inner')
    if 'OppTeamID' in teamstats_df.columns:
        merged_oppID = pd.merge(team_df, teamstats_df, left_on='TeamID', right_on='OppTeamID', how='inner')
    else:
        merged_oppID = pd.DataFrame() 
    if not merged_oppID.empty:
        merged_df = pd.concat([merged_ID, merged_oppID], axis=0) 
    else:
        merged_df = merged_ID 
    columns_to_drop = [col for col in ['TeamID_x', 'TeamID_y', 'Team'] if col in merged_df.columns]
    merged_df.drop(columns=columns_to_drop, inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    merged_dfs.append(merged_df)
    print(merged_df.head())
final_merged_df = pd.concat(merged_dfs, ignore_index=True)


mens_df = pd.read_csv(os.path.join(comp_dir, 'MTeams.csv'))
menstourney_df = pd.read_csv(os.path.join(comp_dir, 'MNCAATourneyCompactResults.csv'))
mteamstats_df = pd.read_csv('/kaggle/working/MNCAAteamstats.csv')
womens_df = pd.read_csv(os.path.join(comp_dir, 'WTeams.csv'))
womenstourney_df = pd.read_csv(os.path.join(comp_dir, 'WNCAATourneyCompactResults.csv'))
wteamstats_df = pd.read_csv('/kaggle/working/WNCAAteamstats.csv')

# Men's Teams
merged_df = pd.merge(menstourney_df, mens_df[['TeamID', 'TeamName']], 
                      left_on='WTeamID', right_on='TeamID')
wincounts = merged_df.groupby(['Season', 'TeamName']).size().reset_index(name='WinCount')
season_winner = wincounts.groupby('Season')['WinCount'].transform(max)
top_teams = wincounts[wincounts['WinCount'] == season_winner]
top_teams = top_teams[['Season', 'TeamName']].rename(columns={'TeamName': 'Winner'})
merged_stats = pd.merge(mens_df, mteamstats_df, on='TeamID', how='inner')
MNCAAchampionshipteamstats = pd.merge(top_teams, merged_stats, 
                                      left_on=['Season', 'Winner'], right_on=['Season', 'TeamName'], how='inner')
MNCAAchampionshipteamstats.drop(columns=['Winner', 'FirstD1Season', 'LastD1Season'], errors='ignore', inplace=True)

# Women's Teams
merged_women_df = pd.merge(womenstourney_df, womens_df[['TeamID', 'TeamName']], 
                            left_on='WTeamID', right_on='TeamID')
women_wincounts = merged_women_df.groupby(['Season', 'TeamName']).size().reset_index(name='WinCount')
winner_season_women = women_wincounts.groupby('Season')['WinCount'].transform(max)
topteams_women = women_wincounts[women_wincounts['WinCount'] == winner_season_women]
topteams_women = topteams_women[['Season', 'TeamName']].rename(columns={'TeamName': 'Winner'})
merged_stats_women = pd.merge(womens_df, wteamstats_df, on='TeamID', how='inner')
WNCAAchampionshipteamstats = pd.merge(topteams_women, merged_stats_women, 
                                      left_on=['Season', 'Winner'], right_on=['Season', 'TeamName'], how='inner')
WNCAAchampionshipteamstats.drop(columns=['Winner'], errors='ignore', inplace=True)

MNCAAchampionshipteamstats.to_csv('/kaggle/working/MNCAAchampionshipteamstats.csv', index=False)
WNCAAchampionshipteamstats.to_csv('/kaggle/working/WNCAAchampionshipteamstats.csv', index=False)
print("\nMen's Champions Regular Season Stats\n")
print(MNCAAchampionshipteamstats)
print("\nWomen's Champions Regular Season Stats\n")
print(WNCAAchampionshipteamstats)


mens_df = pd.read_csv(os.path.join(comp_dir, 'MTeams.csv'))
mteamstats_df = pd.read_csv('/kaggle/working/MNCAAteamstats.csv')
womens_df = pd.read_csv(os.path.join(comp_dir, 'WTeams.csv'))
wteamstats_df = pd.read_csv('/kaggle/working/WNCAAteamstats.csv')

# Filter for the 2025 season
mteamstats_df = mteamstats_df[mteamstats_df['Season'] == 2025]
wteamstats_df = wteamstats_df[wteamstats_df['Season'] == 2025]

# men's team stats
merged_df = pd.merge(mens_df, mteamstats_df, on='TeamID', how='inner')
merged_opp_df = pd.merge(mens_df, mteamstats_df, left_on='TeamID', right_on='OppTeamID', how='inner')
MNCAA2025_teamstats = pd.concat([merged_df, merged_opp_df], axis=1)
MNCAA2025_teamstats.reset_index(drop=True, inplace=True)
MNCAA2025_teamstats.drop(columns=['FirstD1Season', 'LastD1Season'], errors='ignore', inplace=True)

# women's team stats
merged_women_df = pd.merge(womens_df, wteamstats_df, on='TeamID', how='inner')
merged_womenopp_df = pd.merge(womens_df, wteamstats_df, left_on='TeamID', right_on='OppTeamID', how='inner')
WNCAA2025_teamstats = pd.concat([merged_women_df, merged_womenopp_df], axis=1)
WNCAA2025_teamstats.reset_index(drop=True, inplace=True)

MNCAA2025_teamstats.to_csv('/kaggle/working/MNCAA2025teamstats.csv', index=False)
WNCAA2025_teamstats.to_csv('/kaggle/working/WNCAA2025teamstats.csv', index=False)
print("\nMen's NCAA 2025 Regular Season Stats\n")
print(MNCAA2025_teamstats)
print("\nWomen's NCAA 2025 Regular Season Stats\n")
print(WNCAA2025_teamstats)


# Average Scoring from Past Champions
mens_df = pd.read_csv('/kaggle/working/MNCAAchampionshipteamstats.csv')
womens_df = pd.read_csv('/kaggle/working/WNCAAchampionshipteamstats.csv')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (gender, df) in enumerate([('Men\'s', mens_df), ('Women\'s', womens_df)]):
    grouped_df = df.groupby(['TeamName', 'Season'])[['OppScore', 'TeamScore']].mean().reset_index()
    axes[idx].scatter(grouped_df['OppScore'], grouped_df['TeamScore'], alpha=0.7)
    axes[idx].grid(True, linestyle='--', alpha=0.7)
    axes[idx].set_title(f'{gender} Average Team Score versus Opponent Score by Champions')
    axes[idx].set_xlabel('Average Opponent Score')
    axes[idx].set_ylabel('Average Team Score')
    for i in range(len(grouped_df)):
        axes[idx].text(grouped_df['OppScore'].iloc[i] + 0.1, grouped_df['TeamScore'].iloc[i] + 0.1, 
                       f'{grouped_df["TeamName"].iloc[i]} ({grouped_df["Season"].iloc[i]})', 
                       fontsize=8, alpha=0.7)
plt.tight_layout()
plt.show()


# Read the data
mens_df = pd.read_csv('/kaggle/working/MNCAA2025teamstats.csv')
womens_df = pd.read_csv('/kaggle/working/WNCAA2025teamstats.csv')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Define circle positions (adjust these values)
circle_positions = {  
    "Men's": (65, 80),
    "Women's": (57.5, 80)}
for idx, (gender, df) in enumerate([('Men\'s', mens_df), ('Women\'s', womens_df)]):
    grouped_df = df.groupby(['TeamName', 'Season'])[['OppScore', 'TeamScore']].mean().reset_index()
    axes[idx].scatter(grouped_df['OppScore'], grouped_df['TeamScore'], alpha=0.7)
    axes[idx].grid(True, linestyle='--', alpha=0.7)
    axes[idx].set_title(f'{gender} 2025 Average Team Score versus Opponent Score')
    axes[idx].set_xlabel('Average Opponent Score')
    axes[idx].set_ylabel('Average Team Score')
    for i in range(len(grouped_df)):
        axes[idx].text(grouped_df['OppScore'].iloc[i] + 0.1, grouped_df['TeamScore'].iloc[i] + 0.1, 
                       f'{grouped_df["TeamName"].iloc[i]} ({grouped_df["Season"].iloc[i]})', 
                       fontsize=8, alpha=0.7)
    circle_x, circle_y = circle_positions[gender]
    circle = patches.Circle((circle_x, circle_y), radius=5, color='firebrick', fill=False, linewidth=2)
    axes[idx].add_patch(circle)
plt.tight_layout()
plt.show()


# Load data separately
mens_df = pd.read_csv('/kaggle/working/MNCAAchampionshipteamstats.csv')
womens_df = pd.read_csv('/kaggle/working/WNCAAchampionshipteamstats.csv')

for df in [mens_df, womens_df]:
    df['Score Difference'] = df['TeamScore'] - df['OppScore']
    df['Team (Season)'] = df['TeamName'] + ' (' + df['Season'].astype(str) + ')'
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (gender, df) in enumerate([("Men's", mens_df), ("Women's", womens_df)]):
    sns.boxplot(x='Team (Season)', y='Score Difference', data=df, ax=axes[idx])
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90)
    axes[idx].set_title(f'{gender} NCAA Champions Scoring Differential')
plt.tight_layout()
plt.show()


mens_df = pd.read_csv('/kaggle/working/MNCAA2025teamstats.csv')
womens_df = pd.read_csv('/kaggle/working/WNCAA2025teamstats.csv')
mens_champ_df = pd.read_csv('/kaggle/working/MNCAAchampionshipteamstats.csv')
womens_champ_df = pd.read_csv('/kaggle/working/WNCAAchampionshipteamstats.csv')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (gender, season_df, champ_df) in enumerate([
    ("Men's", mens_df, mens_champ_df), 
    ("Women's", womens_df, womens_champ_df)]):
    season_df['Score Differential'] = season_df['TeamScore'] - season_df['OppScore']
    champ_df['Score Differential'] = champ_df['TeamScore'] - champ_df['OppScore']
    median_score = season_df.groupby('TeamName')['Score Differential'].median().reset_index()
    top_teams = median_score.sort_values(by='Score Differential', ascending=False).head(10)
    filtered_df = season_df[season_df['TeamName'].isin(top_teams['TeamName'])]
    filtered_df['Team Type'] = 'Top 10 Team'
    champ_df['Team Type'] = 'Champions'
    champ_df['TeamName'] = 'Champions' 
    combined_df = pd.concat([filtered_df, champ_df])
    median_score = combined_df.groupby('TeamName')['Score Differential'].median().reset_index()
    sorted_teams = median_score.sort_values(by='Score Differential', ascending=False)['TeamName'].tolist()
    palette = {team: 'steelblue' for team in sorted_teams if team != 'Champions'}
    palette['Champions'] = 'firebrick'

    sns.boxplot(
        x='TeamName', y='Score Differential', data=combined_df, 
        order=sorted_teams, palette=palette, ax=axes[idx])
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90)
    axes[idx].set_title(f'Winning Score Distributions for Top {gender} Teams & Champion')
    axes[idx].set_xlabel('Team')
    axes[idx].set_ylabel('Score Difference')
plt.tight_layout()
plt.show()


# Average Scoring from Past Champions
mens_df = pd.read_csv('/kaggle/working/MNCAAchampionshipteamstats.csv')
womens_df = pd.read_csv('/kaggle/working/WNCAAchampionshipteamstats.csv')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (gender, df) in enumerate([('Men\'s', mens_df), ('Women\'s', womens_df)]):
    grouped_df = df.groupby(['TeamName', 'Season'])[['OppTO', 'TeamTO']].mean().reset_index()
    axes[idx].scatter(grouped_df['OppTO'], grouped_df['TeamTO'], alpha=0.7)
    axes[idx].grid(True, linestyle='--', alpha=0.7)
    axes[idx].set_title(f'{gender} Average Team Turnovers versus Opponent Turnovers by Champions')
    axes[idx].set_xlabel('Average Opponent Turnovers')
    axes[idx].set_ylabel('Average Team Turnovers')
    for i in range(len(grouped_df)):
        axes[idx].text(grouped_df['OppTO'].iloc[i] + 0.1, grouped_df['TeamTO'].iloc[i] + 0.1, 
                       f'{grouped_df["TeamName"].iloc[i]} ({grouped_df["Season"].iloc[i]})', 
                       fontsize=8, alpha=0.7)
plt.tight_layout()
plt.show()


# Average Scoring from Past Champions
mens_df = pd.read_csv('/kaggle/working/MNCAA2025teamstats.csv')
womens_df = pd.read_csv('/kaggle/working/WNCAA2025teamstats.csv')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Define circle positions (adjust these values)
circle_positions = {  
    "Men's": (13.5, 15),
    "Women's": (18, 13.5)}
for idx, (gender, df) in enumerate([('Men\'s', mens_df), ('Women\'s', womens_df)]):
    grouped_df = df.groupby(['TeamName', 'Season'])[['OppTO', 'TeamTO']].mean().reset_index()
    axes[idx].scatter(grouped_df['OppTO'], grouped_df['TeamTO'], alpha=0.7)
    axes[idx].grid(True, linestyle='--', alpha=0.7)
    axes[idx].set_title(f'{gender} 2025 Average Team Turnovers versus Opponent Turnovers')
    axes[idx].set_xlabel('Average Opponent Turnovers')
    axes[idx].set_ylabel('Average Team Turnovers')
    for i in range(len(grouped_df)):
        axes[idx].text(grouped_df['OppTO'].iloc[i] + 0.1, grouped_df['TeamTO'].iloc[i] + 0.1, 
                       f'{grouped_df["TeamName"].iloc[i]} ({grouped_df["Season"].iloc[i]})', 
                       fontsize=8, alpha=0.7)
        circle_x, circle_y = circle_positions[gender]
    circle = patches.Circle((circle_x, circle_y), radius=2, color='firebrick', fill=False, linewidth=2)
    axes[idx].add_patch(circle)
plt.tight_layout()
plt.show()


mens_df = pd.read_csv('/kaggle/working/MNCAAchampionshipteamstats.csv')
womens_df = pd.read_csv('/kaggle/working/WNCAAchampionshipteamstats.csv')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Loop through each dataset (Men's and Women's)
for idx, (gender, df) in enumerate([("Men's", mens_df), ("Women's", womens_df)]):
    df['DayNum'] = pd.to_numeric(df['DayNum'], errors='coerce')
    df['Season'] = pd.to_numeric(df['Season'], errors='coerce')
    df['Win'] = pd.to_numeric(df['Win'], errors='coerce')
    df = df.dropna(subset=['DayNum', 'Season', 'Win'])
    df = df.sort_values(by=['Season', 'TeamName', 'DayNum'])
    legend_handles = {}
    ax = axes[idx]
    for (team, season), team_data in df.groupby(['TeamName', 'Season']):
        label = f"{team} ({season})"
        line, = ax.plot(team_data['DayNum'], team_data['Win'].cumsum(), label=label, alpha=0.7)
        legend_handles[label] = line

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Day Number')
    ax.set_ylabel('Cumulative Wins')
    ax.set_title(f'{gender} NCAA champions Win Progression Over the Season')
    sorted_legend = sorted(legend_handles.items(), key=lambda x: int(x[0].split('(')[-1][:-1]))
    sorted_labels, sorted_handles = zip(*sorted_legend)
    ax.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=2)

plt.tight_layout()
plt.show()



mens_df = pd.read_csv('/kaggle/working/MNCAA2025teamstats.csv')
womens_df = pd.read_csv('/kaggle/working/WNCAA2025teamstats.csv')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (gender, df) in enumerate([("Men's", mens_df), ("Women's", womens_df)]):
    df['DayNum'] = pd.to_numeric(df['DayNum'], errors='coerce')
    df['Win'] = pd.to_numeric(df['Win'], errors='coerce')
    df = df.dropna(subset=['DayNum', 'Win'])
    df = df.sort_values(by=['TeamName', 'DayNum'])
    total_wins = df.groupby('TeamName')['Win'].sum()
    if {'TeamScore', 'OppScore'}.issubset(df.columns):
        df['Score Difference'] = df['TeamScore'] - df['OppScore']
        median_score_diff = df.groupby('TeamName')['Score Difference'].median().reset_index()
        top_teams = median_score_diff.sort_values(by='Score Difference', ascending=False).head(25)
        top_teams = df[df['TeamName'].isin(top_teams['TeamName'])]
        ax = axes[idx]
        for team, team_data in top_teams.groupby('TeamName'):
            ax.plot(team_data['DayNum'], team_data['Win'].cumsum(), label=team, alpha=0.8)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Day Number')
        ax.set_ylabel('Cumulative Wins')
        ax.set_title(f'2025 NCAA {gender} (Top 25 Teams) Win Progression Over the Season')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
    else:
        print(f"Error: 'TeamScore' and 'OppScore' columns are missing from the {gender} dataset.")
plt.tight_layout()
plt.show()


mens_df = pd.read_csv('/kaggle/working/MNCAA2025teamstats.csv')
womens_df = pd.read_csv('/kaggle/working/WNCAA2025teamstats.csv')
mens_champ_df = pd.read_csv('/kaggle/working/MNCAAchampionshipteamstats.csv')
womens_champ_df = pd.read_csv('/kaggle/working/WNCAAchampionshipteamstats.csv')

for df in [mens_df, womens_df, mens_champ_df, womens_champ_df]:
    df['DayNum'] = pd.to_numeric(df['DayNum'], errors='coerce')
    df['Win'] = pd.to_numeric(df['Win'], errors='coerce')
    if 'TeamScore' in df.columns and 'OppScore' in df.columns:
        df['Score Difference'] = df['TeamScore'] - df['OppScore']

for df in [mens_df, womens_df, mens_champ_df, womens_champ_df]:
    df.dropna(subset=['DayNum', 'Win'], inplace=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (gender, season_df, champ_df) in enumerate([
    ("Men's", mens_df, mens_champ_df), 
    ("Women's", womens_df, womens_champ_df)]):
    ax = axes[idx]
    if 'Score Difference' in season_df.columns:
        median_score_diff = season_df.groupby('TeamName')['Score Difference'].median().reset_index()
        top_10_teams = median_score_diff.sort_values(by='Score Difference', ascending=False).head(10)
        season_df = season_df[season_df['TeamName'].isin(top_10_teams['TeamName'])]
    season_df.sort_values(by=['DayNum'], inplace=True)
    champ_df.sort_values(by=['Season', 'DayNum'], inplace=True)
    season_df['Cumulative_Wins'] = season_df.groupby('TeamName')['Win'].cumsum()
    champ_df['Cumulative_Wins'] = champ_df.groupby(['Season', 'TeamName'])['Win'].cumsum()
    average_wins = champ_df.groupby('DayNum')['Cumulative_Wins'].mean().reset_index()
    average_wins['Smoothed_Wins'] = average_wins['Cumulative_Wins'].rolling(window=5, min_periods=1).mean()
    for team, data in season_df.groupby('TeamName'):
        ax.plot(data['DayNum'], data['Cumulative_Wins'], label=team, alpha=0.8, color='steelblue')
        last_row = data.iloc[-1]
        ax.text(last_row['DayNum'], last_row['Cumulative_Wins'], team, fontsize=8, verticalalignment='bottom', horizontalalignment='left')
    ax.plot(average_wins['DayNum'], average_wins['Smoothed_Wins'], label="Champions", color='firebrick', linewidth=2)
    ax.text(average_wins['DayNum'].iloc[-1], average_wins['Smoothed_Wins'].iloc[-1], "Champions", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(range(20, 121, 20))
    ax.set_xlabel('Day Number')
    ax.set_ylabel('Cumulative Wins')
    ax.set_title(f'Win Progression of Top 2025 {gender} Teams vs. Past Champions')
plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np

# Load the datasets
menschampion_stats = pd.read_csv('/kaggle/working/MNCAAchampionshipteamstats.csv')
womenschampion_stats = pd.read_csv('/kaggle/working/WNCAAchampionshipteamstats.csv')

# Combine both datasets for joint analysis (optional)
menschampion_stats['Gender'] = 'Men'
womenschampion_stats['Gender'] = 'Women'
champion_stats = pd.concat([menschampion_stats, womenschampion_stats], ignore_index=True)

# Calculate additional statistics
champion_stats['ScoreDif'] = champion_stats['TeamScore'] - champion_stats['OppScore']
champion_stats['TeamFG%'] = champion_stats['TeamFGM'] / champion_stats['TeamFGA'].replace(0, np.nan)
champion_stats['TeamFG3%'] = champion_stats['TeamFGM3'] / champion_stats['TeamFGA3'].replace(0, np.nan)
champion_stats['TeamFT%'] = champion_stats['TeamFTM'] / champion_stats['TeamFTA'].replace(0, np.nan)
champion_stats['RebDif'] = (champion_stats['TeamOR'] + champion_stats['TeamDR']) - (champion_stats['OppOR'] + champion_stats['OppDR'])
champion_stats['AstDif'] = champion_stats['TeamAst'] - champion_stats['OppAst']
champion_stats['StlDif'] = champion_stats['TeamStl'] - champion_stats['OppStl']
champion_stats['BlkDif'] = champion_stats['TeamBlk'] - champion_stats['OppBlk']
champion_stats['TODif'] = champion_stats['OppTO'] - champion_stats['TeamTO']
champion_stats['PFDif'] = champion_stats['OppPF'] - champion_stats['TeamPF']

# Handling division by zero and NaN values
champion_stats.fillna(0, inplace=True)

# Compute total wins per team per season
if 'Win' in champion_stats.columns:
    wins_by_team = (
        champion_stats.groupby(['TeamName', 'Season'])['Win']
        .sum()
        .reset_index()
        .rename(columns={'Win': 'TotalWins'})
    )
    champion_stats = champion_stats.merge(wins_by_team, on=['TeamName', 'Season'], how='left')
else:
    print("Warning: 'Win' column not found. Ensure your dataset includes game results.")

# Selecting relevant columns for analysis
selected_columns = ['TeamName', 'Season', 'Gender', 'TotalWins', 'ScoreDif', 'TeamFG%', 'TeamFG3%', 'TeamFT%', 
                    'RebDif', 'AstDif', 'StlDif', 'BlkDif', 'TODif', 'PFDif']
champion_stats = champion_stats[selected_columns]

# Calculate the average for each stat across all teams
average_stats = champion_stats.drop(columns=['TeamName', 'Season', 'Gender']).mean().to_frame().T

# Add a row for "Champions" to represent the average of all championship teams
average_stats.insert(0, 'TeamName', 'Champions')

# Display final output
print("Average Stats Across All Championship Teams:")
print(average_stats)



import pandas as pd
import numpy as np

# Load datasets
mens_2025 = pd.read_csv('/kaggle/working/MNCAA2025teamstats.csv')
womens_2025 = pd.read_csv('/kaggle/working/WNCAA2025teamstats.csv')
mens_champ = pd.read_csv('/kaggle/working/MNCAAchampionshipteamstats.csv')
womens_champ = pd.read_csv('/kaggle/working/WNCAAchampionshipteamstats.csv')

def process_team_data(teamstats, champstats):
    # Compute statistics
    teamstats['ScoreDif'] = teamstats['TeamScore'] - teamstats['OppScore']
    teamstats['TeamFG%'] = np.where(teamstats['TeamFGA'] > 0, teamstats['TeamFGM'] / teamstats['TeamFGA'], 0)
    teamstats['TeamFG3%'] = np.where(teamstats['TeamFGA3'] > 0, teamstats['TeamFGM3'] / teamstats['TeamFGA3'], 0)
    teamstats['TeamFT%'] = np.where(teamstats['TeamFTA'] > 0, teamstats['TeamFTM'] / teamstats['TeamFTA'], 0)
    teamstats['RebDif'] = (teamstats['TeamOR'] + teamstats['TeamDR']) - (teamstats['OppOR'] + teamstats['OppDR'])
    teamstats['AstDif'] = teamstats['TeamAst'] - teamstats['OppAst']
    teamstats['StlDif'] = teamstats['TeamStl'] - teamstats['OppStl']
    teamstats['BlkDif'] = teamstats['TeamBlk'] - teamstats['OppBlk']
    teamstats['TODif'] = teamstats['OppTO'] - teamstats['TeamTO']
    teamstats['PFDif'] = teamstats['OppPF'] - teamstats['TeamPF']

    # Replace infinities and NaN values with 0
    teamstats.replace([float('inf'), float('-inf')], 0, inplace=True)
    teamstats.fillna(0, inplace=True)

    # Compute total wins by team
    wins_by_team = teamstats.groupby('TeamName')['Win'].sum().reset_index()

    # Compute average stats per team
    stat_columns = ['ScoreDif', 'TeamFG%', 'TeamFG3%', 'TeamFT%', 'RebDif', 'AstDif', 'StlDif', 'BlkDif', 'TODif', 'PFDif']
    average_by_team = teamstats.groupby('TeamName')[stat_columns].mean().reset_index()

    # Merge wins and average stats
    team_stats = pd.merge(wins_by_team, average_by_team, on='TeamName')

    # Compute statistics for championship teams
    champstats['ScoreDif'] = champstats['TeamScore'] - champstats['OppScore']
    champstats['TeamFG%'] = np.where(champstats['TeamFGA'] > 0, champstats['TeamFGM'] / champstats['TeamFGA'], 0)
    champstats['TeamFG3%'] = np.where(champstats['TeamFGA3'] > 0, champstats['TeamFGM3'] / champstats['TeamFGA3'], 0)
    champstats['TeamFT%'] = np.where(champstats['TeamFTA'] > 0, champstats['TeamFTM'] / champstats['TeamFTA'], 0)
    champstats['RebDif'] = (champstats['TeamOR'] + champstats['TeamDR']) - (champstats['OppOR'] + champstats['OppDR'])
    champstats['AstDif'] = champstats['TeamAst'] - champstats['OppAst']
    champstats['StlDif'] = champstats['TeamStl'] - champstats['OppStl']
    champstats['BlkDif'] = champstats['TeamBlk'] - champstats['OppBlk']
    champstats['TODif'] = champstats['OppTO'] - champstats['TeamTO']
    champstats['PFDif'] = champstats['OppPF'] - champstats['TeamPF']

    champstats.replace([float('inf'), float('-inf')], 0, inplace=True)
    champstats.fillna(0, inplace=True)

    champion_avg_stats = champstats.mean(numeric_only=True).to_frame().T
    champion_avg_stats.insert(0, 'TeamName', 'Champions')

    # Compute final score
    team_stats['FinalScore'] = sum(
        team_stats[col] - champion_avg_stats[col].values[0] for col in stat_columns + ['Win']
    )

    min_score, max_score = team_stats['FinalScore'].min(), team_stats['FinalScore'].max()
    team_stats['Champion_Probability'] = 1 if max_score == min_score else (team_stats['FinalScore'] - min_score) / (max_score - min_score)

    team_stats = team_stats.sort_values(by='FinalScore', ascending=False)
    return team_stats.nlargest(1, 'FinalScore')['TeamName'].iloc[0]

# Predict Champions
mens_champion = process_team_data(mens_2025, mens_champ)
womens_champion = process_team_data(womens_2025, womens_champ)

print("\n" + "="*50)
print(f"ğŸ�†MEN'S 2025 NCAA Champion Prediction: {mens_champion.upper()} ğŸ�€ğŸ”µ")
print("="*50 + "\n")
print("\n" + "="*50)
print(f"ğŸ�†WOMEN'S 2025 NCAA Champion Prediction: {womens_champion.upper()} ğŸ�€ğŸ¤ ")
print("="*50 + "\n")

