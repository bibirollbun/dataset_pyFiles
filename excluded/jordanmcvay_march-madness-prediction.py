import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

input_dir = '../input/march-machine-learning-mania-2025/'
Seeds = pd.read_csv(input_dir + 'MNCAATourneySeeds.csv')  # Tournament seed rankings
Coaches = pd.read_csv(input_dir + 'MTeamCoaches.csv')  # Team coach information
Conferences = pd.read_csv(input_dir + 'MTeamConferences.csv')  # Team conference affiliations
RegularDetail = pd.read_csv(input_dir + 'MRegularSeasonDetailedResults.csv')  # Regular season game stats for every team
TourneyCompact = pd.read_csv(input_dir + 'MNCAATourneyCompactResults.csv')  # Past tournament game results
MasseyOrdinals = pd.read_csv(input_dir + 'MMasseyOrdinals.csv')  # Team ranking system from Massey Ordinals
display(RegularDetail.columns.values)


WinTeams = pd.DataFrame()
LoseTeams = pd.DataFrame()

# Define the columns to be used for both WinTeams and LoseTeams that we want to preserve
columns = ['Season', 'TeamID', 'Points', 'OppPoints',
       'Loc', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'OppFGM', 'OppFGA',
       'OppFGM3', 'OppFGA3', 'OppFTM', 'OppFTA', 'OppOR', 'OppDR', 'OppAst', 'OppTO',
       'OppStl', 'OppBlk', 'OppPF']

#  Extracts statistics for winning teams(team we passed in with ID is team that won
WinTeams[columns] = RegularDetail[['Season',  'WTeamID', 'WScore', 'LScore',
       'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
       'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA',
       'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO',
       'LStl', 'LBlk', 'LPF']]

WinTeams['Wins'] = 1  # Add wins column(they always win)
WinTeams['Losses'] = 0  # Losing column remains 0 for winners

# Extract statistics for losing teams(replace W's with L's and L's with W's because we are swapping the ID )
LoseTeams[columns] = RegularDetail[['Season',  'LTeamID', 'LScore', 'WScore',
       'WLoc', 'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
       'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA',
       'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO',
       'WStl', 'WBlk', 'WPF']]

# Adjust location column (switch home/away for losing teams)
def change_loc(loc):
    if loc == 'H': 
        return 'A'
    elif loc == 'A': 
        return 'H'
    else: return 'N'  # 'N' is neutral location
# flips every home and away
LoseTeams['Loc'] = LoseTeams['Loc'].apply(change_loc)

LoseTeams['Wins'] = 0  # ID we care about is the loser
LoseTeams['Losses'] = 1  # Add 1 to losses column

# Combine both winning and losing teams into one dataset
WinLoseTeams = pd.concat([WinTeams, LoseTeams]).drop(columns=['Loc']) # Drop 'Loc' column since it was modified
# Group by TeamID and Season to compute total stats
combinedTeams = WinLoseTeams.groupby(['Season', 'TeamID']).sum() # each team id has its own line in a season
combinedTeams['NumGames'] = combinedTeams['Wins'] + combinedTeams['Losses'] # Compute total games played to calculate per-game stats
display(combinedTeams.columns.values) # Display computed column names


# now we have all these features but need to make them comparable to each other
# ex: you may have more points scored just because you played more games, not because you were better
RegularSeasonInput = pd.DataFrame()

# Create RegularSeasonInput DataFrame for per-game stats
RegularSeasonInput['WinRatio'] = combinedTeams['Wins'] / combinedTeams['NumGames']
RegularSeasonInput['PointsPerGame'] = combinedTeams['Points'] / combinedTeams['NumGames']
RegularSeasonInput['PointsAllowedPerGame'] = combinedTeams['OppPoints'] / combinedTeams['NumGames']
RegularSeasonInput['PointsRatio'] = combinedTeams['Points'] / combinedTeams['OppPoints']
RegularSeasonInput['OTsPerGame'] = combinedTeams['NumOT'] / combinedTeams['NumGames']

RegularSeasonInput['FGPerGame'] = combinedTeams['FGM'] / combinedTeams['NumGames']
RegularSeasonInput['FGRatio'] = combinedTeams['FGM'] / combinedTeams['FGA']
RegularSeasonInput['FGsAllowedPerGame'] = combinedTeams['OppFGM'] / combinedTeams['NumGames']

RegularSeasonInput['FG3PerGame'] = combinedTeams['FGM3'] / combinedTeams['NumGames']
RegularSeasonInput['FG3Ratio'] = combinedTeams['FGM3'] / combinedTeams['FGA3']
RegularSeasonInput['FG3AllowedPerGame'] = combinedTeams['OppFGM3'] / combinedTeams['NumGames']

RegularSeasonInput['FTPerGame'] = combinedTeams['FTM'] / combinedTeams['NumGames']
RegularSeasonInput['FTRatio'] = combinedTeams['FTM'] / combinedTeams['FTA']
RegularSeasonInput['FTAllowedPerGame'] = combinedTeams['OppFTM'] / combinedTeams['NumGames']

RegularSeasonInput['ORRatio'] = combinedTeams['OR'] / (combinedTeams['OR'] + combinedTeams['OppDR'])
RegularSeasonInput['DRRatio'] = combinedTeams['DR'] / (combinedTeams['DR'] + combinedTeams['OppOR'])

RegularSeasonInput['AstPerGame'] = combinedTeams['Ast'] / combinedTeams['NumGames']
RegularSeasonInput['TOPerGame'] = combinedTeams['TO'] / combinedTeams['NumGames']
RegularSeasonInput['StlPerGame'] = combinedTeams['Stl'] / combinedTeams['NumGames']
RegularSeasonInput['BlkPerGame'] = combinedTeams['Blk'] / combinedTeams['NumGames']
RegularSeasonInput['PFPerGame'] = combinedTeams['PF'] / combinedTeams['NumGames']
display(RegularSeasonInput) # Display computed stats


# Filters MasseyOrdinals for final pre-tournament rankings (day 133)   
MasseyOrdinals = MasseyOrdinals[MasseyOrdinals['RankingDayNum'] == 133]
team_rankings = {} # dictionary for team rankingsfrom Massey Ordinals
for i in range(len(MasseyOrdinals)):
    idx = (MasseyOrdinals["Season"].iloc[i], MasseyOrdinals["TeamID"].iloc[i]) # Create a (Season, TeamID) key
    team_rankings[idx] = MasseyOrdinals["OrdinalRank"].iloc[i] # Store the team's ordinal rank

# Convert tournament seed data into a dictionary for quick lookup    
seed_dict = Seeds.set_index(['Season', 'TeamID'])

# Initialize the TourneyInput dataframe to store tournament matchups
TourneyInput = pd.DataFrame()

# Process tournament winners and losers separately
winners = pd.DataFrame()
# do not want to know whether team1 or team2 is winner since that would bias the model
winners[['Season' , 'Team1' , 'Team2']] = TourneyCompact[['Season', 'WTeamID', 'LTeamID']]
winners['Result'] = 1 # Mark winners with Result = 1
losers = pd.DataFrame()
losers[['Season' , 'Team1' , 'Team2']] = TourneyCompact[['Season', 'LTeamID', 'WTeamID']]
losers['Result'] = 0  # Mark losers with Result = 0

#  Combine winners and losers into TourneyInput
TourneyInput = pd.concat([winners, losers])
# Keep only seasons from 2003 onwards since RegularSeasonInput starts at 2003
TourneyInput = TourneyInput[TourneyInput['Season']>=2003].reset_index(drop=True) 

# Create dictionaries to store conference and coach information for each team
team_conference = {}
team_coach = {}
# Map team conferences from the Conferences dataset
for i in range(len(Conferences)):
    idx = (Conferences["Season"][i], Conferences["TeamID"][i])
    team_conference[idx] = Conferences["ConfAbbrev"][i] # stores conference abbreviation
#  Map team coaches from the Coaches dataset
for i in range(len(Coaches)):
    idx = (Coaches["Season"][i], Coaches["TeamID"][i])
    team_coach[idx] = Coaches["CoachName"][i] # stores coach name
    
#  Initialize lists to store tournament features for each team
team1_confs = []
team2_confs = []
team1_coaches = []
team2_coaches = []
team1seeds = []
team2seeds = []
team1_ranks = []
team2_ranks = []

# Loop through each tournament matchup to extract team features
for i in range(len(TourneyInput)):
    idx = (TourneyInput["Season"][i], TourneyInput["Team1"][i])
    team1_confs.append(team_conference[idx]) # Get Team 1's conference
    team1_coaches.append(team_coach[idx]) # Get Team 1's coach
    team1_ranks.append(team_rankings[idx]) # Get Team 1's rank
    idx = (TourneyInput["Season"][i], TourneyInput["Team2"][i])
    team2_confs.append(team_conference[idx])  # Get Team 2's conference
    team2_coaches.append(team_coach[idx]) # Get Team 2's coach
    team2_ranks.append(team_rankings[idx]) # Get Team 2's rank
 

# Extract and convert tournament seeds
for i in range(len(TourneyInput)):   
    idx = (TourneyInput["Season"][i], TourneyInput['Team1'][i])
    seed = seed_dict.loc[idx].values[0] # Extract seed
    if(len(seed)==4): # Convert to integer
        seed = int(seed[1:-1]) # seed has a letter to left and right of seed corresponding to its conference, which we don't need
    else:
        seed = int(seed[1:]) # seed has a letter to left of seed corresponding to its conference, which we don't need
    team1seeds.append(seed)  
    idx = (TourneyInput["Season"][i], TourneyInput['Team2'][i])
    seed = seed_dict.loc[idx].values[0]  # Extracts seed
    if(len(seed)==4): # Convert to integer
        seed = int(seed[1:-1]) # removes conference letter to right and left of seed if length is 4
    else:
        seed = int(seed[1:]) # removes conference letter to left of seed if length is 3
    team2seeds.append(seed)  

# Add extracted features to TourneyInput
TourneyInput['Team1seed'] = team1seeds
TourneyInput['Team2seed'] = team2seeds
TourneyInput["Team1Conf"] = team1_confs
TourneyInput["Team2Conf"] = team2_confs
TourneyInput["Team1Coach"] = team1_coaches
TourneyInput["Team2Coach"] = team2_coaches
TourneyInput['Team1Rank'] = team1_ranks
TourneyInput['Team2Rank'] = team2_ranks

# Initialize dictionaries to track conference strength and coaching experience
conf_strength = {}
coach_experience = {}
# Count total games per conference
conf_games = Conferences.groupby(['Season', 'ConfAbbrev']).size().to_dict()
# Count tournament wins per team
conf_wins = TourneyCompact.groupby(['Season', 'WTeamID']).size().to_dict()

conf_team_wins = {}
# Compute the total wins for each conference
for i in range(len(TourneyInput)):
    # extracts season and team IDs for matchup
    season = TourneyInput["Season"].iloc[i]
    team1 = TourneyInput["Team1"].iloc[i]
    team2 = TourneyInput["Team2"].iloc[i]
    # Retrieves conference affiliations for both teams
    conf1 = team_conference.get((season, team1), None)
    conf2 = team_conference.get((season, team2), None)
    # updates total wins for both conferences for that season
    conf_team_wins[(season, conf1)] = conf_team_wins.get((season, conf1), 0) + conf_wins.get((season, team1), 0)
    conf_team_wins[(season, conf2)] = conf_team_wins.get((season, conf2), 0) + conf_wins.get((season, team2), 0)
# Compute coaching experience       
for i in range(len(Coaches)):
    season = Coaches["Season"][i]
    coach = Coaches["CoachName"][i]
    # Updates coaching experience dictionary
    # If coach existed in the previous season, add 1 to their experience
    # Otherwise, set their experience to 1 (first recorded season as head coach)
    coach_experience[(season, coach)] = coach_experience.get((season - 1, coach), 0) + 1

# Initialize lists to store calculated conference strength and coaching experience
team1_conf_strength = []
team2_conf_strength = []
team1_coach_experience = []
team2_coach_experience = []

# Compute conference strength and coaching experience for each matchup
for i in range(len(TourneyInput)): 
    season = TourneyInput["Season"][i]
    idx = (season, TourneyInput['Team1Conf'][i]) # Create (season, conference) key
    conf_strength[idx] = conf_team_wins.get(idx) / conf_games.get(idx) # Calculate strength as wins/games
    team1_conf_strength.append(conf_strength[idx])
    
    idx = (season, TourneyInput['Team2Conf'][i]) # Create (season, conference) key
    conf_strength[idx] = conf_team_wins.get(idx) / conf_games.get(idx) # Calculate strength as wins/games
    team2_conf_strength.append(conf_strength[idx])
    
    idx = (season, TourneyInput["Team1Coach"][i]) # Create (season, coach) key
    coach_experience[idx] = coach_experience.get(idx)  # Get experience
    team1_coach_experience.append(coach_experience[idx])

    idx = (season, TourneyInput["Team2Coach"][i]) # Create (season, coach) key
    coach_experience[idx] = coach_experience.get(idx)  # Get experience
    team2_coach_experience.append(coach_experience[idx])

# Add computed strength & experience values to TourneyInput
TourneyInput["Team1ConfStrength"] = team1_conf_strength
TourneyInput["Team2ConfStrength"] = team2_conf_strength
TourneyInput["Team1CoachExperience"] = team1_coach_experience
TourneyInput["Team2CoachExperience"] = team2_coach_experience
#  Compute differences in rankings, conference strength, and coach experience
TourneyInput["ConfStrengthDiff"] = TourneyInput["Team1ConfStrength"] - TourneyInput["Team2ConfStrength"]
TourneyInput["CoachExperienceDiff"] = TourneyInput["Team1CoachExperience"] - TourneyInput["Team2CoachExperience"]
TourneyInput["OrdinalRankDiff"] = TourneyInput["Team2Rank"] - TourneyInput["Team1Rank"]
display(TourneyInput)


# An empty list to store the final game-level differences
outscores = []
for i in range(len(TourneyInput)):
    # Extracts (Season, Team1) index for retrieving team statistics from RegularSeasonInput
    idx = (TourneyInput['Season'][i], TourneyInput['Team1'][i])
    # Retrieves Team 1's regular season statistics at the index
    team1score = RegularSeasonInput.loc[idx].copy()
    # Adds Team 1's tournament seed as an additional feature(from TourneyInput)
    team1score['Seed'] = TourneyInput['Team1seed'][i]
    # Extracts (Season, Team2) index for retrieving Team 2's statistics from RegularSeasonInput
    idx = (TourneyInput['Season'][i], TourneyInput['Team2'][i])
    # Retrieves Team 2's regular season statistics
    team2score = RegularSeasonInput.loc[idx].copy()
    # Adds Team 2's tournament seed as an additional feature(from TourneyInput)
    team2score['Seed'] = TourneyInput['Team2seed'][i]

   
    # Computes feature differences between Team 1 and Team 2
    outscore = team1score - team2score # Subtract Team 2's stats from Team 1's stats
    # Adds conference strength difference between Team 1 and Team 2
    outscore['ConfStrengthDiff'] = TourneyInput['ConfStrengthDiff'][i]
    # Adds coaching experience difference between Team 1 and Team 2
    outscore['CoachExperienceDiff'] = TourneyInput['CoachExperienceDiff'][i]
    # Adds ordinal rank difference (Team 2 rank - Team 1 rank, so lower means better ranking for Team 1)
    outscore['OrdinalRankDiff'] = TourneyInput["OrdinalRankDiff"][i]
    # Stores game result (1 if Team 1 won, 0 if Team 2 won)
    outscore['Result'] = TourneyInput['Result'][i] # Now we have entire complete row
    # Appends computed outscore data for this matchup to the outscores list(gives list of single line data frames)
    outscores.append(outscore)
# Converts list of game-level differences into a Pandas DataFrame
outscores = pd.DataFrame(outscores)

display(outscores)
display(outscores.describe())


# Computes correlation matrix to understand which features actually would affect prediciton the most
correlations = round(outscores.corr(),2) # Rounds correlation values to 2 decimal places
# Display the absolute correlation of each feature with the target variable ('Result')
display(np.abs(correlations['Result'])) # Helps identify which features influence the outcome the most

import seaborn as sns # For visualizing correlations
plt.figure(figsize=(15,10))
# Generate a heatmap to visualize feature correlations
sns.heatmap(correlations)
plt.show() # sns uses pyplot as a backbone


# Extracts feature matrix (X) and target variable (y) for model training
X = outscores[outscores.columns[:-1]].values # Excludes the 'Result' column (last column
y = outscores['Result'].values # Target variable (1 = Team1 won, 0 = Team2 won)

# Sets random seed to ensure reproducibility of results
np.random.seed(1)
# Shuffles dataset indices randomly
idx = np.random.permutation(len(X)) # Randomly mixing up all of our inputs since currently all 1's are on top and all 0's on bottom
# Splits shuffled dataset into training and testing sets (80% training, 20% testing)
train_idx = idx[:int(-.2*len(X))]
test_idx = idx[int(-.2*len(X)):]
# Creates training and testing datasets based on selected indices
X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]
# Normalizes feature values using Min-Max scaling (0 to 1 range)
mins = X_train.min(axis=0)  # Find the minimum value per feature
maxs = X_train.max(axis=0)  # Find the maximum value per feature
# Apply Min-Max scaling transformation to the training and testing sets
X_train = (X_train - mins)/(maxs - mins)
X_test = (X_test - mins)/(maxs - mins)
# Print the shapes of the datasets to verify correct splitting
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# from sci-kit learn
from sklearn.ensemble import AdaBoostClassifier # For training the model
# Initializes AdaBoost model: 200 estimators improves accuracy, 5% speed prevents overfitting, same results each time
model = AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=1)
model.fit(X_train, y_train)
# Evaluates model and print its accuracy on the test dataset
print(model.score(X_test, y_test))


