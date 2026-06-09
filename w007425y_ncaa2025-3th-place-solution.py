import pandas as pd
year = 2025
kaggleFolderPath = '/kaggle/input/march-machine-learning-mania-' + str(year)
fivethirtyeightFolderPath = '/kaggle/input/538data'


#Mens Probability Matrix
#source: https://github.com/gotoConversion/goto_conversion
#Matrices were computed by converting betting odds to probabilities using goto_conversion

mensProbabilities_df = pd.read_csv(fivethirtyeightFolderPath + '/mensProbabilitiesTable.csv', index_col = 'player') #source: https://github.com/gotoConversion/goto_conversion
mensProbabilities_df = mensProbabilities_df.drop('Elo_Rating', axis=1)


#Womens Probability Matrix
#source: https://github.com/gotoConversion/goto_conversion
#Matrices were computed by converting betting odds to probabilities using goto_conversion

womensProbabilities_df = pd.read_csv(fivethirtyeightFolderPath + '/womensProbabilitiesTable.csv', index_col = 'player') #source: https://github.com/gotoConversion/goto_conversion
womensProbabilities_df = womensProbabilities_df.drop('Elo_Rating', axis=1)


#Import team seeds
mensTeamSeeds_df = pd.read_csv(kaggleFolderPath + '/MNCAATourneySeeds.csv')
mensTeamSeeds2025_df = mensTeamSeeds_df.iloc[[x == year for x in mensTeamSeeds_df['Season']]]
womensTeamSeeds_df = pd.read_csv(kaggleFolderPath + '/WNCAATourneySeeds.csv')
womensTeamSeeds2025_df = womensTeamSeeds_df.iloc[[x == year for x in womensTeamSeeds_df['Season']]]


#Implement Optimal Strategy (if you agree)
def get_roundOfMatch(team1, team2, seeds_df):

    slotMap = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

    team1_seed = seeds_df.loc[[x == team1 for x in seeds_df['TeamID']],'Seed'].values[0]
    team2_seed = seeds_df.loc[[x == team2 for x in seeds_df['TeamID']],'Seed'].values[0]

    isFirstFourMatch = team1_seed[:3] == team2_seed[:3]
    if isFirstFourMatch:
        return 1

    team1_region = str(team1_seed[:1])
    team2_region = str(team2_seed[:1])

    team1_seedNumber = int(team1_seed[1:3]) #careful with first four teams
    team2_seedNumber = int(team2_seed[1:3]) #careful with first four teams

    isRegionSame = team1_region == team2_region
    if not isRegionSame:

        isTeam1_regionWX = team1_region in ['W','X']
        isTeam2_regionWX = team2_region in ['W','X']

        if isTeam1_regionWX and isTeam2_regionWX: #both W or X region
            return 6

        elif (not isTeam1_regionWX) and (not isTeam2_regionWX): #both not W or X region
            return 6

        else:
            return 7

    else: #same region

        team1_slot = slotMap.index(team1_seedNumber)
        team2_slot = slotMap.index(team2_seedNumber)

        isRound2 = (team1_slot // 2) == (team2_slot // 2)  #round of 64 or first four (not counted anyway)
        if isRound2:
            return 2

        isRound3 = (team1_slot // 4) == (team2_slot // 4)
        if isRound3: #yet to find why but "elif" throws error
            return 3

        isRound4 = (team1_slot // 8) == (team2_slot // 8)
        if isRound4: #yet to find why but "elif" throws error
            return 4

        else:
            return 5

def get_tourneyFlag(team1, team2, seeds_df):

    tourneyTeams = seeds_df['TeamID'].tolist()

    isTeam1InTourney = team1 in tourneyTeams
    isTeam2InTourney = team2 in tourneyTeams

    if isTeam1InTourney and isTeam2InTourney:
        return get_roundOfMatch(team1, team2, seeds_df)

    else:
        return 0

def get_flag_list(submission_df, mensTeamSeeds2025_df, womensTeamSeeds2025_df):
    flag_list = []
    for i in range(submission_df.shape[0]):

        currRow = submission_df.iloc[i,0].split('_')
        team1 = int(currRow[1])
        team2 = int(currRow[2])

        isWomensMatch = team1 + team2 > 6000
        if isWomensMatch:
            flag = get_tourneyFlag(team1, team2, womensTeamSeeds2025_df)
        else:
            flag = get_tourneyFlag(team1, team2, mensTeamSeeds2025_df)

        flag_list.append(flag)
    return flag_list

def set_optimalStrategy(submission_df, mensTeamSeeds2025_df, womensTeamSeeds2025_df, riskTeams, riskTeamToWinRounds):
    # Get the list of match round flags
    flag_list = get_flag_list(submission_df, mensTeamSeeds2025_df, womensTeamSeeds2025_df)

    # Ensure riskTeams and riskTeamToWinRounds correspond
    if len(riskTeams) != len(riskTeamToWinRounds):
        raise ValueError("The lengths of riskTeams and riskTeamToWinRounds must be the same")

    # Create a dictionary mapping team IDs to their winning round ranges
    risk_dict = dict(zip(riskTeams, riskTeamToWinRounds))

    # Iterate through each row in the submission file
    for i in range(submission_df.shape[0]):
        submission_row = submission_df.iloc[i, 0].split('_')
        submission_round = flag_list[i]

        team1 = int(submission_row[1])
        team2 = int(submission_row[2])

        # Check if team1 is in riskTeams
        if team1 in risk_dict:
            if (0 < submission_round) and (submission_round <= risk_dict[team1]):
                submission_df.at[i, 'Pred'] = 1.0  # team1 wins
                print(submission_df.iloc[i])

        # Check if team2 is in riskTeams
        elif team2 in risk_dict:
            if (0 < submission_round) and (submission_round <= risk_dict[team2]):
                submission_df.at[i, 'Pred'] = 0.0  # team2 wins
                print(submission_df.iloc[i])

    return submission_df

submission_df = pd.read_csv('/kaggle/input/538data/submission.csv')

riskTeams = [1179] 
riskTeamToWinRounds = [2]  
submission_df = set_optimalStrategy(submission_df, mensTeamSeeds2025_df, womensTeamSeeds2025_df, riskTeams, riskTeamToWinRounds)

# Define multiple risk teams and their winning round ranges
riskTeams = [1181, 1120, 1196, 1222, 3163, 3376] 
riskTeamToWinRounds = [6, 5, 5, 5, 6, 6]  

# Call the modified function
submission_df = set_optimalStrategy(submission_df, mensTeamSeeds2025_df, womensTeamSeeds2025_df, riskTeams, riskTeamToWinRounds)
submission_df.to_csv('submission.csv', index=False)

