import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
import pprint


# Helper Functions
def summarize_dataset(df_data):
    summary = pd.DataFrame({
        'Feature': df_data.columns,                        # Feature names
        'Data Type': df_data.dtypes.astype(str).values,   # Keep actual data types
        'Missing Values': df_data.isnull().sum().values,  # Number of missing values
        'Missing Percentage': (df_data.isnull().sum() / len(df_data) * 100).values,  # Percentage of missing values
        'Example Values': df_data.apply(lambda col: ', '.join(map(str, col.dropna().unique()[:3]))),  # Get 3 unique values
        'Distinct Count': df_data.nunique().values,       # Number of distinct values
    })

    return summary

def optimized_generate_id(season, first_team, second_team):
    min_team = np.minimum(first_team, second_team).astype(str)
    max_team = np.maximum(first_team, second_team).astype(str)
    season_str = season.astype(str)
    return season_str + '_' + min_team + '_' + max_team

def optimized_generate_region(seed):
    seed_series = pd.Series(seed).fillna("0")  # Handle NaN values

    def safe_int_conversion(x):
        if x != "0":
            try:
                return int(x)
            except ValueError:
                return 0  # Or another default value
        else:
            return 0

    region = np.where(seed_series != "0", seed_series.str[0], "0")
    seed_num = np.where(seed_series != "0", seed_series.str[1:3].apply(safe_int_conversion), 0)
    return region, seed_num

def optimized_apply_generate_id(df):
    df["ID"] = optimized_generate_id(df['Season'], df['WTeamID'], df['LTeamID'])
    df["TeamID1"] = np.minimum(df['WTeamID'], df['LTeamID'])
    df["TeamID2"] = np.maximum(df['WTeamID'], df['LTeamID'])
    df["Pred"] = np.where(df["TeamID1"] == df['WTeamID'], 1, 0)

    seeds1 = np.where(df["TeamID1"] == df['WTeamID'], df["WSeed"], df["LSeed"])
    regions1, seed_nums1 = optimized_generate_region(seeds1)
    df["Region1"] = regions1
    df["SeedNum1"] = seed_nums1

    seeds2 = np.where(df["TeamID2"] == df['WTeamID'], df["WSeed"], df["LSeed"])
    regions2, seed_nums2 = optimized_generate_region(seeds2)
    df["Region2"] = regions2
    df["SeedNum2"] = seed_nums2

    return df

# Function (Supporting Function for Seed Margin calculation)

def digit_difference(str1, str2):
    digits1 = "".join(filter(str.isdigit, str1))
    digits2 = "".join(filter(str.isdigit, str2))

    if not digits1 or not digits2:
        return 0  # Return 0 if no digits found in either string

    num1 = int(digits1)
    num2 = int(digits2)

    return num1 - num2

def calculate_seed_margin(df):
    df['SeedMargin'] = df['SeedNum1'] - df['SeedNum2']

    return df

def read_file(file_name: str):
    return pd.read_csv(base_path + file_name)


base_path = "/kaggle/input/march-machine-learning-mania-2025/"

# Team
women_team = read_file("WTeams.csv")
men_team = read_file("MTeams.csv")

women_seeds = read_file("WNCAATourneySeeds.csv")
men_seeds = read_file("MNCAATourneySeeds.csv")    
seeds = pd.concat([women_seeds, men_seeds])

# Season
women_season = read_file("WSeasons.csv")
men_season = read_file("MSeasons.csv")

season_df = pd.concat([women_season, men_season])

## Regular Detailed Results
Wregular_DR = read_file("MRegularSeasonDetailedResults.csv")
Mregular_DR = read_file("WRegularSeasonDetailedResults.csv")

regular_DR = pd.concat([Wregular_DR, Mregular_DR])
regular_DR["MatchType"] = "Regular"

## Tourney Detailed Results
Wtourney_DR = read_file("WNCAATourneyDetailedResults.csv")
Mtourney_DR = read_file("MNCAATourneyDetailedResults.csv")

tourney_DR = pd.concat([Wtourney_DR, Mtourney_DR])
tourney_DR["MatchType"] = "Tournament"


# Map Seeds to Winning Team and Losing Team
def map_seed(df: pd.DataFrame, seeds: pd.DataFrame):
    detailed_result_df = df.copy()
    winning_seed = seeds.rename(columns={
        "TeamID": "WTeamID",
        "Seed": "WSeed",
    })
    losing_seed = seeds.rename(columns={
        "TeamID": "LTeamID",
        "Seed": "LSeed",
    })
    
    detailed_result_df = detailed_result_df.merge(
        winning_seed,
        on=["Season", "WTeamID"],
        how="left"
    )
    
    detailed_result_df = detailed_result_df.merge(
        losing_seed,
        on=["Season", "LTeamID"],
        how="left"
    )

    return detailed_result_df


# Fill not available seeds with 0
detailed_result_df = map_seed(pd.concat([regular_DR, tourney_DR]), seeds)
detailed_result_df["WSeed"] = pd.Series(detailed_result_df["WSeed"]).fillna(0)
detailed_result_df["LSeed"] = pd.Series(detailed_result_df["LSeed"]).fillna(0)

# Add Team Id 1, 2, Seed Margin and Seed Num 1 and 2
detailed_result_df = optimized_apply_generate_id(detailed_result_df.copy())
detailed_result_df = calculate_seed_margin(detailed_result_df)


def get_team_statistic(df: pd.DataFrame) -> pd.DataFrame:
    all_tournament_grouped_win = df.copy()
    all_tournament_grouped_win["Wins"] = 0
    all_tournament_grouped_win = all_tournament_grouped_win.groupby(["WTeamID", "Season"]).count().reset_index()[["WTeamID", "Season", "Wins"]]

    all_tournament_grouped_lose = df.copy()
    all_tournament_grouped_lose["Losses"] = 0
    all_tournament_grouped_lose = all_tournament_grouped_lose.groupby(["LTeamID", "Season"]).count().reset_index()[["LTeamID", "Season", "Losses"]]

    ## compute total games played
    team_statistic = all_tournament_grouped_win.merge(
        all_tournament_grouped_lose, 
        left_on=["WTeamID", "Season"], 
        right_on=["LTeamID", "Season"],
        how="inner"
    )
    team_statistic.fillna(0, inplace=True)
    team_statistic["TotalGames"] = team_statistic["Wins"] + team_statistic["Losses"]
    team_statistic["WinRate"] = team_statistic["Wins"] / team_statistic["TotalGames"]

    return team_statistic
    
def generate_top_10_team(df: pd.DataFrame):
    team_statistic = get_team_statistic(df)
    
    team_statistic_mean_10_years = team_statistic[team_statistic["Season"] > 2014]
    team_statistic_mean_10_years = team_statistic_mean_10_years.groupby(['WTeamID'])['WinRate'].mean().reset_index()
    team_statistic_mean_10_years = team_statistic_mean_10_years.sort_values(by='WinRate', ascending=False)
    team_statistic_mean_10_years = team_statistic_mean_10_years[:10]
    order = team_statistic_mean_10_years["WTeamID"]
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x='WTeamID', y='WinRate', data=team_statistic_mean_10_years, order=order)
    
    plt.title('Mean Win Rate by Team ID')
    plt.xlabel('Team ID')
    plt.ylabel('Mean Win Rate')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()



print("Women Team Statistic:")
Wteam_stat = generate_top_10_team(Wtourney_DR)

print("Men Team Statistic:")
Mteam_stat = generate_top_10_team(Mtourney_DR)


def show_consecutive_win(df: pd.DataFrame):
    list_season = list(range(2015,2025,1))
    dic_top5_10year = {}
    list_top5_10year = []

    team_statistic = get_team_statistic(df)
    
    for year in list_season:
        team_statistic_season = team_statistic[team_statistic["Season"] == year]
        team_statistic_season = team_statistic_season.sort_values(by='WinRate', ascending=False)
        team_statistic_season = team_statistic_season.reset_index(drop=True)
        team_statistic_season = team_statistic_season[:5]
        dic_top5_10year["top5" + str(year)] = team_statistic_season
        list_top5_10year.append(team_statistic_season)

    concatenated_df = pd.concat(list_top5_10year, ignore_index=True)
    # Create a pivot table with TeamID as rows, Season as columns, and WinRate as values
    pivot_table = concatenated_df.pivot_table(values='WinRate', index='WTeamID', columns='Season')
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))  # Adjust figure size as needed
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".3f")  # You can change the colormap and formatting
    
    # Customize the plot
    plt.title('Win Rate Heatmap by Team ID and Season')
    plt.xlabel('Season')
    plt.ylabel('Team ID')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    

print("Women Team Statistic:")
Wteam_stat = show_consecutive_win(Wtourney_DR)
# show_consecutive_win(concatenated_df)

print("Men Team Statistic:")
Mteam_stat = show_consecutive_win(Mtourney_DR)


def check_seed_win_correlation(df: pd.DataFrame, seeds: pd.DataFrame):
    df = optimized_apply_generate_id(map_seed(df, seeds))
    
    # Win Margin Between Team
    df["WinMargin"] = df["WScore"] - df["LScore"]
    
    # Seed Margin (In Tournament)
    df["SeedMargin"] = df.apply(lambda row: int(row["WSeed"][1:3]) - int(row["LSeed"][1:3]), axis=1)

    display(df[["SeedMargin", "WinMargin"]].corr())

    sns.scatterplot(data=df, x="SeedMargin", y="WinMargin")
    plt.show()

print("Correlation Statistic:")
check_seed_win_correlation(pd.concat([Wtourney_DR]), seeds)

print("Men Team Statistic:")
check_seed_win_correlation(pd.concat([Mtourney_DR]), seeds)


import pandas as pd
import matplotlib.pyplot as plt

def plot_location_frequency(df, title_prefix):
    # Group by 'WLoc' and count the occurrences of each location
    location_counts = df['WLoc'].value_counts()

    # Calculate the total frequency
    total_frequency = location_counts.sum()

    # Create the histogram (bar plot)
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    bars = location_counts.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])  # Customize colors

    # Customize the plot
    plt.title(f'{title_prefix}: Frequency of Winning Locations (Total: {total_frequency})')
    plt.xlabel('Game Location')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed

    # Add legend with frequency values
    for bar in bars.patches:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')  # Adjust 5 for vertical offset

    # Show the plot
    plt.tight_layout()
    plt.show()

plot_location_frequency(Wtourney_DR, "Tournament Women")
plot_location_frequency(Mtourney_DR, "Tournament Men")


from xgboost import XGBRegressor, XGBClassifier
# read data
from sklearn.model_selection import train_test_split


def append_team_statistic(df):
    detailed_result_df = df
    
    # Define the features without prefixes
    scoring_features = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
    
    # Create WTeam_score
    WTeam_score = detailed_result_df[['Season', 'WTeamID'] + ['W' + feat for feat in scoring_features]].copy()
    WTeam_score.columns = ['Season', 'TeamID'] + scoring_features
    
    # Create LTeam_score
    LTeam_score = detailed_result_df[['Season', 'LTeamID'] + ['L' + feat for feat in scoring_features]].copy()
    LTeam_score.columns = ['Season', 'TeamID'] + scoring_features
    
    Team_score = pd.concat([WTeam_score, LTeam_score])
    
    agg_stat = ["sum", "mean", "skew", "std", "min", "max"]
    # Create the correct aggregation dictionary
    agg_dict = {
        feature: agg_stat  # Apply aggregations to numerical, 'first' to categorical
        for feature in scoring_features
    }
    
    # Apply aggregations
    Team_score_stat = Team_score.groupby(['Season', 'TeamID']).agg(agg_dict)
    Team_score_stat = Team_score_stat.reset_index()
    Team_score_stat.columns = ["_".join(col) for col in Team_score_stat.columns]

    Team_score_stat.to_csv("Team_score_stat.csv", index=False)

    return Team_score_stat
    
def create_team_stats(df, team_stat_path: str = None):
    # Contains TeamId, Season and also aggregation information
    if team_stat_path == None:
        Team_score_stat = append_team_statistic(df)
    else:
        Team_score_stat = pd.read_csv("/kaggle/working/Team_score_stat.csv")
        
    columns = Team_score_stat.columns
    T1_columns = []
    for col in columns:
        if col == 'Season_':
            T1_columns.append("Season")
        elif col == 'TeamID_':
            T1_columns.append("TeamID1")
        else:
            T1_columns.append("T1" + col) 
        
    T2_columns = []
    for col in columns:
        if col == 'Season_':
            T2_columns.append("Season")
        elif col == 'TeamID_':
            T2_columns.append("TeamID2")
        else:
            T2_columns.append('T2' + col)

    T1Team_score_stat = Team_score_stat.copy()
    T1Team_score_stat.columns = T1_columns
    
    T2Team_score_stat = Team_score_stat.copy()
    T2Team_score_stat.columns = T2_columns
    
    df = df.merge(
        T1Team_score_stat,
        on = ["TeamID1", "Season"],
        how = "left",
    )
    
    df = df.merge(
        T2Team_score_stat,
        on = ["TeamID2", "Season"],
        how = "left",
    )
    return df


def append_stat_to_team(detailed_result_feature_df: pd.DataFrame) -> pd.DataFrame:
    Team_score_stat = pd.read_csv("/kaggle/working/Team_score_stat.csv")
    columns = Team_score_stat.columns
    T1_columns = []
    for col in columns:
        if col == 'Season_':
            T1_columns.append("Season")
        elif col == 'TeamID_':
            T1_columns.append("TeamID1")
        else:
            T1_columns.append("T1" + col) 
        
    T2_columns = []
    for col in columns:
        if col == 'Season_':
            T2_columns.append("Season")
        elif col == 'TeamID_':
            T2_columns.append("TeamID2")
        else:
            T2_columns.append('T2' + col)

    T1Team_score_stat = Team_score_stat.copy()
    T1Team_score_stat.columns = T1_columns
    
    T2Team_score_stat = Team_score_stat.copy()
    T2Team_score_stat.columns = T2_columns
    
    df = df.merge(
        T1Team_score_stat,
        on = ["TeamID1", "Season"],
        how = "left",
    )
    
    df = df.merge(
        T2Team_score_stat,
        on = ["TeamID2", "Season"],
        how = "left",
    )

    return df


# Create Team Stats
detailed_result_feature_df = create_team_stats(detailed_result_df)
col = ['SeedNum1', 'SeedNum2', 'SeedMargin', 'T1FGM_sum', 'T1FGM_mean', 'T1FGM_skew', 'T1FGM_std', 'T1FGM_min', 'T1FGM_max', 'T1FGA_sum', 'T1FGA_mean', 'T1FGA_skew', 'T1FGA_std', 'T1FGA_min', 'T1FGA_max', 'T1FGM3_sum', 'T1FGM3_mean', 'T1FGM3_skew', 'T1FGM3_std', 'T1FGM3_min', 'T1FGM3_max', 'T1FGA3_sum', 'T1FGA3_mean', 'T1FGA3_skew', 'T1FGA3_std', 'T1FGA3_min', 'T1FGA3_max', 'T1FTM_sum', 'T1FTM_mean', 'T1FTM_skew', 'T1FTM_std', 'T1FTM_min', 'T1FTM_max', 'T1FTA_sum', 'T1FTA_mean', 'T1FTA_skew', 'T1FTA_std', 'T1FTA_min', 'T1FTA_max', 'T1OR_sum', 'T1OR_mean', 'T1OR_skew', 'T1OR_std', 'T1OR_min', 'T1OR_max', 'T1DR_sum', 'T1DR_mean', 'T1DR_skew', 'T1DR_std', 'T1DR_min', 'T1DR_max', 'T1Ast_sum', 'T1Ast_mean', 'T1Ast_skew', 'T1Ast_std', 'T1Ast_min', 'T1Ast_max', 'T1TO_sum', 'T1TO_mean', 'T1TO_skew', 'T1TO_std', 'T1TO_min', 'T1TO_max', 'T1Stl_sum', 'T1Stl_mean', 'T1Stl_skew', 'T1Stl_std', 'T1Stl_min', 'T1Stl_max', 'T1Blk_sum', 'T1Blk_mean', 'T1Blk_skew', 'T1Blk_std', 'T1Blk_min', 'T1Blk_max', 'T1PF_sum', 'T1PF_mean', 'T1PF_skew', 'T1PF_std', 'T1PF_min', 'T1PF_max', 'T2FGM_sum', 'T2FGM_mean', 'T2FGM_skew', 'T2FGM_std', 'T2FGM_min', 'T2FGM_max', 'T2FGA_sum', 'T2FGA_mean', 'T2FGA_skew', 'T2FGA_std', 'T2FGA_min', 'T2FGA_max', 'T2FGM3_sum', 'T2FGM3_mean', 'T2FGM3_skew', 'T2FGM3_std', 'T2FGM3_min', 'T2FGM3_max', 'T2FGA3_sum', 'T2FGA3_mean', 'T2FGA3_skew', 'T2FGA3_std', 'T2FGA3_min', 'T2FGA3_max', 'T2FTM_sum', 'T2FTM_mean', 'T2FTM_skew', 'T2FTM_std', 'T2FTM_min', 'T2FTM_max', 'T2FTA_sum', 'T2FTA_mean', 'T2FTA_skew', 'T2FTA_std', 'T2FTA_min', 'T2FTA_max', 'T2OR_sum', 'T2OR_mean', 'T2OR_skew', 'T2OR_std', 'T2OR_min', 'T2OR_max', 'T2DR_sum', 'T2DR_mean', 'T2DR_skew', 'T2DR_std', 'T2DR_min', 'T2DR_max', 'T2Ast_sum', 'T2Ast_mean', 'T2Ast_skew', 'T2Ast_std', 'T2Ast_min', 'T2Ast_max', 'T2TO_sum', 'T2TO_mean', 'T2TO_skew', 'T2TO_std', 'T2TO_min', 'T2TO_max', 'T2Stl_sum', 'T2Stl_mean', 'T2Stl_skew', 'T2Stl_std', 'T2Stl_min', 'T2Stl_max', 'T2Blk_sum', 'T2Blk_mean', 'T2Blk_skew', 'T2Blk_std', 'T2Blk_min', 'T2Blk_max', 'T2PF_sum', 'T2PF_mean', 'T2PF_skew', 'T2PF_std', 'T2PF_min', 'T2PF_max', 'Pred']
detailed_result_feature_df = detailed_result_feature_df[col]
detailed_result_feature_df


original_submission_file = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv")

def prepare_feature(df):
    def split_id(id_str):
        parts = id_str.split("_")
        return int(parts[0]), int(parts[1]), int(parts[2])

    df[["Season", "TeamID1", "TeamID2"]] = df["ID"].apply(split_id).tolist()
    df = create_team_stats(df, "/kaggle/working/Team_score_stat.csv")
    display(df)

    # can use imputer, but for now use 0 first
    df.fillna(0, inplace=True)
    
    return df

sub = prepare_feature(original_submission_file)


seed_1 = seeds.rename(columns={
    "TeamID": "TeamID1",
    "Seed": "SeedNum1",
})
seed_2 = seeds.rename(columns={
    "TeamID": "TeamID2",
    "Seed": "SeedNum2",
})

sub = sub.merge(
    seed_1,
    on=["Season", "TeamID1"],
    how="left"
)

sub = sub.merge(
    seed_2,
    on=["Season", "TeamID2"],
    how="left"
)

sub.fillna(0)

sub["SeedNum1"] = sub["SeedNum1"].apply(lambda x: str(x)[1:3] if isinstance(x, str) else 0).astype(int)
sub["SeedNum2"] = sub["SeedNum2"].apply(lambda x: str(x)[1:3] if isinstance(x, str) else 0).astype(int)

sub


sub = calculate_seed_margin(sub)
sub = sub[col]


features_to_use = col[:-1]
features_to_use


df = detailed_result_feature_df # The dataframe use to train
sub = sub[features_to_use] # The dataframe that we need to submit

y = df["Pred"]
X = df[features_to_use]
X.info()


X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 88)


model = XGBClassifier(n_estimator=5, seed=168, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
pred = model.predict_proba(X_test)[:, 1]

print(f'Log Loss: {log_loss(y_test, pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(y_test, pred)}')
print(f'Brier Score: {brier_score_loss(y_test, pred)}')

pd.DataFrame(pred).describe()


sub["Pred"] = model.predict_proba(sub)[:, 1]


sub["ID"] = original_submission_file["ID"]
sub[['ID', 'Pred']].to_csv('submission.csv', index=False)




