# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
DATA_PATH = "../input/march-machine-learning-mania-2025"

#load Men's Data
teams = pd.read_csv(f"{DATA_PATH}/MTeams.csv")
seasons = pd.read_csv(f"{DATA_PATH}/MSeasons.csv")
seeds = pd.read_csv(f"{DATA_PATH}/MNCAATourneySeeds.csv")
regular_results = pd.read_csv(f"{DATA_PATH}/MRegularSeasonCompactResults.csv")
tourney_results = pd.read_csv(f"{DATA_PATH}/MNCAATourneyCompactResults.csv")
detailed_results = pd.read_csv(f"{DATA_PATH}/MRegularSeasonDetailedResults.csv")
massey_ordinals = pd.read_csv(f"{DATA_PATH}/MMasseyOrdinals.csv")

# Load Women’s Data
w_teams = pd.read_csv(f"{DATA_PATH}/WTeams.csv")
w_seasons = pd.read_csv(f"{DATA_PATH}/WSeasons.csv")
w_seeds = pd.read_csv(f"{DATA_PATH}/WNCAATourneySeeds.csv")
w_regular_results = pd.read_csv(f"{DATA_PATH}/WRegularSeasonCompactResults.csv")
w_tourney_results = pd.read_csv(f"{DATA_PATH}/WNCAATourneyCompactResults.csv")
w_detailed_results = pd.read_csv(f"{DATA_PATH}/WRegularSeasonDetailedResults.csv")



display(teams.head())


# Number of rows and columns
print("Teams Shape:", teams.shape)
print("Seasons Shape:", seasons.shape)
print("Seeds Shape:", seeds.shape)
print("Regular Results Shape:", regular_results.shape)
print("Tourney Results Shape:", tourney_results.shape)
print("Detailed Results Shape:", detailed_results.shape)
print("Massey Ordinals Shape:", massey_ordinals.shape)

# General Info
teams.info()
seasons.info()
seeds.info()
regular_results.info()
tourney_results.info()
detailed_results.info()
massey_ordinals.info()



print(teams.describe())
print(seasons.describe())
print(seeds.describe())
print(regular_results.describe())
print(tourney_results.describe())
print(detailed_results.describe())
print(massey_ordinals.describe())


# Merge Teams and Seasons
team_season = pd.merge(teams, seasons, how='inner', left_on='FirstD1Season', right_on='Season')
w_team_season = pd.merge(w_teams, w_seasons, how='inner', left_on='TeamID', right_on='Season')


# Merge Regular Season Results with Seeds
regular_season = pd.merge(regular_results, seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
w_regular_season = pd.merge(w_regular_results, w_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')


# Merge Tournament Data with Team Info
tourney_data = pd.merge(tourney_results, seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
w_tourney_data = pd.merge(w_tourney_results, w_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')


# Mens ranking data. Women's ranking data is not available.
# GOAL : Enrich my regular season data 
rankings = massey_ordinals.groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index()  # Calculate average season rank based on their ordinal rank
combined_data = pd.merge(regular_season, rankings, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')


#final_data = combined_data[['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore', 'Seed', 'OrdinalRank']]
#w_final_data = w_regular_season[['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore', 'Seed']]


# Include WLoc in the final data after merging
final_data = pd.merge(regular_results[['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc']],
                      seeds[['Season', 'TeamID', 'Seed']],
                      left_on=['Season', 'WTeamID'],
                      right_on=['Season', 'TeamID'],
                      how='left')

w_final_data = pd.merge(w_regular_results[['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc']],
                        w_seeds[['Season', 'TeamID', 'Seed']],
                        left_on=['Season', 'WTeamID'],
                        right_on=['Season', 'TeamID'],
                        how='left')


# # Men’s teams
# print(teams.head())
# print(teams.describe(include='all'))
# print(teams.nunique())

# # Women’s teams
# print(w_teams.head())
# print(w_teams.describe(include='all'))
# print(w_teams.nunique())


# Check missing values
print("Missing Values (Men's Data):")
print(final_data.isnull().sum())

print("Missing Values (Women's Data):")
print(w_final_data.isnull().sum())



# Check for duplicates
print("Duplicate Rows in Men's Data:", final_data.duplicated().sum())
print("Duplicate Rows in Women's Data:", w_final_data.duplicated().sum())


print(seeds['Seed'].value_counts())
print(w_seeds['Seed'].value_counts())



print(regular_results['WLoc'].value_counts())
print(w_regular_results['WLoc'].value_counts())



plt.figure(figsize=(12, 6))
sns.histplot(final_data['WScore'], bins=30, kde=True, color='blue', label='Men')
sns.histplot(w_final_data['WScore'], bins=30, kde=True, color='red', label='Women')
plt.title('Distribution of Winning Scores')
plt.legend()
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(data=[final_data['WScore'], w_final_data['WScore']], palette='coolwarm')
plt.xticks([0, 1], ['Men', 'Women'])
plt.title('Boxplot of Winning Scores (Men vs Women)')
plt.show()


avg_men_score = final_data.groupby('Season')['WScore'].mean()
avg_women_score = w_final_data.groupby('Season')['WScore'].mean()

plt.figure(figsize=(12, 6))
sns.lineplot(x=avg_men_score.index, y=avg_men_score.values, label='Men', color='blue')
sns.lineplot(x=avg_women_score.index, y=avg_women_score.values, label='Women', color='red')
plt.title('Average Scores Per Season (Men vs Women)')
plt.show()


# Fixing a strat: Keeping quartiles from 0.25 and 0.75 and removing outliers for both men and women
Q1 = final_data['WScore'].quantile(0.25)
Q3 = final_data['WScore'].quantile(0.75)
IQR = Q3 - Q1

outliers_men = final_data[(final_data['WScore'] < Q1 - 1.5 * IQR) | 
                          (final_data['WScore'] > Q3 + 1.5 * IQR)]
print(f"Outliers in Men's WScore: {outliers_men.shape[0]}")

Q1_w = w_final_data['WScore'].quantile(0.25)
Q3_w = w_final_data['WScore'].quantile(0.75)
IQR_w = Q3_w - Q1_w

outliers_women = w_final_data[(w_final_data['WScore'] < Q1_w - 1.5 * IQR_w) | 
                              (w_final_data['WScore'] > Q3_w + 1.5 * IQR_w)]
print(f"Outliers in Women's WScore: {outliers_women.shape[0]}")


final_data['Score_Diff'] = final_data['WScore'] - final_data['LScore']
w_final_data['Score_Diff'] = w_final_data['WScore'] - w_final_data['LScore']

final_data['Score_Diff'].mean() # 12
w_final_data['Score_Diff'].mean() # 14



win_rate = final_data['WTeamID'].value_counts() / (final_data['WTeamID'].value_counts() + final_data['LTeamID'].value_counts())
final_data['WinRate'] = final_data['WTeamID'].map(win_rate)

win_rate_w = w_final_data['WTeamID'].value_counts() / (w_final_data['WTeamID'].value_counts() + w_final_data['LTeamID'].value_counts())
w_final_data['WinRate'] = w_final_data['WTeamID'].map(win_rate_w)



#3101 is a 16th team in the seed list and having a win rate of 58 is explanatory here
# Filter teams with win rate > 70%
high_win_rate_teams = win_rate_w[win_rate_w > 0.70]
print("Teams with win rate > 70%:")
print(high_win_rate_teams)
# Interpretation: Manually checked 3124 seeds and it is 
# almost placed at 1st or 2nd position. Also a very old team. So seed can be kept in the features and can be given weight.


sns.countplot(data=regular_results, x='WLoc')
plt.title('Winning Location for Men')
plt.show()

sns.countplot(data=w_regular_results, x='WLoc')
plt.title('Winning Location for Women')
plt.show()
# High chances that the team playing at home is more likely to win.


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# For Men's Tournament
# Calculate average win rate per seed
seed_performance_men = pd.DataFrame()
for seed in seeds['Seed'].unique():
    seed_teams = seeds[seeds['Seed'] == seed]['TeamID']
    # Get win rates for teams with this seed
    seed_win_rates = win_rate[win_rate.index.isin(seed_teams)]
    seed_performance_men = pd.concat([seed_performance_men, 
                                    pd.DataFrame({'Seed': seed,
                                                'Avg_Win_Rate': seed_win_rates.mean(),
                                                'Std_Win_Rate': seed_win_rates.std()},
                                               index=[0])])

# Plot for Men
plt.figure(figsize=(15, 7))
sns.scatterplot(data=seed_performance_men, x='Seed', y='Avg_Win_Rate')
plt.fill_between(range(len(seed_performance_men)),
                seed_performance_men['Avg_Win_Rate'] - seed_performance_men['Std_Win_Rate'],
                seed_performance_men['Avg_Win_Rate'] + seed_performance_men['Std_Win_Rate'],
                alpha=0.2)
plt.title("Average Win Rate by Seed Position (Men's Tournament)")
plt.xlabel('Seed')
plt.ylabel('Average Win Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# For Women's Tournament
# Calculate average win rate per seed
seed_performance_women = pd.DataFrame()
for seed in w_seeds['Seed'].unique():
    seed_teams = w_seeds[w_seeds['Seed'] == seed]['TeamID']
    # Get win rates for teams with this seed
    seed_win_rates = win_rate_w[win_rate_w.index.isin(seed_teams)]
    seed_performance_women = pd.concat([seed_performance_women,
                                      pd.DataFrame({'Seed': seed,
                                                  'Avg_Win_Rate': seed_win_rates.mean(),
                                                  'Std_Win_Rate': seed_win_rates.std()},
                                                 index=[0])])

# Plot for Women
plt.figure(figsize=(15, 7))
sns.scatterplot(data=seed_performance_women, x='Seed', y='Avg_Win_Rate')
plt.fill_between(range(len(seed_performance_women)),
                seed_performance_women['Avg_Win_Rate'] - seed_performance_women['Std_Win_Rate'],
                seed_performance_women['Avg_Win_Rate'] + seed_performance_women['Std_Win_Rate'],
                alpha=0.2)
plt.title("Average Win Rate by Seed Position (Women's Tournament)")
plt.xlabel('Seed')
plt.ylabel('Average Win Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Outlier capping
cap_men = final_data['WScore'].quantile(0.99)
cap_women = w_final_data['WScore'].quantile(0.99)

# Apply capping
final_data['WScore'] = final_data['WScore'].clip(upper=cap_men)
w_final_data['WScore'] = w_final_data['WScore'].clip(upper=cap_women)

# Confirm the effect
print(final_data['WScore'].describe())
print(w_final_data['WScore'].describe())


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

# === Read Data ===
def read_data(data_path):
    teams = pd.read_csv(f'{data_path}/MTeams.csv')
    seasons = pd.read_csv(f'{data_path}/MSeasons.csv')
    seeds = pd.read_csv(f'{data_path}/MNCAATourneySeeds.csv')
    regular_results = pd.read_csv(f'{data_path}/MRegularSeasonCompactResults.csv')
    massey_ordinals = pd.read_csv(f'{data_path}/MMasseyOrdinals.csv')

    w_teams = pd.read_csv(f'{data_path}/WTeams.csv')
    w_seasons = pd.read_csv(f'{data_path}/WSeasons.csv')
    w_seeds = pd.read_csv(f'{data_path}/WNCAATourneySeeds.csv')
    w_regular_results = pd.read_csv(f'{data_path}/WRegularSeasonCompactResults.csv')

    return (teams, seasons, seeds, regular_results, massey_ordinals,
            w_teams, w_seasons, w_seeds, w_regular_results)


# === Elo Rating Function ===
K = 20
INITIAL_RATING = 1500

def update_elo(winner, loser, ratings):
    rating_winner = ratings.get(winner, INITIAL_RATING) # Try and retrieve winners rating, if not found just pass in the initial rating
    rating_loser = ratings.get(loser, INITIAL_RATING)

    expected_winner = 1 / (1 + 10 ** ((rating_loser - rating_winner) / 400))
    expected_loser = 1 - expected_winner

    ratings[winner] = rating_winner + K * (1 - expected_winner) # Updates winners rating based on formula
    ratings[loser] = rating_loser + K * (0 - expected_loser) # Similary fer loser

# === Function to Generate Elo Ratings from Game Data ===
def generate_elo(data):
    team_ratings = {}
    for index, row in data.iterrows():
        winner, loser = row['WTeamID'], row['LTeamID']
        update_elo(winner, loser, team_ratings)

# Final dataframe: We have converted the above pandas series to a dataframe
## Make team ID as index
    elo_df = pd.DataFrame.from_dict(team_ratings, orient='index', columns=['EloRating'])
    # Data ready. Scaling for faster model interpretability for eventual training on a tree based algorithm.
    elo_df['OrdinalRank'] = (elo_df['EloRating'] - elo_df['EloRating'].min()) / (elo_df['EloRating'].max() - elo_df['EloRating'].min())
    # This was set to an index. Converting it to a normal column now.
    elo_df.reset_index(inplace=True)
    elo_df.columns = ['TeamID', 'EloRating', 'OrdinalRank']

    return elo_df


# === Feature Transformation ===
def feature_transformation(combined):
    # === 1. Rolling Win Margin ===
    # We do this to capture recent performance trends based on no. of wins by a team
    combined['WinMargin'] = combined['WScore'] - combined['LScore']

    # Rolling average at TeamID level (5-game window)
    #For each group's WinMargin series (x), it applies a .rolling() window of size 5 and calculates the .mean()
    combined['RollingWinMargin'] = combined.groupby(['Season', 'TeamID'])['WinMargin'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # Clipping extreme win margins (5th to 95th percentile)
    combined['WinMargin'] = combined['WinMargin'].clip(
        lower=combined['WinMargin'].quantile(0.05),
        upper=combined['WinMargin'].quantile(0.95)
    )

    # === 2. Cyclic Encoding for SeasonPhase ===
    if 'SeasonPhase' in combined.columns:
        combined['SeasonPhase_Sin'] = np.sin(2 * np.pi * combined['SeasonPhase'].astype('category').cat.codes / 4)
        combined['SeasonPhase_Cos'] = np.cos(2 * np.pi * combined['SeasonPhase'].astype('category').cat.codes / 4)

    # === 3. Interaction Terms ===
    combined['SeedNum_x_HomeAdvantage'] = combined['SeedNum'] * combined['HomeAdvantage']
    combined['WScore_x_LScore'] = combined['WScore'] * combined['LScore']

    #  Reducing influcence of extreme scores which might cause model to hallucinate
    # Interpretation: Adds potentially non-linear relationships to the model.
    # Seed might matter more/less at home and high scores might interact differently than low scores.
    combined['WScore_x_LScore'] = combined['WScore_x_LScore'].clip(
        upper=combined['WScore_x_LScore'].quantile(0.99)
    )

    # === 4. Defense/Offense Balance ===
    # Did not add epsilon (1e-6) as there was no DIVISIONBYZEROERROR
    combined['ScoreRatio'] = combined['WScore'] / combined['LScore']

    #  5. Seed Difference  - Doing this to quantify difference between teams seeds and actual seeds.
    # Some seeds seeds have NaN values seen above. Hence imputing with 0 first before we calc mean
    combined['SeedDiff'] = combined['SeedNum'].fillna(0) - combined.groupby('Season')['SeedNum'].transform('mean')

    # === 6. Mean-Based Imputation for RankDiff ===
    # Doing this to basically a simple way to handle missing rank differences, preserving some season-specific context.
    # Imputing with 0 causes abrupt predictions leading to almost similar percentages in all preds
    combined['RankDiff'] = combined.groupby('Season')['RankDiff'].transform(lambda x: x.fillna(x.mean()))

    # === 7. Standard Scaling for Scores ===
    scaler_standard = StandardScaler()
    combined[['WScore', 'LScore', 'WinMargin']] = scaler_standard.fit_transform(combined[['WScore', 'LScore', 'WinMargin']])

    return combined


# === Scaling and Fixes ===
def scale_combined_ranks(combined):
    pt = PowerTransformer(method='yeo-johnson')

    # Handle skewness for men's data only
    men_data = combined.loc[combined['IsWomen'] == 0, 'OrdinalRank'].fillna(0).values.reshape(-1, 1)
    combined.loc[combined['IsWomen'] == 0, 'OrdinalRank'] = pt.fit_transform(men_data)

    # MinMax scale for both men's and women's data
    scaler = MinMaxScaler()
    combined['OrdinalRank'] = scaler.fit_transform(combined[['OrdinalRank']])

    return combined

def handle_seed_and_rank(combined):
    combined['SeedNum'].fillna(-1, inplace=True)

    combined['IsSeeded'] = (combined['SeedNum'] != -1).astype(int)

    combined['RankDiff'] = combined['OrdinalRank'] - combined['SeedNum']

    return combined

def clean_combined_data(combined):
    combined = scale_combined_ranks(combined)
    combined = handle_seed_and_rank(combined)
    combined['TeamID'].fillna(combined['WTeamID'], inplace=True)
    combined = feature_transformation(combined)
    return combined


# === Generate Combined Dataset ===
def generate_combined_dataset(data_path):
    (teams, seasons, seeds, regular_results, massey_ordinals,
     w_teams, w_seasons, w_seeds, w_regular_results) = read_data(data_path)

    # Men's data processing
    men = pd.merge(regular_results, seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
    men_elo = generate_elo(regular_results)

    men = pd.merge(men, men_elo, left_on='WTeamID', right_on='TeamID', how='left')
    men['IsWomen'] = 0

    # Women's data processing
    women = pd.merge(w_regular_results, w_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
    women['IsWomen'] = 1

    # Combine Data
    combined = pd.concat([men, women], ignore_index=True)

    #  Create HomeAdvantage from WLoc
    combined['HomeAdvantage'] = combined['WLoc'].map({'H': 1, 'A': -1, 'N': 0}).fillna(0)

    #  Extract numeric part from 'Seed' to create 'SeedNum'
    combined['SeedNum'] = combined['Seed'].str.extract(r'(\d+)').astype(float).fillna(-1)

    combined['OrdinalRank'].fillna(combined['OrdinalRank'].median(), inplace=True)


    #  Debugging Output
    print(f"SeedNum value counts:\n{combined['SeedNum'].value_counts()}")

    # Clean combined data
    combined = clean_combined_data(combined)

    return combined
#
import warnings
warnings.filterwarnings('ignore')

# === RUN CODE ===
data_path = "../input/march-machine-learning-mania-2025"
combined = generate_combined_dataset(data_path)

# Show output to verify
print(combined.head())
print(combined.describe())


important_features = [
    'Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT',
    'EloRating', 'OrdinalRank', 'IsWomen', 'TeamID', 'HomeAdvantage',
    'SeedNum', 'IsSeeded', 'RankDiff', 'RollingWinMargin', 'SeedNum_x_HomeAdvantage'
]

# Create the filtered dataset
filtered_combined = combined[important_features]

# Check the new dataset shape and head
print(filtered_combined.shape)
print(filtered_combined.head())



# === CHECK 1: Data Types ===
print("\n[1] Data Types:")
print(combined.dtypes)

# === CHECK 2: Missing Values ===
print("\n[2] Missing Values:")
print(combined.isnull().sum())

# === CHECK 3: Cardinality (Unique Value Count) ===
print("\n[3] Cardinality:")
cardinality = combined.nunique()
print(cardinality)

# === CHECK 4: Summary Statistics ===
print("\n[4] Summary Statistics:")
print(combined.describe())

# === CHECK 5: Unique Values ===
print("\n[5] Unique Value Check:")
for column in combined.columns:
    unique_values = combined[column].unique()
    print(f"{column}: {len(unique_values)} unique values, {unique_values[:5]}")



# Selecting the final set of important features from the combined dataset

selected_features = [ 'TeamID',
    'WScore', 'LScore', 'ScoreRatio', 'NumOT', 'HomeAdvantage', 
    'SeedNum', 'OrdinalRank', 'EloRating', 'DayNum', 'WTeamID', 
    'LTeamID', 'Season', 'RollingWinMargin','SeedNum_x_HomeAdvantage'
]

# Create a new dataset with only the selected features
final_dataset = combined[selected_features]

# Compute correlation matrix for the selected features
correlation_matrix = final_dataset.corr()

# Display correlation matrix as a heatmap to visualize
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Matrix (Selected Features)")
plt.show()

# Identify highly correlated features (threshold > 0.8)
threshold = 0.8
highly_correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated_features.add(colname)

highly_correlated_features



import xgboost as xgb
from sklearn.metrics import brier_score_loss
import numpy as np


X = combined[['RollingWinMargin', 'NumOT', 'SeedNum_x_HomeAdvantage',
                    'OrdinalRank', 'EloRating', 'WTeamID', 'LTeamID', 'Season',
                   'DayNum']]


# Define target variable
y = (combined['WScore'] > combined['LScore']).astype(int)

# Define DMatrix for XGBoost
dtrain = xgb.DMatrix(X, label=y)

# Define parameter grid
param_grid = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'gamma': 0.1,
    'reg_lambda': 1,       # L2 regularization
    'reg_alpha': 1,        # L1 regularization
    'objective': 'binary:logistic'
}

# Cross-validation using xgb.cv
cv_results = xgb.cv(
    params=param_grid,
    dtrain=dtrain,
    num_boost_round=100,
    nfold=3,
    metrics="logloss",         # Use logloss instead of brier directly
    early_stopping_rounds=10,
    seed=42
)

# Best score and best round
best_iteration = cv_results['test-logloss-mean'].idxmin()
best_score = cv_results['test-logloss-mean'].min()

print(f"Best iteration: {best_iteration}")
print(f"Best score (logloss): {best_score}")

# Fit the best model
best_model = xgb.train(
    params=param_grid,
    dtrain=dtrain,
    num_boost_round=best_iteration
)

# Feature importance
importance = best_model.get_score(importance_type='weight')

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=list(importance.values()), y=list(importance.keys()))
plt.title('Feature Importance (Without ScoreRatio)')
plt.show()



from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# ==== Prepare Data ====
X = combined[['RollingWinMargin', 'NumOT', 'OrdinalRank', 'WTeamID', 'LTeamID', 'Season', 'DayNum', 'SeedNum_x_HomeAdvantage']]
y = (combined['WScore'] > combined['LScore']).astype(int)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== Define Tuned Models ====
xgb = XGBClassifier(
    colsample_bytree=0.8,
    gamma=0.1,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    reg_alpha=0.1,
    reg_lambda=1,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)

rf = RandomForestClassifier(
    max_depth=15,
    max_features='log2',
    min_samples_leaf=3,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)

lgb = LGBMClassifier(
    colsample_bytree=0.8,
    learning_rate=0.1,
    max_depth=10,
    num_leaves=31,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    random_state=42
)

# ==== Stacking Classifier ====
stacked_clf = StackingClassifier(
    estimators=[
        ('xgb', xgb),
        ('rf', rf),
        ('lgb', lgb)
    ],
    final_estimator=LogisticRegression(max_iter=500),
    passthrough=True,
    n_jobs=-1
)

# ==== Fit Stacking Model ====
stacked_clf.fit(X_train, y_train)

# ==== Predict and Evaluate ====
y_pred = stacked_clf.predict(X_test)
y_proba = stacked_clf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Brier Score:", brier_score_loss(y_test, y_proba))
print("Log Loss:", log_loss(y_test, y_proba))


import pandas as pd
import numpy as np

# === Read Submission File ===
submission_file = "../input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv"
sample_submission = pd.read_csv(submission_file)

# === Split ID into Season, Team1, and Team2 ===
sample_submission[['Season', 'Team1', 'Team2']] = sample_submission['ID'].str.split('_', expand=True).astype(int)

# === Merge using Team1 ===
X_kaggle_1 = sample_submission.merge(
    combined,
    left_on=['Season', 'Team1'],
    right_on=['Season', 'WTeamID'],
    how='left'
)

# === Merge using Team2 ===
X_kaggle_2 = sample_submission.merge(
    combined,
    left_on=['Season', 'Team2'],
    right_on=['Season', 'LTeamID'],
    how='left'
)

# === Combine both matches ===
X_kaggle = pd.concat([X_kaggle_1, X_kaggle_2]).drop_duplicates(subset=['Season', 'Team1', 'Team2']).reset_index(drop=True)



# === Handle Missing Values ===
for col in ['RollingWinMargin', 'NumOT', 'OrdinalRank', 'SeedNum_x_HomeAdvantage']:
    if col in X_kaggle.columns:
        X_kaggle[col] = X_kaggle[col].fillna(combined.groupby('WTeamID')[col].transform('mean'))



# Fallback for remaining missing values
X_kaggle.fillna(-1, inplace=True)



# Drop existing LTeamID if it already exists
if 'LTeamID' in X_kaggle.columns:
    X_kaggle.drop('LTeamID', axis=1, inplace=True)

# Rename Team2 to LTeamID
X_kaggle = X_kaggle.rename(columns={'Team2': 'LTeamID'})




# === Predict probabilities ===
features = ['RollingWinMargin', 'NumOT', 'OrdinalRank', 'WTeamID', 'LTeamID', 'Season', 'DayNum', 'SeedNum_x_HomeAdvantage']



X_kaggle[features]


 pred = stacked_clf.predict_proba(X_kaggle[features])[:, 1]


pred


# Ensure length consistency
sample_submission['Pred'] = pd.to_numeric(pred[:len(sample_submission)], errors='coerce')


# === Show Results ===
print(sample_submission[['ID', 'Pred']][sample_submission['Pred'] != 0.0].head(10))


# === Store Submission File ===
output_file = "the_best_prediction_ever_made.csv"
sample_submission[['ID', 'Pred']].to_csv(output_file, index=False)

print(f"Submission file saved to: {output_file}")


