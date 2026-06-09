import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


teams = pd.read_csv("MTeams.csv")
seasons = pd.read_csv("MSeasons.csv")
seeds = pd.read_csv("MNCAATourneySeeds.csv")
regular_results = pd.read_csv("MRegularSeasonCompactResults.csv")
tourney_results = pd.read_csv("MNCAATourneyCompactResults.csv")
detailed_results = pd.read_csv("MRegularSeasonDetailedResults.csv")
massey_ordinals = pd.read_csv("MMasseyOrdinals.csv")


print(seeds)


tourney_results.head()


detailed_results.shape


teams.head()


teams.shape


seasons.head()


seasons.tail()


seasons.shape


plt.figure(figsize=(12, 6))
sns.countplot(x="Season", data=seasons, palette="coolwarm")
plt.xlabel("Season")
plt.ylabel("Frequency")
plt.title("Count of Records per Season")
plt.xticks(rotation=90)
plt.show()


seeds.shape


seeds.head()


regular_results['Margin'] = regular_results['WScore'] - regular_results['LScore']

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(regular_results['WScore'], kde=True, color='green')
plt.title("Winning Scores Distribution")
plt.xlabel("Winning Score")

plt.subplot(1, 3, 2)
sns.histplot(regular_results['LScore'], kde=True, color='red')
plt.title("Losing Scores Distribution")
plt.xlabel("Losing Score")

plt.subplot(1, 3, 3)
sns.histplot(regular_results['Margin'], kde=True, color='purple')
plt.title("Margin of Victory Distribution")
plt.xlabel("Margin (WScore - LScore)")
plt.tight_layout()
plt.show()
tourney_results['Margin'] = tourney_results['WScore'] - tourney_results['LScore']

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(tourney_results['WScore'], kde=True, color='blue')
plt.title("Tournament Winning Scores Distribution")
plt.xlabel("Winning Score")

plt.subplot(1, 3, 2)
sns.histplot(tourney_results['LScore'], kde=True, color='orange')
plt.title("Tournament Losing Scores Distribution")
plt.xlabel("Losing Score")

plt.subplot(1, 3, 3)
sns.histplot(tourney_results['Margin'], kde=True, color='brown')
plt.title("Tournament Margin of Victory Distribution")
plt.xlabel("Margin")
plt.tight_layout()
plt.show()


sns.countplot(data=regular_results, x='WLoc', palette='magma')
plt.title("Distribution of Winning Locations (Regular Season)")
plt.xlabel("Winning Location (H = Home, A = Away, N = Neutral)")
plt.show()


tourney_results.head()


tourney_results.shape


tourney_results.describe(include='all')


plt.figure(figsize=(10, 6))
sns.scatterplot(x=tourney_results['LScore'], y=tourney_results['WScore'], alpha=0.5)
plt.title("Scatter Plot: Winning Score vs Losing Score (Tournament)")
plt.xlabel("Losing Score")
plt.ylabel("Winning Score")
plt.show()



avg_wscore_by_season = tourney_results.groupby('Season')['WScore'].mean()
plt.figure(figsize=(12, 6))
plt.plot(avg_wscore_by_season.index, avg_wscore_by_season.values, marker='o', linestyle='-', color='purple')
plt.title("Trend of Average Tournament Winning Score Over Seasons")
plt.xlabel("Season")
plt.ylabel("Average Winning Score")
plt.grid(True)
plt.show()


detailed_results.head()


detailed_results.columns


detailed_results['Possession_Control'] = (
    detailed_results['WOR'] +
    detailed_results['WDR'] +
    detailed_results['WStl'] -
    detailed_results['WTO']
)



print("=== Possession_Control Descriptive Statistics ===")
print(detailed_results['Possession_Control'].describe())


# Histogram
plt.figure(figsize=(8,6))
sns.histplot(detailed_results['Possession_Control'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Possession_Control")
plt.xlabel("Possession_Control")
plt.ylabel("Frequency")
plt.show()


# Boxplot
plt.figure(figsize=(6,6))
sns.boxplot(y=detailed_results['Possession_Control'], color='lightgreen')
plt.title("Boxplot of Possession_Control")
plt.ylabel("Possession_Control")
plt.show()



detailed_results['Shooting_Efficiency'] = (
    detailed_results['WFGM'] +
    0.5 * detailed_results['WFGM3'] +
    0.5 * detailed_results['WFTM']
) / (detailed_results['WFGA'] + 0.5 * detailed_results['WFTA'])



print("=== Shooting_Efficiency Descriptive Statistics ===")
print(detailed_results['Shooting_Efficiency'].describe())


plt.figure(figsize=(10, 6))
sns.histplot(detailed_results['Shooting_Efficiency'], bins=30, kde=True, color='dodgerblue')
plt.title("Distribution of Shooting Efficiency")
plt.xlabel("Shooting Efficiency")
plt.ylabel("Frequency")
plt.show()



# ----------------------------
# 2. Boxplot to Spot Outliers
# ----------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(y=detailed_results['Shooting_Efficiency'], color='lightcoral')
plt.title("Boxplot of Shooting Efficiency")
plt.ylabel("Shooting Efficiency")
plt.show()


if 'Season' in detailed_results.columns:
    avg_shoot_eff_by_season = detailed_results.groupby('Season')['Shooting_Efficiency'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(avg_shoot_eff_by_season.index, avg_shoot_eff_by_season.values, marker='o', linestyle='-', color='purple')
    plt.title("Average Shooting Efficiency Over Seasons")
    plt.xlabel("Season")
    plt.ylabel("Average Shooting Efficiency")
    plt.grid(True)
    plt.show()


detailed_results['Shooting_Efficiency'] = (
    detailed_results['WFGM'] +
    0.5 * detailed_results['WFGM3'] +
    0.5 * detailed_results['WFTM']
) / (detailed_results['WFGA'] + 0.5 * detailed_results['WFTA'])


# Calculate Possession_Control
detailed_results['Possession_Control'] = (
    detailed_results['WOR'] + detailed_results['WDR'] + detailed_results['WStl']
) - detailed_results['WTO']




# Calculate Assist Ratio 
detailed_results['Assist_Ratio'] = detailed_results['WAst'] / (detailed_results['WFGM'] + detailed_results['WFTM'] + epsilon)


# Calculate Balanced_Score
detailed_results['Balanced_Score'] = (
    0.4 * detailed_results['Shooting_Efficiency'] +
    0.4 * detailed_results['Possession_Control'] +
    0.2 * detailed_results['Assist_Ratio']
)


print("=== Balanced_Score Descriptive Statistics ===")
print(detailed_results['Balanced_Score'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(detailed_results['Balanced_Score'], bins=30, kde=True, color='mediumpurple')
plt.title("Distribution of Balanced_Score")
plt.xlabel("Balanced_Score")
plt.ylabel("Frequency")
plt.show()


# ----------------------------
# 2. Boxplot for Balanced_Score
# ----------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(y=detailed_results['Balanced_Score'], color='lightblue')
plt.title("Boxplot of Balanced_Score")
plt.ylabel("Balanced_Score")
plt.show()


if 'Season' in detailed_results.columns:
    avg_balanced_by_season = detailed_results.groupby('Season')['Balanced_Score'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(avg_balanced_by_season.index, avg_balanced_by_season.values, marker='o', linestyle='-', color='teal')
    plt.title("Average Balanced_Score Over Seasons")
    plt.xlabel("Season")
    plt.ylabel("Average Balanced_Score")
    plt.grid(True)
    plt.show()


metrics = ["WScore", "LScore"]
plt.figure(figsize=(16, 10))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 4, i)
    sns.histplot(data=detailed_results, x=metric, kde=True)
    plt.title(f"Distribution of {metric}")
plt.tight_layout()
plt.show()


detailed_results["WFGPerc"] = detailed_results["WFGM"] / detailed_results["WFGA"]
detailed_results["LFGPerc"] = detailed_results["LFGM"] / detailed_results["LFGA"]
sns.histplot(detailed_results["WFGPerc"], color="green", label="Winning FG%", kde=True)
sns.histplot(detailed_results["LFGPerc"], color="red", label="Losing FG%", kde=True)
plt.title("Distribution of FG% (Winning vs. Losing Teams)")
plt.legend()
plt.show()


detailed_results["Margin"] = detailed_results["WScore"] - detailed_results["LScore"]
plt.figure(figsize=(6, 4))
sns.histplot(detailed_results["Margin"], kde=True, color="purple")
plt.title("Distribution of Margin of Victory (Regular Season)")
plt.xlabel("Margin (WScore - LScore)")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=tourney_results['Margin'], color='green')
plt.title("Boxplot of Tournament Score Margin")
plt.xlabel("Margin (Difference between WScore and LScore)")
plt.show()



sns.boxplot(data=regular_results, x='WLoc', y='WScore', palette='Set2')
plt.title("Winning Scores by Location (Regular Season)")
plt.xlabel("Winning Location")
plt.ylabel("Winning Score")
plt.show()


def feature_engineering(regular_results, detailed_results, massey_ordinals):

    team_stats = regular_results.groupby('WTeamID').agg({'WScore': ['mean', 'count']})
    team_stats.columns = ['AvgPointsScored', 'GamesWon']
    team_stats['AvgPointsAllowed'] = regular_results.groupby('LTeamID')['LScore'].mean()
    team_stats['GamesLost'] = regular_results.groupby('LTeamID')['LScore'].count()
    team_stats['TotalGames'] = team_stats['GamesWon'] + team_stats['GamesLost']
    team_stats['WinRatio'] = team_stats['GamesWon'] / team_stats['TotalGames']


    detailed_results['Shooting_Efficiency'] = (
        detailed_results['WFGM'] +
        0.5 * detailed_results['WFGM3'] +
        0.5 * detailed_results['WFTM']
    ) / (detailed_results['WFGA'] + 0.5 * detailed_results['WFTA'])

    detailed_results['Possession_Control'] = (
        detailed_results['WOR'] +
        detailed_results['WDR'] +
        detailed_results['WStl']
    ) - detailed_results['WTO']

    detailed_results['Balanced_Score'] = (
        0.4 * detailed_results['Shooting_Efficiency'] +
        0.4 * detailed_results['Possession_Control'] +
        0.2 * (detailed_results['WAst'] / (detailed_results['WFGM'] + detailed_results['WFTM']))
    )


    advanced_metrics = detailed_results.groupby('WTeamID').agg({
        'Shooting_Efficiency': 'mean',
        'Possession_Control': 'mean',
        'Balanced_Score': 'mean'
    }).rename(columns={
        'Shooting_Efficiency': 'AvgShootingEff',
        'Possession_Control': 'AvgPossessionControl',
        'Balanced_Score': 'AvgBalancedScore'
    })


    team_stats = team_stats.merge(advanced_metrics, left_index=True, right_index=True, how='left')


    latest_rankings = massey_ordinals[massey_ordinals['RankingDayNum'] == 133]
    team_stats = team_stats.merge(latest_rankings[['TeamID', 'OrdinalRank']], left_index=True, right_on='TeamID', how='left')

    return team_stats



print(massey_ordinals.columns)


team_stats = feature_engineering(regular_results, detailed_results, massey_ordinals)


team_stats


team_stats.duplicated().sum()


# Display dataset info
print("Dataset Information:\n")
print(team_stats.info())



# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(team_stats.head())


print("\nMissing Values:")
print(team_stats.isnull().sum())


print("\nSummary Statistics:")
print(team_stats.describe())



team_stats.hist(figsize=(14, 12), bins=30, edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=team_stats)
plt.xticks(rotation=90)
plt.title("Boxplot of Numerical Features")
plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(team_stats.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


team_stats.fillna(team_stats.median(numeric_only=True), inplace=True)



from scipy.stats import zscore



z_scores = np.abs(zscore(team_stats.select_dtypes(include=[np.number])))
threshold = 3  # Common threshold for outliers
team_stats_cleaned = team_stats[(z_scores < threshold).all(axis=1)]




def plot_boxplots(data, title):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=data)
    plt.title(title)
    plt.show()


plot_boxplots(team_stats["TotalGames"], "TotalGames - Before Removing Outliers")

Q1 = team_stats["TotalGames"].quantile(0.25)
Q3 = team_stats["TotalGames"].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


team_stats_cleaned = team_stats[(team_stats["TotalGames"] >= lower_bound) & (team_stats["TotalGames"] <= upper_bound)]


plot_boxplots(team_stats_cleaned["TotalGames"], "TotalGames - After Removing Outliers")


team_stats_cleaned.to_csv("/content/team_stats_cleaned.csv", index=False)

print(f"Outliers removed for TotalGames using IQR. Cleaned data saved as 'team_stats_cleaned.csv'.")


def plot_boxplots(data, title):
    plt.figure(figsize=(12, 6))

    for column in data.select_dtypes(include=np.number).columns:
        sns.boxplot(x=data[column])
        plt.title(title + " - " + column)
        plt.show()


plot_boxplots(team_stats_cleaned, "Boxplot After Removing Outliers")


import pandas as pd

summary_stats = team_stats_cleaned.describe()


stats_to_display = ['std', 'mean', '50%', 'min', 'max']
summary_table = summary_stats.loc[stats_to_display]

summary_table = summary_table.rename(index={
    'std': 'Standard Deviation',
    'mean': 'Mean',
    '50%': 'Variance',
    'min': 'Min',
    'max': 'Max'
})

summary_table




from sklearn.preprocessing import MinMaxScaler


columns_to_scale = ['AvgPointsScored', 'GamesWon', 'AvgPointsAllowed', 'GamesLost',
                    'TotalGames', 'WinRatio', 'AvgShootingEff', 'AvgPossessionControl',
                    'AvgBalancedScore', 'OrdinalRank']

scaler = MinMaxScaler()
team_stats_cleaned[columns_to_scale] = scaler.fit_transform(team_stats_cleaned[columns_to_scale])



team_stats_cleaned


team_stats.to_csv('team_stats.csv', index=False)



print("Dataset Information:\n")
print(team_stats.info())


print("\nMissing Values:")
print(team_stats.isnull().sum())


team_stats.shape


team_stats.head()


print("\nDuplicate Rows:")
print(team_stats.duplicated().sum())


!pip install tensorflow



import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled!")
    except RuntimeError as e:
        print(f"GPU memory configuration failed: {e}")


df = pd.read_csv("team_stats_cleaned.csv")

teams = df["TeamID"].unique()
df.drop(columns=["OrdinalRank"], inplace=True)
df_features = df.drop(columns=["TeamID"])


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_features)
scaled_df = pd.DataFrame(scaled_data, columns=df_features.columns)
scaled_df["TeamID"] = df["TeamID"].values


X = scaled_df.drop(columns=["WinRatio"]).values
y = scaled_df["WinRatio"].values


sequence_length = 5
X_seq, y_seq = [], []

for i in range(len(X) - sequence_length):
    X_seq.append(X[i:i+sequence_length])
    y_seq.append(y[i+sequence_length])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)


def create_model():
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


model_path = "lstm_winratio_model.h5"
if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model = load_model(model_path, compile=False)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
else:
    print("Training new LSTM model...")
    model = create_model()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_seq, y_seq, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stop])
    model.save(model_path)
    print("Model training complete and saved!")


sample_submission = pd.read_csv("SampleSubmissionStage2.csv")


predictions = []
batch_size = 64
all_team_data = {}

for team in teams:
    team_data = scaled_df[scaled_df["TeamID"] == team].drop(columns=["TeamID"]).values[-sequence_length:]

    if len(team_data) < sequence_length:
        padding = np.zeros((sequence_length - len(team_data), team_data.shape[1]))
        team_data = np.vstack((padding, team_data))

    team_data = team_data.reshape(1, sequence_length, -1)
    all_team_data[team] = team_data


batch_team1, batch_team2, match_ids = [], [], []

for idx, match_id in enumerate(sample_submission["ID"]):
    season, team1, team2 = match_id.split("_")
    team1, team2 = int(team1), int(team2)


    if team1 not in all_team_data or team2 not in all_team_data:
        print(f"Missing data for TeamID {team1} or {team2}, skipping...")
        continue

    batch_team1.append(all_team_data[team1])
    batch_team2.append(all_team_data[team2])
    match_ids.append(match_id)

    if len(batch_team1) >= batch_size or idx == len(sample_submission) - 1:
        batch_team1 = np.vstack(batch_team1)
        batch_team2 = np.vstack(batch_team2)

        team1_winratios = model.predict(batch_team1, batch_size=batch_size).flatten()
        team2_winratios = model.predict(batch_team2, batch_size=batch_size).flatten()

        win_probs = team1_winratios / (team1_winratios + team2_winratios)

        predictions.extend(zip(match_ids, win_probs))

        batch_team1, batch_team2, match_ids = [], [], []

    if idx % 1000 == 0:
        print(f"Processed {idx}/{len(sample_submission)} matchups...")


output_df = pd.DataFrame(predictions, columns=["ID", "Pred"])
output_df.to_csv("winratio_predictions_2025.csv", index=False)

print("Predictions saved successfully!")



import pandas as pd

def compare_pred_columns(file_path1, file_path2):

    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)


    df1.columns = df1.columns.str.lower()
    df2.columns = df2.columns.str.lower()


    if 'id' not in df1.columns or 'id' not in df2.columns:
        raise ValueError("Both CSV files must have an 'id' column (case-insensitive).")


    if 'pred' not in df1.columns:
        if 'Pred' in df1.columns:
            df1.rename(columns={'Pred': 'pred'}, inplace=True)
        else:
            raise ValueError(f"File '{file_path1}' is missing the 'pred' or 'Pred' column.")
    if 'pred' not in df2.columns:
        if 'Pred' in df2.columns:
            df2.rename(columns={'Pred': 'pred'}, inplace=True)
        else:
            raise ValueError(f"File '{file_path2}' is missing the 'pred' or 'Pred' column.")


    df1['id'] = df1['id'].astype(str)
    df2['id'] = df2['id'].astype(str)


    merged_df = pd.merge(df1[['id', 'pred']], df2[['id', 'pred']], on='id', suffixes=('_cv1', '_cv2'))


    merged_df['pred_cv1'] = pd.to_numeric(merged_df['pred_cv1'], errors='coerce')
    merged_df['pred_cv2'] = pd.to_numeric(merged_df['pred_cv2'], errors='coerce')


    merged_df['error_difference'] = merged_df['pred_cv1'] - merged_df['pred_cv2']


    mse = (merged_df['error_difference'] ** 2).mean()


    output_file = "comparison_results.csv"
    merged_df[['id', 'pred_cv1', 'pred_cv2', 'error_difference']].to_csv(output_file, index=False)

    return merged_df[['id', 'pred_cv1', 'pred_cv2', 'error_difference']], mse, output_file


compare_pred_columns("/content/predictions.csv","/content/winratio_predictions_2025_final.csv")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_error(csv_file):

    df = pd.read_csv(csv_file)


    df['id'] = df['id'].astype(str)


    sns.set_style("whitegrid")


    plt.figure(figsize=(8, 5))
    sns.histplot(df['error_difference'], bins=30, kde=True, color='blue')
    plt.axvline(df['error_difference'].mean(), color='red', linestyle='dashed', label='Mean Error')
    plt.title("Distribution of Error Differences")
    plt.xlabel("Error Difference (Pred_CV1 - Pred_CV2)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()





visualize_error("comparison_results.csv")








