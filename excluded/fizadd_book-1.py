import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss


# Load important files (Update path if needed)
seeds = pd.read_csv("/kaggle/input/big-data/MNCAATourneySeeds.csv")
tourney_results = pd.read_csv("/kaggle/input/big-data/MNCAATourneyCompactResults.csv")
teams = pd.read_csv("/kaggle/input/big-data/MTeams.csv")
regular_results = pd.read_csv("/kaggle/input/big-data/MRegularSeasonCompactResults.csv")


# Remove 'W' from Season for joins and clean seed
seeds['Seed'] = seeds['Seed'].str.extract('(\d+)').astype(int)


# Create training data from past tournament results
def create_train_data(results, seeds):
    X = []
    y = []

    for index, row in results.iterrows():
        season = row['Season']
        team1 = min(row['WTeamID'], row['LTeamID'])
        team2 = max(row['WTeamID'], row['LTeamID'])

        # Get seeds
        seed1 = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == team1)]['Seed']
        seed2 = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == team2)]['Seed']
        
        if seed1.empty or seed2.empty:
            continue

        # Feature: Seed difference
        feature = [int(seed1.values[0]) - int(seed2.values[0])]

        # Label: 1 if team1 wins
        label = 1 if row['WTeamID'] == team1 else 0

        X.append(feature)
        y.append(label)

    return np.array(X), np.array(y)

X_train, y_train = create_train_data(tourney_results, seeds)



# Use Random Forest (simple, works well)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



import pandas as pd
import numpy as np

# Load data
sample_sub = pd.read_csv("/kaggle/input/big-data/SampleSubmissionStage1.csv")
seeds = pd.read_csv("/kaggle/input/big-data/WNCAATourneySeeds.csv")

# Clean Seed column
seeds['Seed'] = seeds['Seed'].str.extract(r'(\d+)').astype(int)

# Extract year, team1, team2
split_df = sample_sub['ID'].str.split('_', expand=True).astype(int)
sample_sub['Season'] = split_df[0]
sample_sub['Team1'] = split_df[[1, 2]].min(axis=1)
sample_sub['Team2'] = split_df[[1, 2]].max(axis=1)

# Merge seed info
seeds.columns = ['Season', 'TeamID', 'Seed']
sample_sub = sample_sub.merge(seeds, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
sample_sub = sample_sub.rename(columns={'Seed': 'Seed1'}).drop(columns='TeamID')

sample_sub = sample_sub.merge(seeds, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left')
sample_sub = sample_sub.rename(columns={'Seed': 'Seed2'}).drop(columns='TeamID')

# Compute feature (seed diff)
sample_sub['SeedDiff'] = sample_sub['Seed1'].fillna(0) - sample_sub['Seed2'].fillna(0)

# Final test set
X_test = sample_sub[['SeedDiff']].values
test_ids = sample_sub['ID'].tolist()

# Predict
preds = model.predict_proba(X_test)[:, 1]



import pandas as pd
import numpy as np

# Step 1: Make sure `test_ids` and `preds` are equal in length
assert len(test_ids) == len(preds), "Mismatch in test IDs and predictions!"

# Step 2: Convert predictions to flat list or numpy array if needed
preds = np.asarray(preds).flatten()

# Step 3: Create the DataFrame
submission = pd.DataFrame({
    'ID': test_ids,
    'Pred': preds
})

# Step 4: Save as CSV (optimized with chunking for large datasets)
submission.to_csv("submission.csv", index=False, chunksize=50000)

print("Submission file created successfully!")

