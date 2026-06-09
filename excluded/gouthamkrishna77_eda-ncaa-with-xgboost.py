import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# Step 1: Load datasets
regular_season = pd.read_csv('MRegularSeasonDetailedResults.csv')
tourney_results = pd.read_csv('MNCAATourneyDetailedResults.csv')
teams = pd.read_csv('MTeams.csv')

# Step 2: Concatenate regular season and tournament results
all_games = pd.concat([regular_season, tourney_results], axis=0, ignore_index=True)


all_games.head(10)


all_games.describe()


all_games = pd.merge(all_games, teams[['TeamID', 'TeamName']], left_on='LTeamID', right_on='TeamID', how='left')
all_games.rename(columns={'TeamName': 'LTeamName'}, inplace=True)
all_games.drop(columns=['TeamID'], inplace=True)


all_games.info()


all_games.info()


all_games = all_games.dropna()



all_games['WLoc'].unique()


wloc_mapping = {'H': 1, 'A': -1, 'N': 0}  # Home = 1, Away = -1, Neutral = 0
all_games['WLoc'] = all_games['WLoc'].map(wloc_mapping)



# Create a new column for point difference
all_games['PointDiff'] = all_games['WScore'] - all_games['LScore']

# Display the first few rows to check the new column
print(all_games[['WScore', 'LScore', 'PointDiff']].head())



all_games['LTeamName'].unique()



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
all_games['LTeamName_encoded'] = le.fit_transform(all_games['LTeamName'])
all_games = all_games.drop('LTeamName', axis=1)



plt.figure(figsize=(10, 6))
sns.boxplot(x=all_games['PointDiff'])
plt.title('Boxplot of Point Differential (PointDiff)')
plt.xlabel('Point Differential')
plt.show()


plt.figure(figsize=(10, 5))
sns.histplot(all_games['WTeamID'], bins=30, kde=True)
plt.title("Distribution of Winning Team ID (WTeamID)")
plt.xlabel("WTeamID")
plt.ylabel("Frequency")
plt.show()


scaler =StandardScaler()
scaled_df =scaler.fit_transform(all_games)


x =all_games.drop(columns=['WTeamID'])
y =all_games['WTeamID']


pca =PCA(n_components=0.90)
df_pca =pca.fit_transform(x)
df_pca


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(df_pca,y,test_size=0.2,random_state=42)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize the regression model
regressor = RandomForestRegressor(random_state=42)

# Train the model on your training data
regressor.fit(x_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(x_test)

# Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and RÂ² score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("RÂ² Score:", r2)



from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Initialize and train the XGBoost regressor
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(x_test)

# Calculate performance metrics
r2_xgb = r2_score(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print("XGBoost Regressor:")
print("RÂ² Score:", r2_xgb)
print("MSE:", mse_xgb)
print("MAE:", mae_xgb)


