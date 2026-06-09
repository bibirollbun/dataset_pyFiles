import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#mount drive
from google.colab import drive
drive.mount('/content/drive')


#load in data
data = pd.read_csv('/content/drive/MyDrive/RedsComp25/redscomp_data.csv')

#filter out 2021, as we have no previous year data here (more discussion to follow)
data = data[data['year'] > 2021]


#use one hot encoding to encode position (categorical variable)
data = pd.get_dummies(data, columns=['prev_year_primary_position'], prefix='position')

#use one hot encoding to encode bats column
data = pd.get_dummies(data, columns=['bats'], prefix='bats')


#convert throws to 0-1 (0 = lefty, 1 = righty)
data['throws'] = data['throws'].map({'L': 0, 'R': 1})


#features for our batting model
features_batting = ['weight', 'height', 'age', 'throws',
            'prev_year_pitches_played_in_field', 'prev_year_pa', 'prev_year_tto', 'bats_L', 'bats_R', 'bats_B',
            'position_0.0','position_2.0', 'position_3.0', 'position_4.0', 'position_5.0',
              'position_6.0', 'position_7.0', 'position_8.0', 'position_9.0']
target = ['pa']


#features for pitching model
features_pitching = ['weight', 'height', 'age', 'throws', 'prev_year_pitches_thrown',
             'prev_year_bf', 'prev_year_pct_relief']
target_pitching = ['bf']


data_batting_short = data[['player', 'weight', 'height', 'age', 'bats_L', 'bats_R', 'bats_B','throws', 'position_0.0', 'position_1.0',
       'position_2.0', 'position_3.0', 'position_4.0', 'position_5.0',
       'position_6.0', 'position_7.0', 'position_8.0', 'position_9.0',
            'prev_year_pitches_played_in_field', 'prev_year_tto', 'prev_year_pa', 'pa']].dropna()

#filter pitchers out of our batting df
data_batting_short = data_batting_short[data_batting_short['position_1.0'] != True]


data_pitching_short = data[['player','weight', 'height', 'age', 'throws', 'prev_year_pitches_thrown',
            'prev_year_fastball_avg_velo', 'prev_year_bf',
                            'prev_year_pct_relief', 'bf', 'position_1.0']].dropna()


#split dataset into features and target
X_batting = data_batting_short[features_batting].dropna()
y_batting = data_batting_short[target].dropna()


X_pitching = data_pitching_short[features_pitching].dropna()
y_pitching = data_pitching_short[target_pitching]


# Train-test split with 25% test size
X_train_batting, X_test_batting, y_train_batting, y_test_batting = train_test_split(X_batting, y_batting, test_size=0.25, random_state=33)
X_train_pitching, X_test_pitching, y_train_pitching, y_test_pitching = train_test_split(X_pitching, y_pitching, test_size=0.25, random_state=42)

# Check the shapes to confirm the split
print('BATTING')
print(f"X_train shape: {X_train_batting.shape}")
print(f"X_test shape: {X_test_batting.shape}")
print(f"y_train shape: {y_train_batting.shape}")
print(f"y_test shape: {y_test_batting.shape}")

print('PITCHING')
print(f"X_train shape: {X_train_pitching.shape}")
print(f"X_test shape: {X_test_pitching.shape}")
print(f"y_train shape: {y_train_pitching.shape}")
print(f"y_test shape: {y_test_pitching.shape}")


# Define the hyperparameter grid for our batting model
param_grid_batting = {
    'n_estimators': [35, 50],
    'max_depth': [3, 5, 7 , 9],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [3],
}

rf_model_batting = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV object
grid_search = GridSearchCV(estimator=rf_model_batting, param_grid=param_grid_batting,
                           cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train_batting, y_train_batting.values.ravel())


# Best hyperparameters
print(f"Best Parameters for batting grid search: {grid_search.best_params_}")

# Best MSE score
best_mse = -grid_search.best_score_  # Convert back from negative MSE
print(f"Best Cross-Validated MSE: {best_mse}")


# Define the hyperparameter grid for pitching
param_grid_pitching = {
    'n_estimators': [15, 20],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [2,3],
}
rf_model_pitching = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search_pitching = GridSearchCV(estimator=rf_model_pitching, param_grid=param_grid_pitching,
                           cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)

# Fit GridSearchCV
grid_search_pitching.fit(X_train_pitching, y_train_pitching.values.ravel())



# Best hyperparameters
print(f"Best Parameters for pitching model: {grid_search_pitching.best_params_}")

# Best MSE score
best_mse_pitching = -grid_search_pitching.best_score_  # Convert back from negative MSE
print(f"Best Cross-Validated MSE: {best_mse_pitching}")


# Make predictions using the best estimator
best_rf_model_batting = grid_search.best_estimator_
y_pred_batting = best_rf_model_batting.predict(X_test_batting)

# Plotting the predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_batting, y_pred_batting, alpha=0.7, color='black')
plt.plot([y_test_batting.min(), y_test_batting.max()], [y_test_batting.min(), y_test_batting.max()], 'r--', lw=2)
plt.xlabel('Observed PA')
plt.ylabel('Predicted PA')
plt.title('Batting Model - Actual vs Predicted PA')
plt.grid(True)
plt.show()


# Calculate RMSE
mse = mean_squared_error(y_test_batting, y_pred_batting)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# Make predictions using the best estimator
best_rf_model_pitching = grid_search_pitching.best_estimator_
y_pred_pitching = best_rf_model_pitching.predict(X_test_pitching)

# Plotting the predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_pitching, y_pred_pitching, alpha=0.7, color='blue')
plt.plot([y_test_pitching.min(), y_test_pitching.max()], [y_test_pitching.min(), y_test_pitching.max()], 'r--', lw=2)
plt.xlabel('Observed BF')
plt.ylabel('Predicted BF')
plt.title('Pitching Model - Actual vs Predicted BF')
plt.grid(True)
plt.show()


# Calculate RMSE
mse = mean_squared_error(y_test_pitching, y_pred_pitching)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# Get feature importances from the best model
importances = best_rf_model_batting.feature_importances_

# Pair feature names with their importances
feature_importance = pd.Series(importances, index=features_batting).sort_values(ascending=False)

# Plot the feature importances
plt.figure(figsize=(8, 6))
feature_importance.plot(kind='barh', color='black')
plt.xlabel('Importance')
plt.title('Feature Importances in Batting Random Forest Model')
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.show()




# Get feature importances from the best pitching model
importances_pitching = best_rf_model_pitching.feature_importances_

# Pair feature names with their importances
feature_importances_pitching = pd.Series(importances_pitching, index=features_pitching).sort_values(ascending=False)

# Plot the feature importances
plt.figure(figsize=(8, 6))
feature_importances_pitching.plot(kind='barh', color='blue')
plt.xlabel('Importance')
plt.title('Feature Importances in Pitching Random Forest Model')
plt.gca().invert_yaxis()  # Invert to show the most important feature on top
plt.grid(axis='x')
plt.show()


sample_submission = pd.read_csv('/content/drive/MyDrive/RedsComp25/sample_submission.csv')


#get 2023 data, as it is relevant to 2024 predictions
data_23 = data[data['year'] == 2023]
data_to_merge = data_23[['player', 'weight', 'height', 'age', 'bats_L', 'bats_R', 'bats_B', 'throws', 'primary_position', 'prev_year_tto',
            'pitches_played_in_field', 'pitches_thrown', 'ff_avg', 'position_0.0', 'position_1.0',
       'position_2.0', 'position_3.0', 'position_4.0', 'position_5.0',
       'position_6.0', 'position_7.0', 'position_8.0', 'position_9.0', 'pct_relief','bf', 'pa']]

#merge data with sample submission
sample_submission = pd.merge(sample_submission, data_to_merge,
                              left_on = 'PLAYER_ID', right_on='player', how='left')

#drop 'player' column as it is a duplicate
sample_submission = sample_submission.drop('player', axis=1)

#fill na values with zero
sample_submission = sample_submission.fillna(0)



#rename columns according to model architecture
sample_submission = sample_submission.rename(columns={'primary_position': 'prev_year_primary_position', 'pitches_played_in_field': 'prev_year_pitches_played_in_field',
                                  'pa': 'prev_year_pa', 'bf': 'prev_year_bf', 'pitches_thrown': 'prev_year_pitches_thrown',
                                  'ff_avg': 'prev_year_fastball_avg_velo', 'pct_relief': 'prev_year_pct_relief',
                                 'ff_avg': 'prev_year_fastball_avg_velo'
                                  })


#make plate appearance predictions
pred_plate_appearance = best_rf_model_batting.predict(sample_submission[features_batting])
sample_submission['predicted_pa'] = pred_plate_appearance

#make batters faced predictions
pred_batters_faced = best_rf_model_pitching.predict(sample_submission[features_pitching])
sample_submission['predicted_bf'] = pred_batters_faced

#make pa and bf zero respectively if prev year pa/bf is zero
sample_submission['predicted_pa'] = np.where(sample_submission['prev_year_pa'] == 0, 0, sample_submission['predicted_pa'])
sample_submission['predicted_bf'] = np.where(sample_submission['prev_year_bf'] == 0, 0, sample_submission['predicted_bf'])

#get final predicted playing time column through addition
sample_submission['PREDICTED_PLAYING_TIME'] = sample_submission['predicted_pa'] + sample_submission['predicted_bf']

#predict 100 for anybody who's prediction is zero (likely rookies)
sample_submission['PREDICTED_PLAYING_TIME'] = np.where(sample_submission['PREDICTED_PLAYING_TIME'] == 0, 100, sample_submission['PREDICTED_PLAYING_TIME'])



#shorten columns for final submission
final_submission = sample_submission[['PLAYER_ID', 'PREDICTED_PLAYING_TIME']]
final_submission = final_submission.rename(columns={'PREDICTED_PLAYING_TIME': 'PLAYING_TIME'})


#dump our final submission to csv
final_submission.to_csv('/content/drive/MyDrive/RedsComp25/tufts_submission.csv', index=False)

