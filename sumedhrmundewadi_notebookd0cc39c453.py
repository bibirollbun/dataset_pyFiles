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


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

print(train.shape)

print("----------Dataset Description-----------")
print(train.describe())

print("----------Dataset Information-----------")
print(train.info())


import matplotlib.pyplot as plt
import seaborn as sns

train_df_analysis = train.drop('id', axis=1)

#1. Distribution of BeatsPerMinute
plt.figure(figsize=(10, 6))
sns.histplot(train_df_analysis['BeatsPerMinute'], bins=50, kde=True)
plt.title("Distribution of BeatsPerMinute")
plt.xlabel("BeatsPerMInute(BPM)")
plt.ylabel("Count")

# distribution of other scores for example RythmScore and AudioLoudness
# You actually want to loop through all the features or plot them individually
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
axes = axes.flatten() # flatten to iterate easily

features_to_plot = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                   'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs', 'Energy']

for i, col in enumerate(features_to_plot):
    sns.histplot(train_df_analysis[col], bins=50, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
plt.tight_layout()
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(12, 10))
correlational_matrix = train_df_analysis.corr()
sns.heatmap(correlational_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correaltional matrix of Features and BeatsPerMinute")
plt.show()


# dummy dataset or dataframe and replace it after loading
'''
data = {
    'id': range(524164),
    'RhythmScore': np.random.uniform(0.0769, 0.975, 524164),
    'AudioLoudness': np.random.uniform(-27.509725, -1.357000, 524164),
    'VocalContent': np.random.uniform(0.0235, 0.256401, 524164),
    'AcousticQuality': np.random.uniform(0.000005, 0.995, 524164),
    'InstrumentalScore': np.random.uniform(0.000001, 0.869258, 524164),
    'LivePerformanceLikelihood': np.random.uniform(0.0243, 0.599924, 524164),
    'MoodScore': np.random.uniform(0.0256, 0.978, 524164),
    'TrackDurationMs': np.random.uniform(63973.0, 464723.2281, 524164),
    'Energy': np.random.uniform(0.000067, 1.0, 524164),
    'BeatsPerMinute': np.random.uniform(46.718, 206.037, 524164)
}
df = pd.dataframe(data)
'''

#Actual Dataframe continuation
train_new_features = train.copy()

# Feature Engineering based on selected features and general ideas

#1 Loudness per unit duration (using TrackDurationMs)
# Convert duration frommilliseconds to minutes for a more intuitive ratio

train_new_features['TrackDurationMinutes'] = train_new_features['TrackDurationMs'] / 60000
train_new_features['LoudnessPerMinute'] = train_new_features['AudioLoudness'] / train_new_features['TrackDurationMinutes']

#Rhythm-Energy Interaction
train_new_features['RhythmEnergyProduct'] = train_new_features['RhythmScore'] * train_new_features['Energy']
train_new_features['RhytmEnergyRatio'] = train_new_features['RhythmScore'] / (train_new_features['Energy'] + 1e-6)

# Log Transformation for TrackDurationMs (if it was heavily skewed)
train_new_features['LogTrackDurationMs'] = np.log1p(train_new_features['TrackDurationMs'])

# Mood-Energy interaction
train_new_features['MoodEnergyProduct'] = train_new_features['MoodScore'] * train_new_features['Energy']

# Vocal vs Instrumental Balance Difference (Ratio and Difference)
# Adding epsilon for safety against division by zero with InstrumentalScore
train_new_features['VocalInstrumentalProduct'] = train_new_features['VocalContent'] / (train_new_features['InstrumentalScore'] + 1e-6)
train_new_features['VocalInstrumentalDiff'] = train_new_features['VocalContent'] / train_new_features['InstrumentalScore']

# Displaying info() and describe() for new feature dataframe'
print("-------New Dataframe Info-------")
train_new_features.info()

print("-------New Dataframe Description-------")
train_new_features.describe()

# Re-run correlation matrix with new features
plt.figure(figsize=(14, 12))
# Drop 'id' and the temporary 'TrackDurationinMinutes' column for correlation
correlation_matrix_new = train_new_features.drop(['id', 'TrackDurationMinutes'], axis=1).corr()
sns.heatmap(correlation_matrix_new, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation matrix of Features(including new ones) and BeatsPerMinute")
plt.show()
plt.savefig("myplot.png")


from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assuming df_new_features is the new dataframe in the previous step

#1. Select Features X and y(Target)
features = [col for col in train_new_features.columns if not col in ['id', 'BeatsPerMinute', 'TrackDurationMs', 'TrackDurationMinutes']]
X = train_new_features[features]
y = train_new_features['BeatsPerMinute']

print(f"\nFeatures used for modelling {features}")
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

#2. Split data into training and validation sets
# Using a 80/20 split, random_state for reproducibility

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(f"\nTrain set size: {len(X_train)} samples")
print(f"Validation set size: {len(X_val)} samples")

#3. Choose a model: LightGBMRegressor
lgbm = LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31)
# Starting with some resaonable defaults

#4 Train the model
print("\nTraining the LGBM model...")
lgbm.fit(X_train, y_train)
print("Model Training complete!")

#5. Make predictions on the validation set
y_pred = lgbm.predict(X_val)

#6. Evaluate performance
mae = mean_absolute_error(y_val, y_pred) 
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print("Model Evaluation on Validation set")
print(f"Mean Abosulte Error (MAE) = {mae:.4f} BPM")
print(f"Root Mean Square Error (RMSE) = {rmse:.4f} BPM")

plt.figure(figsize=(10, 8))
feature_importances = pd.Series(lgbm.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title('LightGBM Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


import lightgbm as lgbm

print("\n Hyper parameter tuning attempt1")

lgbm_tuned = LGBMRegressor(
    n_estimators=2000, 
    random_state=42, 
    learning_rate=0.02, 
    num_leaves=64, 
    max_depth=-1, 
    min_child_samples = 20
)

print("\nTraining LightGBM model with Tuned Parameters")
# For early stopping we need to pass a validation set to the fit method

lgbm_tuned.fit(X_train, y_train,
               eval_set=[(X_val, y_val)],
               eval_metric='mae', # Evaluate using MAE
               callbacks=[lgbm.early_stopping(stopping_rounds=100, verbose=False)])
# Stop if mae doesn't improve for 10 rounds

print("Model Training Complete!")

y_tuned_lgbm = lgbm_tuned.predict(X_val)

mae_tuned = mean_absolute_error(y_val, y_tuned_lgbm)
rmse_tuned  = np.sqrt(mean_squared_error(y_val, y_tuned_lgbm))

print("Model Evalution (Tuned) on model Validation set!")
print(f"Mean Absolute Error (MAE): {mae_tuned:.4f} BPM")
print(f"Root Mean Squared Error (RMSE): {rmse_tuned:.4f} BPM")

print("\nBaseline mae: 21.2140 BPM")
print(f"Tuned MAE: {mae_tuned:.4f} BPM")


submission = pd.DataFrame({
    "id": test["id"],            # use the ID column from test set
    "Calories": lgbm_tuned # predictions from your final model
})

# ✅ Save to Kaggle working directory (so Kaggle sees it)
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("Submission file created successfully!")
print(submission.head())

