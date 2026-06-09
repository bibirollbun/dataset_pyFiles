import pandas as pd

train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

print("Training Data:")
display(train_df.head())

print("\nTest Data:")
display(test_df.head())


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Basic engineered numeric features
train_df['LoudnessEnergyInteraction'] = train_df['AudioLoudness'] * train_df['Energy']
test_df['LoudnessEnergyInteraction'] = test_df['AudioLoudness'] * test_df['Energy']

train_df["rs_energy"]    = train_df["RhythmScore"] * train_df["Energy"]
test_df["rs_energy"]     = test_df["RhythmScore"] * test_df["Energy"]

train_df["loud_acoustic"]= train_df["AudioLoudness"] * train_df["AcousticQuality"]
test_df["loud_acoustic"] = test_df["AudioLoudness"] * test_df["AcousticQuality"]

train_df["voc_vs_inst"]  = train_df["VocalContent"] / (train_df["InstrumentalScore"] + 1e-6)
test_df["voc_vs_inst"]   = test_df["VocalContent"] / (test_df["InstrumentalScore"] + 1e-6)

train_df["mood_sq"]      = train_df["MoodScore"] ** 2
test_df["mood_sq"]       = test_df["MoodScore"] ** 2

train_df["duration_rhythm"] = train_df["TrackDurationMs"] * train_df["RhythmScore"]
test_df["duration_rhythm"]  = test_df["TrackDurationMs"] * test_df["RhythmScore"]
train_df["live_energy"]  = train_df["LivePerformanceLikelihood"] * train_df["Energy"]
test_df["live_energy"]   = test_df["LivePerformanceLikelihood"] * test_df["Energy"]
train_df["energy_sq"]    = train_df["Energy"] ** 2
test_df["energy_sq"]     = test_df["Energy"] ** 2

train_df["is_live"]   = (train_df["LivePerformanceLikelihood"] > 0.5).astype(int)
test_df["is_live"]    = (test_df["LivePerformanceLikelihood"] > 0.5).astype(int)

train_df["is_instrumental"] = (train_df["InstrumentalScore"] > 0.7).astype(int)
test_df["is_instrumental"]  = (test_df["InstrumentalScore"] > 0.7).astype(int)

train_df["is_vocal_heavy"]  = (train_df["VocalContent"] > 0.7).astype(int)
test_df["is_vocal_heavy"]   = (test_df["VocalContent"] > 0.7).astype(int)


print("Missing values in train_df:")
print(train_df.isnull().sum())
print("\nMissing values in test_df:")
print(test_df.isnull().sum())


print("Data types in train_df:")
print(train_df.dtypes)
print("\nData types in test_df:")
print(test_df.dtypes)

numerical_features = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# Exclude the 'id' and 'BeatsPerMinute' columns for outlier detection and scaling
numerical_features.remove('id')
if 'BeatsPerMinute' in numerical_features:
    numerical_features.remove('BeatsPerMinute')


print("\nNumerical features for outlier detection and scaling:")
print(numerical_features)

# Check for outliers using the describe method to see min/max values
print("\nDescriptive statistics for numerical features in train_df:")
display(train_df[numerical_features].describe())

print("\nDescriptive statistics for numerical features in test_df:")
display(test_df[numerical_features].describe())

categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns:", categorical_cols)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Scale numerical features in train_df
train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])

# Scale numerical features in test_df
test_df[numerical_features] = scaler.transform(test_df[numerical_features])

print("Scaled train_df:")
display(train_df.head())
print("\nScaled test_df:")
display(test_df.head())


# Display the heads of both dataframes to show the newly engineered features
print("Train DataFrame with new feature:")
display(train_df.head())

print("\nTest DataFrame with new feature:")
display(test_df.head())


from sklearn.ensemble import RandomForestRegressor as RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define features (X) and target (y) for training and testing
features = [col for col in train_df.columns if col not in ['id', 'BeatsPerMinute']]
X_train = train_df[features]
y_train = train_df['BeatsPerMinute']
X_test = test_df[features]

# Create the stacking regressor
rf_model = RandomForestRegressor()

# Train the stacked model
rf_model.fit(X_train, y_train)

print("Random Forest Regressor model trained.")


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Evaluate Random Forest Regressor model
y_train_pred_rf = rf_model.predict(X_train)
mse_rf = mean_squared_error(y_train, y_train_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_train, y_train_pred_rf)
r2_rf = r2_score(y_train, y_train_pred_rf)
print(f"\nRandom Forest Regressor Metrics on Training Data:")
print(f"  MSE: {mse_rf:.4f}")
print(f"  RMSE: {rmse_rf:.4f}")
print(f"  MAE: {mae_rf:.4f}")
print(f"  R-squared: {r2_rf:.4f}")


# Predict on X_test using the trained models
y_pred = rf_model.predict(X_test)

# Print the first 5 predictions for each model
print("First 5 predictions for Random Forest Regressor:")
print(y_pred[:5])


import matplotlib.pyplot as plt
import seaborn as sns

# Create scatter plots of predictions vs actual values for each model

plt.subplot(2, 2, 3)
sns.scatterplot(x=y_train, y=y_train_pred_rf, alpha=0.6)
plt.title('Random Forest Regressor: Actual vs. Predicted')
plt.xlabel('Actual BeatsPerMinute')
plt.ylabel('Predicted BeatsPerMinute')

plt.tight_layout()
plt.show()

# Calculate residuals for each model
residuals_rf = y_train - y_train_pred_rf

# Create scatter plots of residuals vs predicted values for each model

plt.subplot(2, 2, 3)
sns.scatterplot(x=y_train_pred_rf, y=residuals_rf, alpha=0.6)
plt.title('Random Forest Regressor: Residuals vs. Predicted')
plt.xlabel('Predicted BeatsPerMinute')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()


submission_df = test_df[['id']].copy()
submission_df['BeatsPerMinute'] = y_pred
print(submission_df.head())


submission_df.to_csv('submission.csv', index=False)

