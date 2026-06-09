import numpy as np 
import pandas as pd 
import warnings

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

warnings.simplefilter(action='ignore', category=Warning)


df = pd.read_csv("/kaggle/input/mlx-2-0-regression/train.csv")
test = pd.read_csv("/kaggle/input/mlx-2-0-regression/test.csv")


df.head()


test.head()


df.shape


df.describe()


test.describe()


desc = pd.DataFrame({
    'feature': df.columns,
    'type': df.dtypes.values,
    'count': df.count().values,
    'nunique': df.nunique().values,
    'null': df.isnull().sum().values
})

pd.set_option('display.max_rows', None)

print(desc)
desc.to_excel("column_info.xlsx", index=False)


desc_test = pd.DataFrame({
    'feature': test.columns,
    'type': test.dtypes.values,
    'count': test.count().values,
    'nunique': test.nunique().values,
    'null': test.isnull().sum().values
})

pd.set_option('display.max_rows', None)

print(desc_test)
desc_test.to_excel("column_info_test.xlsx", index=False)


duplicates = df[df.duplicated()]
duplicates


duplicates_test = test[test.duplicated()]
duplicates_test


missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
print(missing_df[missing_df['Missing Values'] > 0])


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Data Heatmap")
plt.show()


missing_values_test = test.isnull().sum()
missing_percentage_test = (missing_values_test / len(df)) * 100

missing_test = pd.DataFrame({'Missing Values': missing_values_test, 'Percentage': missing_percentage_test})
print(missing_test[missing_test['Missing Values'] > 0])


numerical_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
categorical_cols = list(df.select_dtypes(include=["object"]).columns)
numerical_cols.remove("target")


# Replacing missing values with the mean for numerical columns and with the mode for the categorical columns

df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

test[numerical_cols] = test[numerical_cols].fillna(df[numerical_cols].mean())
test[categorical_cols] = test[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])


df.isnull().sum()


df.shape


new_df = pd.concat([df, test], axis =0) 


new_df.head()


print(df['composition_label_0'].nunique())
print(df['composition_label_1'].nunique())
print(df['creator_collective'].nunique())
print(df['composition_label_2'].nunique())
print(df['track_identifier'].nunique())


new_df['weekday_of_release'].unique()


new_df['season_of_release'].unique()


new_df['lunar_phase'].unique()


print("ts0:", new_df['time_signature_0'].value_counts(), "\n")
print("ts1:", new_df['time_signature_1'].value_counts(), "\n")
print("ts2:", new_df['time_signature_2'].value_counts())


# new_df.loc[(new_df['time_signature_1'].between(3.91291467 - 1e-5, 3.91291467 + 1e-5)), 'time_signature_1'] = 3.91291467
new_df['time_signature_0'] = new_df['time_signature_0'].round()
new_df['time_signature_1'] = new_df['time_signature_1'].round()
new_df['time_signature_2'] = new_df['time_signature_2'].round()

print("ts0:", new_df['time_signature_0'].value_counts(), "\n")
print("ts1:", new_df['time_signature_1'].value_counts(), "\n")
print("ts2:", new_df['time_signature_2'].value_counts())


new_df['time_signature_2'] = pd.to_numeric(new_df['time_signature_2'], errors='coerce')
new_df['time_signature_2'] = new_df['time_signature_2'].round()
print("ts2:", new_df['time_signature_2'].value_counts())


print("hs0:", new_df['harmonic_scale_0'].value_counts(), "\n")
print("hs1:", new_df['harmonic_scale_1'].value_counts(), "\n")
print("hs2:", new_df['harmonic_scale_2'].value_counts())


new_df['harmonic_scale_0'] = new_df['harmonic_scale_0'].round()
new_df['harmonic_scale_1'] = new_df['harmonic_scale_1'].round()
new_df['harmonic_scale_2'] = new_df['harmonic_scale_2'].round()

print("hs0:", new_df['harmonic_scale_0'].value_counts(), "\n")
print("hs1:", new_df['harmonic_scale_1'].value_counts(), "\n")
print("hs2:", new_df['harmonic_scale_2'].value_counts())


print("tm0:", new_df['tonal_mode_0'].value_counts(), "\n")
print("tm1:", new_df['tonal_mode_1'].value_counts(), "\n")
print("tm2:", new_df['tonal_mode_2'].value_counts())


new_df['tonal_mode_0'] = new_df['tonal_mode_0'].round()
new_df['tonal_mode_1'] = new_df['tonal_mode_1'].round()
new_df['tonal_mode_2'] = new_df['tonal_mode_2'].round()

print("tm0:", new_df['tonal_mode_0'].value_counts(), "\n")
print("tm1:", new_df['tonal_mode_1'].value_counts(), "\n")
print("tm2:", new_df['tonal_mode_2'].value_counts())


new_df['key_variety'].value_counts()


new_df['key_variety'] = new_df['key_variety'].round()
new_df['key_variety'].value_counts()


new_df['publication_timestamp'] = pd.to_datetime(new_df['publication_timestamp'])


new_df['year'] = new_df['publication_timestamp'].dt.year
new_df['month'] = new_df['publication_timestamp'].dt.month
new_df['day'] = new_df['publication_timestamp'].dt.day


# Create new features for the correlated features
# new_df["emotional_resonance_charge_2_combined"] = new_df["emotional_resonance_2"] / (new_df["emotional_charge_2"]+ 1e-6) 
# new_df["emotional_resonance_charge_1_combined"] = new_df["emotional_resonance_1"] / (new_df["emotional_charge_1"]+ 1e-6) 
# new_df["emotional_resonance_charge_0_combined"] = new_df["emotional_resonance_0"] / (new_df["emotional_charge_0"]+ 1e-6) 


freq_encoding_features = [
    'composition_label_0', 'composition_label_1', 'creator_collective', 'composition_label_2',
    'track_identifier', 'creator_collective'
]

for feature in freq_encoding_features:
    freq = new_df[feature].value_counts()
    new_df[f'{feature}_encoded'] = new_df[feature].map(freq)


weekday_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

new_df['weekday_of_release_num'] = new_df['weekday_of_release'].map(weekday_mapping)


# # Cyclical encoding for the weekday_of_release
# new_df['weekday_sin'] = np.sin(2 * np.pi * new_df['weekday_of_release_num'] / 7)
# new_df['weekday_cos'] = np.cos(2 * np.pi * new_df['weekday_of_release_num'] / 7)


season_mapping = {
    'spring': 0,
    'summer': 1,
    'autumn': 2,
    'winter': 3
}

new_df['season_of_release_num'] = new_df['season_of_release'].map(season_mapping)


# #Cyclical encoding
# new_df['season_sin'] = np.sin(2 * np.pi * new_df['season_of_release_num'] / 4)
# new_df['season_cos'] = np.cos(2 * np.pi * new_df['season_of_release_num'] / 4)


# Ordinal Encoding
lunar_mapping = {
    'new': 0,
    'waxing': 1,
    'full': 2,
    'waning': 3
}

new_df['lunar_phase_encoded'] = new_df['lunar_phase'].map(lunar_mapping)


new_df.drop([
    'composition_label_0', 'composition_label_1', 'composition_label_2', 'track_identifier', 
    'weekday_of_release', 'season_of_release', 'creator_collective', 'publication_timestamp',
    'lunar_phase'
], axis=1, inplace=True)


# new_df.drop([
#     'weekday_of_release_num', 'season_of_release_num'
# ], axis=1, inplace=True)


pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)          # Do not wrap lines
pd.set_option('display.max_colwidth', None)   # Show full content inside each cell

new_df.head()


# pd.reset_option('display.max_columns')
# pd.reset_option('display.width')
# pd.reset_option('display.max_colwidth')


df_copy = new_df.iloc[:61609].copy()
test_copy = new_df.iloc[61609:].copy()
test_copy = test_copy.drop(columns=['target'],  axis=1)


df_copy.shape


test_copy.shape


# encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
# test[categorical_cols] = encoder.transform(test[categorical_cols])


corr_matrix = df_copy.corr()

plt.figure(figsize=(14,12))

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)

plt.title("Feature Correlation Heatmap")
plt.show()


corr_matrix.to_excel('correlation_table.xlsx')

high_corr = corr_matrix[((corr_matrix > 0.8) | (corr_matrix < -0.8)) & (corr_matrix != 1)]

high_corr.to_excel('high_correlation_table.xlsx')


correlations = df_copy.corr()["target"].drop("target")  # Drop the self-correlation

plt.figure(figsize=(8, 12))
sns.heatmap(correlations.to_frame(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature vs Target Correlation")
plt.show()

correlations = correlations.sort_values(ascending=False)
correlations.to_excel("feature_target_correlations.xlsx")


# df_copy = df_copy.drop([
#     'emotional_charge_2', 
#     'emotional_charge_1',
#     'emotional_charge_0'
# ], axis=1)

# test_copy = test_copy.drop([
#     'emotional_charge_2', 
#     'emotional_charge_1',
#     'emotional_charge_0'
# ], axis=1)


df_copy.columns


# Removing features that are not very helpful to the model
df_copy = df_copy.drop([
    'groove_efficiency_1', 'beat_frequency_1', 'intensity_index_0', 'groove_efficiency_2',
    'composition_label_0_encoded', 'time_signature_2', 'tonal_mode_1', 'month',
    'harmonic_scale_0', 'tonal_mode_0', 'duration_ms_1', 'organic_texture_1'
], axis=1)
test_copy = test_copy.drop([
    'groove_efficiency_1', 'beat_frequency_1', 'intensity_index_0', 'groove_efficiency_2',
    'composition_label_0_encoded', 'time_signature_2', 'tonal_mode_1', 'month',
    'harmonic_scale_0', 'tonal_mode_0', 'duration_ms_1', 'organic_texture_1'
], axis=1)


# def drop_highly_correlated_features(df, threshold=0.8):
#     corr_matrix = df.corr().abs()

#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

#     to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

#     df_reduced = df.drop(columns=to_drop)

#     return df_reduced, to_drop


# reduced_df, dropped_features = drop_highly_correlated_features(df_copy, threshold=0.8)

# print(f"Dropped features: {dropped_features}")
# print(f"New shape: {reduced_df.shape}")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


X = df_copy.drop(columns=["target", "id"])
y = df_copy["target"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# df_copy.describe()


# features_to_scale = [
#     'groove_efficiency_1', 'beat_frequency_1', 'duration_ms_0', 'album_name_length', 'beat_frequency_0',
#     'beat_frequency_2', 'album_component_count', 'duration_ms_2', 'organic_immersion_0',
#     'groove_efficiency_2', 'organic_immersion_2', 'duration_consistency', 'tempo_volatility',
#     'organic_immersion_1', 'groove_efficiency_0', 'duration_ms_1', 'composition_label_0_encoded',
#     'composition_label_2_encoded', 'track_identifier_encoded', 
#     'composition_label_1_encoded', 'creator_collective_encoded', 'year', 'month', 'day'	
# ]

# scaler = StandardScaler()

# X_train_scaled = X_train.copy()
# X_val_scaled = X_val.copy()
# X_test_scaled = test_copy.drop(columns=["id"]).copy()

# X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
# X_val_scaled[features_to_scale] = scaler.transform(X_val[features_to_scale])
# X_test_scaled[features_to_scale] = scaler.transform(X_test_scaled[features_to_scale])


# model = LinearRegression()
# model.fit(X_train_scaled, y_train)   # Train on scaled training data
# y_pred = model.predict(X_val_scaled) # Predict on scaled validation data

# rmse = np.sqrt(mean_squared_error(y_val, y_pred))
# print("Root Mean Squared Error:", rmse)


# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(test_copy.drop(columns=["id"]))


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor  

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)


from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

y_val_pred = grid_search.predict(X_val)

# RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("RMSE:", rmse)

# MAE 
mae = mean_absolute_error(y_val, y_val_pred)
print("MAE:", mae)

# R² 
r2 = r2_score(y_val, y_val_pred)
print("R^2 Score:", r2)


final_model = RandomForestRegressor(
    max_depth=None,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)

final_model.fit(X, y)  


X_test = test_copy.drop(columns=["id"])

test_predictions = final_model.predict(X_test)
test["target"] = test_predictions
submission = test[["id", "target"]]


submission.head()


submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")

