# imports
import pandas as pd
import numpy as np


# Load all the data
train_wildfire = pd.read_csv('/kaggle/input/forest-fire-prediction-epoch-hackathon/wildfire_sizes_before_2010.csv')
train_weather = pd.read_csv('/kaggle/input/forest-fire-prediction-epoch-hackathon/weather_monthly_state_aggregates.csv')
state_data = pd.read_csv('/kaggle/input/forest-fire-prediction-epoch-hackathon/merged_state_data.csv')
submission_template = pd.read_csv('/kaggle/input/forest-fire-prediction-epoch-hackathon/zero_submission.csv')

# Rename columns for consistency
train_wildfire = train_wildfire.rename(columns={'STATE': 'State', 'month': 'year_month'})
submission_template = submission_template.rename(columns={'STATE': 'State', 'month': 'year_month'})


# Merge wildfires and weather
train_data = train_wildfire.merge(train_weather, left_on=['State', 'year_month'], right_on=['State', 'year_month'])
train_data = train_data.merge(state_data, left_on=['State'], right_on=['State'])

# Construct test data (2011–2015) from weather and state data
test_data = submission_template[['State', 'year_month']].copy()
test_data = test_data.merge(train_weather, left_on=['State', 'year_month'], right_on=['State', 'year_month'], how='left')
# weather_test = train_weather[(pd.to_datetime(train_weather['year_month']).dt.year >= 2011) & (pd.to_datetime(train_weather['year_month']).dt.year <= 2015)]
test_data = test_data.merge(state_data, left_on=['State'], right_on=['State'])

# Feature engineering
def extract_features(df):
    df['year'] = pd.to_datetime(df['year_month']).dt.year
    df['month'] = pd.to_datetime(df['year_month']).dt.month
    # df = df.drop('year_month', axis=1)
    df['temp_range'] = df['TMAX'] - df['TMIN']
    # df['dryness_index'] = df['EVAP'] / (df['PRCP'] + 1e-5)  # Avoid division by zero
    df['Percentage Water'] = 100*df['Water Area (sq mi)']/df['Total Area (sq mi)']
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['Percentage of Federal Land'] = df['Percentage of Federal Land'].apply(lambda x: float(x.strip('%')))
    df['Urbanization Rate (%)'] = df['Urbanization Rate (%)']
    df = df.drop(labels=['Water Area (sq mi)', 'Land Area (sq mi)'], axis=1)
    return df

train_data = extract_features(train_data)
test_data = extract_features(test_data)

# Interpolate missing values for each state separately in test_data
numeric_cols = ['TMIN', 'TMAX', 'PRCP', 'EVAP','temp_range', 'mean_elevation', 'month_sin', 'month_cos',
                'Total Area (sq mi)', 'Percentage of Federal Land', 'Urbanization Rate (%)']
for state in test_data['State'].unique():
    state_mask = test_data['State'] == state
    test_data.loc[state_mask, numeric_cols] = test_data.loc[state_mask, numeric_cols].interpolate(method='linear')

# Fill remaining NaNs with state-specific medians from train_data
for state in test_data['State'].unique():
    state_train = train_data[train_data['State'] == state]
    state_test_mask = test_data['State'] == state
    for col in numeric_cols:
        if test_data.loc[state_test_mask, col].isna().any():
            median_val = state_train[col].median() if not state_train[col].isna().all() else train_data[col].median()
            test_data.loc[state_test_mask, col] = test_data.loc[state_test_mask, col].fillna(median_val)

# One-hot encode State column
train_data = pd.get_dummies(train_data, columns=["State"], prefix="State")
test_data = pd.get_dummies(test_data, columns=["State"], prefix="State")

# Align columns between train_df and test_df
all_state_cols = sorted(set(train_data.columns) | set(test_data.columns))
all_state_cols = [col for col in all_state_cols if col.startswith('State_')]
for col in all_state_cols:
    if col not in train_data.columns:
        train_data[col] = False
    if col not in test_data.columns:
        test_data[col] = False

# Ensure test_df has same feature columns as train_df (excluding target)
test_data = test_data[[col for col in train_data.columns if col != 'total_fire_size']]

# Handle missing values (if any)
# train_data.fillna(train_data.median(numeric_only=True), inplace=True)
display(train_data)

display(test_data)


# train_data['Total Area (sq mi)'] = train_data['Total Area (sq mi)'].apply(np.log)
# test_data['Total Area (sq mi)'] = test_data['Total Area (sq mi)'].apply(np.log)


# train_data['mean_elevation'] = (train_data['mean_elevation'] - train_data['mean_elevation'].min())/(train_data['mean_elevation'].max()-train_data['mean_elevation'].min())


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """
    Computes the Kaggle competition metric for predicting forest fire sizes.

    Parameters:
        solution (pd.DataFrame): A DataFrame with columns ["ID", "STATE", "month", "total_fire_size"].
        submission (pd.DataFrame): A DataFrame with columns ["ID", "STATE", "month", "total_fire_size"],
                                  where "ID" is formatted as "STATE_month".

    Returns:
        float: The mean of min(abs(log(pred / true)), 10) over all valid entries.
    """
    # Merge submission with ground truth on (STATE, month)
    merged = solution.merge(submission, on=["STATE", "month"], how="left", suffixes=("_true", "_pred"))

    # Identify missing predictions and assign a score of 10 for them
    missing_pred_mask = merged["total_fire_size_pred"].isna()
    zero_pred_mask = merged["total_fire_size_pred"] <= 0

    # Compute log error where prediction is valid
    valid_pred_mask = ~missing_pred_mask & ~zero_pred_mask
    log_errors = np.full(len(merged), 10.0)  # Default to max penalty

    # Compute actual log error only for valid predictions
    log_errors[valid_pred_mask] = np.abs(np.log(merged.loc[valid_pred_mask, "total_fire_size_pred"] /
                                                 merged.loc[valid_pred_mask, "total_fire_size_true"]))

    # Apply the min operation
    final_scores = np.minimum(log_errors, 10)

    # Return the mean score (if no valid entries, return 10)
    return np.mean(final_scores) if len(final_scores) > 0 else 10.0


from sklearn.ensemble import RandomForestRegressor
# from sklearn_quantile import RandomForestQuantileRegressor
from sklearn.model_selection import train_test_split

train_df = train_data.copy()
test_df = test_data.copy()

# Feature list (excluding dropped columns and target)
features = [col for col in train_df.columns if col not in ['total_fire_size', 'year_month', 'year', 'month']]

# Split for validation
X = train_df[features]
y = train_df['total_fire_size']




# # from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# # outl = IsolationForest(random_state=42)
# outl = LocalOutlierFactor(n_neighbors=20)
# pca = PCA(n_components=5)
[f for f in features if not f.startswith("State_")]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_train[fs])

# X_scaled = pca.fit_transform(X_scaled)

# outlier = outl.fit_predict(X_scaled)
# X_train = X_train[outlier == 1]
# y_train = y_train[outlier==1]


import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor


for i in [1500]:# Train regression model
    X_train, X_val, y_train, y_val = X[:int(.8*len(X))], X[int(.8*len(X)):], y[:int(.8*len(X))], y[int(.8*len(X)):]
    model = RandomForestRegressor(n_estimators=i, random_state=42, max_features='auto')
    # model  = ExtraTreesRegressor(n_estimators=i, random_state=42)
    # model = GaussianProcessRegressor()
    # reg = RandomForestQuantileRegressor(n_estimators=100, random_state=42, quantile=0.5)
    # reg.fit(X_train, np.log(y_train))

    # model = lgb.LGBMRegressor(
    # n_estimators=1000,
    # learning_rate=0.05,
    # num_leaves=31,
    # random_state=42
    # )
    # model.fit(
    # X_train.drop(['month', 'year'], axis=1), np.log(y_train),
    # eval_set=[(X_val.drop(['month', 'year'], axis=1), np.log(y_val))],
    # eval_metric='rmse',
    # )
    model.fit(X_train, np.log(y_train))
    
    # Validation predictions
    val_preds = np.exp(model.predict(X_val))
    
    # Format for scoring (reconstruct State from dummies for validation)
    def get_state_from_dummies(row, state_cols):
        for col in state_cols:
            if row[col] == 1:
                return col.replace('State_', '')
        return None
    
    state_cols = [col for col in train_df.columns if col.startswith('State_')]
    solution_val = pd.DataFrame({
        'ID': [f"{get_state_from_dummies(train_df.iloc[i], state_cols)}_{train_df.iloc[i]['year_month']}" 
               for i in X_val.index],
        'STATE': [get_state_from_dummies(train_df.iloc[i], state_cols) for i in X_val.index],
        'month': train_df.iloc[X_val.index]['year_month'],
        'total_fire_size': y_val
    })
    submission_val = pd.DataFrame({
        'ID': [f"{get_state_from_dummies(train_df.iloc[i], state_cols)}_{train_df.iloc[i]['year_month']}" 
               for i in X_val.index],
        'STATE': [get_state_from_dummies(train_df.iloc[i], state_cols) for i in X_val.index],
        'month': train_df.iloc[X_val.index]['year_month'],
        'total_fire_size': val_preds
    })
    
    # Evaluate
    val_score = score(solution_val, submission_val)
    print(f"Validation Score {i}: {val_score:.4f}")


# Predict on test data
model = RandomForestRegressor(n_estimators=1500, random_state=42)
model.fit(X, np.log(y))
X_test = test_df[features]
test_preds = np.exp(model.predict(X_test))

# Prepare submission (without ID yet)
submission_df = test_df[['year_month']].copy()
submission_df['total_fire_size'] = test_preds
submission_df['State'] = [get_state_from_dummies(test_df.iloc[i], state_cols) for i in range(len(test_df))]
print("Modeling complete! Intermediate submission shape:", submission_df.shape)


display(submission_df)


# add ID column that kaggle wants (order does not matter though, items are match by (STATE, month) pair)
# Clip predictions to avoid zeros or negatives
submission_df['total_fire_size'] = submission_df['total_fire_size'].clip(lower=0.01)

# Create an ID column by assigning a unique integer to each row
submission_df['ID'] = range(len(submission_df))

# Reorder the DataFrame so that 'ID' is the first column
cols = ['ID'] + [col for col in submission_df.columns if col != 'ID']
submission_df = submission_df[cols]

submission_df = submission_df.rename(columns={'State': 'STATE', 'year_month': 'month'})
submission_df = submission_df[['ID', 'STATE', 'month', 'total_fire_size']]

# Save the DataFrame as a CSV file with a header
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'.")

# # order columns
# submission.to_csv('submission.csv', index=False)

display(submission_df)

