import pandas as pd
import numpy as np


# Load the data
train_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/train_df.csv')
test_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/test_df.csv')
submit_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/submission.csv')


# Combine train and test data into a single DataFrame
all_df = pd.concat([train_df, test_df]).reset_index(drop=True)

# Convert datetime column to datetime type
all_df['datetime'] = pd.to_datetime(all_df['datetime'])

# Convert datetime column to datetime type
all_df['datetime'] = pd.to_datetime(all_df['datetime'])

# Apply log transformation to promotion features
all_df['promotion_1_log'] = np.log1p(all_df['promotion_1'])
all_df['promotion_2_log'] = np.log1p(all_df['promotion_2'])
all_df['promotion_3_log'] = np.log1p(all_df['promotion_3'])


%%time

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Define exogenous variables and target variable
exog_vars = [
    'promotion_1',
    'promotion_2_log',
    'promotion_3_log',
]
target_var = 'e_users'

# Cross-validation time splits (train/test periods)
cv_configs = [
    # train
    {
        "train_end": "2022-12-31 23:00:00",
        "test_start": "2023-01-01 00:00:00",
        "test_end": "2023-04-30 23:00:00",
    },
    {
        "train_end": "2023-04-30 23:00:00",
        "test_start": "2023-05-01 00:00:00",
        "test_end": "2023-08-31 23:00:00",
    },
    {
        "train_end": "2023-08-31 23:00:00",
        "test_start": "2023-09-01 00:00:00",
        "test_end": "2023-12-31 23:00:00",
    },
    {
        "train_end": "2023-12-31 23:00:00",
        "test_start": "2024-01-01 00:00:00",
        "test_end": "2024-04-30 23:00:00",
    },
    {
        "train_end": "2024-04-30 23:00:00",
        "test_start": "2024-05-01 00:00:00",
        "test_end": "2024-08-31 23:00:00",
    },
    # test
    {
        "train_end": "2024-08-31 23:00:00",
        "test_start": "2024-09-01 00:00:00",
        "test_end": "2024-12-31 23:00:00",
    },
]

results = []

# Loop through each cross-validation fold
for i, cfg in enumerate(cv_configs, 1):
    train =all_df[(all_df['datetime'] <= cfg["train_end"])].copy()
    test = all_df[(all_df['datetime'] >= cfg["test_start"]) & (all_df['datetime'] <= cfg["test_end"])].copy()

    y_train = train.set_index('datetime')[target_var].asfreq('h')
    X_train = train.set_index('datetime')[exog_vars].asfreq('h')

    y_test = test.set_index('datetime')[target_var].asfreq('h')
    X_test = test.set_index('datetime')[exog_vars].asfreq('h')

    try:
        # Fit SARIMAX model
        model = SARIMAX(
            y_train, exog=X_train,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 24),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit()

        # Forecast and evaluate using RMSE
        y_pred = model_fit.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(i, rmse)
        print(model_fit.summary())

        # Save result for current fold
        results.append({
            "fold": i,
            "train_end": cfg["train_end"],
            "test_start": cfg["test_start"],
            "test_end": cfg["test_end"],
            "rmse": rmse
        })

    except Exception as e:
        # Handle and record errors
        results.append({
            "fold": i,
            "error": str(e)
        })

# Extract RMSE values (exclude final test fold if needed)
rmse_values = [entry['rmse'] for entry in results[:-1]]

# Compute mean RMSE
mean_rmse = np.mean(rmse_values)

print(f"RMSE: {mean_rmse:.2f}")


# Add predictions to the submission DataFrame and export to CSV
submit_df['e_users'] = y_pred.reset_index()['predicted_mean']
submit_df.to_csv('submission.csv', index=False)

