# List all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd

# Read the Training & Test Sets and set the "row_id" Column as the Index Column
full_training_set: pd.DataFrame = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv", index_col = "row_id")
test_set: pd.DataFrame = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/example_test_files/test.csv", index_col = "row_id")

print("Full Training Set:")
display(full_training_set.head())
print(f"{full_training_set.shape[0]} rows and {full_training_set.shape[1]} columns\n")
print(full_training_set.info())
print()

print("Test Set:")
display(test_set.head())
print(f"{test_set.shape[0]} rows and {test_set.shape[1]} columns\n")


# Confirm the length of the "row_id" column is equal to the number of rows of the Training Set so that it
# can be used as the index for the DataFrame
full_training_set.index.nunique() == full_training_set.shape[0]


full_training_set.dropna(axis = "rows", subset = ["target"], inplace = True)


full_training_targets: pd.Series = full_training_set.pop(item = "target")


# Inspect the size of the Training Set now
print(f"{full_training_set.shape[0]} rows and {full_training_set.shape[1]} columns")


for col in full_training_set.columns:
    # Calculate the no. Missing Values for a Column
    no_miss_values: int = full_training_set[col].isna().sum() 
    
    if no_miss_values > 0:
        # Calculate the proportion of Missing Values for a Column
        proportion_missing_values = (no_miss_values / len(full_training_set[col])) * 100

        print(f"The {col} Column has {no_miss_values} NaN Values which is "
              f"{proportion_missing_values:.2f}% of the Training Set\n")


from sklearn.model_selection import train_test_split

train_data, valid_data, train_targets, valid_targets = train_test_split(
    full_training_set,
    full_training_targets,
    train_size = 0.7,
    test_size = 0.3
)


from sklearn.impute import SimpleImputer
impute_mean = SimpleImputer(strategy = "mean")

# Impute the Missing Values for the remaining Columns by the Average (Mean)
train_data_imputed = pd.DataFrame(impute_mean.fit_transform(train_data))
valid_data_imputed = pd.DataFrame(impute_mean.fit_transform(valid_data))
test_data_imputed = pd.DataFrame(impute_mean.fit_transform(test_set))

# Assign back the Index Column Names to the Training, Validation & Test Sets after they were removed after Imputation
train_data_imputed.index = train_data.index
train_data_imputed.columns = train_data.columns

valid_data_imputed.index = valid_data.index
valid_data_imputed.columns = valid_data.columns

test_data_imputed.index = test_set.index
test_data_imputed.columns = test_set.columns


# Inspect the Training, Validation & Test Data now
print("Training Set:")
display(train_data_imputed.head())
print(f"{train_data_imputed.shape[0]} rows and {train_data_imputed.shape[1]} columns\n")
print()

print("Validation Set:")
display(valid_data_imputed.head())
print(f"{valid_data_imputed.shape[0]} rows and {valid_data_imputed.shape[1]} columns\n")
print()

print("Test Set:")
display(test_data_imputed.head())
print(f"{test_data_imputed.shape[0]} rows and {test_data_imputed.shape[1]} columns\n")


test_data_imputed.drop(labels = ["time_id"], axis = "columns", inplace = True)


from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

def best_params_lgbm(lr, num_estimators):
    """
    Training & Validation Steps:
    1. Initialise the LightGBM Algorithm with a set learning rate and number of estimators according to
    the argument supplied to the parameters
    2. Fit the LightGBM Algorithm on the Training Data & Targets
    3. Use the Trained LightGBM Model to make predictions on the Validation Set
    4. Return the Mean Absolute Error (MAE) of the Validation Set predictions
    """
    lgbm_model = LGBMRegressor(
        learning_rate = lr,
        n_estimators = num_estimators,
        objective = "mae",
        device_type = "gpu"
    )

    lgbm_model.fit(train_data_imputed, train_targets)

    valid_preds = lgbm_model.predict(valid_data_imputed)
    return mean_absolute_error(valid_targets, valid_preds)


# Set the range of estimators to be: [150, 250]
# Set a fixed learning rate to be the default: 0.1
estimator_results: dict = {num_estimators: best_params_lgbm(0.1, num_estimators) for num_estimators in range(150, 250 + 10, 10)}

# Show all the results
print("Results:")
display(estimator_results)

from statistics import stdev
print("\nStandard Deviation:", stdev(estimator_results.values()))
print()

# Get the number of estimators that produced the lowest MAE
best_no_estimators: float = min(estimator_results, key = estimator_results.get)

print(f"Best No. LightGBM Estimators: {best_no_estimators} with an MAE of: {(estimator_results[best_no_estimators]):.4f}")


import numpy as np

# Set the learning rate range to be: [0.05, 0.15]
# Use the best number of estimators from above 
lr_results: dict = {learning_rate: best_params_lgbm(learning_rate, best_no_estimators) for learning_rate in np.arange(0.05, 0.15 + 0.01, 0.01)}

# Show all the results
print("Results:")
display(lr_results)

from statistics import stdev
print("\nStandard Deviation:", stdev(lr_results.values()))
print()

# Get the learning rate that produced the lowest MAE
best_lr: float = min(lr_results, key = lr_results.get)

print(f"Best Learning Rate: {best_lr} with an MAE of: {(lr_results[best_lr]):.4f}")


# Impute the Missing Values of the Full Training set for the remaining Columns by the Average (Mean)
full_training_set_imputed = pd.DataFrame(impute_mean.fit_transform(full_training_set))

# Assign back the Index Column Names to the Full Training Set after they were removed after Imputation
full_training_set_imputed.index = full_training_set.index
full_training_set_imputed.columns = full_training_set.columns


from sklearn.pipeline import make_pipeline

lgbm_model = make_pipeline(
    LGBMRegressor(
        # Use the best parameter values found in the previous section
        learning_rate = best_lr,
        n_estimators = best_no_estimators,
        objective = "mae",
        device_type = "gpu"
    )
)

lgbm_model.fit(full_training_set_imputed, full_training_targets)


test_preds: np.ndarray = lgbm_model.predict(test_data_imputed)
test_preds


import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()


count = 0
for (test, revealed_targets, sample_prediction) in iter_test:
    sample_prediction["target"] = lgbm_model.predict(test.drop("row_id", axis = "columns"))
    env.predict(sample_prediction)
    count += 1

