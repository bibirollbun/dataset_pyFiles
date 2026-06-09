!pip install "flaml[automl]"
!pip install "catboost"


from flaml import AutoML


# Use FLAML for automated model and hyperparameter optimization
automl = AutoML()


from pathlib import Path
import pandas as pd
import warnings; warnings.simplefilter("ignore", RuntimeWarning)

DATA_DIR = Path("/kaggle/input/hill-of-towie-wind-turbine-power-prediction")
OUTPUT_DIR = Path("/kaggle/working")
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data_fpath = DATA_DIR / "training_dataset.parquet"
submission_data_fpath = DATA_DIR / "submission_dataset.parquet"


TARGET_COL = "target"

class Cols:
    """Class to hold the column names of the dataset. Which allows for tab-completion and minimizes typos."""

    TIMESTAMP = "TimeStamp_StartFormat"
    TURBINE_ID = "turbine_id"
    WINDSPEED_MEAN = "wtc_AcWindSp_mean"
    ACTIVEPOWER_MEAN = "wtc_ActPower_mean"
    OP_TIMEON = "wtc_ScReToOp_timeon"
    SHUTDOWN_DURATION = "ShutdownDuration"


input_df = pd.read_parquet(train_data_fpath)
input_df.head(3)


test_fraction = 0.3
ts_min, ts_max = input_df[Cols.TIMESTAMP].min(), input_df[Cols.TIMESTAMP].max()
test_subset_start = ts_min + (ts_max - ts_min) * (1 - test_fraction)
is_train_subset = input_df[Cols.TIMESTAMP] < test_subset_start

feature_cols = [i for i in input_df.columns if not i.endswith(";1") and i != TARGET_COL]


def _split_into_X_y(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return d[feature_cols], d[TARGET_COL].mask(~d["is_valid"])


# Filter out invalid rows and rows with NaN values
input_df = input_df.dropna()
input_df = input_df[input_df["is_valid"] == True]


X_train, y_train = input_df[is_train_subset].pipe(_split_into_X_y)
X_test, y_test = input_df[~is_train_subset].pipe(_split_into_X_y)


display(X_train.head(3))
display(y_train.head(3))


display(X_test.head(3))
display(y_test.head(3))


FULL_10MIN = 600  # full counter value for 10min interval
other_turbines_ids = [2, 3, 4, 5, 7]
other_turbines_power_cols = [f"{Cols.ACTIVEPOWER_MEAN};{i}" for i in other_turbines_ids]
other_turbines_op_time_cols = [f"{Cols.OP_TIMEON};{i}" for i in other_turbines_ids]
other_turbines_shutdown_duration_cols = [f"{Cols.SHUTDOWN_DURATION};{i}" for i in other_turbines_ids]


# we could write it as a simple function...
def mean_active_power_of_fully_operating_non_target_turbines(X: pd.DataFrame) -> pd.Series:
    """Calculate mean of Active Power for all other turbines (non-target) that are fully operating."""
    others_active_power = X[other_turbines_power_cols]
    others_is_fully_operating = X[other_turbines_op_time_cols] == FULL_10MIN
    others_not_shutdown = X[other_turbines_shutdown_duration_cols] == 0
    is_ok = others_is_fully_operating.to_numpy() & others_not_shutdown.to_numpy()
    return (
        others_active_power
        .mask(~is_ok)  # if "not ok", correponding data will be set to NaN
        .mean(axis=1)  # mean across all non-target turbines
        .clip(0)       # clipping with lower bound at 0
    )


# but better yet, we can make it a class in the style of the scikit-learn library
class SimpleModel:
    """Simple model that takes Active Power from all other fully operating wind turbines and averages them."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SimpleModel":
        # this simple model does not need training (coefficient fitting)
        # but this is where you can put your logic to train your model
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return mean_active_power_of_fully_operating_non_target_turbines(X)


simple_model = SimpleModel()
simple_model.fit(X=X_train, y=y_train)
y_pred = simple_model.predict(X_test)


mean_absolute_error = (y_pred - y_test).abs().mean()
print(f"The MAE for model is {mean_absolute_error:.2f}")


ax = pd.DataFrame({"predicted": y_pred, "actual": y_test}).plot.scatter(
    x="actual", y="predicted", s=1, alpha=0.2, grid=True
)
ax.plot([0, y_test.max()], [0, y_test.max()], "--r");  # adding red 1-to-1 line


X_train.columns[1:-2]


# Define AutoML settings
automl_settings = {
    "time_budget": 60*10,             # Seconds to spend
    "ensemble": True,                # Use ensemble of best
    "task": "regression",           # Type of task
    "metric": "mae",               # Evaluation metric
    #"estimator_list": ["catboost", "xgboost", "xgb_limitdepth", "lgbm", "rf", "extra_tree"],
    "estimator_list": ["catboost", "lgbm"],
    "log_file_name": "automl.log",  # Save logs
    "seed": 42,
}


# Fit AutoML
automl.fit(X_train=X_train[X_train.columns[1:-2]], y_train=y_train, **automl_settings)



y_test_pred = automl.predict(X_test)



mean_absolute_error = (y_test_pred - y_test).abs().mean()
print(f"The MAE for model is {mean_absolute_error:.2f}")


ax = pd.DataFrame({"predicted": y_test_pred, "actual": y_test}).plot.scatter(
    x="actual", y="predicted", s=1, alpha=0.2, grid=True
)
ax.plot([0, y_test.max()], [0, y_test.max()], "--r");  # adding red 1-to-1 line


X_submission = pd.read_parquet(submission_data_fpath).set_index("id")
X_submission.head(3)


submission_predictions = simple_model.predict(X_submission)
submission_predictions


submission_predictions = automl.predict(X_submission)


submission_predictions = pd.DataFrame.from_dict({"id":X_submission.index,"prediction":submission_predictions}).set_index("id")


submission_predictions


output_fpath = (OUTPUT_DIR / "sample_model_submission.csv").as_posix()

(submission_predictions.fillna(0).to_csv(output_fpath))
print(f"Submission file saved to {output_fpath}")


!head -n 5 {output_fpath}


_df = pd.read_csv(output_fpath)

# checking the columns are the expected ones
assert _df.columns.to_list() == ["id", "prediction"], (
    f'Expected columns ["id", "prediction"], found: {_df.columns.to_list()}'
)

# checking no nulls in the data
assert _df.isna().sum().sum() == 0, "There are NA values in the data!"

# checking the row ids are unique and within expected range
duplicated_ids = _df["id"].duplicated()
assert not duplicated_ids.any(), f"There are duplicated ids: {_df['id'][duplicated_ids].values}"
invalid_ids = set(_df["id"].unique()) - set(range(52704))
assert not invalid_ids, f"The following row IDs are not within the expected ones: {invalid_ids}"




