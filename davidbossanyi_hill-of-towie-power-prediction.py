import datetime as dt

start_time = dt.datetime.now(tz=dt.UTC)
MAX_SECONDS = 3600 * 12


# Comment these lines out when developing locally
! pip install ephem 'flaml[automl]'
%cd /kaggle/working


import math
from pathlib import Path

import ephem
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
from flaml.automl import AutoML
from kagglehub.config import DEFAULT_CACHE_FOLDER


class SunPosition:
    def __init__(self, *, latitude: float, longitude: float) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self._observer = self._create_ephem_observer()
        self._sun = ephem.Sun()

    def _create_ephem_observer(self) -> ephem.Observer:
        observer = ephem.Observer()
        observer.lat = str(self.latitude)
        observer.lon = str(self.longitude)
        return observer

    def altitude(self, *, timestamp_utc: dt.datetime) -> float:
        self._observer.date = timestamp_utc
        self._sun.compute(self._observer)
        return self._sun.alt


CACHE_DIR = Path(DEFAULT_CACHE_FOLDER) / "competitions" / "hill-of-towie-wind-turbine-power-prediction"


def load_training_dataset(*, force_download: bool = False) -> pl.LazyFrame:
    file_path = kagglehub.competition_download(
        handle="hill-of-towie-wind-turbine-power-prediction",
        path="training_dataset.parquet",
        force_download=force_download,
    )
    return pl.scan_parquet(Path(file_path))


def load_submission_dataset(*, force_download: bool = False) -> pl.LazyFrame:
    file_path = kagglehub.competition_download(
        handle="hill-of-towie-wind-turbine-power-prediction",
        path="submission_dataset.parquet",
        force_download=force_download,
    )
    return pl.scan_parquet(Path(file_path))


def load_turbine_metadata(*, force_download: bool = False) -> pl.LazyFrame:
    file_path = CACHE_DIR / "turbine_metadata.csv"
    if not file_path.exists() or force_download:
        response = requests.get(
            "https://zenodo.org/records/14870023/files/Hill_of_Towie_turbine_metadata.csv?download=1",
            headers={"Accept": "text/csv"},
            timeout=10,
        )
        response.raise_for_status()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(response.content.decode("utf-8-sig"), encoding="utf-8")
    return pl.scan_csv(file_path)


df_train = load_training_dataset().collect()
df_train.head(5)


def preprocess(X: pl.DataFrame, *, ref_wtgs: list[int], lat: float, lon: float) -> pl.DataFrame:
    sun_position = SunPosition(latitude=lat, longitude=lon)
    X = (
        X.lazy()
        .with_columns(
            pl.col("TimeStamp_StartFormat")
            .sub(dt.datetime(2016, 1, 1, tzinfo=dt.UTC))
            .dt.total_seconds()
            .alias("seconds_since_2016"),
            *[
                pl.col(f"wtc_ScYawPos_mean;{wtg}").radians().sin().alias(f"wtc_ScYawPos_mean_sin;{wtg}")
                for wtg in ref_wtgs
            ],
            *[
                pl.col(f"wtc_ScYawPos_mean;{wtg}").radians().cos().alias(f"wtc_ScYawPos_mean_cos;{wtg}")
                for wtg in ref_wtgs
            ],
            *[
                pl.col(f"wtc_AcWindSp_stddev;{wtg}")
                .truediv(f"wtc_AcWindSp_mean;{wtg}")
                .fill_nan(0)
                .alias(f"turbulence_intensity;{wtg}")
                for wtg in ref_wtgs
            ],
            pl.concat_list([pl.col(f"wtc_AmbieTmp_mean;{wtg}") for wtg in ref_wtgs])
            .list.mean()
            .alias("ambient_temp_mean"),
        )
        .collect()
    )
    X = (
        X.lazy()
        .with_columns(
            *[
                pl.col(col).shift(1).alias(col + "_lag10min")
                for col in X.columns
                if col not in ["TimeStamp_StartFormat", "seconds_since_2016"]
            ]
        )
        .with_columns(
            pl.col("TimeStamp_StartFormat").dt.minute().mul(2 * math.pi / 60).sin().alias("minutes_sin"),
            pl.col("TimeStamp_StartFormat").dt.minute().mul(2 * math.pi / 60).cos().alias("minutes_cos"),
            pl.col("TimeStamp_StartFormat").dt.hour().mul(2 * math.pi / 24).sin().alias("hours_sin"),
            pl.col("TimeStamp_StartFormat").dt.hour().mul(2 * math.pi / 24).cos().alias("hours_cos"),
            pl.col("TimeStamp_StartFormat").dt.ordinal_day().mul(2 * math.pi / 365).sin().alias("days_sin"),
            pl.col("TimeStamp_StartFormat").dt.ordinal_day().mul(2 * math.pi / 365).cos().alias("days_cos"),
            pl.col("TimeStamp_StartFormat").dt.month().mul(2 * math.pi / 12).sin().alias("months_sin"),
            pl.col("TimeStamp_StartFormat").dt.month().mul(2 * math.pi / 12).cos().alias("months_cos"),
        )
        .collect()
        .with_columns(
            pl.col("TimeStamp_StartFormat")
            .map_elements(lambda ts: sun_position.altitude(timestamp_utc=ts), return_dtype=pl.Float64)
            .mul(180 / math.pi)
            .alias("sun_altitude"),
        )
    )
    for wtg in ref_wtgs:
        X = append_turbine_status(X, turbine=wtg)
        X = modify_power_by_status(X, turbine=wtg)
    return calculate_icing_indicator(X, ref_wtgs=ref_wtgs)


def append_turbine_status(df: pl.DataFrame, turbine: int) -> pl.DataFrame:
    df_pc = (
        df.lazy()
        .select(f"wtc_AcWindSp_mean;{turbine}", f"wtc_ActPower_mean;{turbine}", f"wtc_AmbieTmp_mean;{turbine}", "id")
        .rename(
            {
                f"wtc_AcWindSp_mean;{turbine}": "ws",
                f"wtc_ActPower_mean;{turbine}": "power",
                f"wtc_AmbieTmp_mean;{turbine}": "temp",
            }
        )
        .filter(pl.col("power").gt(0))
        .with_columns(
            pl.col("ws").truediv(1).round().cast(pl.Int32).mul(1).alias("ws_bin"),
            pl.col("temp").le(1).alias("ice_likely"),
        )
        .collect()
    )
    df_pc_binned = (
        df_pc.lazy()
        .group_by("ws_bin")
        .agg(
            pl.col("ws").mean().alias("ws"),
            pl.col("power").mean().alias("power"),
            pl.col("power").quantile(0.95).alias("power_ub"),
            pl.col("power").quantile(0.05).alias("power_lb"),
        )
        .with_columns(
            pl.when(pl.col("ws").lt(15)).then(pl.col("power_ub")).otherwise(2300).alias("power_ub"),
            pl.when(pl.col("ws").lt(15)).then(pl.col("power_lb")).otherwise(2300).alias("power_lb"),
        )
        .sort("ws")
        .collect()
    )

    df_pc = (
        df_pc.with_columns(
            lower_bound=np.interp(
                df_pc.select("ws").to_numpy().flatten(),
                df_pc_binned.select("ws").to_numpy().flatten(),
                df_pc_binned.select("power_lb").to_numpy().flatten(),
            ),
            upper_bound=np.interp(
                df_pc.select("ws").to_numpy().flatten(),
                df_pc_binned.select("ws").to_numpy().flatten(),
                df_pc_binned.select("power_ub").to_numpy().flatten(),
            ),
        )
        .lazy()
        .with_columns(
            pl.when(pl.col("power").lt(pl.col("lower_bound").mul(0.9)).and_(pl.col("ice_likely")))
            .then(pl.lit(True))  # noqa: FBT003
            .otherwise(pl.lit(False))  # noqa: FBT003
            .alias("ICING"),
            pl.when(pl.col("power").lt(pl.col("lower_bound").mul(0.85)).and_(pl.col("ws").gt(5)))
            .then(pl.lit(True))  # noqa: FBT003
            .otherwise(pl.lit(False))  # noqa: FBT003
            .alias("UNDERPERFORMANCE"),
            pl.when(pl.col("power").gt(pl.col("upper_bound").mul(1.35)).and_(pl.col("ws").gt(3)))
            .then(pl.lit(True))  # noqa: FBT003
            .otherwise(pl.lit(False))  # noqa: FBT003
            .alias("OVERPERFORMANCE"),
        )
        .with_columns(
            pl.when(pl.col("ICING"))
            .then(pl.lit("Icing"))
            .when(pl.col("UNDERPERFORMANCE").or_(pl.col("OVERPERFORMANCE")))
            .then(pl.lit("Abnormal"))
            .otherwise(pl.lit("Normal"))
            .alias(f"status;{turbine}"),
        )
        .collect()
    )
    return df.join(df_pc.select("id", f"status;{turbine}"), on="id", how="left").with_columns(
        pl.col(f"status;{turbine}").fill_null("Offline").alias(f"status;{turbine}")
    )


def filter_is_valid(X: pl.DataFrame, y: pl.Series) -> tuple[pl.DataFrame, pl.Series]:
    y = y.filter(X.select("is_valid").to_series())
    X = X.filter(pl.col("is_valid"))
    return X, y


def modify_power_by_status(X: pl.DataFrame, turbine: int) -> pl.DataFrame:
    return X.with_columns(
        pl.when(pl.col(f"status;{turbine}").eq("Offline"))
        .then(pl.lit(0.0))
        .when(pl.col(f"status;{turbine}").eq("Abnormal"))
        .then(pl.lit(None))
        .otherwise(pl.col(f"wtc_ActPower_mean;{turbine}"))
        .alias(f"wtc_ActPower_mean;{turbine}"),
        pl.when(pl.col(f"status;{turbine}").eq("Offline"))
        .then(pl.lit(0.0))
        .when(pl.col(f"status;{turbine}").eq("Abnormal"))
        .then(pl.lit(None))
        .otherwise(pl.col(f"wtc_ActPower_max;{turbine}"))
        .alias(f"wtc_ActPower_max;{turbine}"),
        pl.when(pl.col(f"status;{turbine}").eq("Offline"))
        .then(pl.lit(0.0))
        .when(pl.col(f"status;{turbine}").eq("Abnormal"))
        .then(pl.lit(None))
        .otherwise(pl.col(f"wtc_ActPower_min;{turbine}"))
        .alias(f"wtc_ActPower_min;{turbine}"),
        pl.when(pl.col(f"status;{turbine}").eq("Offline"))
        .then(pl.lit(0.0))
        .when(pl.col(f"status;{turbine}").eq("Abnormal"))
        .then(pl.lit(None))
        .otherwise(pl.col(f"wtc_ActPower_stddev;{turbine}"))
        .alias(f"wtc_ActPower_stddev;{turbine}"),
    )


def calculate_icing_indicator(X: pl.DataFrame, ref_wtgs: list[int]) -> pl.DataFrame:
    return X.with_columns(
        pl.concat_list([pl.col(f"status;{wtg}").eq("Icing") for wtg in ref_wtgs])
        .list.any()
        .cast(pl.Int32)
        .alias("icing_indicator")
    )


def select_features(X: pl.DataFrame, *, ref_wtgs: list[int]) -> pl.DataFrame:
    cols = [
        *[pl.col(f"wtc_AcWindSp_mean;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"turbulence_intensity;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_AcWindSp_min;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_AcWindSp_max;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_ScYawPos_mean_sin;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_ScYawPos_mean_cos;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_ActPower_mean;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_ActPower_stddev;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_ActPower_min;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_ActPower_max;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_GenRpm_mean;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_PitcPosA_mean;{ref_wtg}") for ref_wtg in ref_wtgs],
        *[pl.col(f"ShutdownDuration;{ref_wtg}").truediv(600) for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_AcWindSp_mean;{ref_wtg}_lag10min") for ref_wtg in ref_wtgs],
        *[pl.col(f"wtc_ActPower_mean;{ref_wtg}_lag10min") for ref_wtg in ref_wtgs],
        pl.col("is_valid_lag10min").fill_null(False),  # noqa: FBT003
        pl.col("icing_indicator"),
        pl.col("ambient_temp_mean"),
        pl.col("sun_altitude"),
        pl.col("seconds_since_2016"),
        pl.col("hours_sin"),
        pl.col("hours_cos"),
        pl.col("days_sin"),
        pl.col("days_cos"),
        pl.col("months_sin"),
        pl.col("months_cos"),
    ]
    return X.select(*cols)


def plot_generalization(
    automl: AutoML,
    *,
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_validation: pl.DataFrame,
    y_validation: pl.Series,
    variable_name: str,
    unit: str,
) -> None:
    train_prediction = pl.Series(values=automl.predict(X_train.to_pandas())).clip(lower_bound=0)
    validation_prediction = pl.Series(values=automl.predict(X_validation.to_pandas())).clip(lower_bound=0)

    mae_train = abs(y_train - train_prediction).mean()
    mae_validation = abs(y_validation - validation_prediction).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(
        x=y_train.to_numpy().flatten(),
        y=train_prediction.to_numpy().flatten(),
        alpha=0.1,
    )
    ax1.text(0.05, 0.95, f"MAE: {mae_train:.2f} {unit}", ha="left", va="top", transform=ax1.transAxes)
    ax1.set_xlabel(f"True {variable_name} [{unit}]")
    ax1.set_ylabel(f"Predicted {variable_name} [{unit}]")
    ax1.set_title("Training set")
    ax1.grid(visible=True)

    ax2.scatter(
        x=y_validation.to_numpy().flatten(),
        y=validation_prediction.to_numpy().flatten(),
        alpha=0.1,
    )
    ax2.text(0.05, 0.95, f"MAE: {mae_validation:.2f} {unit}", ha="left", va="top", transform=ax2.transAxes)
    ax2.set_xlabel(f"True {variable_name} [{unit}]")
    ax2.set_ylabel(f"Predicted {variable_name} [{unit}]")
    ax2.set_title("Validation set")
    ax2.grid(visible=True)


def plot_feature_importance(automl: AutoML) -> None:
    feature_importance = pl.DataFrame(
        {
            "Name": automl.feature_names_in_,
            "Importance": automl.feature_importances_,
        },
    ).sort("Importance")

    fig, ax = plt.subplots(figsize=(10, 0.17 * len(feature_importance)))
    bars = ax.barh(
        y=feature_importance.select("Name").to_numpy().flatten(),
        width=feature_importance.select("Importance").to_numpy().flatten(),
        alpha=0.7,
    )
    ax.bar_label(bars)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_ylim(0.1, len(feature_importance))
    fig.tight_layout()
    print(len(feature_importance), " features used in the model")


X_train_ws = df_train.select(pl.exclude("wtc_AcWindSp_mean;1"))
y_train_ws = df_train.select("wtc_AcWindSp_mean;1").to_series()

wf_lat_lon = load_turbine_metadata().select(pl.col("Latitude").mean(), pl.col("Longitude").mean()).collect()

training_mask = X_train_ws.select(
    pl.col("TimeStamp_StartFormat").lt(dt.datetime(2019, 1, 1, tzinfo=dt.UTC))
).to_series()
X_validation_ws = X_train_ws.filter(~training_mask)
y_validation_ws = y_train_ws.filter(~training_mask)
X_train_ws = X_train_ws.filter(training_mask)
y_train_ws = y_train_ws.filter(training_mask)

X_train_ws = preprocess(
    X_train_ws,
    ref_wtgs=[2, 3, 4, 5, 7],
    lat=wf_lat_lon.select("Latitude").item(),
    lon=wf_lat_lon.select("Longitude").item(),
)

X_train_ws, y_train_ws = filter_is_valid(X_train_ws, y_train_ws)
X_train_ws = select_features(X_train_ws, ref_wtgs=[2, 3, 4, 5, 7])

X_validation_ws = preprocess(
    X_validation_ws,
    ref_wtgs=[2, 3, 4, 5, 7],
    lat=wf_lat_lon.select("Latitude").item(),
    lon=wf_lat_lon.select("Longitude").item(),
)

X_validation_ws, y_validation_ws = filter_is_valid(X_validation_ws, y_validation_ws)
X_validation_ws = select_features(X_validation_ws, ref_wtgs=[2, 3, 4, 5, 7])

automl_ws = AutoML()
automl_settings = {
    "time_budget": 3600 * 4,
    "task": "regression",
    "metric": "mae",
    "estimator_list": [
        "catboost",
    ],
    "log_file_name": "automl_ws.log",
    "seed": 42,
    "eval_method": "cv",
    "n_splits": 5,
    "split_type": "uniform",
    "early_stop": True,
}
automl_ws.fit(
    X_train=X_train_ws.to_pandas(),
    y_train=y_train_ws.to_pandas(),
    **automl_settings,
)


plot_generalization(
    automl_ws,
    X_train=X_train_ws,
    y_train=y_train_ws,
    X_validation=X_validation_ws,
    y_validation=y_validation_ws,
    variable_name="Wind Speed",
    unit="m/s",
)


plot_feature_importance(automl_ws)


X_train = df_train.select(pl.exclude("target"))
y_train = df_train.select("target").to_series()

X_test = load_submission_dataset().collect()

wf_lat_lon = load_turbine_metadata().select(pl.col("Latitude").mean(), pl.col("Longitude").mean()).collect()

training_mask = X_train.select(pl.col("TimeStamp_StartFormat").lt(dt.datetime(2019, 1, 1, tzinfo=dt.UTC))).to_series()
X_validation = X_train.filter(~training_mask)
y_validation = y_train.filter(~training_mask)
X_train = X_train.filter(training_mask)
y_train = y_train.filter(training_mask)

X_train = preprocess(
    X_train,
    ref_wtgs=[2, 3, 4, 5, 7],
    lat=wf_lat_lon.select("Latitude").item(),
    lon=wf_lat_lon.select("Longitude").item(),
)

X_train, y_train = filter_is_valid(X_train, y_train)
X_train = select_features(X_train, ref_wtgs=[2, 3, 4, 5, 7])

assert len([col for col in X_train.columns if col.endswith(";1")]) == 0, (
    "Test turbine features should not be in training set"
)

X_validation = preprocess(
    X_validation,
    ref_wtgs=[2, 3, 4, 5, 7],
    lat=wf_lat_lon.select("Latitude").item(),
    lon=wf_lat_lon.select("Longitude").item(),
)

X_validation, y_validation = filter_is_valid(X_validation, y_validation)
X_validation = select_features(X_validation, ref_wtgs=[2, 3, 4, 5, 7])

X_train = X_train.with_columns(
    engineered_wind_speed=pl.Series(values=automl_ws.model.predict(X_train.to_pandas())).clip(lower_bound=0),
)
X_validation = X_validation.with_columns(
    engineered_wind_speed=pl.Series(values=automl_ws.model.predict(X_validation.to_pandas())).clip(lower_bound=0),
)

seconds_so_far = (dt.datetime.now(tz=dt.UTC) - start_time).total_seconds()
buffer = 7200  # 2 hour buffer

automl = AutoML()
automl_settings = {
    "time_budget": MAX_SECONDS - seconds_so_far - buffer,
    "task": "regression",
    "metric": "mae",
    "estimator_list": [
        "catboost",
    ],
    "log_file_name": "automl.log",
    "seed": 42,
    "eval_method": "cv",
    "n_splits": 5,
    "split_type": "uniform",
    "early_stop": True,
}
automl.fit(
    X_train=X_train.to_pandas(),
    y_train=y_train.to_pandas(),
    **automl_settings,
)


plot_generalization(
    automl,
    X_train=X_train,
    y_train=y_train,
    X_validation=X_validation,
    y_validation=y_validation,
    variable_name="Active Power",
    unit="kW",
)


plot_feature_importance(automl)


automl_ws.model.fit(
    X_train=pl.concat([X_train_ws, X_validation_ws]).to_pandas(),
    y_train=pl.concat([y_train_ws, y_validation_ws]).to_pandas(),
)

X_train = X_train.with_columns(
    engineered_wind_speed=pl.Series(values=automl_ws.model.predict(X_train_ws.to_pandas())).clip(lower_bound=0),
)
X_validation = X_validation.with_columns(
    engineered_wind_speed=pl.Series(values=automl_ws.model.predict(X_validation_ws.to_pandas())).clip(lower_bound=0),
)

automl.model.fit(
    X_train=pl.concat([X_train, X_validation]).to_pandas(),
    y_train=pl.concat([y_train, y_validation]).to_pandas(),
)


train_prediction = pl.Series(values=automl_ws.model.predict(pl.concat([X_train_ws, X_validation_ws]).to_pandas())).clip(
    lower_bound=0
)
mae_train = abs(pl.concat([y_train_ws, y_validation_ws]) - train_prediction).mean()

fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

ax1.scatter(
    x=pl.concat([y_train_ws, y_validation_ws]).to_numpy().flatten(),
    y=train_prediction.to_numpy().flatten(),
    alpha=0.1,
)
ax1.text(0.05, 0.95, f"MAE: {mae_train:.2f}", ha="left", va="top", transform=ax1.transAxes)
ax1.set_xlabel("True Wind Speed [m/s]")
ax1.set_ylabel("Predicted Wind Speed [m/s]")
ax1.set_title("Full Train set")
ax1.grid(visible=True)


train_prediction = pl.Series(values=automl.model.predict(pl.concat([X_train, X_validation]).to_pandas())).clip(
    lower_bound=0
)
mae_train = abs(pl.concat([y_train, y_validation]) - train_prediction).mean()

fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

ax1.scatter(
    x=pl.concat([y_train, y_validation]).to_numpy().flatten(),
    y=train_prediction.to_numpy().flatten(),
    alpha=0.1,
)
ax1.text(0.05, 0.95, f"MAE: {mae_train:.2f} kW", ha="left", va="top", transform=ax1.transAxes)
ax1.set_xlabel("True Active Power [kW]")
ax1.set_ylabel("Predicted Active Power [kW]")
ax1.set_title("Full Train set")
ax1.grid(visible=True)


X_test = load_submission_dataset().collect()
df_id = X_test.select("id")

X_test = preprocess(
    X_test,
    ref_wtgs=[2, 3, 4, 5, 7],
    lat=wf_lat_lon.select("Latitude").item(),
    lon=wf_lat_lon.select("Longitude").item(),
)

X_test = select_features(X_test, ref_wtgs=[2, 3, 4, 5, 7])

X_test = X_test.with_columns(
    engineered_wind_speed=pl.Series(values=automl_ws.model.predict(X_test.to_pandas())).clip(lower_bound=0),
)
y_test = pl.Series(values=automl.model.predict(X_test.to_pandas())).clip(lower_bound=0)

submission = df_id.with_columns(prediction=y_test)


# checking the columns are the expected ones
assert submission.columns == ["id", "prediction"], f'Expected columns ["id", "prediction"], found: {submission.columns}'

# checking no nulls in the data
assert submission.select(pl.col("id").is_null().sum()).item() == 0, "There are null values in the 'id' column"
assert submission.select(pl.col("id").is_nan().sum()).item() == 0, "There are nan values in the 'id' column"
assert submission.select(pl.col("prediction").is_null().sum()).item() == 0, (
    "There are null values in the 'prediction' column"
)
assert submission.select(pl.col("prediction").is_nan().sum()).item() == 0, (
    "There are nan values in the 'prediction' column"
)

# checking the row ids are unique and within expected range
duplicated_ids = submission.select("id").is_duplicated()
assert not duplicated_ids.any(), (
    f"There are duplicated ids: {submission.select('id').filter(duplicated_ids).to_series().unique()}"
)
invalid_ids = set(submission.select("id").unique().to_series().to_list()) - set(range(52704))
assert not invalid_ids, f"The following row IDs are not within the expected ones: {invalid_ids}"

print("Submission file is valid and ready for submission.")

submission.write_csv("submission.csv")

