!pip install -qU "polars[plot]"
# for altair charts
!pip install -qU "vegafusion-python-embed>=1.5.0" "vegafusion>=1.5.0" "vl-convert-python>=1.6.0"


import polars as pl
import altair as alt
import optuna  # for hyperparameter tuning


alt.data_transformers.enable("vegafusion")


train = pl.read_csv("/kaggle/input/playground-series-s5e1/train.csv", try_parse_dates=True).with_columns(
        pl.col("num_sold").cast(pl.Int64),
    )
test = pl.read_csv("/kaggle/input/playground-series-s5e1/test.csv", try_parse_dates=True)
print(train.shape, test.shape)
train.head()


print(train.shape, test.shape)


# null_count in dataframe
train.select(pl.all().null_count())


# Distribution of sales across country for selected year
(
    train.filter(
        pl.col("date").dt.year().eq(2014),  # selected year
    )
    .group_by("date", "country")
    .agg(pl.sum("num_sold"))
    .plot.line("date", "num_sold", color="country")
    .properties(width=500)
)


# Distribution of sales across different stores for selected year
(
    train.filter(
        pl.col("date").dt.year().eq(2014),  # selected year
    )
    .group_by("date", "store")
    .agg(pl.sum("num_sold"))
    .plot.line("date", "num_sold", color="store")
    .properties(width=500)
)


def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    min_max_scaler = lambda expr, name: expr.sub(pl.max(name)).truediv(pl.max(name) - pl.min(name))

    return (
        df.with_columns(
            # Fetaures from date column
            year=pl.col("date").dt.year(),
            month=pl.col("date").dt.month(),
            weeknum=pl.col('date').dt.week(),
            day_of_year=pl.col("date").dt.ordinal_day(),
            is_weekend=pl.col("date").dt.weekday().is_in([6, 7]).cast(pl.UInt8),
        )
        .with_columns(
            pl.col("year").pipe(min_max_scaler, "year"),
            pl.col("month").pipe(min_max_scaler, "month"),
            pl.col("day_of_year").pipe(min_max_scaler, "day_of_year"),
            pl.col("weeknum").pipe(min_max_scaler, "weeknum"),
        )
        .to_dummies(("country", "store", "product"))
    )


import math


def __add_fourier_terms(df: pl.DataFrame, period: int, terms: int) -> pl.DataFrame:
    exprs = []
    for k in range(1, terms + 1):
        exprs += [
            pl.col("day_of_year").truediv(period).mul(2*math.pi*k).sin().alias(f"sin_{k}_{period}"),
            pl.col("day_of_year").truediv(period).mul(2*math.pi*k).cos().alias(f"cos_{k}_{period}"),
        ]
    return df.with_columns(exprs)


def training_fetaure_engineering(df: pl.DataFrame) -> pl.DataFrame:
    add_lag = lambda expr, lag: expr.shift(lag).backward_fill(lag).alias(f"target_lag_{lag}")
    add_rolling = lambda expr, period: expr.rolling("date", period=period).alias(f"target_rolling_{period}")
    standard_scaler = lambda expr, name: expr.sub(pl.mean(name)).truediv(pl.std(name))

    return (
        df
        .with_columns(
            pl.col("num_sold").pipe(add_lag, 7),
            pl.col("num_sold").pipe(add_lag, 14),
            pl.col("num_sold").pipe(add_lag, 30),
            # rolling mean
            pl.col("num_sold").sum().pipe(add_rolling, "7d"),
            pl.col("num_sold").sum().pipe(add_rolling, "14d"),
            pl.col("num_sold").sum().pipe(add_rolling, "30d"),
        )
        .pipe(__add_fourier_terms, period=365, terms=3)  # yearly seasonality
        .pipe(__add_fourier_terms, period=7, terms=2)  # weekly seasonality
        .with_columns(
            [
                pl.col(name).pipe(standard_scaler, name) for name in 
                 ("target_lag_7", "target_lag_14", "target_lag_30",
                  "target_rolling_7d", "target_rolling_14d", "target_rolling_30d")
            ]
        )
    )


_train_data = (
    train.drop_nulls("num_sold")
    .pipe(feature_engineering)
    # Currently I don't know how to make prediction on this data
    # .pipe(training_fetaure_engineering)
    .drop("id", "date")  # dropping "date" column
)
X = _train_data.drop("num_sold")
y = _train_data["num_sold"]
X.sample(5, seed=42)


# Cross val score function
import time
import functools
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error

k_fold = KFold(7, shuffle=False)
scoring = "neg_mean_absolute_percentage_error"

def calc_cv_time(func):
    """Calculate cross validation time and print right there."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        scores = func(*args, **kwargs)
        final = time.perf_counter() - start

        print(f"Time taken: {final:.2f} seconds")
        print(f"Score Mean: {scores.mean():.4f}")
        print(f"Score Std: {scores.std():.4f}")

        return scores
    
    return wrapper


@calc_cv_time
def calc_cv_scores(X: pl.DataFrame, y: pl.Series, model):
    scores = cross_val_score(model, X, y, cv=k_fold, scoring=scoring)
    return scores


from sklearn.linear_model import Ridge

calc_cv_scores(X, y, Ridge(tol=1e-2, max_iter=10_00_000, random_state=42))


from sklearn.ensemble import RandomForestRegressor

calc_cv_scores(X, y, RandomForestRegressor(random_state=42))


from sklearn.ensemble import HistGradientBoostingRegressor

calc_cv_scores(X, y, HistGradientBoostingRegressor(scoring=scoring, random_state=42))


# Use optuna for hyperparameter tuning of model best model
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 5, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "max_depth": trial.suggest_int("max_depth", 10, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.001, 3),
    }

    scores = calc_cv_scores(X, y, HistGradientBoostingRegressor(**params, scoring=scoring, random_state=42))
    return scores.mean()


# Create study object and optimize the `objective` function
# We aim to maximize accuracy
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
# Run 50 trials to find the best hyperparameters
study.optimize(objective, n_trials=7)


model = HistGradientBoostingRegressor()#**study.best_trial.params, random_state=42)
model.fit(X, y)


from sklearn.metrics import mean_absolute_percentage_error, r2_score

y_pred = model.predict(X)
print(
    "Train scores:",
    mean_absolute_percentage_error(y, y_pred),
    r2_score(y, y_pred),
)


test_data = test.pipe(feature_engineering)
test_data.head()


submission_data = test_data.select(
    "id",
    num_sold=pl.lit(model.predict(test_data.drop("id", "date"))).ceil().cast(pl.Int64),
)
submission_data.head()


submission_data.write_csv("/kaggle/working/submission.csv")


pl.scan_csv("/kaggle/working/submission.csv").head(10).collect()

