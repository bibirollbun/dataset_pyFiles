import altair as alt

# alt.renderers.enable("jupyter", offline=True)
alt.data_transformers.disable_max_rows()

from pathlib import Path
from dataclasses import dataclass
import altair as alt
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@dataclass
class CFG:
    train_path: Path = Path("/kaggle/input/playground-series-s5e4/train.csv")
    test_path: Path = Path("/kaggle/input/playground-series-s5e4/test.csv")
    pltpd_path: Path = Path("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")

cfg = CFG()


def plot_point(df: pl.DataFrame):
    transforms = [pl.col(col).cast(pl.Float64) for col in ["Episode_Length_minutes", "Listening_Time_minutes"]]
    df = df.with_columns(transforms)

    print("Length:", len(df))
    return df.plot.point(
        x="Episode_Length_minutes",
        y="Listening_Time_minutes",
    ).properties(
        width=600,
        height=300,
    )


def plot_linear_regression(df: pl.DataFrame, col_x = "Episode_Length_minutes"):
    print("len", len(df))
    df_f = df.filter(~pl.col(col_x).is_null())
    X = df_f.select(col_x).to_numpy()
    y = df_f.select("Listening_Time_minutes").to_numpy()

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y, predictions))

    print(f"Coefficient: {model.coef_[0][0]:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    print(f"RMSE: {rmse:.4f}")

    df_f = df_f.with_columns(pl.Series("Predicted", predictions.flatten()))
    if len(df_f) > 10000:
        df_f = df_f.sample(10000)

    scatter_plot = (
        alt.Chart(df_f)
        .mark_circle()
        .encode(
            x=alt.X(col_x),
            y=alt.Y("Listening_Time_minutes"),
            tooltip=[col_x, "Listening_Time_minutes"],
        )
    )

    regression_line = alt.Chart(df_f.to_pandas()).mark_line(color="red").encode(x=col_x, y="Predicted")

    chart = (scatter_plot + regression_line).properties(
        width=600,
        height=300,
    )

    return chart


def display_three_dfs(filter_con: pl.expr.expr.Expr, col_x = "Episode_Length_minutes"):
    global df_train, df_pltpd, df_test
    
    display(df_train.filter(filter_con))
    if len(df_train.filter(filter_con)) > 0:
        display(plot_linear_regression(df_train.filter(filter_con), col_x))
        
    display(df_pltpd.filter(filter_con))
    if len(df_pltpd.filter(filter_con)) > 0:
        display(plot_linear_regression(df_pltpd.filter(filter_con), col_x))

    display(df_test.filter(filter_con))
    

df_test = pl.read_csv(cfg.test_path)

df_train = pl.read_csv(cfg.train_path)

df_pltpd = pl.read_csv(cfg.pltpd_path)
df_pltpd = df_pltpd.drop_nulls(subset=["Listening_Time_minutes"])
df_pltpd = df_pltpd.with_columns(pl.col("Number_of_Ads").cast(pl.Float64))
df_pltpd = df_pltpd.with_columns(pl.Series(range(1_000_000, 1_000_000 + len(df_pltpd))).alias("id"))



plot_linear_regression(df_train)


plot_linear_regression(df_pltpd)


df_pltpd = df_pltpd.filter(pl.col("Episode_Length_minutes") != pl.col("Listening_Time_minutes"))
plot_linear_regression(df_pltpd)


df_train = pl.read_csv(cfg.train_path, infer_schema_length=0).with_columns(pl.all().cast(pl.String, strict=False))
df_pltpd = pl.read_csv(cfg.pltpd_path, infer_schema_length=0).with_columns(pl.all().cast(pl.String, strict=False))
df_test = pl.read_csv(cfg.test_path, infer_schema_length=0).with_columns(pl.all().cast(pl.String, strict=False))

df_train = df_train.drop_nulls(subset=["Episode_Length_minutes", "Number_of_Ads", "Guest_Popularity_percentage", "Listening_Time_minutes"])
df_pltpd = df_pltpd.drop_nulls(subset=["Episode_Length_minutes", "Number_of_Ads", "Guest_Popularity_percentage", "Listening_Time_minutes"])
df_test = df_test.drop_nulls(subset=["Episode_Length_minutes", "Guest_Popularity_percentage"])

df_pltpd = df_pltpd.with_columns(pl.col("Number_of_Ads").cast(pl.Float64).cast(pl.String))

transforms = []
for col in ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads", "Listening_Time_minutes"]:
    transforms += [
        pl.when(pl.col(col).str.contains("\\."))
            .then(
                pl.col(col).str.extract(r"\.(\d+)$").str.len_chars().cast(pl.Int64)
            )
            .otherwise(pl.lit(0))
            .alias(f"{col}_Decimal_Len"),
        
        pl.when(pl.col(col).str.contains("\\."))
            .then(
                pl.col(col).str.extract(r"\.(\d+)$").cast(pl.Int64)
            )
            .otherwise(pl.lit(0))
            .alias(f"{col}_Decimal"),
        
        pl.col(col).cast(pl.Float64)
    ]

df_train = df_train.with_columns(transforms)
df_pltpd = df_pltpd.with_columns(transforms)
df_test = df_test.with_columns(transforms[:-3])

display(df_train)
display(df_pltpd)
display(df_test)


filter_con = (pl.col("Episode_Length_minutes_Decimal_Len") > 2)
display_three_dfs(filter_con)


df_same = df_train.filter((pl.col("Episode_Length_minutes_Decimal_Len") > 5) & (pl.col("Episode_Length_minutes").round(3) == pl.col("Listening_Time_minutes").round(3)))
df_diff = df_train.filter((pl.col("Episode_Length_minutes_Decimal_Len") > 5) & (pl.col("Episode_Length_minutes").round(3) != pl.col("Listening_Time_minutes").round(3)))
display(df_same)
display(df_diff)


plot_linear_regression(df_diff)


df_same.describe()


df_diff.describe()


df_same["Episode_Length_minutes_Decimal_Len"].plot.hist()


df_diff["Episode_Length_minutes_Decimal_Len"].plot.hist()


filter_con = (pl.col("Episode_Length_minutes_Decimal_Len") >= 5)
display_three_dfs(filter_con)


filter_con = (pl.col("Number_of_Ads") > 3.0)
display_three_dfs(filter_con, col_x="Number_of_Ads")




