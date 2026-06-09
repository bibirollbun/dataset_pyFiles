!pip install hvplot scikit_learn==1.7.0 -q


import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

import hvplot
import hvplot.polars
from hvplot.polars import hvPlotTabularPolars as hvpl
from tqdm.auto import tqdm

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

sklearn.set_config(transform_output='polars')

random_state = 42

LOG_PRICE = True


from typing import overload, Literal


@overload
def clean_names(
    df: pl.DataFrame,
    remove_special: bool = False,
    truncate=-1,
) -> pl.DataFrame: ...


@overload
def clean_names(
    df: pl.LazyFrame,
    remove_special: bool = False,
    truncate=-1,
) -> pl.LazyFrame: ...


def clean_names(
    df,
    remove_special: bool = False,
    truncate=-1,
):
    columns = pl.Series(df.collect_schema().names())
    if remove_special:
        columns = columns.str.replace_all(r"[^A-Za-z0-9_]", " ")
    columns = (
        columns.str.normalize("NFKC")
        .str.strip_chars()
        .str.replace_all('"', "")
        .str.replace_all("'", "")
        # credits: [@jtaylor](https://gist.github.com/jaytaylor/3660565)
        .str.replace_all(r"(.)([A-Z][a-z]+)", r"${1}_${2}")
        .str.replace_all(r"([a-z0-9])([A-Z])", r"${1}_${2}")
        .str.replace_all(r"[ _]+", "_")
        .str.to_lowercase()
    )
    if truncate > 0:
        columns = columns.str.slice(0, truncate)
    return df.rename({k: s for k, s in zip(df.collect_schema(), columns)})


def scantt_csv(
    train_source,
    test_source,
    *,
    indicator_column="train",
    rechunk=False,
    **scan_csv_kwa,
) -> pl.LazyFrame:
    """Lazily read from the training and testing CSV files."""
    train = pl.scan_csv(train_source, **scan_csv_kwa)
    test = pl.scan_csv(test_source, **scan_csv_kwa)

    train_schema = train.collect_schema()
    test_schema = test.collect_schema()
    y_columns = [name for name in train_schema if name not in test_schema]
    return pl.concat(
        [
            train.select(
                pl.exclude(y_columns),
                pl.col(y_columns),
                pl.lit(True).alias(indicator_column),
            ),
            test.with_columns(
                [pl.lit(None).alias(name) for name in y_columns]
                + [pl.lit(False).alias(indicator_column)]
            ),
        ],
        rechunk=rechunk,
    )


def feature_influence(
    model,
    return_as: Literal["polars", "pandas", "list", "dict"] = "polars",
):
    """Extract feature importances or coefficients from a trained model, returning a list, whose elements appear in the following order:
    - If `model` has `feature_names_in`, include it.
    - If `model` has `feature_importances_`, include it.
    - If `model` has `coef_`, include it.
    """

    # TransformedTargetRegressor and TransformedTargetClassifier
    if (reg := getattr(model, "regressor_", None)) or (clf := getattr(model, "classifier_", None)):
        model = reg or clf

    attrs = {
        "feature_name": [
            "feature_names_in_",
            "feature_names_",
        ],
        "feature_influence": [
            "feature_importances_",
            "coef_",
        ],
    }
    result = {}
    for att, aliases in attrs.items():
        for alias in aliases:
            if (v := getattr(model, alias, None)) is not None:
                result[att] = v

    if return_as == "polars":
        import polars as pl

        df = pl.DataFrame(result)
        if "feature_name" in df.columns:
            return df.sort("feature_influence", descending=True)
        return df

    elif return_as == "pandas":
        import pandas as pd

        df = pd.DataFrame(result)
        if "feature_name" in df.columns:
            return df.sort_values("feature_influence", ascending=False)
        return df

    elif return_as == "list":
        return list(result.values())
    return result


def winkler_interval_score(
    y_true: np.ndarray,
    y_pred_low: np.ndarray,
    y_pred_high: np.ndarray,
    mean=True,
    expm1=LOG_PRICE,
) -> float:
    """Winkler Interval score with alpha=0.1"""
    if expm1:
        y_true = np.expm1(y_true)
        y_pred_low = np.expm1(y_pred_low)
        y_pred_high = np.expm1(y_pred_high)

    s = np.abs(y_pred_high - y_pred_low) + (2 / 0.1) * (
        np.maximum(y_true - y_pred_high, 0) + np.maximum(y_pred_low - y_true, 0)
    )
    return np.mean(s) if mean else s




df = clean_names(
    pl.concat(
        [
            scantt_csv(
                "/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv",
                "/kaggle/input/prediction-interval-competition-ii-house-price/test.csv",
                schema_overrides={
                    "sale_date": pl.Date,
                },
            ),
            pl.scan_csv("/kaggle/input/kingcountysales/kingco_sales.csv")
            .drop("sale_id", "pinx", "")
            .filter(pl.col("grade").ne(20))
            .with_columns(
                pl.lit(True).alias("train"),
                pl.lit(0).alias("id"),
                pl.col("sale_date").cast(pl.Date),
                pl.col("sale_nbr").replace({"NA": None}).cast(float),
            ),
        ],
        how="align",
        rechunk=True,
    )
    .select(pl.exclude("id").shrink_dtype())
    .with_columns(  ### Normalize city names
        pl.col("city").replace(
            {
                "BEAUX ARTS": "BEAUX ARTS VILLAGE",
                "SEA-TAC": "SEATAC",
                "SeaTac": "SEATAC",
            }
        )
    )
).collect()
if LOG_PRICE:
    df = df.with_columns(pl.col("sale_price", "land_val", "imp_val").log1p())
df.head()


from sklearn.neighbors import NearestNeighbors


def get_nearest_transit_distances(X, k=1):
    df_bus = clean_names(pl.read_csv("/kaggle/input/pi-ii-extras/gtfs_stops.txt"))
    est = NearestNeighbors(n_neighbors=k, metric="haversine").fit(
        df_bus[["stop_lat", "stop_lon"]].to_numpy()
    )
    dists, _ = est.kneighbors(X)
    return dists


k = 1
df_transit_d = pl.from_numpy(
    get_nearest_transit_distances(df[["latitude", "longitude"]], k),
    schema=[f"dist_metro{i}" for i in range(k)],
)
df = pl.concat([df, df_transit_d], how="horizontal")
df.head()


from itertools import product
import json

with open("/kaggle/input/pi-ii-extras/warning_codes.json", "r") as f:
    warning_codes = json.load(f)

with open("/kaggle/input/pi-ii-extras/zoning_type_map.json", "r") as f:
    zoning_type_map = json.load(f)

with open("/kaggle/input/pi-ii-extras/cat_kv.json", "r") as f:
    cat_kv = json.load(f)

for k, v in cat_kv.items():
    cat_kv[k] = tuple(set(str(s) for s in v + ["Other", "Missing"]))
    
cross_kv = {
    (cat_a, cat_b): tuple(f"{a} {b}" for a, b in product(it_a, it_b))
    for cat_a, cat_b, it_a, it_b in [
        ("grade", "condition", range(1, 14), range(1, 6)),
        ("city", "area", cat_kv["city"], cat_kv["area"]),
    ]
}


def prep(df: pl.DataFrame | pl.LazyFrame, *_) -> pl.DataFrame:
    views_db = [
        "view_olympics",
        "view_cascades",
        "view_territorial",
        "view_skyline",
        "view_sound",
        "view_lakewash",
        "view_lakesamm",
        "view_other",
    ]
    lux_db = views_db + ["golf", "greenbelt"]  # luxury

    return (
        df.lazy()
        .with_columns(
            pl.col("sale_date").dt.year().alias("sale_year"),
            pl.col("sale_date").dt.month().add(9).truediv(3).floor().alias("sale_season"),
            pl.col("sale_date").dt.quarter().alias("sale_quarter"),
            pl.col("sale_date").dt.month().alias("sale_month"),
            pl.col("sale_date").dt.epoch().alias("sale_epoch"),
        )
        .with_columns(
            pl.col("sale_year").sub("join_year").alias("join_ydiff"),
            pl.col("sale_year").sub("year_built").alias("built_ydiff"),
            pl.col("sale_year").sub("year_reno").alias("reno_ydiff"),
        )
        .with_columns(
            pl.when(pl.col("sqft") < pl.col("sqft_1"))
            .then(pl.col("sqft_1") + pl.col("sqft_fbsmt"))
            .otherwise(pl.col("sqft"))
            .alias("sqft")
        )
        .with_columns(
            pl.col("land_val").truediv(pl.col("sqft")).alias("land_dens"),
            pl.col("imp_val").truediv(pl.col("sqft")).alias("imp_dens"),
            pl.when(LOG_PRICE)
            .then(pl.col("imp_val").exp().sub(1).add(pl.col("land_val").exp().sub(1)).log1p())
            .otherwise(pl.col("imp_val").add(pl.col("land_val")))
            .alias("imp_p_land"),
            pl.col("imp_val")
            .sub(pl.col("land_val"))
            .alias("imp_s_land"),  # log may produce null if imp_val < land_val
            pl.col("imp_val")
            .truediv(pl.col("land_val"))
            .replace([np.inf, np.nan], [-1, -1])
            .alias("imp_d_land"),
        )
        .with_columns(
            pl.col("sqft").sub(pl.col("sqft_1") + pl.col("sqft_fbsmt")).alias("sqft_2_above"),
            pl.col("sqft").truediv(pl.col("sqft_lot")).replace(np.inf, 1).alias("floor_area_ratio"),
            pl.col("sqft_1")
            .truediv(pl.col("sqft_lot"))
            .replace(np.inf, 1)
            .alias("build_cover_ratio"),
            pl.col("garb_sqft").add(pl.col("gara_sqft")).alias("garage_sqft"),
            (pl.col("sqft") + pl.col("gara_sqft") + pl.col("garb_sqft")).alias("sqft_total"),
            pl.col("sqft_lot").sub(pl.col("sqft")).alias("sqft_yard"),
            pl.col("bath_full")
            .add(pl.col("bath_3qtr").mul(0.75))
            .add(pl.col("bath_half").mul(0.5))
            .alias("total_baths_scaled"),
            pl.col("bath_full")
            .add(pl.col("bath_3qtr"))
            .add(pl.col("bath_half"))
            .alias("total_baths"),
        )
        .with_columns(
            pl.col("sqft").truediv(pl.col("beds")).alias("sqft_per_bed"),
            pl.col("sqft").truediv(pl.col("total_baths_scaled")).alias("sqft_per_bath"),
            pl.col("beds").truediv(pl.col("total_baths_scaled")).alias("bed_bath_ratio"),
        )
        .with_columns(
            ### _db: had positive linear correlation based on data
            pl.mean_horizontal(pl.col(r"^view.*$")).alias("avg_view"),
            pl.mean_horizontal(pl.col(r"^view.*$").replace(0, None)).alias("avg_view_nz"),
            pl.mean_horizontal(pl.col(views_db)).alias("avg_view_db"),
            pl.mean_horizontal(pl.col(views_db).replace(0, None)).alias("avg_view_nz_db"),
            pl.mean_horizontal(pl.col("wfnt", "golf", "greenbelt", r"^view.*$")).alias("avg_lux"),
            pl.mean_horizontal(
                pl.col("wfnt", "golf", "greenbelt", r"^view.*$").replace(0, None)
            ).alias("avg_lux_nz"),
            pl.mean_horizontal(pl.col(lux_db)).alias("avg_lux_db"),
            pl.mean_horizontal(pl.col(lux_db).replace(0, None)).alias("avg_lux_nz_db"),
            pl.mean_horizontal(
                pl.col("wfnt", "view_lakewash", "view_lakesamm", "view_otherwater")
            ).alias("avg_water"),
        )
        .with_columns(pl.col(r"^avg_.*_nz.*$").fill_null(0))
        ### special attractions
        .with_columns(
            ### Haversine distance
            pl.when(pl.col("city").ne(city))
            .then(pl.lit(1e9))  ### Set to ~inf for locations not in the same city
            .otherwise(
                (
                    pl.col("latitude").sub(lat).radians().truediv(2).sin().pow(2)
                    + pl.col("latitude").radians().cos()
                    * pl.lit(lat).radians().cos()
                    * pl.col("longitude").sub(long).radians().truediv(2).sin().pow(2)
                )
                .arcsin()
                .sqrt()
                .mul(6371 * 2)
            )
            .alias(f"temp__dist_{name}")
            for name, city, lat, long in [
                ### tourist attractions
                ("ta__space_needle", "SEATTLE", 47.620564319437946, -122.34925479474991),
                ("ta__pike_market", "SEATTLE", 47.609454615640196, -122.34181434232742),
                ("ta__bot_garden", "BELLEVUE", 47.609234182629706, -122.17867066930958),
                ### industry centers
                ("tech__amazon", "SEATTLE", 47.62197468565063, -122.33641910325747),
                ("tech__microsoft", "REDMOND", 47.6459181655192, -122.13195093335484),
                ("tech__meta", "SEATTLE", 47.62913385782651, -122.3429023609527),
                ("tech__google_6_st_camp", "KIRKLAND", 47.669429269757416, -122.19698165907319),
                ### schools: top ~20 https://www.usnews.com/education/best-high-schools/washington/rankings
                ("sch__tesla_stem_high", "REDMOND", 47.64858584883487, -122.03723111674697),
                ("sch__international_school", "BELLEVUE", 47.60438631833371, -122.17132073209477),
                ("sch__issaquah_high", "ISSAQUAH", 47.522514151945366, -122.02864121675421),
                ("sch__interlake_high", "BELLEVUE", 47.629023536853076, -122.12401261674819),
                ("sch__newport_high", "BELLEVUE", 47.56801344352629, -122.17221734558795),
                ("sch__raisbeck_aviation_high", "TUKWILA", 47.520818680023275, -122.30060801860841),
                ("sch__mercer_high", "MERCER ISLAND", 47.57212590952493, -122.21967226093288),
                ("sch__bellevue_high", "BELLEVUE", 47.60456180504359, -122.19855090325846),
                ("sch__lincoln_high", "SEATTLE", 47.65997980167359, -122.33987479261197),
                ("sch__ballard_high", "SEATTLE", 47.67670039054923, -122.37541096092693),
                ("sch__redmond_high", "REDMOND", 47.694916028514854, -122.10710427441705),
                ("sch__garfield_high", "SEATTLE", 47.605370139891896, -122.30182846093102),
                ("sch__woodinville_high", "WOODINVILLE", 47.77129815732598, -122.16103081674002),
                ("sch__roosevelt_high", "SEATTLE", 47.67755222370462, -122.31276341674533),
                ("sch__lakewash_high", "KIRKLAND", 47.673185014788956, -122.1808636455819),
                ("sch__liberty_high", "RENTON", 47.47956433127914, -122.11734533210192),
                ("sch__north_creek_high", "BOTHELL", 47.82433704402086, -122.18273465906431),
                ("sch__skyline_high", "SAMMAMISH", 47.60014240904159, -122.03228934558602),
            ]
        )
        .with_columns(pl.min_horizontal(r"^temp__dist_.*$").alias("dist_attraction"))
        .with_columns(
            pl.min_horizontal(rf"^temp__dist_{s}_.*$").alias(f"dist_{s}")
            for s in ["ta", "tech", "sch"]
        )
        .drop(r"^temp__dist_.*$")
        ### warning codes
        .with_columns(
            pl.col("sale_warning")
            .str.strip_chars(" ")
            .str.split(" ")
            .cast(pl.List(pl.Int64), strict=False)
            .list.drop_nulls()
            .list.eval(pl.element().replace(warning_codes))
            .list.sum(),
        )
        ### Zoning
        .with_columns(
            pl.col("zoning")
            .str.replace_all(r"([-0-9][.0-9]*)", "")
            .str.strip_chars()
            .alias("zoning_str"),
            pl.col("zoning")
            .str.extract(r"([-0-9][.0-9]*)")
            .alias("zoning_n")
            .fill_null("0")
            .cast(float, strict=False)
            .abs(),
        )
        .with_columns(
            pl.col("zoning_str").replace(zoning_type_map).alias("zoning_type"),
        )
        ### Cross Category
        .with_columns(
            pl.concat_str(a, b, separator=" ").cast(pl.Enum(v)).alias(f"{a}_{b}")
            for (a, b), v in cross_kv.items()
        )
        ### Impute
        .with_columns(
            pl.col("sale_nbr").fill_null(0),
        )
        ### Mark nominal as category
        .with_columns(
            pl.col(cat_k).cast(pl.String).fill_null("Missing").cast(pl.Enum(cat_v))
            for cat_k, cat_v in cat_kv.items()
        )
    ).collect()


from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    TargetEncoder,
    FunctionTransformer,
)

target_enc_features = ["subdivision"]
ordinal_enc_features = list(  # "Everything else"
    (set(cat_kv.keys()) | set(f"{a}_{b}" for a, b in cross_kv)) - set(target_enc_features)
)

prep_pipe = make_pipeline(
    FunctionTransformer(prep),
    make_column_transformer(
        (
            TargetEncoder(random_state=random_state),
            target_enc_features,
        ),
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ordinal_enc_features,
        ),
        remainder="passthrough",
        verbose_feature_names_out=False,
    ),
    FunctionTransformer(
        lambda df, *_: df.drop(
            "train",  # Only a marker
            "sale_date",  # Decomposed into other time features
            "join_year",  # Is decided completely by join_status
            "zoning",  # Decomposed into zoning_*
            r"^view.*$",  # Empirically not useful
        )
    ),
)

prep_pipe.fit_transform(
    df.head(10000).drop('sale_price'), 
    df.head(10000)['sale_price']
).head()


from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

cat_opts = {  
    "n_estimators": 2000,
    "learning_rate": 0.2,
    "subsample": 0.25,
    "early_stopping_rounds": 50,
    "bootstrap_type": "Bernoulli",
    "thread_count": 5,
    "task_type": "GPU",
    "random_state": random_state,
    "verbose": 500,
}

ALPHA = 0.1
a_low, a_high = ALPHA / 2, 1 - ALPHA / 2
models = [
    CatBoostRegressor(objective=f"Quantile:alpha={a_low}", **cat_opts),
    CatBoostRegressor(objective=f"Quantile:alpha={a_high}", **cat_opts),
]

knn_features = ["longitude", "latitude"]
knn_reg = [
    "sqft",
    "sqft_lot",
    "grade",
    "condition",
    "imp_p_land",
    "imp_val",
    "land_val",
]
knn_clf = None

detrender = make_pipeline(
    StandardScaler(),
    LinearRegression(),
)
detrender_features = ["sale_year"]


# Modeling function
from sklearn.model_selection import train_test_split


def model_detrend_cqr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models,
    *,
    alpha=ALPHA,
    detrender=None,
    detrender_features=None,
    knn_features=None,
    knn_reg=None,
    knn_clf=None,
    plot_feature_influence=True,
):
    if detrender is None:  # No detrending
        detrender = DummyRegressor(strategy="constant", constant=0)
    if detrender_features is None:
        detrender_features = ["sale_year"]

    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
    )

    _ = detrender.fit(X_train[detrender_features], y_train)
    y_train_trend = detrender.predict(X_train[detrender_features])
    y_calib_trend = detrender.predict(X_calib[detrender_features])
    y_test_trend = detrender.predict(X_test[detrender_features])

    knns = []
    for knn_target, knn in zip(
        [knn_reg, knn_clf],
        [KNeighborsRegressor(15), KNeighborsClassifier(15)],
    ):
        if knn_target is None:
            continue

        knn_y = pd.DataFrame()
        for k in knn_target:
            if k == "sale_price":
                knn_y[k] = y_train
            else:
                knn_y[k] = X_train[k]
        _ = knn.fit(X_train[knn_features], knn_y)
        knns.append(knn)

        knn_new_cols = [f"nn_{k}" for k in knn_target]
        X_train = pd.concat(
            [
                X_train.reset_index(drop=True),
                pd.DataFrame(knn.predict(X_train[knn_features]), columns=knn_new_cols),
            ],
            axis=1,
        )
        X_calib = pd.concat(
            [
                X_calib.reset_index(drop=True),
                pd.DataFrame(knn.predict(X_calib[knn_features]), columns=knn_new_cols),
            ],
            axis=1,
        )
        X_test = pd.concat(
            [
                X_test.reset_index(drop=True),
                pd.DataFrame(knn.predict(X_test[knn_features]), columns=knn_new_cols),
            ],
            axis=1,
        )

    train_preds, calib_preds, test_preds = [], [], []
    for m in models:
        _ = m.fit(X_train, y_train - y_train_trend)
        train_preds.append(m.predict(X_train) + y_train_trend)
        calib_preds.append(m.predict(X_calib) + y_calib_trend)
        test_preds.append(m.predict(X_test) + y_test_trend)

    nonconf_low = calib_preds[0] - y_calib
    nonconf_high = y_calib - calib_preds[1]
    q_low = np.quantile(nonconf_low, (1 - alpha / 2))
    q_high = np.quantile(nonconf_high, (1 - alpha / 2))

    for p in [train_preds, calib_preds, test_preds]:
        p[0] -= q_low
        p[1] += q_high

    plt = None
    if plot_feature_influence:
        plt = (
            hvpl(feature_influence(models[0]).filter(cs.numeric().ne(0))).bar(
                x="feature_name", rot=75, title=f"Fold {i} Feature Importances (Lower Bound)"
            )
            + hvpl(feature_influence(models[1]).filter(cs.numeric().ne(0))).bar(
                x="feature_name", rot=75, title=f"Fold {i} Feature Importances (Upper Bound)"
            )
        ).opts(shared_axes=False)

    return {
        "train_score": winkler_interval_score(y_train, train_preds[0], train_preds[1]),
        "calib_score": winkler_interval_score(y_calib, calib_preds[0], calib_preds[1]),
        "test_score": winkler_interval_score(y_test, test_preds[0], test_preds[1]),
        "knns": knns,
        "train_X": X_train,
        "test_X": X_test,
        "train_pred": train_preds,
        "train_true": y_train.to_numpy(),
        "calib_X": X_calib,
        "calib_pred": calib_preds,
        "calib_true": y_calib.to_numpy(),
        "test_pred": test_preds,
        "test_true": y_test.to_numpy(),
        "plt": plt,
    }


# For some reason, I can't exactly replicate the results I get on my local machine
# One possible explanation is due to hardware differences
# If someone knows more about this / ways to make code more reproducible please let me know

### For reference, on my machine, the losses of the first fold reported are:
# 0:	learn: 0.0438892	total: 16.7ms	remaining: 33.3s
# 500:	learn: 0.0137729	total: 3.32s	remaining: 9.93s
# 1000:	learn: 0.0126503	total: 6.58s	remaining: 6.56s
# 1500:	learn: 0.0120192	total: 9.82s	remaining: 3.26s
# 1999:	learn: 0.0116170	total: 13s	remaining: 0us
# 0:	learn: 0.0559174	total: 6.71ms	remaining: 13.4s
# 500:	learn: 0.0117977	total: 3.36s	remaining: 10.1s
# 1000:	learn: 0.0107650	total: 6.66s	remaining: 6.65s
# 1500:	learn: 0.0101947	total: 9.89s	remaining: 3.29s
# 1999:	learn: 0.0098567	total: 13.1s	remaining: 0us
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
res = []
dat = df.filter("train")

for i, (train_idx, test_idx) in tqdm(
    enumerate(kf.split(dat, groups=dat.select(pl.col("sale_date").dt.year()))),
    "Cross Validation",
    total=kf.get_n_splits(),
):
    df_train = dat[train_idx]
    df_test = dat[test_idx]
    y_train = df_train["sale_price"]
    X_train = prep_pipe.fit_transform(df_train.drop("sale_price"), y_train)
    y_test = df_test["sale_price"]
    X_test = prep_pipe.transform(df_test.drop("sale_price"))
    if isinstance(X_train, pl.DataFrame):
        X_train = X_train.to_pandas()
        y_train = y_train.to_pandas()
        X_test = X_test.to_pandas()
        y_test = y_test.to_pandas()

    res.append(
        model_detrend_cqr(
            X_train,
            y_train,
            X_test,
            y_test,
            models,
            detrender=detrender,
            detrender_features=detrender_features,
            knn_features=knn_features,
            knn_reg=knn_reg,
            knn_clf=knn_clf,
        )
    )
res[-1]["plt"]


# Scores & other statistics

### For reference, the Overall stats on my machine were:
# score: 263032.253 ± 627.277 / 316012.322 ± 2036.903 / 315381.761 ± 3310.168
# coverage: 0.937 ± 0.001 / 0.900 ± 0.000 / 0.900 ± 0.002
# interval: 222204.930 ± 432.337 / 223117.333 ± 981.634 / 223256.988 ± 733.610
print("Stats (train / test):")
stats: dict[str, list[list]]
stats = {k: list() for k in ["score", "coverage", "interval"]}

for i, r in enumerate(res):
    for k in stats:
        stats[k].append(list())
    for split in ["train", "calib", "test"]:
        if f"{split}_true" not in r:
            continue
        stats["score"][-1].append(r[f"{split}_score"])
        stats["coverage"][-1].append(
            np.mean(
                (r[f"{split}_pred"][0] <= r[f"{split}_true"])
                & (r[f"{split}_true"] <= r[f"{split}_pred"][1])
            )
        )
        if LOG_PRICE:
            stats["interval"][-1].append(
                np.mean(np.expm1(r[f"{split}_pred"][1]) - np.expm1(r[f"{split}_pred"][0]))
            )
        else:
            stats["interval"][-1].append(np.mean(r[f"{split}_pred"][1] - r[f"{split}_pred"][0]))

    print(f"Fold {i}:")
    for k, v in stats.items():
        print(f'\t{k}: {" / ".join(str(np.round(vv, 3)) for vv in v[-1])}')

print("\nOverall:")
for k, v in stats.items():
    transposed = np.array(v).T
    means = transposed.mean(axis=1)
    stds = transposed.std(axis=1)
    print(f'\t{k}: {" / ".join(f"{m:.3f} ± {s:.3f}" for m, s in zip(means, stds))}')


# Plot of true price & predicted interval ordered by sale_price on the x-axis
# (Commented out to prevent immense lag)

# plt_dataset = "test"
# ys = {k: list() for k in ["lower", "true", "upper"]}
# for r in res:
#     lower, upper = r[f"{plt_dataset}_pred"]
#     ys["lower"].append(lower)
#     ys["true"].append(r[f"{plt_dataset}_true"])
#     ys["upper"].append(upper)
# ys = pl.DataFrame({k: np.concatenate(v) for k, v in ys.items()}).sort("true")
# ys = ys.with_columns(
#     pl.col("lower", "upper").rolling_mean(10).name.suffix("_MA10"),
#     score=winkler_interval_score(ys["true"], ys["lower"], ys["upper"], mean=False),
# )
# (
#     hvpl(ys).line(
#         y=["lower", "upper", "lower_MA10", "upper_MA10"],
#         color=["skyblue", "salmon", "royalblue", "crimson"],
#         title="True Sale Prices vs. Predicted Intervals Sorted by Price",
#         ylabel="Sale Price",
#     )
#     * hvpl(ys).line(y="true", color="black", label="true", line_width=5)
# )


df_train = df.filter("train")
df_test = df.filter(~pl.col("train"))

y_train = df_train["sale_price"].to_pandas()
X_train = prep_pipe.fit_transform(df_train.drop("sale_price"), y_train).to_pandas()
y_test = df_test["sale_price"].to_pandas()
X_test = prep_pipe.transform(df_test.drop("sale_price")).to_pandas()

r = model_detrend_cqr(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    detrender=detrender,
    knn_features=knn_features,
    knn_reg=knn_reg,
    knn_clf=knn_clf,
)


(
    hvpl(feature_influence(models[0]).filter(cs.numeric().ne(0))).bar(
        x="feature_name", rot=75, title=f"Feature Importances (Lower Bound)"
    )
    + hvpl(feature_influence(models[1]).filter(cs.numeric().ne(0))).bar(
        x="feature_name", rot=75, title=f"Feature Importances (Upper Bound)"
    )
).opts(shared_axes=False)

# (Commented out to prevent immense lag)
# r["calib_score"]
# ys = {
#     "lower": r["calib_pred"][0],
#     "true": r["calib_true"],
#     "upper": r["calib_pred"][1],
# }
# ys = pl.DataFrame(ys).sort("true")
# ys = ys.with_columns(
#     pl.col("lower", "upper").rolling_mean(10).name.suffix("_MA10"),
#     score=winkler_interval_score(ys["true"], ys["lower"], ys["upper"], mean=False),
# )
# (
#     hvpl(ys.drop("true")).line(
#         y=["lower", "upper", "lower_MA10", "upper_MA10"],
#         color=["skyblue", "salmon", "royalblue", "crimson"],
#         title="Calibration Sale Prices vs. Predicted Intervals Sorted by Price",
#         ylabel="Sale Price",
#     )
#     * hvpl(ys).line(y="true", color="black", label="true", line_width=5)
# )


sub = pl.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv").with_columns(
    pi_lower=np.expm1(r["test_pred"][0]),
    pi_upper=np.expm1(r["test_pred"][1]),
)
sub.head()
sub.write_csv("/kaggle/working/submission.csv")




