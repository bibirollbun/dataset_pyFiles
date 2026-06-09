# Data
import polars as pl
import numpy as np

# ML
## Metrics & tools
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold

## Models
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

# Tools
from pathlib import Path
from dataclasses import dataclass
from tqdm.notebook import tqdm


@dataclass
class CFG:
    data_path: Path = Path("/kaggle/input/playground-series-s5e2")
    external_path: Path = Path('/kaggle/input/student-bag-price-prediction-dataset')
    seed: int = 2025

    cat_features: tuple = ("Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color")
    num_features: tuple = ("Compartments", "Weight Capacity (kg)")
    drop_cols: tuple = ("id")
    target: str = "Price"

cfg = CFG()


train_df = pl.read_csv(cfg.data_path / 'train.csv')
external_df = pl.read_csv(cfg.external_path / 'Noisy_Student_Bag_Price_Prediction_Dataset.csv')

test_df = pl.read_csv(cfg.data_path / 'test.csv')
subm_df = pl.read_csv(cfg.data_path / 'sample_submission.csv')


train_df.head()


external_df.head()


train_df.shape[0], external_df.shape[0]


extended_df = pl.concat([
    train_df.drop(cfg.drop_cols).with_columns(is_train=1),
    external_df.with_columns(is_train=0)
])
extended_df.head()


extended_df.null_count()


extended_df = (
    extended_df
    .filter(~pl.col("Price").is_null())
)
y_extended = extended_df["Price"]
extended_df = extended_df.drop("Price")


def inplace_null(data, columns: list, inplace: str="None"):
    return data.with_columns(
        pl.col(columns).cast(pl.String).fill_null(inplace), 
        pl.col("Compartments", "Weight Capacity (kg)").fill_null(0)
    )


adv_df = extended_df.pipe(inplace_null, columns=cfg.cat_features)


adv_df = adv_df.sample(
    fraction=1,
    shuffle=True,
    seed=cfg.seed
)


X = adv_df.drop("is_train")
y = adv_df["is_train"]


cv = KFold(shuffle=True, random_state=cfg.seed)


models = []
metrics = []
for train_idx, valid_idx in tqdm(cv.split(X), total=cv.get_n_splits()):
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    train_pool = Pool(X_train.to_pandas(), y_train.to_pandas(), cat_features=cfg.cat_features)
    valid_pool = Pool(X_valid.to_pandas(), y_valid.to_pandas(), cat_features=cfg.cat_features)

    model = CatBoostClassifier(
        iterations=100, # Ğ¢Ğ¾Ğ»ÑŒĞºĞ¾ Ğ¾Ñ†ĞµĞ½Ğ¸Ñ‚ÑŒ Ğ¼Ğ°Ñ�ÑˆÑ‚Ğ°Ğ± Ğ±ĞµĞ´Ñ�Ñ‚Ğ²Ğ¸Ñ� :) 
        eval_metric="AUC",
        random_state=cfg.seed,
        verbose=100,
        early_stopping_rounds=15
    )

    model.fit(train_pool, eval_set=valid_pool)

    y_pred = model.predict(valid_pool)
    metric = roc_auc_score(y_valid, y_pred)

    metrics.append(metric)
    models.append(model)


np.mean(metrics) - np.std(metrics)


train_df = X.with_columns(Price=y_extended)
train_df.write_csv("extended.csv")


def inplace_null(data, columns: list, inplace: str="None"):
    return data.with_columns(pl.col(columns).cast(pl.String).fill_null(inplace))


def min_max_scaler(data, columns: list):
    min = pl.col(columns).min()
    max = pl.col(columns).max()
    
    return data.with_columns((pl.col(columns) - min) / (max - min))


train_prep = (
    train_df
    .pipe(inplace_null, columns=cfg.cat_features) # Ğ’Ñ�Ñ‚Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ null
    .pipe(min_max_scaler, columns=cfg.num_features) # ĞœĞ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¸Ñ€ÑƒĞµĞ¼ num_features - Ñ…Ğ¾Ñ‚ÑŒ Ğ¸ Ğ½ĞµĞ¾Ğ±Ñ�Ğ·Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ğ¾
)


test_prep = (
    test_df
    .pipe(inplace_null, columns=cfg.cat_features) # Ğ’Ñ�Ñ‚Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ null
    .pipe(min_max_scaler, columns=cfg.num_features) # ĞœĞ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¸Ñ€ÑƒĞµĞ¼ num_features
)


X = train_prep.drop(cfg.target)
y = train_prep[cfg.target]

X_test = test_prep.clone()


cv = KFold(shuffle=True, random_state=cfg.seed)


models = []
metrics = []
for train_idx, valid_idx in tqdm(cv.split(X), total=cv.get_n_splits()):
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    train_pool = Pool(X_train.to_pandas(), y_train.to_pandas(), cat_features=cfg.cat_features)
    valid_pool = Pool(X_valid.to_pandas(), y_valid.to_pandas(), cat_features=cfg.cat_features)

    model = CatBoostRegressor(
        eval_metric="RMSE",
        random_state=cfg.seed,
        verbose=100,
        early_stopping_rounds=30
    )

    model.fit(train_pool, eval_set=valid_pool)

    y_pred = model.predict(valid_pool)
    metric = mean_squared_error(y_valid, y_pred, squared=False)

    metrics.append(metric)
    models.append(model)


np.mean(metrics) - np.std(metrics)


weights = np.array(metrics)
weights = weights / weights.sum()
weights


test_pool = Pool(X_test.to_pandas(), cat_features=cfg.cat_features)


y_pred = np.zeros(subm_df.shape[0])
for model, weight in tqdm(zip(models, weights), total=len(models)):
    y_pred += weight * model.predict(test_pool)


subm_df = subm_df.with_columns(Price=y_pred)
subm_df.head()


subm_df.write_csv('submission.csv')




