
import os
import gc
import sys
import math
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go

from tqdm.auto import tqdm
import lightgbm as lgb
import category_encoders as ce

from typing import Dict, List

from IPython.display import display

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor




# Paths
KAGGLE_INPUT_PATH = Path("/kaggle/input/playground-series-s5e10")
LOCAL_INPUT_PATH = Path("playground-series-s5e10")

DATA_DIR = KAGGLE_INPUT_PATH if KAGGLE_INPUT_PATH.exists() else LOCAL_INPUT_PATH

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SAMPLE_SUBMISSION_FILENAME = "sample_submission.csv"

TARGET = "accident_risk"
ID_COLUMN = "id"
PRIMARY_METRIC = "rmse"
SEED = 200
N_SPLITS = 5
PAIRPLOT_SAMPLE = 5000

# Palette reproduced from Plantilla.ipynb
yellow = "#F7C53E"
cyan_g = "#0CF7AF"
cyan_dark = "#11AB7C"
purple = "#D826F8"
purple_dark = "#9309AB"
purple_light = "#b683d6"
blue = "#0C97FA"
red = "#FA1D19"
orange = "#FA9F19"
green = "#0CFA58"
light_blue = "#01FADC"
soft_blue = "#81c9e6"
dark_blue = "#394be6"

PALETTE_2 = [cyan_g, purple]
PALETTE_3 = [yellow, cyan_g, purple]
PALETTE_4 = [yellow, orange, purple, light_blue]
PALETTE_5 = [purple_dark, purple_light, purple, blue, light_blue]
PALETTE_6 = [blue, red, orange, green, light_blue, purple]
PALETTE_7 = [purple_dark, purple_light, purple, blue, light_blue, dark_blue, soft_blue]
PALETTE_7_C = [purple_dark, blue, purple, light_blue, purple_light, soft_blue, dark_blue]

sns.set_style("whitegrid")
sns.set_palette(PALETTE_7)
plt.style.use({"figure.facecolor": "#f8fafc"})

pd.set_option("display.float_format", "{:.4f}".format)
warnings.filterwarnings("ignore")




def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and OS-level randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))




def get_paths() -> Dict[str, Path]:
    paths = {
        "train": DATA_DIR / TRAIN_FILENAME,
        "test": DATA_DIR / TEST_FILENAME,
        "sample": DATA_DIR / SAMPLE_SUBMISSION_FILENAME,
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found at {path}")
    return paths




def load_datasets(paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path) for name, path in paths.items()}




def bilingual_print(en: str, es: str) -> None:
    print(f"EN> {en}")
    print(f"ES> {es}")




def memory_usage_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024**2




def dataset_profile(df: pd.DataFrame, name: str) -> pd.DataFrame:
    profile = pd.DataFrame({
        "dataset": name,
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(dropna=False),
        "missing_pct": (df.isna().mean() * 100).round(3),
    })
    return profile




def convert_booleans_to_category(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in columns:
        df_copy[col] = df_copy[col].map({True: "True", False: "False"}).astype("category")
    return df_copy




def get_feature_groups(train_df: pd.DataFrame, target: str, id_column: str) -> Dict[str, List[str]]:
    features_df = train_df.drop(columns=[target])
    numeric = features_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = features_df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in [id_column]:
        if col in numeric:
            numeric.remove(col)
        if col in categorical:
            categorical.remove(col)
    return {"numeric": numeric, "categorical": categorical}




def build_preprocessor(feature_groups: Dict[str, List[str]]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if feature_groups["numeric"]:
        transformers.append(("numeric", numeric_pipeline, feature_groups["numeric"]))
    if feature_groups["categorical"]:
        transformers.append(("categorical", categorical_pipeline, feature_groups["categorical"]))

    return ColumnTransformer(transformers=transformers, remainder="drop")




def plot_missing_values(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 4))
    msno.bar(df, color=PALETTE_7_C)
    plt.title("Missing Values Overview / Valores Faltantes")
    plt.tight_layout()
    plt.show()




def plot_target_distribution(df: pd.DataFrame, target: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(df[target], bins=30, kde=True, color=PALETTE_7[2], ax=axes[0])
    axes[0].set_title("Target Distribution")
    sns.boxplot(x=df[target], color=PALETTE_7[4], ax=axes[1])
    axes[1].set_title("Target Boxplot")
    plt.tight_layout()
    plt.show()




def plot_correlation_heatmap(df: pd.DataFrame, target: str) -> None:
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, mask=mask, cmap=PALETTE_2, annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation / CorrelaciÃ³n de Variables")
    plt.tight_layout()
    plt.show()




def plot_categorical_summary(df: pd.DataFrame, categorical_cols: List[str], target: str) -> None:
    if not categorical_cols:
        bilingual_print("No categorical features to profile.", "No hay variables categÃ³ricas para perfilar.")
        return

    for col in categorical_cols:
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, palette=PALETTE_7, ax=axes[0])
        axes[0].set_title(f"{col} â€” counts")
        axes[0].tick_params(axis='x', rotation=30)
        sns.barplot(data=df, x=col, y=target, order=order, palette=PALETTE_7_C, ax=axes[1])
        axes[1].set_title(f"{col} vs {target} (mean)")
        axes[1].tick_params(axis='x', rotation=30)
        plt.tight_layout()
        plt.show()




def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: List[str], target: str) -> None:
    if not numeric_cols:
        bilingual_print("No numeric features to profile.", "No hay variables numÃ©ricas para perfilar.")
        return

    n_cols = 2
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    plt.figure(figsize=(16, 5 * n_rows))
    for idx, col in enumerate(numeric_cols, 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        sns.scatterplot(data=df, x=col, y=target, color=PALETTE_7[idx % len(PALETTE_7)], alpha=0.6, ax=ax)
        sns.regplot(data=df, x=col, y=target, scatter=False, lowess=True, color=PALETTE_7_C[idx % len(PALETTE_7_C)], ax=ax)
        ax.set_title(f"{col} vs {target}")
    plt.tight_layout()
    plt.show()




def plot_pairgrid(df: pd.DataFrame, columns: List[str], sample_size: int = PAIRPLOT_SAMPLE) -> None:
    if len(columns) < 2:
        bilingual_print("Not enough numeric features for pairplot.", "No hay suficientes variables numÃ©ricas para pairplot.")
        return
    sample_df = df[columns].sample(min(len(df), sample_size), random_state=SEED)
    sns.pairplot(sample_df, diag_kind="kde", corner=True, plot_kws={"alpha": 0.5, "s": 20, "color": PALETTE_7[3]})
    plt.suptitle("Pairplot of Numeric Features", y=1.02)
    plt.show()




def feature_group_statistics(df: pd.DataFrame, group_col: str, target: str) -> pd.DataFrame:
    agg = df.groupby(group_col)[target].agg(["count", "mean", "median", "std"]).reset_index()
    agg = agg.sort_values("mean", ascending=False)
    return agg




def evaluate_models(models: Dict[str, object], X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer, n_splits: int = N_SPLITS) -> pd.DataFrame:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = []
    for name, estimator in models.items():
        fold_metrics = []
        start = time.time()
        for train_idx, valid_idx in cv.split(X, y):
            pipeline = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ])
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = pipeline.predict(X.iloc[valid_idx])
            rmse = mean_squared_error(y.iloc[valid_idx], preds, squared=False)
            mae = mean_absolute_error(y.iloc[valid_idx], preds)
            fold_metrics.append((rmse, mae))
        duration = time.time() - start
        rmses, maes = zip(*fold_metrics)
        results.append({
            "model": name,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "mae_mean": float(np.mean(maes)),
            "fit_time_sec": duration,
        })
    leaderboard = pd.DataFrame(results).sort_values("rmse_mean").reset_index(drop=True)
    return leaderboard




def train_best_pipeline(model: object, preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", clone(model)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline




def evaluate_holdout(pipeline: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> Dict[str, float]:
    preds = pipeline.predict(X_valid)
    return {
        "rmse": mean_squared_error(y_valid, preds, squared=False),
        "mae": mean_absolute_error(y_valid, preds),
        "r2": r2_score(y_valid, preds),
    }




def clip_predictions(preds: np.ndarray, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    return np.clip(preds, lower, upper)




def generate_submission(pipeline: Pipeline, test_features: pd.DataFrame, test_ids: pd.Series, target: str, filename: str = "submission.csv") -> pd.DataFrame:
    predictions = clip_predictions(pipeline.predict(test_features))
    submission = pd.DataFrame({ID_COLUMN: test_ids, target: predictions})
    submission.to_csv(filename, index=False)
    return submission




set_seed(SEED)
paths = get_paths()
data = load_datasets(paths)

train_df = data["train"].copy()
test_df = data["test"].copy()
sample_submission_df = data["sample"].copy()

bilingual_print(
    f"Loaded train shape {train_df.shape} and test shape {test_df.shape}.",
    f"Cargadas train {train_df.shape} y test {test_df.shape}."
)

train_memory = memory_usage_mb(train_df)
test_memory = memory_usage_mb(test_df)
bilingual_print(
    f"Train memory usage: {train_memory:.2f} MB | Test memory usage: {test_memory:.2f} MB.",
    f"Uso de memoria train: {train_memory:.2f} MB | Uso de memoria test: {test_memory:.2f} MB."
)

bilingual_print("Train head (top 5 rows).", "Train head (primeras 5 filas).")
display(train_df.head())

bilingual_print("Train tail (last 5 rows).", "Train tail (Ãºltimas 5 filas).")
display(train_df.tail())

bilingual_print("Sample submission preview.", "Vista previa del sample submission.")
display(sample_submission_df.head())

validate_columns = set(sample_submission_df.columns)
assert TARGET in validate_columns, "Target column missing in sample submission"




profile_table = dataset_profile(train_df, "train")
summary_stats = train_df.describe(include="all").T

bilingual_print(
    "Dataset profile with dtypes, uniques, and missing percentages.",
    "Perfil del dataset con tipos, valores Ãºnicos y porcentajes de nulos."
)
display(profile_table)

bilingual_print(
    "Descriptive statistics (numeric and categorical).",
    "EstadÃ­sticos descriptivos (numÃ©ricos y categÃ³ricos)."
)
display(summary_stats)

categorical_cols = train_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)
if ID_COLUMN in numeric_cols:
    numeric_cols.remove(ID_COLUMN)




plot_missing_values(train_df)
plot_target_distribution(train_df, TARGET)
plot_correlation_heatmap(train_df, TARGET)
plot_numeric_distributions(train_df, numeric_cols, TARGET)
plot_categorical_summary(train_df, categorical_cols, TARGET)
numeric_for_pairplot = numeric_cols + [TARGET]
plot_pairgrid(train_df[numeric_for_pairplot], numeric_for_pairplot)




aggs = {}
for col in categorical_cols:
    stats = feature_group_statistics(train_df, col, TARGET)
    bilingual_print(
        f"Target statistics grouped by {col}.",
        f"EstadÃ­sticas del target agrupadas por {col}."
    )
    display(stats.head())
    aggs[col] = stats

bilingual_print(
    "Correlation between target and numeric features.",
    "CorrelaciÃ³n entre el objetivo y las variables numÃ©ricas."
)
correlations = train_df[numeric_cols + [TARGET]].corr()[TARGET].sort_values(ascending=False)
display(correlations)




bool_columns = train_df.select_dtypes(include=["bool"]).columns.tolist()
if bool_columns:
    bilingual_print(
        f"Casting boolean columns to categorical: {bool_columns}",
        f"Convirtiendo columnas booleanas a categÃ³ricas: {bool_columns}"
    )
    train_df = convert_booleans_to_category(train_df, bool_columns)
    test_df = convert_booleans_to_category(test_df, bool_columns)

feature_groups = get_feature_groups(train_df, TARGET, ID_COLUMN)
feature_columns = [col for col in train_df.columns if col not in [TARGET, ID_COLUMN]]

preprocessor = build_preprocessor(feature_groups)

X = train_df[feature_columns].copy()
y = train_df[TARGET].copy()

test_features = test_df[feature_columns].copy()
test_ids = test_df[ID_COLUMN].copy()

bilingual_print(
    f"Prepared {len(feature_columns)} features for modelling.",
    f"Se prepararon {len(feature_columns)} caracterÃ­sticas para el modelado."
)

feature_catalog = pd.DataFrame({
    "feature": feature_columns,
    "role": ["numeric" if col in feature_groups["numeric"] else "categorical" for col in feature_columns]
})
display(feature_catalog)




candidate_models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=SEED),
    "LGBMRegressor": LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=SEED
    ),
    "XGBRegressor": XGBRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=SEED,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1
    ),
}

bilingual_print(
    f"Evaluating {len(candidate_models)} candidate models with {N_SPLITS}-fold CV.",
    f"Evaluando {len(candidate_models)} modelos candidatos con CV de {N_SPLITS} folds."
)
leaderboard = evaluate_models(candidate_models, X, y, preprocessor, n_splits=N_SPLITS)
display(leaderboard)

best_model_name = leaderboard.loc[0, "model"]
bilingual_print(
    f"Best CV model: {best_model_name}",
    f"Mejor modelo segÃºn CV: {best_model_name}"
)




X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

best_model = candidate_models[best_model_name]
best_pipeline = train_best_pipeline(best_model, preprocessor, X_train, y_train)

holdout_metrics = evaluate_holdout(best_pipeline, X_valid, y_valid)
bilingual_print(
    f"Hold-out RMSE: {holdout_metrics['rmse']:.4f} | MAE: {holdout_metrics['mae']:.4f} | R2: {holdout_metrics['r2']:.4f}",
    f"Hold-out RMSE: {holdout_metrics['rmse']:.4f} | MAE: {holdout_metrics['mae']:.4f} | R2: {holdout_metrics['r2']:.4f}"
)




final_pipeline = train_best_pipeline(best_model, preprocessor, X, y)
submission_df = generate_submission(final_pipeline, test_features, test_ids, TARGET)

bilingual_print(
    f"Submission saved with shape {submission_df.shape}",
    f"Submission guardada con forma {submission_df.shape}"
)

display(submission_df.head())
print("Missing values per column:")
print(submission_df.isna().sum())


