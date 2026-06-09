import os 
import re
import gc
import sys
import math
import time
import random
import warnings
import datetime
import numpy as np 
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import lightgbm as lgb
import missingno as msno
import plotly.express as px
import category_encoders as ce
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from itertools import combinations

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Put theme of notebook 
from colorama import Fore, Style

# Colors
red = Fore.RED + Style.BRIGHT
mgta = Fore.MAGENTA + Style.BRIGHT
yllw = Fore.YELLOW + Style.BRIGHT
cyn = Fore.CYAN + Style.BRIGHT
blue = Fore.BLUE + Style.BRIGHT

# Reset
res = Style.RESET_ALL
plt.style.use({"figure.facecolor": "#282a36"})


# Colors
YELLOW = "#F7C53E"

CYAN_G = "#0CF7AF"
CYAB_DARK = "#11AB7C"

PURPLE = "#D826F8"
PURPLE_DARJ = "#9309AB"
PURPLE_L = "#b683d6"

BLUE = "#0C97FA"
RED = "#FA1D19"
ORANGE = "#FA9F19"
GREEN = "#0CFA58"
LIGTH_BLUE = "#01FADC"
S_BLUE = "#81c9e6"
DARK_BLUE = "#394be6"
# Palettes
PALETTE_2 = [CYAN_G, PURPLE]
PALETTE_3 = [YELLOW, CYAN_G, PURPLE]
PALETTE_4 = [YELLOW, ORANGE, PURPLE, LIGTH_BLUE]
PALETTE_5 = [PURPLE_DARJ, PURPLE_L, PURPLE, BLUE, LIGTH_BLUE]
PALETTE_6 = [BLUE, RED, ORANGE, GREEN, LIGTH_BLUE, PURPLE]

# Vaporwave palette by Francesc Oliveras
PALETTE_7 = [PURPLE_DARJ, PURPLE_L, PURPLE, BLUE, LIGTH_BLUE, DARK_BLUE, S_BLUE]
PALETTE_7_C = [PURPLE_DARJ, BLUE, PURPLE, LIGTH_BLUE, PURPLE_L, S_BLUE, DARK_BLUE]
sns.palplot(sns.color_palette(PALETTE_7))

# Set Style
sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)

cmap = mcolors.LinearSegmentedColormap.from_list("", PALETTE_2)
cmap_2 = mcolors.LinearSegmentedColormap.from_list("", [S_BLUE, PURPLE_DARJ])

font_family = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=10), width=1000, height=500))

warnings.filterwarnings('ignore')


PATH = "/kaggle/input/playground-series-s5e9"
SUBMISSION_FILENAME = "sample_submission.csv"
TEST_FILENAME = "test.csv"
TRAIN_FILENAME = "train.csv"

TARGET = "BeatsPerMinute"

SUBMISSION_DIR = os.path.join(PATH, SUBMISSION_FILENAME)
TRAIN_DIR = os.path.join(PATH, TRAIN_FILENAME) 
TEST_DIR = os.path.join(PATH, TEST_FILENAME)

SEED = 180


def show_corr_heatmap(df, title):
    
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize = (15, 10))
    plt.title(title)
    # sns.heatmap(corr, annot = False, linewidths=.5, fmt=".2f", square=True, mask = mask, cmap=cmap_2)
    if df.shape[1] < 25:
        sns.heatmap(corr, annot=True, linewidths=.5, fmt=".2f", square=True, mask=mask, cmap=cmap_2)
    else:
        sns.heatmap(corr, annot=False, linewidths=.5, square=True, mask=mask, cmap=cmap_2)

    plt.show()


def data_description(df):
    print("Data description")
    print(f"Total number of records {df.shape[0]}")
    print(f'number of features {df.shape[1]}\n\n')
    columns = df.columns
    data_type = []
    
    # Get the datatype of features
    for col in df.columns:
        data_type.append(df[col].dtype)
        
    n_uni = df.nunique()
    # Number of NaN values
    n_miss = df.isna().sum()
    
    names = list(zip(columns, data_type, n_uni, n_miss))
    variable_desc = pd.DataFrame(names, columns=["Name","Type","Unique levels","Missing"])
    print(variable_desc)


def plot_cont(col, ax, color=PALETTE_7[0]):
    sns.histplot(data=comb_df, x=col,
                hue="set",ax=ax, hue_order=labels,
                common_norm=False, **histplot_hyperparams)
    
    ax_2 = ax.twinx()
    ax_2 = plot_cont_dot(
        comb_df.query('set=="train"'),
        col, TARGET, ax_2,
        color=color
    )
    
    ax_2 = plot_cont_dot(
        comb_df, col,
        TARGET, ax_2,
        color=color
    )


def show_pie_mult(dataframe, target = TARGET):
    target_counts = dataframe[target].sum()

    # Creando el grÃ¡fico de pastel con un agujero en el centro
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(target_counts, labels=target, autopct='%1.1f%%', startangle=140, colors=PALETTE_7_C)

    # Agregando un cÃ­rculo blanco en el centro para hacer un agujero
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Ajustando el aspecto para que sea un cÃ­rculo y mostrando el grÃ¡fico
    plt.title('DistribuciÃ³n de los Targets')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def show_pie_categorical(dataframe, target=TARGET):
    target_counts = dataframe[target].value_counts()

    # Creando el grÃ¡fico de pastel con un agujero en el centro
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=140, colors=[PALETTE_7_C[0],PALETTE_7_C[1],PALETTE_7_C[2],
                                                                                                                            PALETTE_7_C[3],PALETTE_7_C[4],PALETTE_7_C[5]])

    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Ajustando el aspecto para que sea un cÃ­rculo y mostrando el grÃ¡fico
    plt.title('DistribuciÃ³n de los Targets')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def show_box_plot(dataframe):
    # numerical_features_for_boxplot = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('id')
    numerical_features_for_boxplot = dataframe.select_dtypes(include=['int64', 'float64'])

    plt.figure(figsize=(20, 15))

    for i, feature in enumerate(numerical_features_for_boxplot, 1):
        plt.subplot(7, 5, i)
        sns.boxplot(y=train_df[feature], color=PALETTE_7_C[i % len(PALETTE_7_C)])
        plt.title(feature)

    plt.tight_layout()
    plt.show()


def show_hist(dataframe):
    # Filtrando las columnas numÃ©ricas para sus histogramas
    numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns

    # Configurando el tamaÃ±o de la figura
    plt.figure(figsize=(20, 15))

    # Creando un histograma para cada caracterÃ­stica numÃ©rica
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(7, 5, i) # Ajustar segÃºn el nÃºmero de caracterÃ­sticas numÃ©ricas
        dataframe[feature].hist(bins=20, color=PALETTE_7_C[int(i%7)])
        plt.title(feature)

    plt.tight_layout()
    plt.show()


def show_hist(dataframe):
    # Filtrando las columnas numÃ©ricas para sus histogramas
    numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns

    # Configurando el tamaÃ±o de la figura
    plt.figure(figsize=(20, 15))

    # Creando un histograma para cada caracterÃ­stica numÃ©rica
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(7, 5, i) # Ajustar segÃºn el nÃºmero de caracterÃ­sticas numÃ©ricas
        dataframe[feature].hist(bins=20, color=PALETTE_7_C[int(i%7)])
        plt.title(feature)

    plt.tight_layout()
    plt.show()


train_df = pd.read_csv(TRAIN_DIR, index_col="id")
test_df = pd.read_csv(TEST_DIR, index_col = "id")
submission_df = pd.read_csv(SUBMISSION_DIR, index_col = "id")


train_df.head()


test_df.head()


show_corr_heatmap(train_df, "Train heatmap")
show_corr_heatmap(test_df, "Test heatmap")


cmap = mcolors.LinearSegmentedColormap.from_list("", PALETTE_2)
cmap_2 = mcolors.LinearSegmentedColormap.from_list("", [S_BLUE, PURPLE_DARJ])

# ConfiguraciÃ³n para Plotly
font_family = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=10), width=1000, height=500))


plt.figure(figsize=(6,4))
sns.countplot(x=TARGET, data=train_df, palette=PALETTE_3)
plt.title("Distribution of the target variable", fontsize=14, fontweight="bold")
plt.show()


features = ["RhythmScore", "AudioLoudness", "Energy", "BeatsPerMinute"]
for col in features:
    plt.figure(figsize=(6,4))
    sns.histplot(train_df[col], kde=True, color=PURPLE, bins=30)
    plt.title(f"Distribution {col}", fontsize=13, fontweight="bold")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.show()


plt.figure(figsize=(12,6))
sns.boxplot(data=train_df[features], palette=PALETTE_4)
plt.title("Distribution and outliers in selected variables", fontsize=14, fontweight="bold")
plt.show()


import pandas as pd
import numpy as np
from pathlib import Path

train = pd.read_csv(TRAIN_DIR)
test = pd.read_csv(TEST_DIR)
sample_sub = pd.read_csv(SUBMISSION_DIR)

# Descubre columnas clave desde sample_submission
id_col = sample_sub.columns[0]
target_col = sample_sub.columns[1]

assert id_col in train.columns and id_col in test.columns, "Falta la columna id en train/test"
assert target_col in train.columns, f"Falta objetivo {target_col} en train"

X = train.drop(columns=[target_col, id_col], errors="ignore")
y = train[target_col].astype(float)
X_test = test.drop(columns=[id_col], errors="ignore")

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"ID: {id_col} | Target: {target_col}")
print("Features:", list(X.columns))


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

poly_ridge = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),  # ~55-65 features
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=2.0, random_state=SEED))
])

# CV rÃ¡pido para sanity-check
kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
for tr_idx, va_idx in kf.split(X, y):
    poly_ridge.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    oof[va_idx] = poly_ridge.predict(X.iloc[va_idx])
rmse = mean_squared_error(y, oof, squared=False)
print(f"[PolyRidge d=2] CV RMSE (3-fold): {rmse:.5f}")

# Entrena full y predice test
poly_ridge.fit(X, y)
pred = poly_ridge.predict(X_test)

# Submission
sub = sample_sub.copy()
pred_df = pd.DataFrame({id_col: test[id_col].values, target_col: pred})
sub = sub.drop(columns=[target_col], errors="ignore").merge(pred_df, on=id_col, how="left")
sub.to_csv("submission_poly_ridge.csv", index=False)
print("âœ… submission_poly_ridge.csv generado.")



sub.head()


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

hgb = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.07,       # 0.06â€“0.10
    max_iter=250,             # subir a 400 si tienes margen
    max_leaf_nodes=31,
    min_samples_leaf=30,      # 20â€“50
    l2_regularization=0.15,   # 0.05â€“0.25
    validation_fraction=0.1,
    n_iter_no_change=30,
    random_state=SEED,
    max_bins=255
)

# CV rÃ¡pido (puedes saltarlo si vas con prisa)
kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
for tr_idx, va_idx in kf.split(X, y):
    hgb.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    oof[va_idx] = hgb.predict(X.iloc[va_idx])
rmse = mean_squared_error(y, oof, squared=False)
print(f"[HGB fast] CV RMSE (3-fold): {rmse:.5f}")

# Train full + predict
hgb.fit(X, y)
pred = hgb.predict(X_test)

sub = sample_sub.copy()
pred_df = pd.DataFrame({id_col: test[id_col].values, target_col: pred})
sub = sub.drop(columns=[target_col], errors="ignore").merge(pred_df, on=id_col, how="left")
sub.to_csv("submission_hgb_fast.csv", index=False)
print("âœ… submission_hgb_fast.csv generado.")



from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd

# Modelos base
poly_ridge = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=2.0, random_state=SEED))
])

elastic = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", ElasticNet(alpha=0.0008, l1_ratio=0.12, max_iter=2000, random_state=SEED))
])

hgb = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.07,
    max_iter=250,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    l2_regularization=0.15,
    validation_fraction=0.1,
    n_iter_no_change=30,
    random_state=SEED,
    max_bins=255
)

# Holdout pequeÃ±o para aprender pesos sin gastar mucho tiempo
Xs, Xv, ys, yv = train_test_split(X, y, test_size=0.2, random_state=SEED)

poly_ridge.fit(Xs, ys); pr_v = poly_ridge.predict(Xv)
elastic.fit(Xs, ys);     el_v = elastic.predict(Xv)
hgb.fit(Xs, ys);         hg_v = hgb.predict(Xv)

stack_X = np.vstack([pr_v, el_v, hg_v]).T
stacker = LinearRegression()
stacker.fit(stack_X, yv)
pred_stack_v = stacker.predict(stack_X)
rmse_stack = mean_squared_error(yv, pred_stack_v, squared=False)
print(f"[Stack holdout] RMSE: {rmse_stack:.5f} | Coefs [PolyRidge, Elastic, HGB]: {stacker.coef_}")

# Re-entrena en FULL y predice test
poly_ridge.fit(X, y); pr_t = poly_ridge.predict(X_test)
elastic.fit(X, y);     el_t = elastic.predict(X_test)
hgb.fit(X, y);         hg_t = hgb.predict(X_test)

stack_test = np.vstack([pr_t, el_t, hg_t]).T
pred_blend = stacker.predict(stack_test)

sub = sample_sub.copy()
pred_df = pd.DataFrame({id_col: test[id_col].values, target_col: pred_blend})
sub = sub.drop(columns=[target_col], errors="ignore").merge(pred_df, on=id_col, how="left")
sub.to_csv("submission.csv", index=False)
print("âœ… submission_stacked.csv generado.")



from __future__ import annotations
import os, warnings, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor

# Intentos de import opcionales
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")

# -------------------
# ConfiguraciÃ³n global
# -------------------
SEED = 42
K_FOLDS = 5
HPO_OPTUNA = False         # Pon True si quieres tunear LGB/XGB/CAT (caro en tiempo)
OPTUNA_TRIALS = 100        # Sube a 300+ si tu tiempo lo permite





rng = np.random.RandomState(SEED)

train = pd.read_csv(TRAIN_DIR)
test = pd.read_csv(TEST_DIR)
sample_sub = pd.read_csv(SUBMISSION_DIR)


# Detecta columnas id/target de forma robusta
id_col = sample_sub.columns[0]
if len(sample_sub.columns) >= 2:
    target_col = sample_sub.columns[1]
else:
    candidates = [c for c in train.columns if c not in test.columns and c != id_col]
    assert len(candidates) == 1, f"No se puede inferir target: {candidates}"
    target_col = candidates[0]
    if target_col not in sample_sub.columns:
        sample_sub[target_col] = np.nan




# Datasets
X = train.drop(columns=[target_col, id_col], errors="ignore")
y = train[target_col].astype(float)
X_test = test.drop(columns=[id_col], errors="ignore")

FEATURES = X.columns.tolist()
print(f"[INFO] ID={id_col} | TARGET={target_col} | n_features={len(FEATURES)} | train={train.shape} | test={test.shape}")




# -----------------
# Utilidades comunes
# -----------------
def rmse(a, b) -> float:
    return mean_squared_error(a, b, squared=False)

def kfold_indices(n: int, k: int, seed: int = SEED):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(np.arange(n)):
        yield tr_idx, va_idx

def save_submission(name: str, preds: np.ndarray):
    sub = sample_sub.copy()
    pred_df = pd.DataFrame({id_col: test[id_col].values, target_col: preds})
    sub = sub.drop(columns=[target_col], errors="ignore").merge(pred_df, on=id_col, how="left")
    out = f"{name}.csv"
    sub.to_csv(out, index=False)
    print(f"[SAVE] {out}")




# --------------------------------------
# Modelos base (con opciones de HPO)
# --------------------------------------
def train_ridge_poly(X, y, X_test, folds=K_FOLDS, degree=2, alphas=(0.5,1,2,3,5,8,12)):
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    for i, (tr, va) in enumerate(kfold_indices(len(X), folds, SEED), 1):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas, store_cv_values=False))
        ])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict(X.iloc[va])
        preds += pipe.predict(X_test) / folds
        print(f"[PolyRidge d={degree}] fold {i}/{folds} OK")
    return oof, preds

def train_elastic(X, y, X_test, folds=K_FOLDS, alpha=0.0012, l1_ratio=0.15, max_iter=4000):
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    for i, (tr, va) in enumerate(kfold_indices(len(X), folds, SEED), 1):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("pt", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("scale", StandardScaler()),
            ("elastic", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=SEED))
        ])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict(X.iloc[va])
        preds += pipe.predict(X_test) / folds
        print(f"[ElasticNet] fold {i}/{folds} OK")
    return oof, preds

def train_histgbr(X, y, X_test, folds=K_FOLDS, params: Dict | None = None):
    if params is None:
        params = dict(
            loss="squared_error",
            learning_rate=0.06,
            max_iter=600,               # sube si quieres apretar mÃ¡s
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=0.15,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=SEED,
            max_bins=255,
        )
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    for i, (tr, va) in enumerate(kfold_indices(len(X), folds, SEED), 1):
        model = HistGradientBoostingRegressor(**params)
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict(X.iloc[va])
        preds += model.predict(X_test) / folds
        print(f"[HistGBR] fold {i}/{folds} OK")
    return oof, preds

# LightGBM con HPO opcional
def train_lgbm(X, y, X_test, folds=K_FOLDS, hpo=False, trials=OPTUNA_TRIALS):
    if not HAS_LGBM:
        print("[WARN] lightgbm no disponible.")
        return np.zeros(len(X)), np.zeros(len(X_test))

    base_params = dict(
        objective="rmse",
        metric="rmse",
        boosting_type="gbdt",
        verbose=-1,
        seed=SEED,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        learning_rate=0.03,
        num_leaves=64,           # valor base razonable; HPO lo sobreescribe
    )

    def _objective(trial, Xtr, ytr, Xva, yva):
        params = base_params.copy()
        params.update(dict(
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 200),
            max_depth=trial.suggest_int("max_depth", -1, 14),
            lambda_l1=trial.suggest_float("lambda_l1", 0.0, 5.0),
            lambda_l2=trial.suggest_float("lambda_l2", 0.0, 5.0),
        ))
        dtr = lgb.Dataset(Xtr, label=ytr)
        dva = lgb.Dataset(Xva, label=yva)
        model = lgb.train(
            params,
            dtr,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            num_boost_round=20000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=300),
                lgb.log_evaluation(period=200)
            ],
        )
        p = model.predict(Xva, num_iteration=model.best_iteration)
        return mean_squared_error(yva, p, squared=False)

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for i, (tr, va) in enumerate(kfold_indices(len(X), folds, SEED), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        best_params = {}
        if hpo and HAS_OPTUNA:
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: _objective(trial, Xtr, ytr, Xva, yva), n_trials=trials, show_progress_bar=False)
            best = study.best_params
            best_params = dict(
                num_leaves=best["num_leaves"],
                min_data_in_leaf=best["min_data_in_leaf"],
                max_depth=best["max_depth"],
                lambda_l1=best["lambda_l1"],
                lambda_l2=best["lambda_l2"],
            )
            print(f"[LGB HPO] fold {i} best:", best_params)

        params = base_params.copy()
        params.update(best_params)

        dtr = lgb.Dataset(Xtr, label=ytr)
        dva = lgb.Dataset(Xva, label=yva)
        model = lgb.train(
            params,
            dtr,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            num_boost_round=20000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=300),
                lgb.log_evaluation(period=200)
            ],
        )
        oof[va] = model.predict(Xva, num_iteration=model.best_iteration)
        preds += model.predict(X_test, num_iteration=model.best_iteration) / folds
        print(f"[LightGBM] fold {i}/{folds} OK | best_iter={model.best_iteration}")

    return oof, preds


# XGBoost con HPO opcional
def train_xgb(X, y, X_test, folds=K_FOLDS, hpo=False, trials=OPTUNA_TRIALS):
    if not HAS_XGB:
        print("[WARN] xgboost no disponible.")
        return np.zeros(len(X)), np.zeros(len(X_test))
    base_params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        seed=SEED,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.9,
        n_estimators=20000
    )
    def _objective(trial, Xtr, ytr, Xva, yva):
        params = base_params.copy()
        params.update(dict(
            max_depth=trial.suggest_int("max_depth", 4, 14),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
        ))
        model = xgb.XGBRegressor(**params)
        model.fit(Xtr, ytr,
                  eval_set=[(Xtr, ytr), (Xva, yva)],
                  verbose=False,
                  early_stopping_rounds=300)
        p = model.predict(Xva)
        return rmse(yva, p)

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    for i, (tr, va) in enumerate(kfold_indices(len(X), folds, SEED), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        best_params = {}
        if hpo and HAS_OPTUNA:
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: _objective(trial, Xtr, ytr, Xva, yva), n_trials=trials, show_progress_bar=False)
            best = study.best_params
            best_params = dict(
                max_depth=best["max_depth"],
                min_child_weight=best["min_child_weight"],
                reg_alpha=best["reg_alpha"],
                reg_lambda=best["reg_lambda"],
                gamma=best["gamma"],
            )
            print(f"[XGB HPO] fold {i} best:", best_params)

        params = base_params.copy()
        params.update(best_params)
        model = xgb.XGBRegressor(**params)
        model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=200, early_stopping_rounds=300)
        oof[va] = model.predict(Xva)
        preds += model.predict(X_test) / folds
        print(f"[XGBoost] fold {i}/{folds} OK | best_ntrees={model.best_ntree_limit if hasattr(model,'best_ntree_limit') else 'n/a'}")
    return oof, preds

# CatBoost con HPO opcional
def train_cat(X, y, X_test, folds=K_FOLDS, hpo=False, trials=OPTUNA_TRIALS):
    if not HAS_CAT:
        print("[WARN] catboost no disponible.")
        return np.zeros(len(X)), np.zeros(len(X_test))
    base_params = dict(
        loss_function="RMSE",
        random_state=SEED,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        iterations=20000,
        od_type="Iter",
        od_wait=300,
        verbose=200
    )
    # HPO con Optuna (simple)
    def _objective(trial, Xtr, ytr, Xva, yva):
        params = base_params.copy()
        params.update(dict(
            depth=trial.suggest_int("depth", 6, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        ))
        model = CatBoostRegressor(**params)
        model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True, verbose=False)
        p = model.predict(Xva)
        return rmse(yva, p)

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    for i, (tr, va) in enumerate(kfold_indices(len(X), folds, SEED), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        best_params = {}
        if hpo and HAS_OPTUNA:
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: _objective(trial, Xtr, ytr, Xva, yva), n_trials=trials, show_progress_bar=False)
            best = study.best_params
            best_params = dict(depth=best["depth"], l2_leaf_reg=best["l2_leaf_reg"])
            print(f"[CAT HPO] fold {i} best:", best_params)

        params = base_params.copy()
        params.update(best_params)
        model = CatBoostRegressor(**params)
        model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True, verbose=200)
        oof[va] = model.predict(Xva)
        preds += model.predict(X_test) / folds
        print(f"[CatBoost] fold {i}/{folds} OK")
    return oof, preds




# ----------------------
# Entrenamiento modelos
# ----------------------
oof_dict: Dict[str, np.ndarray] = {}
pred_dict: Dict[str, np.ndarray] = {}

# 1) Lineales/Polinomiales
oof_pr, pred_pr = train_ridge_poly(X, y, X_test, folds=K_FOLDS, degree=2)
oof_dict["poly_ridge"] = oof_pr; pred_dict["poly_ridge"] = pred_pr
print("[PolyRidge] OOF RMSE:", rmse(y, oof_pr))

oof_el, pred_el = train_elastic(X, y, X_test, folds=K_FOLDS, alpha=0.0012, l1_ratio=0.15)
oof_dict["elastic"] = oof_el; pred_dict["elastic"] = pred_el
print("[ElasticNet] OOF RMSE:", rmse(y, oof_el))

# 2) Ã�rboles
oof_hg, pred_hg = train_histgbr(X, y, X_test, folds=K_FOLDS)
oof_dict["histgbr"] = oof_hg; pred_dict["histgbr"] = pred_hg
print("[HistGBR] OOF RMSE:", rmse(y, oof_hg))

if HAS_LGBM:
    oof_lgb, pred_lgb = train_lgbm(X, y, X_test, folds=K_FOLDS, hpo=HPO_OPTUNA)
    oof_dict["lgb"] = oof_lgb; pred_dict["lgb"] = pred_lgb
    print("[LGB] OOF RMSE:", rmse(y, oof_lgb))

if HAS_XGB:
    oof_xgb, pred_xgb = train_xgb(X, y, X_test, folds=K_FOLDS, hpo=HPO_OPTUNA)
    oof_dict["xgb"] = oof_xgb; pred_dict["xgb"] = pred_xgb
    print("[XGB] OOF RMSE:", rmse(y, oof_xgb))

if HAS_CAT:
    oof_cat, pred_cat = train_cat(X, y, X_test, folds=K_FOLDS, hpo=HPO_OPTUNA)
    oof_dict["cat"] = oof_cat; pred_dict["cat"] = pred_cat
    print("[CAT] OOF RMSE:", rmse(y, oof_cat))




# --------------------------------
# Stacking (RidgeCV) + NNLS blender
# --------------------------------
from numpy.linalg import lstsq

models = list(oof_dict.keys())
OOF = np.vstack([oof_dict[m] for m in models]).T  # (n_samples, n_models)
TST = np.vstack([pred_dict[m] for m in models]).T

# 1) Stacker L2 (RidgeCV sobre OOF)
import numpy as np
stacker = RidgeCV(alphas=np.logspace(-6, 2, 20))  # 1e-6 â€¦ 1e2
stacker.fit(OOF, y)
oof_stack = stacker.predict(OOF)
rmse_stack = mean_squared_error(y, oof_stack, squared=False)
pred_stack = stacker.predict(TST)
print(f"[Stack L2] OOF RMSE: {rmse_stack:.6f} | alpha={stacker.alpha_}")

# 2) NNLS (Non-Negative Least Squares) para pesos >= 0 y suma no forzada
# ImplementaciÃ³n simple vÃ­a SciPy opcional; aquÃ­ aproximamos con l-bfgs si SciPy no estÃ¡:
try:
    import scipy.optimize as sopt
    def nnls_weights(A, b):
        # Minimiza ||A w - b||^2 con w>=0
        n = A.shape[1]
        x0 = np.full(n, 1.0/n)
        bounds = [(0, None)] * n
        def obj(w):
            r = A.dot(w) - b
            return (r @ r)
        res = sopt.minimize(obj, x0=x0, method="L-BFGS-B", bounds=bounds)
        w = res.x
        return w / (w.sum() + 1e-12)
    w_nnls = nnls_weights(OOF, y.values)
except Exception:
    # Fallback: least squares + clipping a >=0 + renormalizaciÃ³n
    w_ls, *_ = lstsq(OOF, y.values, rcond=None)
    w_nnls = np.clip(w_ls, 0, None)
    w_nnls = w_nnls / (w_nnls.sum() + 1e-12)

oof_blend_nnls = OOF.dot(w_nnls)
rmse_nnls = rmse(y, oof_blend_nnls)
pred_blend_nnls = TST.dot(w_nnls)

print("[NNLS] OOF RMSE:", rmse_nnls, "| pesos =", dict(zip(models, np.round(w_nnls, 4))))

# SelecciÃ³n del mejor blender por OOF
if rmse_stack <= rmse_nnls:
    best_name = "stack_l2"
    best_pred = pred_stack
    best_rmse = rmse_stack
else:
    best_name = "blend_nnls"
    best_pred = pred_blend_nnls
    best_rmse = rmse_nnls

print(f"[BEST] {best_name} con OOF RMSE={best_rmse:.6f}")




# -------------
# Guardar CSVs
# -------------
# individuales
for m in models:
    save_submission(f"submission_{m}", pred_dict[m])

# blenders
save_submission("submission_stack_l2", pred_stack)
save_submission("submission_blend_nnls", pred_blend_nnls)

# ganador
save_submission("submission", best_pred)
print("âœ… submission_best.csv generado.")


best_pred.head()

