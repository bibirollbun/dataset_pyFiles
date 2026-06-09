#Install all dependencies
!pip install rdkit > /dev/null 2>&1
!pip install torch torchvision torchaudio torch-geometric > /dev/null 2>&1


#For handling data, data visualisation, basic math and algebra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Handling directory
import os

#For statistical analysis
import seaborn as sns

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors

#For Machine learning
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import random, numpy as np, torch


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Check the training datasets
df_train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
print("--- Display the training dataset top 5 rows---")
print(df_train.head())
print("-----------------------------------------------------------------------------")
#Check the test datasets
df_test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
print("--- Display the test dataset top 5 rows---")
print(df_test.head())


# Define target_variables
target_variables = ["Tg", "FFV", "Tc", "Density", "Rg"] 

# Use plain white background
sns.set_style("whitegrid")  # or "white" if you want no grid

# Clean inf values in target columns
df_train[target_variables] = df_train[target_variables].replace([np.inf, -np.inf], np.nan)

print("\n ---- Distribution of target variables ----")
num_targets = len(target_variables)
rows = (num_targets + 2) //3 #3 plots per row

plt.figure(figsize = (15, 5 * rows))

for i, col in enumerate(target_variables):
    plt.subplot(rows, 3, i + 1)
    sns.histplot(df_train[col].dropna(), kde=True, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of {col}", fontsize=12)
    plt.xlabel(col, fontsize = 10)
    plt.ylabel("Count", fontsize = 10)
    plt.xticks(fontsize = 9)
    plt.yticks(fontsize = 9)
plt.tight_layout
plt.show()


#Display all the missing entries in the dataframe pertaining to the columns 
import matplotlib.pyplot as plt
# Count missing values per column
missing_values = df_train.isna().sum()

# Bar chart of missing values per column
plt.figure(figsize=(8, 5))
missing_values.plot(kind='bar')
plt.title("Number of Missing Values per Column")
plt.ylabel("Count of NaN")
plt.xlabel("Column")
plt.xticks(rotation=45)
plt.tight_layout()
#plt.grid(True)
plt.show()


# Assuming your DataFrame is called df
total_rows = len(df_train)

# Calculate % of non-missing and missing for each column
percent_non_missing = 100 * df_train.notnull().sum() / total_rows
percent_missing = 100 * df_train.isnull().sum() / total_rows

# Combine into a nice summary table
summary = pd.DataFrame({
    'Non-missing %': percent_non_missing.round(2),
    'Missing %': percent_missing.round(2),
    'Non-missing Count': df_train.notnull().sum(),
    'Missing Count': df_train.isnull().sum()
})

print(summary)


# Correlation Matrix for Target Variables
print("\n--- Correlation Matrix of Target Variables ---")
plt.figure(figsize=(8, 6))
sns.heatmap(df_train[target_variables].corr(), annot=True, cmap='crest', fmt=".2f")
plt.title("Correlation Matrix of Target Properties", color="#FFFFFF")


# file: scripts/tg_xgb_rdkit_xgboost_fixed.py
"""Predict Tg from SMILES using RDKit Morgan fingerprints + XGBoost.

Notes
-----
- Expects a DataFrame named `df` (or `df_train` fallback) with columns: `SMILES`, `Tg`.
- Plots use matplotlib only (no seaborn) and each plot is separate.
- Invalid SMILES are mapped to all-zero fingerprints to avoid crashes.
- Cross-validation reports RMSE (mean ± std) across 5 shuffled folds.
"""
from __future__ import annotations

import sys
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from xgboost import XGBRegressor


# -----------------------------------------------------------------------------
# Configuration (adjust as needed)
# -----------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
RADIUS: int = 2
N_BITS: int = 1024
N_SPLITS: int = 5

# RDKit Morgan fingerprint generator (avoids deprecated AllChem API)
GEN = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


# -----------------------------------------------------------------------------
# Data acquisition utilities
# -----------------------------------------------------------------------------

def _pick_dataframe() -> pd.DataFrame:
    """Return df from the global namespace.

    Why: users may name the input DataFrame either `df` or `df_train`.
    """
    g = globals()
    if "df" in g and isinstance(g["df"], pd.DataFrame):
        return g["df"].copy()
    if "df_train" in g and isinstance(g["df_train"], pd.DataFrame):
        return g["df_train"].copy()
    raise NameError(
        "Please define a pandas DataFrame `df` (or `df_train`) with columns: SMILES, Tg.\n"
        "Example: df = pd.read_csv('your_data.csv')"
    )


def _validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(frame.columns)}")


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def smiles_to_fp(smiles: str, gen=GEN) -> np.ndarray:
    """Convert SMILES to Morgan bit vector using rdFingerprintGenerator (no deprecation)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Why: keep sample count aligned and avoid exceptions.
        return np.zeros((N_BITS,), dtype=np.uint8)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_features(df_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    _validate_columns(df_in, ["SMILES", "Tg"]) 
    df_tg = df_in.dropna(subset=["Tg"]).reset_index(drop=True)
    X = np.vstack([smiles_to_fp(s) for s in df_tg["SMILES"].astype(str)])
    y = df_tg["Tg"].to_numpy(dtype=float)
    return X, y


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

def make_model(random_state: int = RANDOM_STATE) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=2.0,
        reg_lambda=8.0,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# -----------------------------------------------------------------------------
# Plotting (each figure separate)
# -----------------------------------------------------------------------------

def plot_parity(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.6, edgecolor="none")
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")  # 45° reference
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.xlabel("True Tg")
    plt.ylabel("Predicted Tg")
    plt.title(f"Parity Plot — MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    res = y_pred - y_true
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, res, s=14, alpha=0.6, edgecolor="none")
    plt.axhline(0, linestyle="--")
    plt.xlabel("True Tg")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residuals vs. True Tg")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main execution (designed for notebooks or scripts)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # If running as a script, ensure the user provided/defined `df`/`df_train` in the same runtime.
    try:
        df_source = _pick_dataframe()
    except Exception as e:
        sys.exit(str(e))

    X, y = build_features(df_source)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = make_model(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)

    print(
        f"Test — MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | R²: {metrics['r2']:.3f}"
    )

    plot_parity(y_test, y_pred)
    plot_residuals(y_test, y_pred)

    # Cross-validation (5-fold, shuffled)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = -cross_val_score(
        make_model(random_state=RANDOM_STATE),
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    print(f"CV RMSE: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")



# file: scripts/tg_xgb_rdkit_xgboost_fixed.py
"""Predict Tg from SMILES using RDKit Morgan fingerprints + XGBoost.

Additions in this version
-------------------------
- Splits the input DataFrame into **labeled** (Tg not NA) and **unlabeled** (Tg NA).
- Trains an initial model on labeled data, evaluates it, and predicts Tg for the unlabeled rows.
- Trains a **final model** on the union of labeled data and pseudo-labeled (predicted) unlabeled data.
- Produces a combined parity plot: true-vs-pred for labeled points, and (pred, pred) markers for unlabeled points to visualize coverage.

Notes
-----
- Expects a DataFrame named `df` (or `df_train` fallback) with columns: `SMILES`, `Tg`.
- Plots use matplotlib only (no seaborn) and each plot is separate.
- Invalid SMILES are mapped to all-zero fingerprints to avoid crashes.
- Cross-validation reports RMSE (mean ± std) across 5 shuffled folds **on the labeled subset only**.
"""
from __future__ import annotations

import sys
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from xgboost import XGBRegressor


# -----------------------------------------------------------------------------
# Configuration (adjust as needed)
# -----------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
RADIUS: int = 2
N_BITS: int = 1024
N_SPLITS: int = 5

# RDKit Morgan fingerprint generator (avoids deprecated AllChem API)
GEN = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


# -----------------------------------------------------------------------------
# Data acquisition utilities
# -----------------------------------------------------------------------------

def _pick_dataframe() -> pd.DataFrame:
    """Return df from the global namespace.

    Why: users may name the input DataFrame either `df` or `df_train`.
    """
    g = globals()
    if "df" in g and isinstance(g["df"], pd.DataFrame):
        return g["df"].copy()
    if "df_train" in g and isinstance(g["df_train"], pd.DataFrame):
        return g["df_train"].copy()
    raise NameError(
        "Please define a pandas DataFrame `df` (or `df_train`) with columns: SMILES, Tg.\n"
        "Example: df = pd.read_csv('your_data.csv')"
    )


def _validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(frame.columns)}")


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def smiles_to_fp(smiles: str, gen=GEN) -> np.ndarray:
    """Convert SMILES to Morgan bit vector using rdFingerprintGenerator (no deprecation)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Why: keep sample count aligned and avoid exceptions.
        return np.zeros((N_BITS,), dtype=np.uint8)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_features(df_in: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build X, y for **labeled** rows (drops NA Tg).

    Returns
    -------
    X : np.ndarray, shape (n_samples, N_BITS)
    y : np.ndarray, shape (n_samples,)
    """
    _validate_columns(df_in, ["SMILES", "Tg"])
    df_tg = df_in.dropna(subset=["Tg"]).reset_index(drop=True)
    X = np.vstack([smiles_to_fp(s) for s in df_tg["SMILES"].astype(str)])
    y = df_tg["Tg"].to_numpy(dtype=float)
    return X, y


def build_features_unlabeled(df_in: pd.DataFrame) -> np.ndarray:
    """Build X for **unlabeled** rows (Tg can be NA)."""
    _validate_columns(df_in, ["SMILES"])  # Tg may be absent/NA here
    X = np.vstack([smiles_to_fp(s) for s in df_in["SMILES"].astype(str)])
    return X


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

def make_model(random_state: int = RANDOM_STATE) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=2.0,
        reg_lambda=8.0,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# -----------------------------------------------------------------------------
# Plotting (each figure separate)
# -----------------------------------------------------------------------------

def plot_parity(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.6, edgecolor="none")
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")  # 45° reference
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.xlabel("True Tg")
    plt.ylabel("Predicted Tg")
    plt.title(f"Parity Plot — MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    res = y_pred - y_true
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, res, s=14, alpha=0.6, edgecolor="none")
    plt.axhline(0, linestyle="--")
    plt.xlabel("True Tg")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residuals vs. True Tg")
    plt.tight_layout()
    plt.show()


def plot_parity_with_all_data(
    y_true_labeled: np.ndarray,
    y_pred_labeled: np.ndarray,
    y_pred_unlabeled: np.ndarray | None = None,
) -> None:
    """Parity plot overlaying labeled data (true vs pred) and unlabeled predictions.

    Unlabeled points are plotted at (pred, pred) on the 45° line to visualize
    their coverage; metrics in the title are computed **only on labeled** data.
    """
    r2 = r2_score(y_true_labeled, y_pred_labeled)
    mae = mean_absolute_error(y_true_labeled, y_pred_labeled)
    rmse = mean_squared_error(y_true_labeled, y_pred_labeled, squared=False)

    # Determine plot bounds using everything available
    vals = [y_true_labeled.min(), y_true_labeled.max(), y_pred_labeled.min(), y_pred_labeled.max()]
    if y_pred_unlabeled is not None and y_pred_unlabeled.size > 0:
        vals.extend([y_pred_unlabeled.min(), y_pred_unlabeled.max()])
    mn, mx = float(min(vals)), float(max(vals))

    plt.figure(figsize=(6, 6))
    # Labeled points
    plt.scatter(y_true_labeled, y_pred_labeled, s=18, alpha=0.7, edgecolor="none", label="Labeled")
    # Unlabeled (pseudo-labeled) points on the diagonal
    if y_pred_unlabeled is not None and y_pred_unlabeled.size > 0:
        plt.scatter(
            y_pred_unlabeled,
            y_pred_unlabeled,
            s=10,
            alpha=0.25,
            edgecolor="none",
            label="Unlabeled (pred)"
        )
    # 45° reference
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.xlabel("True Tg (labeled) / Pred Tg (unlabeled)")
    plt.ylabel("Predicted Tg")
    plt.title(f"Parity Plot (All Data) — Labeled MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main execution (designed for notebooks or scripts)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # If running as a script, ensure the user provided/defined `df`/`df_train` in the same runtime.
    try:
        df_source = _pick_dataframe()
    except Exception as e:
        sys.exit(str(e))

    _validate_columns(df_source, ["SMILES", "Tg"])  # confirm expected columns

    # Split into labeled/unlabeled by Tg
    mask_labeled = df_source["Tg"].notna()
    df_labeled = df_source.loc[mask_labeled].reset_index(drop=False).rename(columns={"index": "orig_index"})
    df_unlabeled = df_source.loc[~mask_labeled].reset_index(drop=False).rename(columns={"index": "orig_index"})

    print(f"Labeled rows: {len(df_labeled)} | Unlabeled rows: {len(df_unlabeled)}")

    # ------------------------------------------------------------------
    # 1) Train/evaluate initial model on labeled data
    # ------------------------------------------------------------------
    X_lab, y_lab = build_features(df_labeled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_lab, y_lab, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    base_model = make_model(random_state=RANDOM_STATE)
    base_model.fit(X_train, y_train)

    y_test_pred = base_model.predict(X_test)
    metrics = evaluate_regression(y_test, y_test_pred)

    print(
        f"Initial (labeled-only) Test — MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | R²: {metrics['r2']:.3f}"
    )

    # Plots for initial model (labeled-only)
    plot_parity(y_test, y_test_pred)
    plot_residuals(y_test, y_test_pred)

    # Cross-validation on labeled subset
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = -cross_val_score(
        make_model(random_state=RANDOM_STATE),
        X_lab,
        y_lab,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    print(f"Labeled-only CV RMSE: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ------------------------------------------------------------------
    # 2) Predict unlabeled Tg and create pseudo-labels
    # ------------------------------------------------------------------
    if len(df_unlabeled) > 0:
        X_unlab = build_features_unlabeled(df_unlabeled)
        y_unlab_pred = base_model.predict(X_unlab)
        df_unlabeled_out = df_unlabeled.copy()
        df_unlabeled_out["Tg_pred"] = y_unlab_pred
        # Optional: write out pseudo-labeled predictions alongside original rows order
        # df_unlabeled_out.to_csv("unlabeled_with_predictions.csv", index=False)
    else:
        X_unlab = np.empty((0, N_BITS), dtype=np.uint8)
        y_unlab_pred = np.array([])
        df_unlabeled_out = df_unlabeled.copy()
        df_unlabeled_out["Tg_pred"] = []

    # ------------------------------------------------------------------
    # 3) Train final model on ALL data: labeled true + unlabeled pseudo-labels
    # ------------------------------------------------------------------
    if X_unlab.shape[0] > 0:
        X_all = np.vstack([X_lab, X_unlab])
        y_all = np.concatenate([y_lab, y_unlab_pred])
    else:
        X_all = X_lab
        y_all = y_lab

    final_model = make_model(random_state=RANDOM_STATE)
    final_model.fit(X_all, y_all)

    # Evaluate final model on the **labeled** subset (only place with ground-truth)
    y_lab_pred_final = final_model.predict(X_lab)
    final_metrics = evaluate_regression(y_lab, y_lab_pred_final)
    print(
        f"Final model (trained on labeled + pseudo-labeled) — Eval on labeled: "
        f"MAE: {final_metrics['mae']:.2f} | RMSE: {final_metrics['rmse']:.2f} | R²: {final_metrics['r2']:.3f}"
    )

    # Predict the unlabeled with the final model too (can differ slightly from base_model)
    if X_unlab.shape[0] > 0:
        y_unlab_pred_final = final_model.predict(X_unlab)
    else:
        y_unlab_pred_final = np.array([])

    # Combined parity plot with all data (labeled points + unlabeled on diagonal as (pred, pred))
    plot_parity_with_all_data(y_lab, y_lab_pred_final, y_pred_unlabeled=y_unlab_pred_final)

    # ------------------------------------------------------------------
    # 4) (Optional) Attach predictions to original df for downstream use
    # ------------------------------------------------------------------
    # Start with the original rows; add a column Tg_pred_final filled where we have predictions
    df_out = df_source.copy()
    df_out["Tg_pred_final"] = np.nan

    # Fill predictions for labeled rows (from final model)
    if len(df_labeled) > 0:
        # df_labeled was reset_index with orig_index preserved; map back
        df_out.loc[df_labeled["orig_index"].to_numpy(), "Tg_pred_final"] = y_lab_pred_final

    # Fill predictions for unlabeled rows (from final model)
    if len(df_unlabeled) > 0:
        df_out.loc[df_unlabeled["orig_index"].to_numpy(), "Tg_pred_final"] = y_unlab_pred_final

    # You may want to persist this
    # df_out.to_csv("df_with_final_predictions.csv", index=False)

    print("Done. Columns added: 'Tg_pred_final' (predictions from final model).")



# file: scripts/tg_xgb_rdkit_xgboost_fixed.py
"""Predict Tg from SMILES using RDKit Morgan fingerprints + XGBoost.

Additions in this version
-------------------------
- Splits the input DataFrame into **labeled** (Tg not NA) and **unlabeled** (Tg NA).
- Trains an initial model on labeled data, evaluates it, and predicts Tg for the unlabeled rows.
- Trains a **final model** on the union of labeled data and pseudo-labeled (predicted) unlabeled data.
- Produces a combined parity plot: true-vs-pred for labeled points, and (pred, pred) markers for unlabeled points to visualize coverage.
- Adds a **probabilistic pseudo-labeling** step via an XGB ensemble to estimate mean ± std for unlabeled, keeps the most confident, and uses **sample weights**.
- Adds a **model comparison** suite (RF, ExtraTrees, HistGB, Ridge, SVR, optional KNN) vs the baseline XGB with a leaderboard.

Notes
-----
- Expects a DataFrame named `df` (or `df_train` fallback) with columns: `SMILES`, `Tg`.
- Plots use matplotlib only (no seaborn) and each plot is separate.
- Invalid SMILES are mapped to all-zero fingerprints to avoid crashes.
- Cross-validation reports RMSE (mean ± std) across 5 shuffled folds **on the labeled subset only**.
"""
from __future__ import annotations

import sys
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor


# -----------------------------------------------------------------------------
# Configuration (adjust as needed)
# -----------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
RADIUS: int = 2
N_BITS: int = 1024
N_SPLITS: int = 5

# Probabilistic pseudo-labeling defaults
ENSEMBLE_SIZE: int = 7                 # number of models in the ensemble
PSEUDO_KEEP_QUANTILE: float = 0.5      # keep the most confident 50% of unlabeled by std
PSEUDO_WEIGHT_SCALING_Q: float = 0.9   # scale for weights based on 90th percentile of std
MIN_PSEUDO_WEIGHT: float = 0.2         # floor for pseudo-label sample weights
MAX_PSEUDO_WEIGHT: float = 1.0         # ceiling for pseudo-label sample weights
EPS: float = 1e-9

# Model comparison settings
COMPARE_MODELS = ("xgb", "rf", "hgb", "etr", "ridge", "svr")  # add "knn" if desired
PLOT_ALL_MODELS: bool = False  # set True to draw parity/residuals for every model

# RDKit Morgan fingerprint generator (avoids deprecated AllChem API)
GEN = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


# -----------------------------------------------------------------------------
# Data acquisition utilities
# -----------------------------------------------------------------------------

def _pick_dataframe() -> pd.DataFrame:
    """Return df from the global namespace.

    Why: users may name the input DataFrame either `df` or `df_train`.
    """
    g = globals()
    if "df" in g and isinstance(g["df"], pd.DataFrame):
        return g["df"].copy()
    if "df_train" in g and isinstance(g["df_train"], pd.DataFrame):
        return g["df_train"].copy()
    raise NameError(
        "Please define a pandas DataFrame `df` (or `df_train`) with columns: SMILES, Tg.\n"
        "Example: df = pd.read_csv('your_data.csv')"
    )


def _validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(frame.columns)}")


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def smiles_to_fp(smiles: str, gen=GEN) -> np.ndarray:
    """Convert SMILES to Morgan bit vector using rdFingerprintGenerator (no deprecation)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Why: keep sample count aligned and avoid exceptions.
        return np.zeros((N_BITS,), dtype=np.uint8)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_features(df_in: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build X, y for **labeled** rows (drops NA Tg).

    Returns
    -------
    X : np.ndarray, shape (n_samples, N_BITS)
    y : np.ndarray, shape (n_samples,)
    """
    _validate_columns(df_in, ["SMILES", "Tg"])
    df_tg = df_in.dropna(subset=["Tg"]).reset_index(drop=True)
    X = np.vstack([smiles_to_fp(s) for s in df_tg["SMILES"].astype(str)])
    y = df_tg["Tg"].to_numpy(dtype=float)
    return X, y


def build_features_unlabeled(df_in: pd.DataFrame) -> np.ndarray:
    """Build X for **unlabeled** rows (Tg can be NA)."""
    _validate_columns(df_in, ["SMILES"])  # Tg may be absent/NA here
    X = np.vstack([smiles_to_fp(s) for s in df_in["SMILES"].astype(str)])
    return X


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

def make_model(random_state: int = RANDOM_STATE) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=2.0,
        reg_lambda=8.0,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# -----------------------------------------------------------------------------
# Probabilistic / ensemble helpers
# -----------------------------------------------------------------------------

def train_ensemble(X: np.ndarray, y: np.ndarray, seeds: Iterable[int]) -> list[XGBRegressor]:
    models: list[XGBRegressor] = []
    for s in seeds:
        m = make_model(random_state=int(s))
        m.fit(X, y)
        models.append(m)
    return models


def predict_ensemble(models: list[XGBRegressor], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(models) == 0:
        return np.array([]), np.array([])
    preds = np.column_stack([m.predict(X) for m in models])  # shape = (n, k)
    mean = preds.mean(axis=1)
    std = preds.std(axis=1, ddof=1) if preds.shape[1] > 1 else np.zeros_like(mean)
    return mean, std


def compute_pseudo_weights(
    std: np.ndarray,
    scaling_quantile: float = PSEUDO_WEIGHT_SCALING_Q,
    min_w: float = MIN_PSEUDO_WEIGHT,
    max_w: float = MAX_PSEUDO_WEIGHT,
) -> np.ndarray:
    if std.size == 0:
        return std
    scale = np.quantile(std, scaling_quantile) + EPS
    w = np.exp(- (std / scale) ** 2)  # in (0, 1]
    w = min_w + (max_w - min_w) * w   # map to [min_w, max_w]
    return w


# -----------------------------------------------------------------------------
# Model registry / utilities for comparison
# -----------------------------------------------------------------------------

def get_model_builders():
    """Return dict mapping model key to a builder callable."""
    def make_xgb():
        return make_model(random_state=RANDOM_STATE)

    def make_rf():
        return RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def make_hgb():
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_depth=None,
            l2_regularization=0.0,
            random_state=RANDOM_STATE,
        )

    def make_etr():
        return ExtraTreesRegressor(
            n_estimators=800,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def make_ridge():
        return Ridge(alpha=5.0)

    def make_svr():
        return SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")

    def make_knn():
        return KNeighborsRegressor(n_neighbors=10, weights="distance", p=2)

    registry = {
        "xgb": make_xgb,
        "rf": make_rf,
        "hgb": make_hgb,
        "etr": make_etr,
        "ridge": make_ridge,
        "svr": make_svr,
        "knn": make_knn,
    }
    return registry


def safe_fit(model, X, y, sample_weight=None):
    """Fit model; pass sample_weight if supported, else fall back silently."""
    try:
        if sample_weight is not None:
            return model.fit(X, y, sample_weight=sample_weight)
        return model.fit(X, y)
    except TypeError:
        return model.fit(X, y)


# -----------------------------------------------------------------------------
# Plotting (each figure separate)
# -----------------------------------------------------------------------------

def plot_parity(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.6, edgecolor="none")
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")  # 45° reference
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.xlabel("True Tg")
    plt.ylabel("Predicted Tg")
    plt.title(f"Parity Plot — MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    res = y_pred - y_true
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, res, s=14, alpha=0.6, edgecolor="none")
    plt.axhline(0, linestyle="--")
    plt.xlabel("True Tg")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residuals vs. True Tg")
    plt.tight_layout()
    plt.show()


def plot_parity_with_all_data(
    y_true_labeled: np.ndarray,
    y_pred_labeled: np.ndarray,
    y_pred_unlabeled: np.ndarray | None = None,
    y_std_unlabeled: np.ndarray | None = None,
) -> None:
    """Parity plot overlaying labeled data (true vs pred) and unlabeled predictions.

    Unlabeled points are plotted at (pred, pred) on the 45° line. If standard
    deviations are provided, draw ±1σ error bars on both axes.
    """
    r2 = r2_score(y_true_labeled, y_pred_labeled)
    mae = mean_absolute_error(y_true_labeled, y_pred_labeled)
    rmse = mean_squared_error(y_true_labeled, y_pred_labeled, squared=False)

    # Determine plot bounds using everything available
    vals = [y_true_labeled.min(), y_true_labeled.max(), y_pred_labeled.min(), y_pred_labeled.max()]
    if y_pred_unlabeled is not None and y_pred_unlabeled.size > 0:
        vals.extend([y_pred_unlabeled.min(), y_pred_unlabeled.max()])
    mn, mx = float(min(vals)), float(max(vals))

    plt.figure(figsize=(6, 6))
    # Labeled points
    plt.scatter(y_true_labeled, y_pred_labeled, s=18, alpha=0.7, edgecolor="none", label="Labeled")

    # Unlabeled (pseudo-labeled) points on the diagonal
    if y_pred_unlabeled is not None and y_pred_unlabeled.size > 0:
        if y_std_unlabeled is not None and y_std_unlabeled.size == y_pred_unlabeled.size:
            plt.errorbar(
                y_pred_unlabeled,
                y_pred_unlabeled,
                xerr=y_std_unlabeled,
                yerr=y_std_unlabeled,
                fmt="o",
                ms=4,
                alpha=0.25,
                capsize=0,
                linestyle="none",
                label="Unlabeled (mean ± 1σ)"
            )
        else:
            plt.scatter(
                y_pred_unlabeled,
                y_pred_unlabeled,
                s=10,
                alpha=0.25,
                edgecolor="none",
                label="Unlabeled (pred)"
            )

    # 45° reference
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.xlabel("True Tg (labeled) / Pred Tg (unlabeled)")
    plt.ylabel("Predicted Tg")
    plt.title(f"Parity Plot (All Data) — Labeled MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_parity_unlabeled(y_pred_unlabeled: np.ndarray | None, y_std_unlabeled: np.ndarray | None = None) -> None:
    """Standalone parity-style plot for unlabeled data using (pred, pred).

    If there are no unlabeled predictions, this is a no-op.
    """
    if y_pred_unlabeled is None or getattr(y_pred_unlabeled, "size", 0) == 0:
        return

    mn = float(np.min(y_pred_unlabeled))
    mx = float(np.max(y_pred_unlabeled))

    plt.figure(figsize=(6, 6))
    if y_std_unlabeled is not None and y_std_unlabeled.size == y_pred_unlabeled.size:
        plt.errorbar(
            y_pred_unlabeled,
            y_pred_unlabeled,
            xerr=y_std_unlabeled,
            yerr=y_std_unlabeled,
            fmt="o",
            ms=4,
            alpha=0.35,
            capsize=0,
            linestyle="none"
        )
    else:
        plt.scatter(y_pred_unlabeled, y_pred_unlabeled, s=12, alpha=0.35, edgecolor="none")

    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.xlabel("Predicted Tg")
    plt.ylabel("Predicted Tg")
    plt.title(f"Parity Plot — Unlabeled (mean ± 1σ if available), n={y_pred_unlabeled.size}")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main execution (designed for notebooks or scripts)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # If running as a script, ensure the user provided/defined `df`/`df_train` in the same runtime.
    try:
        df_source = _pick_dataframe()
    except Exception as e:
        sys.exit(str(e))

    _validate_columns(df_source, ["SMILES", "Tg"])  # confirm expected columns

    # Split into labeled/unlabeled by Tg
    mask_labeled = df_source["Tg"].notna()
    df_labeled = df_source.loc[mask_labeled].reset_index(drop=False).rename(columns={"index": "orig_index"})
    df_unlabeled = df_source.loc[~mask_labeled].reset_index(drop=False).rename(columns={"index": "orig_index"})

    print(f"Labeled rows: {len(df_labeled)} | Unlabeled rows: {len(df_unlabeled)}")

    # ------------------------------------------------------------------
    # 1) Train/evaluate initial model on labeled data
    # ------------------------------------------------------------------
    X_lab, y_lab = build_features(df_labeled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_lab, y_lab, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    base_model = make_model(random_state=RANDOM_STATE)
    base_model.fit(X_train, y_train)

    y_test_pred = base_model.predict(X_test)
    metrics = evaluate_regression(y_test, y_test_pred)

    print(
        f"Initial (labeled-only) Test — MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | R²: {metrics['r2']:.3f}"
    )

    # Plots for initial model (labeled-only)
    plot_parity(y_test, y_test_pred)
    plot_residuals(y_test, y_test_pred)

    # Cross-validation on labeled subset
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = -cross_val_score(
        make_model(random_state=RANDOM_STATE),
        X_lab,
        y_lab,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    print(f"Labeled-only CV RMSE: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ------------------------------------------------------------------
    # 2) Probabilistic pseudo-labels via ensemble uncertainty
    # ------------------------------------------------------------------
    if len(df_unlabeled) > 0:
        seeds = [RANDOM_STATE + i for i in range(ENSEMBLE_SIZE)]
        ensemble = train_ensemble(X_lab, y_lab, seeds)

        X_unlab = build_features_unlabeled(df_unlabeled)
        y_unlab_mean, y_unlab_std = predict_ensemble(ensemble, X_unlab)

        # Keep only the most confident fraction by std
        std_threshold = float(np.quantile(y_unlab_std, PSEUDO_KEEP_QUANTILE))
        keep_mask = y_unlab_std <= std_threshold
        kept_fraction = float(keep_mask.mean())

        X_unlab_kept = X_unlab[keep_mask]
        y_unlab_mean_kept = y_unlab_mean[keep_mask]
        y_unlab_std_kept = y_unlab_std[keep_mask]

        w_unlab_kept = compute_pseudo_weights(
            y_unlab_std_kept,
            scaling_quantile=PSEUDO_WEIGHT_SCALING_Q,
            min_w=MIN_PSEUDO_WEIGHT,
            max_w=MAX_PSEUDO_WEIGHT,
        )

        print(
            f"Unlabeled total: {len(df_unlabeled)} | kept: {keep_mask.sum()} "
            f"({kept_fraction:.0%}) with std ≤ {std_threshold:.3f}"
        )
    else:
        X_unlab = np.empty((0, N_BITS), dtype=np.uint8)
        y_unlab_mean = np.array([])
        y_unlab_std = np.array([])
        X_unlab_kept = X_unlab
        y_unlab_mean_kept = y_unlab_mean
        y_unlab_std_kept = y_unlab_std
        w_unlab_kept = np.array([])

    # ------------------------------------------------------------------
    # 3) Train final model on labeled true + CONFIDENT pseudo-labels (weighted)
    #    (Baseline XGB)
    # ------------------------------------------------------------------
    if X_unlab_kept.shape[0] > 0:
        X_all = np.vstack([X_lab, X_unlab_kept])
        y_all = np.concatenate([y_lab, y_unlab_mean_kept])
        w_all = np.concatenate([np.ones_like(y_lab, dtype=float), w_unlab_kept])
    else:
        X_all = X_lab
        y_all = y_lab
        w_all = np.ones_like(y_lab, dtype=float)

    # Baseline XGB final model
    final_model = make_model(random_state=RANDOM_STATE)
    final_model.fit(X_all, y_all, sample_weight=w_all)

    # Evaluate baseline final model on the **labeled** subset (only place with ground-truth)
    y_lab_pred_final = final_model.predict(X_lab)
    final_metrics = evaluate_regression(y_lab, y_lab_pred_final)
    print(
        f"Final XGB (labeled + confident pseudo-labels) — Eval on labeled: "
        f"MAE: {final_metrics['mae']:.2f} | RMSE: {final_metrics['rmse']:.2f} | R²: {final_metrics['r2']:.3f}"
    )

    # 3b) Model comparison against baseline XGB
    builders = get_model_builders()
    to_run = [m for m in COMPARE_MODELS if m in builders]

    print("\n=== Model Comparison (CV on labeled; final fit on labeled+pseudo with weights) ===")
    results = []
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for key in to_run:
        make_fn = builders[key]
        # CV on labeled (no weights)
        est_cv = make_fn()
        cv_scores = -cross_val_score(
            est_cv, X_lab, y_lab, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        cv_rmse_mean = float(cv_scores.mean())
        cv_rmse_std = float(cv_scores.std())

        # Final fit on labeled+pseudo (weighted where available)
        est_final = make_fn()
        safe_fit(est_final, X_all, y_all, sample_weight=w_all)

        y_lab_pred_this = est_final.predict(X_lab)
        met = evaluate_regression(y_lab, y_lab_pred_this)

        results.append({
            "model": key,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
            "eval_mae": float(met["mae"]),
            "eval_rmse": float(met["rmse"]),
            "eval_r2": float(met["r2"]),
            "estimator": est_final,
            "y_lab_pred": y_lab_pred_this,
        })

        if PLOT_ALL_MODELS:
            print(f"Plotting parity for model: {key}")
            plot_parity(y_lab, y_lab_pred_this)

    # Leaderboard
    results_sorted = sorted(results, key=lambda d: d["cv_rmse_mean"])  # lower is better
    print("\nLeaderboard (sorted by CV RMSE on labeled):")
    for i, r in enumerate(results_sorted, 1):
        print(
            f"{i:>2}. {r['model']:<4} | CV RMSE: {r['cv_rmse_mean']:.3f} ± {r['cv_rmse_std']:.3f} | "
            f"Eval RMSE: {r['eval_rmse']:.3f} | MAE: {r['eval_mae']:.3f} | R²: {r['eval_r2']:.3f}"
        )

    # Choose best by CV RMSE
    if len(results_sorted) > 0:
        best = results_sorted[0]
        print(f"\nBest model by CV RMSE: {best['model']}")
        # Optional: parity plot for best model on labeled set
        if not PLOT_ALL_MODELS:
            plot_parity(y_lab, best["y_lab_pred"])  # labeled-only

        # Predict unlabeled with the best model for reference (point preds)
        if X_unlab.shape[0] > 0:
            y_unlab_pred_best = best["estimator"].predict(X_unlab)
        else:
            y_unlab_pred_best = np.array([])

    # Predict the unlabeled with the baseline final model (point predictions)
    if X_unlab.shape[0] > 0:
        y_unlab_pred_final = final_model.predict(X_unlab)
    else:
        y_unlab_pred_final = np.array([])

    # Combined parity plot with all data (labeled points + unlabeled on diagonal)
    plot_parity_with_all_data(
        y_lab,
        y_lab_pred_final,
        y_pred_unlabeled=y_unlab_mean,
        y_std_unlabeled=y_unlab_std,
    )

    # Standalone parity plot for the unlabeled predictions (mean ± 1σ)
    plot_parity_unlabeled(y_unlab_mean, y_unlab_std)

    # ------------------------------------------------------------------
    # 4) (Optional) Attach predictions to original df for downstream use
    # ------------------------------------------------------------------
    df_out = df_source.copy()
    # Final point predictions (from final_model)
    df_out["Tg_pred_final"] = np.nan
    # Ensemble uncertainty summaries (for unlabeled rows)
    df_out["Tg_pred_mean_ensemble"] = np.nan
    df_out["Tg_pred_std_ensemble"] = np.nan

    # Fill predictions for labeled rows (from final model)
    if len(df_labeled) > 0:
        # df_labeled was reset_index with orig_index preserved; map back
        df_out.loc[df_labeled["orig_index"].to_numpy(), "Tg_pred_final"] = y_lab_pred_final

    # Fill predictions for unlabeled rows (from final model + ensemble stats)
    if len(df_unlabeled) > 0:
        df_out.loc[df_unlabeled["orig_index"].to_numpy(), "Tg_pred_final"] = y_unlab_pred_final
        if y_unlab_mean.size > 0:
            df_out.loc[df_unlabeled["orig_index"].to_numpy(), "Tg_pred_mean_ensemble"] = y_unlab_mean
        if y_unlab_std.size > 0:
            df_out.loc[df_unlabeled["orig_index"].to_numpy(), "Tg_pred_std_ensemble"] = y_unlab_std

    # Add per-model predictions as extra columns (optional, useful for audits)
    if 'results' in locals():
        for r in results:
            col_lab = f"Tg_pred_final_{r['model']}_labeled"
            col_unlab = f"Tg_pred_final_{r['model']}_unlabeled"
            df_out[col_lab] = np.nan
            df_out[col_unlab] = np.nan
            # labeled rows
            if len(df_labeled) > 0:
                df_out.loc[df_labeled["orig_index"].to_numpy(), col_lab] = r["y_lab_pred"]
            # unlabeled rows (point preds)
            if len(df_unlabeled) > 0:
                try:
                    y_unlab_pred_model = r["estimator"].predict(X_unlab)
                except Exception:
                    y_unlab_pred_model = np.array([])
                if getattr(y_unlab_pred_model, 'size', 0) == len(df_unlabeled):
                    df_out.loc[df_unlabeled["orig_index"].to_numpy(), col_unlab] = y_unlab_pred_model

    # You may want to persist this
    # df_out.to_csv("df_with_final_predictions.csv", index=False)

    print("Done. Columns added: 'Tg_pred_final' (final XGB model), per-model prediction columns, and ensemble uncertainty (unlabeled only).")



# file: scripts/tg_bayesian_active_learning.py
"""
Bayesian Active Learning for Tg-from-SMILES with RDKit Morgan fingerprints.

What this script provides
-------------------------
1) Train a BayesianRidge on the **labeled** subset (Tg present) using scaled fingerprints.
2) Score **unlabeled** pool with Bayesian posterior:
   - Predictive mean and std (aleatoric + parameter uncertainty)
   - Mutual Information (BALD-style for Gaussian regression):
       I[y;w|x] = 0.5 * log(1 + alpha * x^T Sigma x)
     (alpha is noise precision; Sigma is posterior covariance of weights)
3) Batch acquisition: select a set of candidates to send to lab using
   uncertainty or MI, with an optional **diversity** term based on Tanimoto distance
   between fingerprints via greedy farthest-first.
4) (Demo option) If you don't have new labels yet, it trains a final model using
   labeled + pseudo-labeled (Bayesian predictive means) to generate parity plots.
   In a *real* AL loop, replace those pseudo-labels with lab-measured Tg after each round.

Inputs
------
- Expects a DataFrame named `df` (or `df_train`) with columns: `SMILES`, `Tg` (Tg can be NA for pool).

Outputs
-------
- Prints a ranked table of acquisition candidates with SMILES, acquisition score, predictive mean, std.
- Generates parity plots for the demo final model (labeled truth vs pred, plus unlabeled on diagonal).

Notes
-----
- Plots: matplotlib only; each plot is separate.
- Invalid SMILES → all-zero fingerprints to avoid crashes.
"""
from __future__ import annotations

import sys
from typing import Iterable, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# Optional downstream predictor
from xgboost import XGBRegressor

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
RADIUS: int = 2
N_BITS: int = 1024
N_SPLITS: int = 5

ACQ: Literal["variance", "mi"] = "mi"          # acquisition: predictive variance or mutual information
BATCH_SIZE: int = 32                              # number to suggest per round
DIVERSITY_LAMBDA: float = 0.25                    # weight for diversity term (0 → no diversity)
ROUNDS: int = 1                                   # how many AL rounds to *suggest*

# Demo: build a final predictor for parity plots using pseudo-labels (means)
BUILD_DEMO_FINAL: bool = True
FINAL_MODEL: Literal["xgb", "bayes"] = "bayes"

# RDKit Morgan fingerprint generator
GEN = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _pick_dataframe() -> pd.DataFrame:
    g = globals()
    if "df" in g and isinstance(g["df"], pd.DataFrame):
        return g["df"].copy()
    if "df_train" in g and isinstance(g["df_train"], pd.DataFrame):
        return g["df_train"].copy()
    raise NameError(
        "Please define a pandas DataFrame `df` (or `df_train`) with columns: SMILES, Tg.\n"
        "Example: df = pd.read_csv('your_data.csv')"
    )


def _validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(frame.columns)}")


def smiles_to_fp(smiles: str, gen=GEN) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((N_BITS,), dtype=np.uint8)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_features(df_in: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    _validate_columns(df_in, ["SMILES", "Tg"])
    df_tg = df_in.dropna(subset=["Tg"]).reset_index(drop=True)
    X = np.vstack([smiles_to_fp(s) for s in df_tg["SMILES"].astype(str)])
    y = df_tg["Tg"].to_numpy(dtype=float)
    return X, y


def build_features_unlabeled(df_in: pd.DataFrame) -> np.ndarray:
    _validate_columns(df_in, ["SMILES"])  # Tg can be NA
    X = np.vstack([smiles_to_fp(s) for s in df_in["SMILES"].astype(str)])
    return X

# -----------------------------------------------------------------------------
# Bayesian core
# -----------------------------------------------------------------------------

def fit_bayesian_ridge(X: np.ndarray, y: np.ndarray) -> tuple[BayesianRidge, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.astype(float))
    br = BayesianRidge(n_iter=800, tol=1e-4, compute_score=True)
    br.fit(Xs, y)
    return br, scaler


def posterior_predict(br: BayesianRidge, scaler: StandardScaler, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        return np.array([]), np.array([])
    Xs = scaler.transform(X.astype(float))
    mean, std = br.predict(Xs, return_std=True)
    return mean, std


def mutual_information_scores(br: BayesianRidge, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """BALD-style MI for Gaussian regression: 0.5*log(1 + alpha * x^T Sigma x).

    For scikit-learn BayesianRidge, attributes:
      - alpha_: noise precision; noise variance = 1/alpha_
      - sigma_: posterior covariance matrix of weights (shape d x d)
    We compute s2_param = diag(Xs * Sigma * Xs^T). MI is monotone in s2_param.
    """
    if X.size == 0:
        return np.array([])
    Xs = scaler.transform(X.astype(float))  # (n, d)
    Sigma = getattr(br, "sigma_", None)
    alpha = getattr(br, "alpha_", None)
    if Sigma is None or alpha is None:
        # Fallback: use predictive std^2 minus noise variance
        mean, std = br.predict(Xs, return_std=True)
        noise_var = 1.0 / max(getattr(br, "alpha_", 1.0), 1e-12)
        s2_param = np.maximum(std**2 - noise_var, 0.0)
    else:
        # s2_param = diag(Xs * Sigma * Xs^T)
        s2_param = np.einsum("ij,jk,ik->i", Xs, Sigma, Xs)
    # MI ~ 0.5 * log(1 + alpha * s2_param)
    if alpha is None:
        alpha = getattr(br, "alpha_", 1.0)
    mi = 0.5 * np.log1p(np.maximum(alpha, 1e-12) * np.maximum(s2_param, 0.0))
    return mi

# -----------------------------------------------------------------------------
# Diversity & batch selection
# -----------------------------------------------------------------------------

def tanimoto_distance(a: np.ndarray, b: np.ndarray) -> float:
    # a,b are binary 0/1 vectors (uint8). Handle empty union.
    inter = int((a & b).sum())
    sa = int(a.sum()); sb = int(b.sum())
    denom = sa + sb - inter
    if denom <= 0:
        return 1.0  # treat as maximally distant if both zero
    sim = inter / float(denom)
    return 1.0 - sim


def greedy_acquire(
    X_bits_pool: np.ndarray,
    scores: np.ndarray,
    k: int,
    lambda_div: float = DIVERSITY_LAMBDA,
) -> np.ndarray:
    """Greedy batch selection maximizing score + diversity (farthest-first).

    Diversity measured via Tanimoto distance between bit fingerprints.
    """
    n = X_bits_pool.shape[0]
    if n == 0 or k <= 0:
        return np.array([], dtype=int)
    k = min(k, n)

    # Start from the highest-scoring item
    order = np.argsort(-scores)
    selected = [int(order[0])]

    # Precompute distances lazily as needed
    min_dist = np.full(n, np.inf)
    ref = X_bits_pool[selected[0]]
    for i in range(n):
        if i == selected[0]:
            min_dist[i] = 0.0
        else:
            min_dist[i] = tanimoto_distance(X_bits_pool[i], ref)

    while len(selected) < k:
        # Score with diversity bonus
        total_score = scores + lambda_div * np.sqrt(np.maximum(min_dist, 0.0))
        total_score[selected] = -np.inf  # don't reselect
        j = int(np.argmax(total_score))
        selected.append(j)
        # Update min_dist with new center j
        ref = X_bits_pool[j]
        for i in range(n):
            if i == j:
                continue
            d = tanimoto_distance(X_bits_pool[i], ref)
            if d < min_dist[i]:
                min_dist[i] = d

    return np.array(selected, dtype=int)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, title_prefix: str = "Parity Plot") -> None:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.6, edgecolor="none")
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.xlabel("True Tg")
    plt.ylabel("Predicted Tg")
    plt.title(f"{title_prefix} — MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    plt.tight_layout()
    plt.show()


def plot_parity_with_unlabeled(
    y_true_labeled: np.ndarray,
    y_pred_labeled: np.ndarray,
    y_pred_unlabeled: np.ndarray | None = None,
    title_prefix: str = "Parity (All Data)",
) -> None:
    r2 = r2_score(y_true_labeled, y_pred_labeled)
    mae = mean_absolute_error(y_true_labeled, y_pred_labeled)
    rmse = mean_squared_error(y_true_labeled, y_pred_labeled, squared=False)

    vals = [y_true_labeled.min(), y_true_labeled.max(), y_pred_labeled.min(), y_pred_labeled.max()]
    if y_pred_unlabeled is not None and y_pred_unlabeled.size > 0:
        vals.extend([y_pred_unlabeled.min(), y_pred_unlabeled.max()])
    mn, mx = float(min(vals)), float(max(vals))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_labeled, y_pred_labeled, s=18, alpha=0.7, edgecolor="none", label="Labeled")
    if y_pred_unlabeled is not None and y_pred_unlabeled.size > 0:
        plt.scatter(y_pred_unlabeled, y_pred_unlabeled, s=10, alpha=0.25, edgecolor="none", label="Unlabeled (pred)")
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.xlabel("True Tg (labeled) / Pred Tg (unlabeled)")
    plt.ylabel("Predicted Tg")
    plt.title(f"{title_prefix} — Labeled MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        df_source = _pick_dataframe()
    except Exception as e:
        sys.exit(str(e))

    _validate_columns(df_source, ["SMILES", "Tg"])  # expected columns

    mask_labeled = df_source["Tg"].notna()
    df_labeled = df_source.loc[mask_labeled].reset_index(drop=False).rename(columns={"index": "orig_index"})
    df_unlabeled = df_source.loc[~mask_labeled].reset_index(drop=False).rename(columns={"index": "orig_index"})

    print(f"Labeled: {len(df_labeled)} | Unlabeled: {len(df_unlabeled)}")

    # Build features
    X_lab, y_lab = build_features(df_labeled)
    X_unlab = build_features_unlabeled(df_unlabeled) if len(df_unlabeled) > 0 else np.empty((0, N_BITS), dtype=np.uint8)

    # Fit Bayesian model on labeled
    br, scaler = fit_bayesian_ridge(X_lab, y_lab)

    # Evaluate on a held-out split of labeled data for sanity
    X_train, X_test, y_train, y_test = train_test_split(X_lab, y_lab, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    br_tmp, sc_tmp = fit_bayesian_ridge(X_train, y_train)
    y_test_mean, _ = posterior_predict(br_tmp, sc_tmp, X_test)
    plot_parity(y_test, y_test_mean, title_prefix="Parity (BayesianRidge, labeled-only)")

    # Cross-validated RMSE on labeled subset
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X_lab_s_all = StandardScaler().fit_transform(X_lab.astype(float))
    # sklearn doesn't have CV wrapper for BR; use scores from simple CV with xgb fallback for speed if needed
    # Here, compute simple baseline with XGB as additional sanity
    xgb = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.7, colsample_bytree=0.6, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)
    cv_scores = -cross_val_score(xgb, X_lab, y_lab, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"Labeled-only XGB CV RMSE: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Score the unlabeled pool
    if X_unlab.size > 0:
        mean_unlab, std_unlab = posterior_predict(br, scaler, X_unlab)
        if ACQ == "mi":
            scores = mutual_information_scores(br, scaler, X_unlab)
        else:
            scores = std_unlab

        # Batch selection with diversity
        sel_idx_pool = greedy_acquire(X_unlab.astype(np.uint8), scores, k=BATCH_SIZE, lambda_div=DIVERSITY_LAMBDA)
        recs = df_unlabeled.iloc[sel_idx_pool].copy()
        recs["acq_score"] = scores[sel_idx_pool]
        recs["Tg_pred_mean_bayes"] = mean_unlab[sel_idx_pool]
        recs["Tg_pred_std_bayes"] = std_unlab[sel_idx_pool]
        recs = recs[["orig_index", "SMILES", "acq_score", "Tg_pred_mean_bayes", "Tg_pred_std_bayes"]].reset_index(drop=True)

        print("\n=== Recommended acquisitions (send to lab) ===")
        print(recs.head(50).to_string(index=False))
    else:
        sel_idx_pool = np.array([], dtype=int)
        mean_unlab = np.array([])
        std_unlab = np.array([])
        recs = pd.DataFrame(columns=["orig_index", "SMILES", "acq_score", "Tg_pred_mean_bayes", "Tg_pred_std_bayes"])    

    # Demo: build final model with pseudo-labels for plotting (replace with real labels in practice)
    if BUILD_DEMO_FINAL:
        if X_unlab.size > 0:
            X_all = np.vstack([X_lab, X_unlab])
            y_all = np.concatenate([y_lab, mean_unlab])
        else:
            X_all = X_lab
            y_all = y_lab

        if FINAL_MODEL == "xgb":
            final_model = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, subsample=0.7, colsample_bytree=0.6, reg_alpha=2.0, reg_lambda=8.0, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)
            final_model.fit(X_all, y_all)
            y_lab_pred = final_model.predict(X_lab)
            y_unlab_pred = final_model.predict(X_unlab) if X_unlab.size > 0 else np.array([])
            title_prefix = "Parity (All Data) — Final XGB (pseudo-labeled)"
        else:
            sc_all = StandardScaler(); X_all_s = sc_all.fit_transform(X_all.astype(float))
            X_lab_s = sc_all.transform(X_lab.astype(float))
            X_unlab_s = sc_all.transform(X_unlab.astype(float)) if X_unlab.size > 0 else X_unlab
            final_model = BayesianRidge(n_iter=800, tol=1e-4, compute_score=True)
            final_model.fit(X_all_s, y_all)
            y_lab_pred = final_model.predict(X_lab_s)
            y_unlab_pred = final_model.predict(X_unlab_s) if X_unlab.size > 0 else np.array([])
            title_prefix = "Parity (All Data) — Final Bayesian (pseudo-labeled)"

        plot_parity_with_unlabeled(y_lab, y_lab_pred, y_pred_unlabeled=y_unlab_pred, title_prefix=title_prefix)

    # Attach recommendation columns to a copy of original df
    df_out = df_source.copy()
    df_out["Tg_pred_mean_bayes"] = np.nan
    df_out["Tg_pred_std_bayes"] = np.nan
    if X_unlab.size > 0:
        idx_unlab = df_unlabeled["orig_index"].to_numpy()
        df_out.loc[idx_unlab, "Tg_pred_mean_bayes"] = mean_unlab
        df_out.loc[idx_unlab, "Tg_pred_std_bayes"] = std_unlab

    print("\nDone. Suggested acquisitions are printed above; demo parity plot generated.")


