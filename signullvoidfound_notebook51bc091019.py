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


# Full pipeline code — run this in your notebook / environment.
# It handles NaNs correctly (impute -> poly -> scale), follows your KNN/mean rules,
# and performs the FE + PCA you requested.

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

# optional helper to display DataFrames (works inside this notebook environment; otherwise prints)
try:
    from caas_jupyter_tools import display_dataframe_to_user
    def show_df(name, df_):
        display_dataframe_to_user(name, df_)
except Exception:
    def show_df(name, df_):
        print(f"\n=== {name} (head) ===")
        print(df_.head().to_string())

# ---------- CONFIG ----------
PATH = "../input/recruitment-task-for-gdsc-ml/MiNDAT.csv"     # change if needed
TARGET = "CORRUCYSTIC_DENSITY"
ID_COL = "LOCAL_IDENTIFIER"

# Hyperparameters
KNN_NEIGHBORS = 3      # reduce if memory/time issues
POLY_TOP_K = 6         # number of unimodal cols to polynomial-expand
KDE_MIN_POINTS = 100   # min samples to attempt KDE modality detection
KDE_GRID = 150
PCA_VARIANCE = 0.95

# ---------- LOAD ----------
df = pd.read_csv(PATH)
print("Loaded:", PATH, "shape:", df.shape)

# ---------- handle identifier ----------
if ID_COL in df.columns:
    ids = df[ID_COL].copy()
    df = df.drop(columns=[ID_COL])
    print("Dropped ID column:", ID_COL)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

y = df[TARGET]
X = df.drop(columns=[TARGET])
# --- Load data ---
df = pd.read_csv("../input/recruitment-task-for-gdsc-ml/MiNDAT.csv")
TARGET = "CORRUCYSTIC_DENSITY"

# Separate target
y = df[TARGET]
X = df.drop(columns=[TARGET])

# --- FIX: drop rows where target is NaN ---
not_nan_idx = ~y.isna()
X = X.loc[not_nan_idx].reset_index(drop=True)
y = y.loc[not_nan_idx].reset_index(drop=True)

print("After dropping target NaNs:", X.shape, y.shape)
# Should print: (9600, 48), (9600,)


print("Shapes after dropping target NaNs:", X.shape, y.shape)

# ---------- utility: multimodality detection ----------
def is_multimodal(series, min_points=KDE_MIN_POINTS, grid_points=KDE_GRID):
    data = series.dropna().values
    if len(data) < min_points or np.all(data == data[0]):
        return False
    try:
        kde = gaussian_kde(data)
        x_eval = np.linspace(np.min(data), np.max(data), grid_points)
        y_eval = kde(x_eval)
        dy = np.diff(y_eval)
        # count changes from + to - (peaks)
        peaks = np.where((np.hstack([0, dy[:-1]]) > 0) & (np.hstack([dy, 0])[1:] <= 0))[0]
        return len(peaks) > 1
    except Exception:
        return False

# ---------- detect numeric/categorical ----------
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_initial = X.select_dtypes(include=['object', 'category']).columns.tolist()

# ---------- classify multimodal vs unimodal ----------
multi_cols = []
uni_cols = []
for c in numeric_cols:
    if is_multimodal(X[c]):
        multi_cols.append(c)
    else:
        uni_cols.append(c)

print("Numeric cols:", len(numeric_cols))
print("Multimodal detected:", len(multi_cols))
print("Unimodal detected:", len(uni_cols))

show_df("multimodal_preview", pd.DataFrame({"multimodal_cols": multi_cols[:50]}))
show_df("unimodal_preview", pd.DataFrame({"unimodal_cols": uni_cols[:50]}))

# ---------- Feature engineering ----------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_cols=("v0rt3X","v1rt3X")):
        self.cluster_cols = list(cluster_cols)
        self.kmeans = None
    def fit(self, X, y=None):
        if all(c in X.columns for c in self.cluster_cols):
            data = X[self.cluster_cols].fillna(0).values
            try:
                self.kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
            except Exception:
                self.kmeans = None
        return self
    def transform(self, X):
        Xt = X.copy()
        if self.kmeans is not None:
            Xt["MOON_CLUSTER"] = self.kmeans.predict(Xt[self.cluster_cols].fillna(0).values)
        else:
            Xt["MOON_CLUSTER"] = 0
        for c in self.cluster_cols:
            if c in Xt.columns:
                vals = Xt[c].fillna(0).astype(float).values
                Xt[f"{c}_sin"] = np.sin(vals)
                Xt[f"{c}_cos"] = np.cos(vals)
        return Xt

fe = FeatureEngineer()
fe.fit(X)
X_fe = fe.transform(X)   # DataFrame, still has column names
print("After FE shape:", X_fe.shape)

# ---------- choose columns for polynomial expansion ----------
# compute Pearson correlation with target on unimodal columns and pick top-K
corrs = {}
for c in uni_cols:
    valid = pd.concat([X_fe[c], y], axis=1).dropna()
    if valid.shape[0] < 50:
        corrs[c] = 0.0
    else:
        try:
            corrs[c] = valid[c].corr(valid[TARGET])
        except Exception:
            corrs[c] = 0.0
sorted_uni = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
poly_cols = [c for c,_ in sorted_uni[:POLY_TOP_K] if c in uni_cols]

print("Selected poly cols (top correlated unimodals):", poly_cols)
show_df("poly_cols", pd.DataFrame({"poly_cols": poly_cols}))

# ---------- final column groups (ensure presence in X_fe) ----------
multi_cols_final = [c for c in multi_cols if c in X_fe.columns]
uni_cols_final = [c for c in uni_cols if c in X_fe.columns and c not in poly_cols]
cat_cols_final = [c for c in cat_cols_initial if c in X_fe.columns]

# include MOON_CLUSTER as categorical
if "MOON_CLUSTER" in X_fe.columns and "MOON_CLUSTER" not in cat_cols_final:
    cat_cols_final.append("MOON_CLUSTER")

# Sanity
print("multi_cols_final:", len(multi_cols_final))
print("uni_cols_final (no poly):", len(uni_cols_final))
print("cat_cols_final:", cat_cols_final)

show_df("multi_cols_final", pd.DataFrame({"multi_cols_final": multi_cols_final}))
show_df("uni_cols_final", pd.DataFrame({"uni_cols_final": uni_cols_final}))
show_df("cat_cols_final", pd.DataFrame({"cat_cols_final": cat_cols_final}))

# ---------- pipelines (IMPUTE BEFORE POLY) ----------
poly_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler())
])

uni_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

multi_pipeline = Pipeline([
    ("imputer", KNNImputer(n_neighbors=KNN_NEIGHBORS)),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

transformers = []
if poly_cols:
    transformers.append(("poly", poly_pipeline, poly_cols))
if uni_cols_final:
    transformers.append(("uni", uni_pipeline, uni_cols_final))
if multi_cols_final:
    transformers.append(("multi", multi_pipeline, multi_cols_final))
if cat_cols_final:
    transformers.append(("cat", cat_pipeline, cat_cols_final))

from sklearn.compose import ColumnTransformer as CT
preprocessor = CT(transformers=transformers, remainder="drop")

# ---------- fit + transform ----------
print("Fitting preprocessor (KNNImputer may be the slowest step)...")
X_proc = preprocessor.fit_transform(X_fe)
print("Preprocessing done. Processed shape:", X_proc.shape)

# ---------- PCA ----------
pca = PCA(n_components=PCA_VARIANCE, random_state=42)
X_pca = pca.fit_transform(X_proc)
print("PCA done. PCA shape:", X_pca.shape)
print("Explained variance (sum):", round(pca.explained_variance_ratio_.sum(), 6))
print("PCA components kept:", X_pca.shape[1])

# ---------- diagnostics & save ----------
# show MOON_CLUSTER counts and MINDSPIKE_VERSION counts (if present)
if "MOON_CLUSTER" in X_fe.columns:
    show_df("MOON_CLUSTER_counts", X_fe["MOON_CLUSTER"].value_counts().reset_index().rename(columns={"index":"cluster","MOON_CLUSTER":"count"}))
if "MINDSPIKE_VERSION" in X_fe.columns:
    show_df("MINDSPIKE_VERSION_counts", X_fe["MINDSPIKE_VERSION"].value_counts().reset_index().rename(columns={"index":"version","MINDSPIKE_VERSION":"count"}))

evr_df = pd.DataFrame({
    "component": np.arange(1, len(pca.explained_variance_ratio_)+1),
    "explained_variance_ratio": pca.explained_variance_ratio_
})
show_df("PCA_explained_variance", evr_df.head(50))

out_dir = "../kaggle/working/"
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "X_preprocessed.npy"), X_proc)
np.save(os.path.join(out_dir, "X_pca.npy"), X_pca)
y.to_csv(os.path.join(out_dir, "y.csv"), index=False)
print("Saved outputs to:", out_dir)

# quick hist for the first two multimodal cols
for c in multi_cols_final[:2]:
    plt.figure(figsize=(6,3))
    plt.hist(X_fe[c].dropna().values, bins=60)
    plt.title(f"Histogram of {c}")
    plt.show()

print("Done. Adjust KNN_NEIGHBORS or POLY_TOP_K if it runs slow or you want fewer features.")
# --- Get feature names out of ColumnTransformer --- #
def get_feature_names(preprocessor):
    output_features = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, 'named_steps') and "poly" in trans.named_steps:
            # Polynomial features
            poly = trans.named_steps["poly"]
            poly_features = poly.get_feature_names_out(cols)
            output_features.extend(poly_features)
        elif hasattr(trans, 'get_feature_names_out'):
            output_features.extend(trans.get_feature_names_out(cols))
        else:
            output_features.extend(cols)
    return output_features

# Fit transform
X_proc = preprocessor.fit_transform(X_fe)

# Rebuild DataFrame with names
feature_names = get_feature_names(preprocessor)
X_proc_df = pd.DataFrame(X_proc, columns=feature_names, index=X_fe.index)

print("Shape:", X_proc_df.shape)
print("Remaining NaNs:", X_proc_df.isnull().sum().sum())  # should be 0
X_proc_df.head()



X_proc_df.isnull().sum()


import pandas as pd
import numpy as np

# Assume df is your dataset BEFORE PCA
# Convert numeric columns to binary based on median
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in numeric_cols:
    df[col] = (df[col] > df[col].median()).astype(int)

print("Data after converting to binary:")
print(df.head())



import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ----------------- Example column groups ----------------- #
# Replace these with your actual dataset columns
uni_cols = ["v0rt3X", "v1rt3X"]      # numeric columns to scale individually
multi_cols = []                      # numeric columns for KNN imputer
cat_cols = []                        # categorical columns (if any)
binary_cols = uni_cols               # columns you want to convert to binary

# ----------------- Feature Engineering ----------------- #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if "v0rt3X" in X.columns and "v1rt3X" in X.columns:
            data = X[["v0rt3X", "v1rt3X"]].fillna(0)
            self.kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
        else:
            self.kmeans = None
        return self

    def transform(self, X):
        X_new = X.copy()
        if self.kmeans:
            X_new["MOON_CLUSTER"] = self.kmeans.predict(
                X_new[["v0rt3X", "v1rt3X"]].fillna(0)
            )
        if "v0rt3X" in X_new.columns:
            X_new["v0rt3X_sin"] = np.sin(X_new["v0rt3X"])
            X_new["v0rt3X_cos"] = np.cos(X_new["v0rt3X"])
        if "v1rt3X" in X_new.columns:
            X_new["v1rt3X_sin"] = np.sin(X_new["v1rt3X"])
            X_new["v1rt3X_cos"] = np.cos(X_new["v1rt3X"])
        return X_new

# ----------------- Convert numeric to binary (optional) ----------------- #
def convert_to_binary(df, columns, method="median"):
    df_bin = df.copy()
    for col in columns:
        if method == "median":
            df_bin[col] = (df_bin[col] > df_bin[col].median()).astype(int)
        elif method == "threshold":
            threshold = 450  # example, customize per column
            df_bin[col] = (df_bin[col] > threshold).astype(int)
    return df_bin

# ----------------- Polynomial features pipeline ----------------- #
poly_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler())
])

# ----------------- ColumnTransformer ----------------- #
fe_preprocessor = ColumnTransformer(
    transformers=[
        ("uni", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), uni_cols),

        ("multi", Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ]), multi_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_cols),

        ("poly_uni", poly_pipe, uni_cols),

        ("pass", "passthrough", ["MOON_CLUSTER","v0rt3X_sin","v0rt3X_cos","v1rt3X_sin","v1rt3X_cos"])
    ],
    remainder="drop"
)

# ----------------- Full Pipeline ----------------- #
# Assume X is your original dataframe
fe = FeatureEngineer()
X_fe = fe.fit_transform(X_proc_df)

# Optional: convert numeric features to binary
# X_fe = convert_to_binary(X_fe, binary_cols, method="median")

# Preprocessing
X_proc = fe_preprocessor.fit_transform(X_fe)
proc_cols = fe_preprocessor.get_feature_names_out()
X_proc_df = pd.DataFrame(X_proc, columns=proc_cols, index=X.index)

# Final imputation safety net
final_imputer = SimpleImputer(strategy="mean")
X_proc_clean = final_imputer.fit_transform(X_proc_df)
X_proc_df = pd.DataFrame(X_proc_clean, columns=proc_cols, index=X.index)

print("NaNs after final imputation:", X_proc_df.isnull().sum().sum())

# PCA
pca = PCA(n_components=0.98, random_state=42)
X_pca = pca.fit_transform(X_proc_df)
pca_cols = [f"PCA{i+1}" for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)

# ----------------- Outputs ----------------- #
print("Original shape after preprocessing:", X_proc_df.shape)
print("After PCA shape:", X_pca_df.shape)
print("Explained variance (first 10 comps):", pca.explained_variance_ratio_[:10])
print("Total explained variance:", pca.explained_variance_ratio_.sum())



import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------- CONFIG ----------------- #
TRAIN_PATH = "../input/recruitment-task-for-gdsc-ml/MiNDAT.csv"   # training data
TEST_PATH  = "../input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"  # test data
ID_COL = "LOCAL_IDENTIFIER"
TARGET = "CORRUCYSTIC_DENSITY"
KNN_NEIGHBORS = 3
POLY_TOP_K = 6
KDE_MIN_POINTS = 100
KDE_GRID = 150
PCA_VARIANCE = 0.98

# ----------------- LOAD DATA ----------------- #
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# Save IDs and remove
train_ids = train_df[ID_COL].copy()
train_df = train_df.drop(columns=[ID_COL])
test_ids  = test_df[ID_COL].copy()
test_df  = test_df.drop(columns=[ID_COL])

# Split X/y
y_train = train_df[TARGET]
X_train = train_df.drop(columns=[TARGET])
X_test  = test_df.copy()  # no target

# ----------------- MULTIMODALITY DETECTION ----------------- #
def is_multimodal(series, min_points=KDE_MIN_POINTS, grid_points=KDE_GRID):
    data = series.dropna().values
    if len(data) < min_points or np.all(data == data[0]):
        return False
    try:
        kde = gaussian_kde(data)
        x_eval = np.linspace(np.min(data), np.max(data), grid_points)
        y_eval = kde(x_eval)
        dy = np.diff(y_eval)
        peaks = np.where((np.hstack([0, dy[:-1]]) > 0) & (np.hstack([dy,0])[1:] <= 0))[0]
        return len(peaks) > 1
    except:
        return False

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_initial = X_train.select_dtypes(include=['object','category']).columns.tolist()

multi_cols = [c for c in numeric_cols if is_multimodal(X_train[c])]
uni_cols   = [c for c in numeric_cols if c not in multi_cols]

# ----------------- FEATURE ENGINEERING ----------------- #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_cols=("v0rt3X","v1rt3X")):
        self.cluster_cols = list(cluster_cols)
        self.kmeans = None
    def fit(self, X, y=None):
        if all(c in X.columns for c in self.cluster_cols):
            data = X[self.cluster_cols].fillna(0).values
            try:
                self.kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
            except:
                self.kmeans = None
        return self
    def transform(self, X):
        Xt = X.copy()
        if self.kmeans is not None:
            Xt["MOON_CLUSTER"] = self.kmeans.predict(Xt[self.cluster_cols].fillna(0).values)
        else:
            Xt["MOON_CLUSTER"] = 0
        for c in self.cluster_cols:
            if c in Xt.columns:
                vals = Xt[c].fillna(0).astype(float).values
                Xt[f"{c}_sin"] = np.sin(vals)
                Xt[f"{c}_cos"] = np.cos(vals)
        return Xt

fe = FeatureEngineer()
fe.fit(X_train)
X_train_fe = fe.transform(X_train)
X_test_fe  = fe.transform(X_test)

# ----------------- POLYNOMIAL SELECTION ----------------- #
corrs = {}
for c in uni_cols:
    valid = pd.concat([X_train_fe[c], y_train], axis=1).dropna()
    corrs[c] = valid[c].corr(valid[TARGET]) if valid.shape[0] >= 50 else 0.0
sorted_uni = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
poly_cols = [c for c,_ in sorted_uni[:POLY_TOP_K] if c in uni_cols]

multi_cols_final = [c for c in multi_cols if c in X_train_fe.columns]
uni_cols_final   = [c for c in uni_cols if c in X_train_fe.columns and c not in poly_cols]
cat_cols_final   = [c for c in cat_cols_initial if c in X_train_fe.columns]
if "MOON_CLUSTER" in X_train_fe.columns:
    cat_cols_final.append("MOON_CLUSTER")

# ----------------- BINARY FEATURE CREATION ----------------- #
binary_cols = uni_cols_final + multi_cols_final
binary_thresholds = {}
for col in binary_cols:
    median_val = X_train_fe[col].median()
    binary_thresholds[col] = median_val
    X_train_fe[col + "_bin"] = (X_train_fe[col] > median_val).astype(int)
    X_test_fe[col + "_bin"]  = (X_test_fe[col] > median_val).astype(int)

# Include binary columns in uni/poly as desired
uni_cols_final += [c + "_bin" for c in binary_cols]
poly_cols += [c + "_bin" for c in binary_cols]

# ----------------- PIPELINES ----------------- #
poly_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler())
])
uni_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
multi_pipeline = Pipeline([
    ("imputer", KNNImputer(n_neighbors=KNN_NEIGHBORS)),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

transformers = []
if poly_cols:
    transformers.append(("poly", poly_pipeline, poly_cols))
if uni_cols_final:
    transformers.append(("uni", uni_pipeline, uni_cols_final))
if multi_cols_final:
    transformers.append(("multi", multi_pipeline, multi_cols_final))
if cat_cols_final:
    transformers.append(("cat", cat_pipeline, cat_cols_final))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

# ----------------- FIT TRAIN ----------------- #
X_train_proc = preprocessor.fit_transform(X_train_fe)
pca = PCA(n_components=PCA_VARIANCE, random_state=42)
X_train_pca = pca.fit_transform(X_train_proc)

# ----------------- TRANSFORM TEST ----------------- #
X_test_proc = preprocessor.transform(X_test_fe)
X_test_pca  = pca.transform(X_test_proc)

# ----------------- GET FEATURE NAMES ----------------- #
def get_feature_names(preprocessor):
    output_features = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, 'named_steps') and "poly" in trans.named_steps:
            poly = trans.named_steps["poly"]
            poly_features = poly.get_feature_names_out(cols)
            output_features.extend(poly_features)
        elif hasattr(trans, 'get_feature_names_out'):
            output_features.extend(trans.get_feature_names_out(cols))
        else:
            output_features.extend(cols)
    return output_features

feature_names_train = get_feature_names(preprocessor)

X_train_proc_df = pd.DataFrame(X_train_proc, columns=feature_names_train, index=X_train_fe.index)
X_train_pca_df  = pd.DataFrame(X_train_pca, columns=[f"PCA{i+1}" for i in range(X_train_pca.shape[1])], index=X_train_fe.index)

X_test_proc_df = pd.DataFrame(X_test_proc, columns=feature_names_train, index=X_test_fe.index)
X_test_pca_df  = pd.DataFrame(X_test_pca, columns=[f"PCA{i+1}" for i in range(X_test_pca.shape[1])], index=X_test_fe.index)

# ----------------- SAVE ----------------- #
out_dir = "../kaggle/working/"
os.makedirs(out_dir, exist_ok=True)
X_train_proc_df.to_csv(os.path.join(out_dir, "X_train_preprocessed.csv"), index=False)
X_train_pca_df.to_csv(os.path.join(out_dir, "X_train_pca.csv"), index=False)
X_test_proc_df.to_csv(os.path.join(out_dir, "X_test_preprocessed.csv"), index=False)
X_test_pca_df.to_csv(os.path.join(out_dir, "X_test_pca.csv"), index=False)
y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)

print("All outputs saved. Shapes:")
print("Train proc:", X_train_proc_df.shape, "| Train PCA:", X_train_pca_df.shape)
print("Test proc :", X_test_proc_df.shape,  "| Test PCA :", X_test_pca_df.shape)



print(X_train_pca.shape)  # should be (#samples, #features)
print(y_train.shape)      # should be (#samples,)



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------- CONFIG -------------------- #
TRAIN_PATH = "../input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TEST_PATH  = "../input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"
ID_COL    = "LOCAL_IDENTIFIER"
TARGET    = "CORRUCYSTIC_DENSITY"
POLY_TOP_K = 6  # top correlated unimodal columns for poly expansion
KNN_NEIGHBORS = 3
PCA_VARIANCE = 0.98

# -------------------- LOAD DATA -------------------- #
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# -------------------- SPLIT TARGET -------------------- #
y = train_df[TARGET]
X = train_df.drop(columns=[TARGET])

# -------------------- DROP TARGET NaNs -------------------- #
not_nan_idx = ~y.isna()
X = X.loc[not_nan_idx].reset_index(drop=True)
y = y.loc[not_nan_idx].reset_index(drop=True)

# -------------------- FEATURE ENGINEERING -------------------- #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_cols=("v0rt3X","v1rt3X")):
        self.cluster_cols = list(cluster_cols)
        self.kmeans = None
    def fit(self, X, y=None):
        if all(c in X.columns for c in self.cluster_cols):
            data = X[self.cluster_cols].fillna(0).values
            try:
                self.kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
            except Exception:
                self.kmeans = None
        return self
    def transform(self, X):
        Xt = X.copy()
        if self.kmeans is not None:
            Xt["MOON_CLUSTER"] = self.kmeans.predict(Xt[self.cluster_cols].fillna(0).values)
        else:
            Xt["MOON_CLUSTER"] = 0
        for c in self.cluster_cols:
            if c in Xt.columns:
                vals = Xt[c].fillna(0).astype(float).values
                Xt[f"{c}_sin"] = np.sin(vals)
                Xt[f"{c}_cos"] = np.cos(vals)
        return Xt

fe = FeatureEngineer()
X_fe = fe.fit_transform(X)
X_test_fe = fe.transform(test_df)  # apply same FE to test

# -------------------- DETECT NUMERIC / CATEGORICAL -------------------- #
numeric_cols = X_fe.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_initial = X_fe.select_dtypes(include=['object', 'category']).columns.tolist()

# -------------------- UNIMODAL / MULTIMODAL -------------------- #
from scipy.stats import gaussian_kde

def is_multimodal(series, min_points=100, grid_points=150):
    data = series.dropna().values
    if len(data) < min_points or np.all(data == data[0]):
        return False
    try:
        kde = gaussian_kde(data)
        x_eval = np.linspace(np.min(data), np.max(data), grid_points)
        y_eval = kde(x_eval)
        dy = np.diff(y_eval)
        peaks = np.where((np.hstack([0, dy[:-1]]) > 0) & (np.hstack([dy, 0])[1:] <= 0))[0]
        return len(peaks) > 1
    except:
        return False

multi_cols = [c for c in numeric_cols if is_multimodal(X_fe[c])]
uni_cols   = [c for c in numeric_cols if c not in multi_cols]

# -------------------- SELECT TOP POLY COLUMNS -------------------- #
corrs = {}
for c in uni_cols:
    valid = pd.concat([X_fe[c], y], axis=1).dropna()
    if valid.shape[0] < 50:
        corrs[c] = 0.0
    else:
        try:
            corrs[c] = valid[c].corr(valid[TARGET])
        except:
            corrs[c] = 0.0
sorted_uni = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
poly_cols = [c for c,_ in sorted_uni[:POLY_TOP_K] if c in uni_cols]

# -------------------- FINAL COLUMN GROUPS -------------------- #
multi_cols_final = [c for c in multi_cols if c in X_fe.columns]
uni_cols_final   = [c for c in uni_cols if c in X_fe.columns and c not in poly_cols]
cat_cols_final   = [c for c in cat_cols_initial if c in X_fe.columns]
if "MOON_CLUSTER" in X_fe.columns and "MOON_CLUSTER" not in cat_cols_final:
    cat_cols_final.append("MOON_CLUSTER")

# -------------------- PIPELINES -------------------- #
poly_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler())
])
uni_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
multi_pipeline = Pipeline([
    ("imputer", KNNImputer(n_neighbors=KNN_NEIGHBORS)),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

transformers = []
if poly_cols:
    transformers.append(("poly", poly_pipeline, poly_cols))
if uni_cols_final:
    transformers.append(("uni", uni_pipeline, uni_cols_final))
if multi_cols_final:
    transformers.append(("multi", multi_pipeline, multi_cols_final))
if cat_cols_final:
    transformers.append(("cat", cat_pipeline, cat_cols_final))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

# -------------------- FIT + TRANSFORM -------------------- #
X_proc = preprocessor.fit_transform(X_fe)
X_proc = np.nan_to_num(X_proc)
X_test_proc = preprocessor.transform(X_test_fe)
X_test_proc = np.nan_to_num(X_test_proc)

# -------------------- PCA -------------------- #
pca = PCA(n_components=PCA_VARIANCE, random_state=42)
X_train_pca = pca.fit_transform(X_proc)
X_test_pca  = pca.transform(X_test_proc)

# -------------------- TRAIN / VALIDATION SPLIT -------------------- #
X_tr, X_val, y_tr, y_val = train_test_split(X_train_pca, y, test_size=0.2, random_state=42)

# -------------------- MODEL -------------------- #
model = GradientBoostingRegressor(random_state=42)
param_grid = {"n_estimators":[100,150], "learning_rate":[0.05,0.1], "max_depth":[3,4]}
grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_tr, y_tr)

best_model = grid.best_estimator_
val_preds = best_model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}, Best Params: {grid.best_params_}")

# -------------------- TRAIN ON FULL TRAINING -------------------- #
best_model.fit(X_train_pca, y)

# -------------------- PREDICT TEST -------------------- #
test_ids = test_df[ID_COL]
test_preds = best_model.predict(X_test_pca)


import pandas as pd
import os

# -------------------- Load specimen CSV -------------------- #
specimen = pd.read_csv("../input/recruitment-task-for-gdsc-ml/SPECIMEN.csv")

# -------------------- Prepare submission -------------------- #
submission = specimen.copy()
submission["CORRUCYSTIC_DENSITY"] = test_preds  # insert your model predictions

# -------------------- Save CSV -------------------- #

submission.to_csv('submission.csv', index=False)

# -------------------- Display first few rows -------------------- #
print("\nFirst 5 rows of submission:")
display(submission.head())

# -------------------- Optional: list files to confirm -------------------- #
print("\nFiles in working directory:")
print(os.listdir("../kaggle/working/"))


