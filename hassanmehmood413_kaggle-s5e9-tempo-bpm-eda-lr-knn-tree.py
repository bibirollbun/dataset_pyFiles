import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from pandas.api.types import (
    is_numeric_dtype, is_bool_dtype, is_string_dtype, is_datetime64_any_dtype
)
from sklearn.base import clone


import math
from typing import Tuple, Optional, List
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Data Preprocessing 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor


# Consistent visualization theme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'danger': '#D62246',
    'neutral': '#6C757D'
}

PALETTE = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
           COLORS['success'], COLORS['danger']]

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(PALETTE)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 4)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


df_train.head()


df_test.head()


CATEGORICAL_FEATURES = ['road_type', 'lighting', 'weather', 'time_of_day']
BOOLEAN_FEATURES = ['road_signs_present', 'public_road', 'holiday', 'school_season']
NUMERICAL_FEATURES = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
TARGET = 'accident_risk'
ID_COL = 'id'


df_train.info()


df_train.duplicated().sum()


cat_cols = CATEGORICAL_FEATURES + BOOLEAN_FEATURES
# 2) Quick readable console summary
for col in cat_cols:
    s = df_train[col]
    n = len(s)
    n_miss = s.isna().sum()
    print(f"\nâ”€â”€ {col} â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    print(f"dtype: {s.dtype} | unique: {s.nunique(dropna=True)} | "
          f"missing: {n_miss} ({n_miss/n:.1%})")
    # top 8 values with counts & %
    vc = s.value_counts(dropna=False).head(8)
    vc_pct = (vc / n).rename("percent")
    display(pd.concat([vc.rename("count"), vc_pct], axis=1))

# 3) Polished summary table (one row per categorical column)
rows = []
for col in cat_cols:
    s = df_train[col]
    n = len(s)
    vc = s.value_counts(dropna=False)
    preview = ", ".join(
        [f"{('âŸ‚ NA' if pd.isna(k) else str(k))} ({v}|{v/n:.1%})" for k, v in vc.head(6).items()]
    )
    rows.append({
        "column": col,
        "dtype": str(s.dtype),
        "n": n,
        "n_missing": int(s.isna().sum()),
        "missing_pct": s.isna().mean(),
        "n_unique": int(s.nunique(dropna=True)),
        "sample_top_values": preview
    })

cat_summary = (
    pd.DataFrame(rows)
    .sort_values(["missing_pct", "n_unique"], ascending=[False, True])
    .reset_index(drop=True)
)

# Pretty display in notebooks (optional)
try:
    display(
        cat_summary.style.format({
            "n": "{:,}",
            "n_missing": "{:,}",
            "missing_pct": "{:.1%}",
            "n_unique": "{:,}",
        }).bar(subset=["missing_pct", "n_unique"], align="left")
    )
except Exception:
    display(cat_summary)



def _ci95(std, n):
    return 1.96 * (std / np.sqrt(np.maximum(n, 1)))

def _colorize(ax):
    for i, bar in enumerate(ax.patches):
        bar.set_facecolor(PALETTE[i % len(PALETTE)])
        bar.set_edgecolor("none")
        bar.set_alpha(0.95)


# === main plot ===
def plot_features_vs_target_with_lists(
    df: pd.DataFrame,
    target: str,
    categorical_cols: list,
    boolean_cols: list,
    numeric_cols: list,
    *,
    ncols: int = 3,
    num_bins: int = 5,
    top_cats: int = 8,
    title: str = 'Features vs "accident_risk"',
):
    # keep only columns that exist (no crashes if a name is missing)
    categorical = [c for c in categorical_cols if c in df.columns]
    boolean     = [c for c in boolean_cols     if c in df.columns]
    numeric     = [c for c in numeric_cols     if c in df.columns]

    # order: categorical â†’ boolean â†’ numeric (change if you prefer)
    features = categorical + boolean + numeric
    if not features:
        print("No matching feature columns found in the DataFrame.")
        return

    y = df[target].astype(float)

    n = len(features)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.6*nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)  # flatten safely

    for i, col in enumerate(features):
        ax = axes[i]

        # ---- numeric branch ----
        if col in numeric:
            d = pd.DataFrame({col: df[col], target: y}).dropna()
            if d.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center"); ax.set_title(col); continue
            # if very few uniques, treat as categorical-like values
            if d[col].nunique() < 3:
                agg = d.groupby(col)[target].agg(["mean", "count", "std"]).sort_index()
                ax.bar(range(len(agg)), agg["mean"].values, width=0.8)
                _colorize(ax)
                ax.set_xticks(range(len(agg)))
                ax.set_xticklabels(agg.index.astype(str), rotation=30, ha="right")
                ax.set_title(f"{col} | mean {target} by value  (n={int(agg['count'].sum())})")
            else:
                # quantile bins (fallback to equal-width)
                try:
                    bins = pd.qcut(d[col], q=num_bins, duplicates="drop")
                except Exception:
                    bins = pd.cut(d[col], bins=min(num_bins, max(2, d[col].nunique())))
                agg = d.groupby(bins, observed=False)[target].agg(["mean", "count", "std"])
                errs = _ci95(agg["std"].fillna(0.0), agg["count"])
                labels = [f"{iv.left:.3g}â€“{iv.right:.3g}" for iv in agg.index]

                ax.bar(range(len(agg)), agg["mean"].values, yerr=errs.values, capsize=3)
                _colorize(ax)
                ax.set_xticks(range(len(agg)))
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_title(f"{col} | mean {target} by bin  (n={int(agg['count'].sum())})")

        # ---- categorical/boolean branch ----
        else:
            s = df[col].astype("object").where(~df[col].isna(), "âŸ‚ NA")
            vc = s.value_counts()
            top = vc.head(top_cats).index
            other_label = f"âŠ– other ({s.nunique()} cats)"
            key = s.where(s.isin(top), other_label)

            d = pd.DataFrame({col: key, target: y}).dropna()
            if d.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center"); ax.set_title(col); continue

            agg = d.groupby(col)[target].mean().sort_values(ascending=False)

            ax.bar(range(len(agg)), agg.values, width=0.8)
            _colorize(ax)
            ax.set_xticks(range(len(agg)))
            ax.set_xticklabels(agg.index.astype(str), rotation=30, ha="right")
            typ = "boolean" if col in boolean_cols else "category"
            ax.set_title(f"{col} ({typ}) | mean {target}  (n={len(d)})")

        # y as percent (probability target)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.25)

    # turn off any unused axes
    for j in range(i + 1, nrows * ncols):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.show()



plot_features_vs_target_with_lists(
    df_train,
    target=TARGET,
    categorical_cols=CATEGORICAL_FEATURES,
    boolean_cols=BOOLEAN_FEATURES,
    numeric_cols=NUMERICAL_FEATURES,
    ncols=3,        
    num_bins=5,     
    top_cats=8,
    title='Features vs "accident_risk"'
)



def _get_colors(n: int, palette=None):
    """Return n colors by cycling strictly through PALETTE."""
    pal = palette if palette is not None else PALETTE
    return [pal[i % len(pal)] for i in range(n)]

def _colorize_bars(ax, palette=None):
    """Apply PALETTE colors to all bar/hist patches in the given Axes."""
    pal = palette if palette is not None else PALETTE
    for i, p in enumerate(ax.patches):
        p.set_facecolor(pal[i % len(pal)])
        p.set_edgecolor("none")
        p.set_alpha(0.95)

def _auto_num_bins(x: np.ndarray, max_bins: int = 50) -> int:
    """Freedmanâ€“Diaconis rule with sane caps."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 5
    q75, q25 = np.percentile(x, [75, 25])
    iqr = max(q75 - q25, 1e-12)
    h = 2 * iqr * (n ** (-1/3))
    if h <= 0:
        return min(10, max_bins)
    bins = int(np.ceil((x.max() - x.min()) / h))
    return int(np.clip(bins, 5, max_bins))

def _choose_time_bucket(s: pd.Series) -> Tuple[pd.Series, str]:
    """Pick a reasonable datetime bucket (day/week/month/quarter) based on span."""
    s = pd.to_datetime(s, errors="coerce")
    s_valid = s.dropna()
    if s_valid.empty:
        return s.astype(object), "raw"
    span = (s_valid.max() - s_valid.min()).days
    if span > 2 * 365:
        grp = s_valid.dt.to_period("Q").astype(str); label = "Quarter"
    elif span > 180:
        grp = s_valid.dt.to_period("M").astype(str); label = "Month"
    elif span > 14:
        grp = s_valid.dt.to_period("W").apply(lambda p: f"{p.start_time.date()}").astype(str); label = "Week"
    else:
        grp = s_valid.dt.date.astype(str); label = "Day"
    return pd.Series(grp, index=s_valid.index).reindex(s.index, fill_value="âŸ‚ NA"), label

def _wrap(text: str, width: int = 18) -> str:
    if len(text) <= width:
        return text
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    return "\n".join(lines)

def plot_column_distributions_grid(
    df: pd.DataFrame,
    *,
    ncols: int = 3,
    max_cats: int = 12,
    fig_scale: float = 1.0,
    show_kde_if_available: bool = True,
    palette=None,
    exclude: Optional[List[str]] = None,  
):
    """
    Numeric   â†’ histogram (+ KDE if SciPy present), mean/median lines, missing% in title.
    Categorical/Bool â†’ bar of top categories (+ 'âŠ– other'), % labels, includes 'âŸ‚ NA'.
    Datetime  â†’ counts by smart period (day/week/month/quarter), includes 'âŸ‚ NA'.
    """
    use_cols = [c for c in df.columns if not (exclude and c in exclude)]
    if len(use_cols) == 0:
        print("No columns to plot.")
        return

    n = len(use_cols)
    nrows = math.ceil(n / ncols)

    base_w, base_h = 5.8, 4.6
    fig_w, fig_h = ncols * base_w * fig_scale, nrows * base_h * fig_scale
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), constrained_layout=True)

    # normalize axes to 2D array
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # optional KDE
    kde_available = False
    if show_kde_if_available:
        try:
            from scipy.stats import gaussian_kde  # noqa: F401
            kde_available = True
        except Exception:
            kde_available = False

    for i, col in enumerate(use_cols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        s = df[col]
        missing_pct = s.isna().mean()

        try:
            # 1) Datetime
            if is_datetime64_any_dtype(s):
                s_bucketed, bucket_label = _choose_time_bucket(s)
                counts = s_bucketed.value_counts(dropna=False).sort_index()
                if counts.size > 30:
                    counts = counts.sort_values(ascending=False).head(30).sort_index()

                ax.bar(np.arange(len(counts)), counts.values)
                _colorize_bars(ax, palette)
                ax.set_xticks(np.arange(len(counts)))
                ax.set_xticklabels([_wrap(str(k)) for k in counts.index], rotation=30, ha="right")
                ax.set_title(f"{col} â€” {bucket_label} counts   |   missing: {missing_pct:.1%}")

            # 2) Categorical / Bool / Strings
            elif is_bool_dtype(s) or is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                s_cat = s.astype("object").where(~s.isna(), other="âŸ‚ NA")
                vc = s_cat.value_counts(dropna=False)
                if vc.size > max_cats:
                    top = vc.head(max_cats)
                    other_n = int(vc.iloc[max_cats:].sum())
                    vc = pd.concat([top, pd.Series({"âŠ– other": other_n})])  # no deprecated .append
                order = vc.index.astype(str)
                ax.bar(np.arange(len(vc)), vc.values)
                _colorize_bars(ax, palette)
                ax.set_xticks(np.arange(len(vc)))
                ax.set_xticklabels([_wrap(o) for o in order], rotation=30, ha="right")
                total = int(len(s))
                for xi, v in enumerate(vc.values):
                    if v > 0:
                        ax.text(xi, v, f"{v/total:.0%}", ha="center", va="bottom", fontsize=8)
                ax.set_title(f"{col} â€” category counts   |   missing: {missing_pct:.1%}")

            # 3) Numeric
            elif is_numeric_dtype(s):
                x = s.dropna().astype(float).values
                if x.size == 0:
                    ax.text(0.5, 0.5, "All values missing", ha="center", va="center")
                    ax.set_title(f"{col} â€” numeric   |   missing: {missing_pct:.1%}")
                else:
                    bins = _auto_num_bins(x)
                    n_vals, bins_edges, patches = ax.hist(x, bins=bins, density=False, edgecolor="white")
                    # color the histogram bars with PALETTE
                    _colorize_bars(ax, palette)

                    ax.set_title(f"{col} â€” histogram   |   missing: {missing_pct:.1%}")
                    mean_v, med_v = float(np.mean(x)), float(np.median(x))
                    # use first two PALETTE colors for lines
                    ax.axvline(mean_v, linestyle="--", linewidth=1.2, label=f"mean={mean_v:.3g}",
                               color=PALETTE[1 % len(PALETTE)])
                    ax.axvline(med_v, linestyle="-.", linewidth=1.0, label=f"median={med_v:.3g}",
                               color=PALETTE[2 % len(PALETTE)])

                    if kde_available and x.size >= 5 and np.isfinite(x).all():
                        from scipy.stats import gaussian_kde
                        try:
                            kde = gaussian_kde(x)
                            xs = np.linspace(x.min(), x.max(), 256)
                            scale = (n_vals.sum() * np.diff(bins_edges).mean())
                            ax.plot(xs, kde(xs) * scale, linewidth=1.6, label="KDE",
                                    color=PALETTE[3 % len(PALETTE)])
                        except Exception:
                            pass
                    ax.legend(loc="upper right", fontsize=8)

            # 4) Fallback
            else:
                s_cat = s.astype("object").where(~s.isna(), other="âŸ‚ NA")
                vc = s_cat.value_counts(dropna=False).head(max_cats)
                ax.bar(np.arange(len(vc)), vc.values)
                _colorize_bars(ax, palette)
                ax.set_xticks(np.arange(len(vc)))
                ax.set_xticklabels([_wrap(str(k)) for k in vc.index], rotation=30, ha="right")
                ax.set_title(f"{col} â€” value counts (fallback)   |   missing: {missing_pct:.1%}")

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", wrap=True)
            ax.set_title(col)

    # turn off any unused axes
    total_ax = nrows * ncols
    for j in range(n, total_ax):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.suptitle("Column Distributions", fontsize=14, y=1.02)
    plt.subplots_adjust(top=0.92, bottom=0.06)
    plt.show()



plot_column_distributions_grid(
    df_train,
    ncols=3,
    max_cats=12,
    fig_scale=1.0,
    show_kde_if_available=True,
    palette=PALETTE,
    exclude=[ID_COL] 
)



def _diverging_cmap_from_theme():
    return LinearSegmentedColormap.from_list(
        "theme_diverging",
        [COLORS["danger"], COLORS["neutral"], COLORS["success"]],
        N=256,
    )

def correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: list,
    bool_cols: list,
    *,
    target: str | None = None,
    exclude: list | None = None,
    method: str = "pearson",   
    annotate: bool = True,
    fmt: str = ".2f",
):
    num = NUMERICAL_FEATURES
    boo = BOOLEAN_FEATURES
    cols = num + boo + ([target] if (target and target in df.columns) else [])
    if exclude:
        cols = [c for c in cols if c not in exclude]
    if not cols:
        print("No columns to correlate.")
        return None

    data = df[cols].copy()
    # treat booleans as 0/1 floats (valid for Pearson/point-biserial)
    for b in boo:
        if b in data.columns:
            data[b] = data[b].astype(float)

    corr = data.corr(method=method, numeric_only=True)

    # order by |corr with target| if available (target first row/col)
    if target in corr.columns:
        order = corr[target].abs().sort_values(ascending=False).index.tolist()
        corr = corr.loc[order, order]

    # mask upper triangle for readability
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # figure size scales with number of columns
    n = len(corr.columns)
    fig_w = max(6, 0.7 * n + 3)
    fig_h = max(5, 0.7 * n + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=_diverging_cmap_from_theme(),
        vmin=-1, vmax=1, center=0,
        annot=annotate, fmt=fmt,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.85, "label": f"{method.title()} correlation"},
        ax=ax,
    )
    title = f"Correlation heatmap ({method})"
    if target in corr.columns:
        title += f" â€” ordered by |corr({target})|"
    ax.set_title(title)
    plt.show()
    return corr



corr_pearson = correlation_heatmap(
    df_train,
    numeric_cols=NUMERICAL_FEATURES,
    bool_cols=BOOLEAN_FEATURES,
    target=TARGET,
    exclude=[ID_COL],     
    method="pearson",
    annotate=True,
)


def make_ohe(drop="if_binary"):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=drop, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


pre_tree = ColumnTransformer(
    transformers=[
        ("num",  SimpleImputer(strategy="median"), NUMERICAL_FEATURES),
        ("bool", "passthrough",                    BOOLEAN_FEATURES),  # ensure these are 0/1
        ("cat",  Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe(drop="if_binary")),
        ]), CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

# Linear/SVR/KNN: scale numerics + OHE categoricals
pre_linear = ColumnTransformer(
    transformers=[
        ("num",  Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERICAL_FEATURES),
        ("bool", "passthrough",                                                                            BOOLEAN_FEATURES),
        ("cat",  Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())]),        CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

# ---------- pipelines ----------
tree_pipe = Pipeline([("prep", pre_tree),   ("model", DecisionTreeRegressor(random_state=RANDOM_STATE))])
knn_pipe  = Pipeline([("prep", pre_linear), ("model", KNeighborsRegressor())])
lin_pipe  = Pipeline([("prep", pre_linear), ("model", LinearRegression())])


for b in BOOLEAN_FEATURES:
    if b in df_train.columns: df_train[b] = df_train[b].astype(int)

X = df_train[CATEGORICAL_FEATURES + BOOLEAN_FEATURES + NUMERICAL_FEATURES].copy()
y = df_train[TARGET].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)


# ---------- evaluation ----------
def fit_eval(estimator, name):
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    rmse = mean_squared_error(y_test, pred, squared=False)
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "pred": pred, "est": estimator}


rows = []
for name, pipe in [
    ("DecisionTree", tree_pipe),
    ("KNN",          knn_pipe),
    ("LinearReg",    lin_pipe),
]:
    rows.append(fit_eval(pipe, name))

df_results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
display(df_results[["model", "rmse", "mae", "r2"]].style.format({"rmse": "{:.4f}", "mae": "{:.4f}", "r2": "{:.3f}"}))

# ---------- plot RMSE (palette-only) ----------
fig, ax = plt.subplots(figsize=(8, 0.6*len(df_results)+3), constrained_layout=True)
bars = ax.barh(df_results["model"], df_results["rmse"])
for i, b in enumerate(bars):
    b.set_facecolor(PALETTE[i % len(PALETTE)])
    b.set_edgecolor("none")
    ax.text(b.get_width(), b.get_y() + b.get_height()/2, f" {b.get_width():.4f}", va="center", ha="left")
ax.invert_yaxis()
ax.set_xlabel("RMSE (lower is better)")
ax.set_title("Model comparison â€” RMSE")
ax.grid(True, axis="x", alpha=0.25)
plt.show()


best = df_results.iloc[0]
best_name = best["model"]
best_est  = best["est"]
best_pred = best["pred"]
resid = y_test - best_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# (1) Predicted vs Actual (should lie near y=x)
axes[0].scatter(y_test, best_pred, s=20, alpha=0.9, color=PALETTE[0], edgecolors="none")
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
axes[0].plot(lims, lims, color=PALETTE[1], linestyle="--", linewidth=1.2)
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")
axes[0].set_title(f"{best_name}: Predicted vs Actual")

# (2) Residuals vs Predicted (should be centered around 0, no pattern)
axes[1].scatter(best_pred, resid, s=20, alpha=0.9, color=PALETTE[2], edgecolors="none")
axes[1].axhline(0, color=PALETTE[1], linestyle="--", linewidth=1.2)
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Residual")
axes[1].set_title(f"{best_name}: Residuals vs Predicted")

plt.show()

print(f"Best by RMSE: {best_name}  |  RMSE={best['rmse']:.4f},  MAE={best['mae']:.4f},  RÂ²={best['r2']:.3f}")


# --- config / helpers ---
feat_cols = CATEGORICAL_FEATURES + BOOLEAN_FEATURES + NUMERICAL_FEATURES
target_col = TARGET
id_col = ID_COL

# Ensure booleans are 0/1 in train (and later for test)
for b in BOOLEAN_FEATURES:
    if b in df_train.columns: df_train[b] = df_train[b].astype(int)

# 1) pick the best model by RMSE from your table
best_row = df_results.iloc[0]
best_name = best_row["model"]
best_template = best_row["est"]         
final_model = clone(best_template)
print(f"Using best model: {best_name}")

# 2) fit on ALL training rows
X_full = df_train[feat_cols].copy()
y_full = df_train[target_col].astype(float)
final_model.fit(X_full, y_full)


test_df = df_test.copy()


# booleans in test â†’ 0/1
for b in BOOLEAN_FEATURES:
    if b in test_df.columns: test_df[b] = test_df[b].astype(int)

# 4) predict on test features
X_sub = test_df[feat_cols].copy()
preds = final_model.predict(X_sub)

oob = ((preds < 0) | (preds > 1)).sum()
if oob:
    preds = np.clip(preds, 0.0, 1.0)

# 5) build submission (prefer sample_submission for exact cols/order)
SUB_PATH = "/kaggle/input/playground-series-s5e10/sample_submission.csv"
try:
    sample = pd.read_csv(SUB_PATH)
    sample_target_col = sample.columns[1]
    if id_col in test_df.columns:
        aligned = pd.Series(preds, index=test_df[id_col]).reindex(sample.iloc[:, 0]).values
        sample.iloc[:, 1] = aligned
    else:
        sample.iloc[:, 1] = preds
    submission = sample.rename(columns={sample_target_col: target_col})
except Exception:
    # fallback: create directly from test_df
    if id_col not in test_df.columns:
        # last resort id
        test_df[id_col] = np.arange(len(test_df))
    submission = pd.DataFrame({id_col: test_df[id_col].values, target_col: preds})

# 6) save
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", submission.shape)
display(submission.head())


