# -------------------------
# Basic / Environment
# -------------------------
import os                          # file paths, env vars
import sys                         # system utilities
import gc                          # garbage collection (memory mgmt)
import random                      # reproducible sampling
import warnings                    # suppress harmless warnings
from datetime import datetime      # timestamps / parsing

# make outputs deterministic where possible
SEED = 42
random.seed(SEED)

# suppress warnings for cleaner notebook output
warnings.filterwarnings("ignore")


# -------------------------
# Data handling
# -------------------------
import numpy as np                 # numerical arrays & math
import pandas as pd                # dataframes / CSV I/O
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 160)


# -------------------------
# Visualization (static + interactive)
# -------------------------
import matplotlib.pyplot as plt    # base plotting
import seaborn as sns              # statistical visualisations
# interactive (very useful on Kaggle to engage readers)
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Statistical tests & utilities
# -------------------------
from scipy import stats            # KS test, chi2, distributions
from collections import Counter    # quick counts / diagnostics
from scipy.stats import fisher_exact
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -------------------------
# Preprocessing & encoding
# -------------------------
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import category_encoders as ce     # target / ordinal / count encoders


# -------------------------
# Modeling (lightweight for EDA: quick baselines / lift charts)
# -------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# model evaluation helpers
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    brier_score_loss,
)


# -------------------------
# Explainability & feature importance (mini-models for EDA)
# -------------------------
import shap                        # SHAP values for feature importance


# -------------------------
# I/O, parallelism & progress
# -------------------------
from tqdm.auto import tqdm         # progress bars
import joblib                      # caching intermediate artifacts


# -------------------------
# Kaggle-specific / polishing
# -------------------------
from IPython.display import display, HTML   # nicer tables / html in notebook



# -------------------------
# Plot styling helper
# -------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)



TRAIN_FILE = "/kaggle/input/playground-series-s5e8/train.csv"
TEST_FILE  = "/kaggle/input/playground-series-s5e8/test.csv"

train = pd.read_csv(TRAIN_FILE, index_col="id", low_memory=False)
test  = pd.read_csv(TEST_FILE,  index_col="id", low_memory=False)

print("Datasets loaded successfully âœ…")
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")



print("ğŸ”¹ First 5 rows:")
display(train.head())




rows, cols = train.shape
mem_mb = train.memory_usage(deep=True).sum() / 1024**2
print(f"ğŸ“Š Overview â€” Train Data")
print(f"Shape: {rows} rows Ã— {cols} cols    |    Memory: {mem_mb:.2f} MB")
train.info()



miss = train.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss):
    print("ğŸ”¹ Missing values (sorted):")
    print(miss)
else:
    print("ğŸ”¹ Missing values: None")



num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(include=['object','category','bool']).columns.tolist()

print(f"ğŸ”¹ Numerical columns ({len(num_cols)}): {num_cols}\n")
print(f"ğŸ”¹ Categorical/other columns ({len(cat_cols)}): {cat_cols}\n")



if num_cols:
    print("ğŸ”¹ Descriptive statistics (numerical):")
    display(train[num_cols].describe().T)



top_n_cat = 8
if cat_cols:
    print(f"ğŸ”¹ Top {top_n_cat} categories (sample) for categorical columns:")
    for c in cat_cols:
        print(f"  -> {c} (nunique={train[c].nunique()}):")
        print(train[c].value_counts(dropna=False).head(top_n_cat).to_string())
        print("-" * 40)



# Target Analysis
def analyze_target(df, target="y"):
    """
    Analyze the target variable distribution.
    Shows counts, percentages, and a simple barplot.
    """
    print("Target Variable Distribution:\n")
    counts = df[target].value_counts()
    percentages = df[target].value_counts(normalize=True) * 100
    
    display(pd.DataFrame({"Counts": counts, "Percentage (%)": percentages.round(2)}))
    
    # Barplot
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title(f"Distribution of Target: {target}")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.show()

# Run for training data
analyze_target(train, target="y")




def univariate_numeric(df, numeric_cols, target=None, sample_frac=1.0, plot=True):
    if sample_frac <= 0 or sample_frac > 1:
        raise ValueError("sample_frac must be in (0, 1].")
    plot_df = df.sample(frac=sample_frac, random_state=SEED) if sample_frac < 1.0 else df
    
    for col in numeric_cols:
        if col not in df.columns:
            print(f"\nâš ï¸�  Column '{col}' not found â€” skip.")
            continue
        
        print(f"\n=== {col} ===")
        s = df[col]
        
        # Essential stats
        mean, median, std = s.mean(), s.median(), s.std()
        skew, kurtosis = s.skew(), s.kurtosis()
        n, nunique = len(s), s.nunique()
        
        # Compact summary
        stats_table = {
            "n": n, "n_unique": nunique,
            "mean": mean, "median": median,
            "std": std, "skew": skew, "kurtosis": kurtosis,
        }
        display(pd.DataFrame.from_dict(stats_table, orient="index", columns=[col]))
        
        # Correlation with target (if binary and provided)
        if target is not None and target in df.columns:
            joined = df[[col, target]].dropna()
            if joined.shape[0] >= 5 and set(joined[target].unique()).issubset({0,1}):
                r, p = stats.pointbiserialr(joined[target], joined[col])
                print(f"  â†’ point-biserial r = {r:.4f}, p = {p:.3g}")
        
        # Skewness note
        if abs(skew) > 1:
            print(f"  âš  Skew={skew:.2f} â†’ consider log1p or rank transform.")
        
        # Plots
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Distribution
            sns.histplot(plot_df[col].dropna(), bins=30, kde=True, ax=axes[0], color="skyblue")
            axes[0].set_title(f"Distribution of {col}")
            
            # Boxplot by target 
            if target is not None and target in df.columns:
                sns.boxplot(x=plot_df[target].astype(str), y=plot_df[col], ax=axes[1], palette="Set2")
                axes[1].set_title(f"{col} by Target")
                axes[1].set_xlabel(target)

                # â�¡ï¸� Ajouter les moyennes par classe
                means = plot_df.groupby(target)[col].mean()
                for i, cls in enumerate(means.index):
                    axes[1].text(i, means[cls], f"Mean={means[cls]:.2f}", 
                                 ha='center', va='bottom', fontsize=9, color="red", fontweight="bold")
            else:
                sns.boxplot(x=plot_df[col].dropna(), ax=axes[1], color="lightgreen")
                axes[1].set_title(f"Boxplot of {col}")
            
            plt.tight_layout()
            plt.show()

numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'] 
univariate_numeric(train, numeric_cols, target="y")




def plot_numeric_correlation_matrix(df, target="y", method="pearson", annotate=True, cmap="vlag"):
    
    # 2) Select numeric columns and separate features vs target
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in num_cols:
        raise ValueError(f"Target '{target}' is not in the numeric columns list.")
    feature_cols = [c for c in num_cols if c != target]
    if len(feature_cols) == 0:
        raise ValueError("No numeric explanatory features found (all numeric cols are only the target).")
    
    # 3) Compute correlation matrix for explanatory features only
    corr_features = df[feature_cols].corr(method=method)
    
    # -------- Full Heatmap for explanatory variables (no mask) --------
    # Dynamically choose a large figure size based on number of features
    n = len(feature_cols)
    # width/height scaling factors - adjust to taste
    width = max(12, 0.45 * n)
    height = max(10, 0.45 * n)
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Annotate only when not too many features (to avoid clutter)
    annotate_cells = annotate and (n <= 20)
    
    sns.heatmap(
        corr_features,
        cmap=cmap,
        center=0,
        annot=annotate_cells,
        fmt=".2f" if annotate_cells else "",
        linewidths=0.5,
        cbar_kws={"shrink": 0.6},
        ax=ax,
        square=False
    )
    ax.set_title(f"Correlation matrix â€” explanatory numeric features (method={method})", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # -------- Correlation with target (barplot) --------
    corr_with_target = df[num_cols].corr(method=method)[target].drop(target)
    corr_with_target = corr_with_target.sort_values(key=lambda x: x.abs(), ascending=False)
    
    plt.figure(figsize=(10, max(4, 0.4*len(corr_with_target))))
    colors = ["crimson" if v < 0 else "steelblue" for v in corr_with_target.values]
    sns.barplot(x=corr_with_target.values, y=corr_with_target.index, palette=colors)
    plt.xlabel(f"Correlation with target ({method})")
    plt.title("Numeric features correlation with target (sorted by absolute value)", fontsize=12)
    plt.axvline(0, color="k", linewidth=0.6)
    plt.tight_layout()
    plt.show()

# Example usage:
plot_numeric_correlation_matrix(train, target="y", method="pearson", annotate=True)



sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
def analyze_categorical_features_improved_no_ci(df, target="y", max_unique=20, top_n=20, sort_by="count"):

    # detect categorical-like columns
    categorical_features = [
        col for col in df.columns 
        if (df[col].dtype == "object" or df[col].nunique() <= max_unique) and col != target
    ]
    
    for col in categorical_features:
        print(f"\n=== Feature: {col} ===")
        
        # aggregated summary: counts, sum (positives), mean
        grp = df.groupby(col)[target].agg(count='count', positives='sum', mean='mean').reset_index()
        grp['pct'] = (grp['mean'] * 100).round(2)
        
        # select top_n based on requested criterion
        if sort_by == "target":
            top = grp.sort_values('mean', ascending=False).head(top_n)
        else:
            top = grp.sort_values('count', ascending=False).head(top_n)
        
        # Display table (sorted by count)
        display_cols = ['count', 'positives', 'pct']
        display(top[[col] + display_cols].sort_values('count', ascending=False).reset_index(drop=True))
        
        # Plot: horizontal bar for counts + dot for target proportion (no CI)
        top = top.reset_index(drop=True)
        y_pos = np.arange(len(top))
        
        fig, ax_count = plt.subplots(figsize=(10, 0.6 * len(top) + 2))
        
        # Horizontal bars = counts
        ax_count.barh(y_pos, top['count'], align='center', color='lightsteelblue', edgecolor='k')
        ax_count.set_yticks(y_pos)
        ax_count.set_yticklabels(top[col].astype(str))
        ax_count.invert_yaxis()
        ax_count.set_xlabel("Count")
        ax_count.set_title(f"{col} â€” Count (bar) and Target Proportion (dots)")
        
        # Secondary axis for target proportion (0..1)
        ax_prop = ax_count.twiny()
        ax_prop.set_xlim(0, 1)  # proportion scale 0..1
        # plot dots (no errorbars)
        ax_prop.scatter(top['mean'], y_pos, color='crimson', s=60, zorder=5)
        ax_prop.set_xlabel("Target Proportion (0..1)")
        
        # annotate proportion pct next to each dot
        for i, (m, p) in enumerate(zip(top['mean'], top['pct'])):
            ax_prop.text(m + 0.02, i, f"{p:.1f}%", va='center', color='crimson', fontsize=9)
        
        plt.tight_layout()
        plt.show()

# Example usage:
analyze_categorical_features_improved_no_ci(train, target="y", max_unique=30, top_n=12, sort_by='count')



# 1) CramÃ©r's V for categorical features vs binary target
def cramers_v_for_categoricals(df, categorical_cols, target="y"):
    results = []
    n = len(df)
    for col in categorical_cols:
        ct = pd.crosstab(df[col], df[target])
        chi2, p, dof, ex = stats.chi2_contingency(ct)
        phi2 = chi2 / n
        r, k = ct.shape
        V = np.sqrt(phi2 / min(k-1, r-1)) if min(k-1, r-1) > 0 else 0.0
        results.append({"feature": col, "cramers_v": V, "chi2_p": p})
    res = pd.DataFrame(results).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    return res


# 2) Interaction heatmap using log-odds 
def interaction_logodds_heatmap(df, row_col, col_col, target="y", min_count=50, cmap="RdBu_r"):
  
    pivot = df.groupby([row_col, col_col])[target].agg(sum='sum', count='count').reset_index()
    pivot['p'] = pivot['sum'] / pivot['count']
    pivot.loc[pivot['count'] < min_count, 'p'] = np.nan
    eps = 1e-6
    pivot['p_clamped'] = pivot['p'].clip(eps, 1-eps)
    pivot['logodds'] = np.log(pivot['p_clamped'] / (1 - pivot['p_clamped']))
    heat = pivot.pivot(index=row_col, columns=col_col, values='logodds')
    # plot
    plt.figure(figsize=(14, max(6, 0.35 * heat.shape[0])))
    sns.heatmap(heat, cmap=cmap, center=0, linewidths=.5, annot=False)
    plt.title(f"Log-odds heatmap: {row_col} Ã— {col_col} (min_count={min_count})")
    plt.ylabel(row_col)
    plt.xlabel(col_col)
    plt.tight_layout()
    plt.show()
    return heat



# 3) Top segments discovery (combinations with high uplift and volume)
def top_segments(df, group_cols, target='y', top_k=20, min_count=100):
    
    baseline = df[target].mean()
    grp = df.groupby(group_cols)[target].agg(count='count', positives='sum').reset_index()
    grp['rate'] = grp['positives'] / grp['count']
    grp = grp[grp['count'] >= min_count].copy()
    grp['uplift'] = grp['rate'] - baseline
    return grp.sort_values(['uplift', 'count'], ascending=[False, False]).head(top_k)


# 6) Fit a compact logistic regression with one interaction (example)
def fit_logistic_interaction(df, formula, sample_frac=1.0):
   
    if sample_frac < 1.0:
        df_fit = df.sample(frac=sample_frac, random_state=1)
    else:
        df_fit = df
    model = smf.logit(formula=formula, data=df_fit).fit(disp=False, maxiter=200)
    params = model.params
    conf = model.conf_int()
    or_df = pd.DataFrame({
        "coef": params,
        "odds_ratio": np.exp(params),
        "ci_lower": np.exp(conf[0]),
        "ci_upper": np.exp(conf[1])
    })
    return model, or_df.sort_values("odds_ratio", key=lambda x: x.abs(), ascending=False)



# Define lists
categorical_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']
numeric_cols = ['age','duration','balance','day','campaign','pdays','previous']  # adapt as needed

# 1) CramÃ©r's V (categorical associations)
cramers = cramers_v_for_categoricals(train, categorical_cols, target='y')
display(cramers.head(9))




#Interaction heatmap job x month
heat0 = interaction_logodds_heatmap(train, 'job', 'month', target='y', min_count=200)



heat1 = interaction_logodds_heatmap(train, 'poutcome', 'month', target='y', min_count=200)



heat2 = interaction_logodds_heatmap(train, 'poutcome', 'contact', target='y', min_count=200)



# CrÃ©er des classes de duration (ex: 5 intervalles par quantiles)
train['duration_bin'] = pd.qcut(train['duration'], q=5, duplicates="drop")

# Puis refaire ton heatmap
heat3 = interaction_logodds_heatmap(train, 'job', 'duration_bin', target='y', min_count=200)



heat4 = interaction_logodds_heatmap(train, 'poutcome', 'duration_bin', target='y', min_count=200)



heat5 = interaction_logodds_heatmap(train, 'contact', 'duration_bin', target='y', min_count=200)




heat6 = interaction_logodds_heatmap(train, 'housing', 'duration_bin', target='y', min_count=200)



heat7 = interaction_logodds_heatmap(train, 'job', 'duration_bin', target='y', min_count=200)



# 5) Top segments 
segs = top_segments(train, ['job','month','poutcome','contact'], target='y', top_k=20, min_count=100)
display(segs)




# Logistic regression with poutcome, month, and duration
formula = 'y ~ duration + C(poutcome) + C(month)'
model, or_table = fit_logistic_interaction(train, formula, sample_frac=0.15)  # sample 15%
print(model.summary())
display(or_table.head(20))



def plot_boxplots(df, num_cols):
   
    for col in num_cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col], color="skyblue")
        plt.title(f"Boxplot - {col}")
        plt.show()

def analyze_outliers(df, num_cols):
   
    results = {}
    for col in num_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        results[col] = round(100 * outliers / len(df), 2)  # %
    return results



numeric_features = train.select_dtypes(include=["int64", "float64"]).columns

# 1. Outlier visualization
plot_boxplots(train, numeric_features)

# 2. Outlier percentage per variable
outlier_summary = analyze_outliers(train, numeric_features)
print("Outlier % per variable:", outlier_summary)



import numpy as np
import pandas as pd

num_cols = ["balance", "duration", "campaign", "previous"] 


def log1p_shift(s):
    min_val = s.min(skipna=True)
    offset = -min_val + 1 if min_val <= 0 else 0
    return np.log1p(s + offset)


train_transformed = train.copy()


for col in num_cols:
    train_transformed[col] = log1p_shift(train_transformed[col])

train_transformed["never_contacted"] = (train_transformed["pdays"] == -1).astype(int)
pdays_pos = train_transformed["pdays"].replace(-1, np.nan)
train_transformed["pdays_log"] = log1p_shift(pdays_pos)

test_transformed = test.copy()

for col in num_cols:
    min_val = train[col].min()
    offset = -min_val + 1 if min_val <= 0 else 0
    test_transformed[col] = np.log1p(test[col] + offset)

test_transformed["never_contacted"] = (test_transformed["pdays"] == -1).astype(int)
pdays_pos_test = test_transformed["pdays"].replace(-1, np.nan)
min_val_pdays = train["pdays"].replace(-1, np.nan).min()
offset_pdays = -min_val_pdays + 1 if min_val_pdays <= 0 else 0
test_transformed["pdays_log"] = np.log1p(pdays_pos_test + offset_pdays)

train_transformed.head()
test_transformed.head()



test_transformed.head()


BINARIZE = ["default", "housing", "loan"]
LABEL_COLS = ["marital", "education", "contact", "poutcome", "job"]
MONTH_ORDER = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

def binarize_columns(train_df, test_df, cols):
    train = train_df.copy()
    test  = test_df.copy()
    created = []
    for c in cols:
        if c in train.columns:
            if train[c].dtype == "O":
               
                mapping = {"yes":1, "y":1, "true":1, "1":1, "no":0, "n":0, "false":0, "0":0}
                train[c + "_bin"] = train[c].astype(str).str.lower().map(mapping).fillna(0).astype(int)
                test[c + "_bin"]  = test[c].astype(str).str.lower().map(mapping).fillna(0).astype(int)
            else:
                
                train[c + "_bin"] = train[c].fillna(0).astype(int)
                test[c + "_bin"]  = test[c].fillna(0).astype(int)
            created.append(c + "_bin")
    return train, test, created

def label_encode_safe(train_df, test_df, cols):
   
    train = train_df.copy()
    test  = test_df.copy()
    mappings = {}
    created = []
    for c in cols:
        if c in train.columns:
            
            tr_vals = train[c].fillna("___NA___").astype(str)
            te_vals = test[c].fillna("___NA___").astype(str)
           
            uniques = pd.Index(tr_vals.unique())
            
            train_cat = pd.Categorical(tr_vals, categories=uniques)
            test_cat  = pd.Categorical(te_vals, categories=uniques)
            train_codes = train_cat.codes.astype(int)   # -1 will not appear for train
            test_codes  = test_cat.codes.astype(int)    # unseen -> -1
            train[c + "_le"] = train_codes
            test[c + "_le"]  = test_codes
            mappings[c] = list(uniques)
            created.append(c + "_le")
    return train, test, mappings, created

def month_cyclic_features(df, month_col="month"):
    out = df.copy()
    if month_col in out.columns:
        mnum = out[month_col].map(MONTH_ORDER).fillna(0).astype(int)
        radians = 2 * np.pi * mnum / 12
        out[month_col + "_sin"] = np.sin(radians)
        out[month_col + "_cos"] = np.cos(radians)
    return out, [month_col + "_sin", month_col + "_cos"] if month_col in out.columns else (out, [])

# ----------------- Apply to your dataframes -----------------
train_tmp = train_transformed.copy()
test_tmp  = test_transformed.copy()

# 1) binarize
train_tmp, test_tmp, bin_cols = binarize_columns(train_tmp, test_tmp, BINARIZE)

# 2) label encode safely
train_tmp, test_tmp, label_mappings, label_cols_created = label_encode_safe(train_tmp, test_tmp, LABEL_COLS)

# 3) month cyclic (adds month_sin, month_cos)
train_tmp, month_cols = month_cyclic_features(train_tmp, "month")
test_tmp, _ = month_cyclic_features(test_tmp, "month")

# 4) package results
train_encoded = train_tmp
test_encoded  = test_tmp
artifacts = {
    "binarized_columns": bin_cols,
    "label_mappings": label_mappings,      # dict col -> list(categories_in_train)
    "label_encoded_columns": label_cols_created,
    "month_cyclic_columns": month_cols
}

# quick info
print("Binarized columns created:", artifacts["binarized_columns"])
print("Label-encoded columns created:", artifacts["label_encoded_columns"])
print("Month cyclic columns:", artifacts["month_cyclic_columns"])
# sample mapping for job (if present)
if "job" in artifacts["label_mappings"]:
    sample_map = {cat:i for i,cat in enumerate(artifacts["label_mappings"]["job"])}
    print("Sample 'job' mapping (train categories -> codes):", dict(list(sample_map.items())[:10]))

train_encoded.head()
test_encoded.head()



orig_categorical_cols = ["job", "marital", "education", "contact", "month", "poutcome", "default", "housing", "loan"]
to_drop_train = [c for c in orig_categorical_cols if c in train_encoded.columns]
to_drop_test  = [c for c in orig_categorical_cols if c in test_encoded.columns]

train_encoded = train_encoded.drop(columns=to_drop_train, errors="ignore")
test_encoded  = test_encoded.drop(columns=to_drop_test, errors="ignore")




train_encoded = train_encoded.drop('duration_bin', axis=1)

