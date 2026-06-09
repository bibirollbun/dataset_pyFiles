!pip install statsmodels > pip_log_statsmodels.txt 2>&1
!pip install scikit_posthocs > pip_log_scikit_posthocs.txt 2>&1
!pip install pingouin > pip_log_pingouin.txt 2>&1


# Core data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import shap

# Statistical functions
from scipy.stats import skew, kurtosis, probplot

# Display utilities for Jupyter notebooks
from IPython.display import display, HTML

# Machine learning preprocessing and modeling
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Metrics
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             precision_recall_curve, classification_report, average_precision_score)

# Statistical
from scipy.stats import chi2_contingency
from scipy.stats import probplot
from scipy.stats import kruskal
import scikit_posthocs as sp
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
from scipy.stats import levene
from scipy import stats
import pingouin as pg
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500) # To display all the columns of dataframe
pd.set_option("max_colwidth", None) # To set the width of the column to maximum


class Config:
    seed = 42
    max_iter = 50000
    n_split_shuffle = 1
    n_split_kfold = 5
    test_size = 0.2
    train_csv = "/kaggle/input/playground-series-s4e1/train.csv"
    test_csv = "/kaggle/input/playground-series-s4e1/test.csv"
    original_csv = "/kaggle/input/churn-modeling/Churn_Modelling.csv"
    target_feature = "Exited"   


# Load the datasets
df_train = pd.read_csv(Config.train_csv)
df_origin = pd.read_csv(Config.original_csv)
df_test = pd.read_csv(Config.test_csv)

# Verify shapes
print("[i] Train Data Shape:", df_train.shape)
print("\n[i] Origin Data Shape:", df_origin.shape)
print("\n[i] Test Data Shape:", df_test.shape)


# Display few rows of each dataset
print("[i] Train Data Preview:")
display(df_train.head())

print("\n[i] Origin Data Preview:")
display(df_origin.head())

print("\n[i] Test Data Preview:")
display(df_test.head())


# Display information about the DataFrames
print("[i] Train Data Info:")
df_train.info()

print("\n[i] Origin Data Info:")
df_origin.info()

print("\n[i] Test Data Info:")
df_test.info()

print("\n[âœ“] Completed displaying DataFrame info.")


# Drop columns Surname, id
df_train.drop(columns="Surname", axis=1, inplace=True)
df_train.drop(columns="id", axis=1, inplace=True)
df_train.drop(columns="CustomerId", axis=1, inplace=True)

df_origin.drop(columns="Surname", axis=1, inplace=True)
df_origin.drop(columns="RowNumber", axis=1, inplace=True)
df_origin.drop(columns="CustomerId", axis=1, inplace=True)

list_test_id = df_test["id"].copy().to_list()
df_test.drop(columns="Surname", axis=1, inplace=True)
df_test.drop(columns="id", axis=1, inplace=True)
df_test.drop(columns="CustomerId", axis=1, inplace=True)

# Remove space in name columns
df_train.columns = (
    df_train.columns
    .str.strip()
    .str.replace(" ", "")
)

df_origin.columns = (
    df_origin.columns
    .str.strip()
    .str.replace(" ", "")
)

df_test.columns = (
    df_test.columns
    .str.strip()
    .str.replace(" ", "")
)


num_features = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
cat_features = ["Geography", "Gender", "HasCrCard", "IsActiveMember", "NumOfProducts"]
cm = sns.light_palette("blue", as_cmap=True)
print("[i] Train Data describe:")
display(df_train[num_features].describe().T.style.background_gradient(cmap=cm))

print("\n[i] Test Data describe:")
display(df_test[num_features].describe().T.style.background_gradient(cmap=cm))

print("\n[i] Origin Data describe:")
display(df_origin[num_features].describe().T.style.background_gradient(cmap=cm))


def cast_features(dfs, dtype_map):
    """
    Cast multiple features in multiple DataFrames to specific dtypes.

    Args:
        dfs (list): List of DataFrames to process.
        dtype_map (dict): {column_name: target_dtype} mapping.
    """
    for df in dfs:
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

# Mapping of columns and their target data types
dtype_map = {
    "CreditScore": "int16",
    "Age": "int8",
    "Tenure": "int8",
    "Balance": "float32",
    "Exited": "int8",
    "EstimatedSalary": "float32",
    "HasCrCard": "int8",
    "IsActiveMember": "int8",
    "NumOfProducts": "int8"
}

df_origin.dropna(inplace=True)

# Apply the function to both df_train and df_test
cast_features([df_train, df_test, df_origin], dtype_map)

def convert_cat(features, df):
    """
    Convert specified columns to categorical dtype.
    This helps reduce memory usage and prepares features
    for encoding steps in the ML pipeline.
    """
    for feature in features:
        if feature in df.columns:
            df[feature] = df[feature].astype("category")

convert_cat(cat_features, df=df_train)
convert_cat(cat_features, df=df_origin)
convert_cat(cat_features, df=df_test)

print("[i] Train Data describe:")
display(df_train[cat_features].describe().T.style.background_gradient(cmap="Blues", subset=["unique", "freq"]))

print("\n[i] Origin Data describe:")
display(df_origin[cat_features].describe().T.style.background_gradient(cmap="Blues", subset=["unique", "freq"]))

print("\n[i] Test Data describe:")
display(df_test[cat_features].describe().T.style.background_gradient(cmap="Blues", subset=["unique", "freq"]))


def displayNULL(df, dataset_name=None):
    total_rows = len(df)

    # Replace blank strings with NaN for completeness
    df_null_check = df.replace(r"^\s*$", np.nan, regex=True)

    missing_df = df_null_check.isnull().sum().reset_index()
    missing_df.columns = ["Feature", "Missing_Count"]
    missing_df = missing_df[missing_df["Missing_Count"] > 0]
    missing_df["Missing_%"] = (missing_df["Missing_Count"] / total_rows * 100).round(2)
    missing_df = missing_df.sort_values(by="Missing_Count", ascending=False).reset_index(drop=True)

    total_missing = missing_df["Missing_Count"].sum()

    print("=" * 80)
    
    if total_missing == 0:
        print(f"[âœ“] No missing values detected in {total_rows:,} rows.")
    else:
        try:
            from tabulate import tabulate
            print(tabulate(missing_df, headers="keys", tablefmt="pretty", showindex=False, colalign=("left", "left", "left")))
        except ImportError:
            print(missing_df.to_string(index=False))
        
        print(f"\n[!] Total missing values: {total_missing:,} out of {total_rows:,} rows.")

print("[i] Missing value train dataset... ")
displayNULL(df_train, dataset_name="Train Set")

print("\n[i] Missing value Origin dataset... ")
displayNULL(df_origin, dataset_name="Origin Set")

print("\n[i] Missing value test dataset... ")
displayNULL(df_test, dataset_name="Test Set")


def check_duplicates_report(df, dataset_name):
    duplicates_count = df.duplicated().sum()
    total_rows = len(df)
    
    # print("=" * 80)
    print(f"[i] {dataset_name} Duplicate Analysis...")
    print("=" * 80)

    if duplicates_count == 0:
        print(f"[âœ“] No duplicate rows found out of {total_rows:,} total rows.")
    else:
        print(f"[!] {duplicates_count:,} duplicate rows detected ({duplicates_count/total_rows:.2%}).")
        print(f"[!] Rows affected: {duplicates_count:,} out of {total_rows:,}.")

datasets = {
    "Training Data": df_train,
    "Origin Data": df_origin,
    "Test Data": df_test
}

duplicate_summary = {}
for name, data in datasets.items():
    check_duplicates_report(data, name)
    duplicate_summary[name] = {
        "duplicates": data.duplicated().sum(),
        "total_rows": len(data)
    }
    print()

print("[âœ“] Duplicate value inspection successfully completed across datasets.")


def checking_outlier(list_feature, df, dataset_name):
    print("=" * 40)
    print(f"[i] Checking for outliers in {dataset_name}...")
    print("=" * 40)
    outlier_info = []
    
    for feature in list_feature:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]
        
        if len(outliers) > 0:
            outlier_info.append({
                "Feature": feature,
                "Outlier Count": len(outliers),
                # "Outlier Detail": outliers.tolist()
            })

    if len(outlier_info) == 0:
        print("[âœ“] No outliers detected in any of the selected features.")
    else:
        return pd.DataFrame(outlier_info)

checking_outlier(list_feature=num_features, df=df_train, dataset_name="Training Data")


checking_outlier(list_feature=num_features, df=df_origin, dataset_name="Origin data")


checking_outlier(list_feature=num_features, df=df_test, dataset_name="Test data")


def color(n_colors=2, tone="diverging"):
    """
    Generate a list of colors based on predefined Seaborn palettes or custom palettes.

    Parameters
    ----------
    n_colors : int, default=2
        Number of colors to generate. If the palette is continuous (e.g., diverging, viridis),
        colors are sampled evenly across the colormap. If the palette is categorical,
        the first `n_colors` colors are returned.

    tone : str, default="diverging"
        The color style or palette to use. Supported values include:
        - "diverging": Seaborn diverging palette (continuous)
        - "pastel": Pastel palette
        - "muted": Muted palette
        - "husl": HUSL palette
        - "Dark2": Dark2 palette
        - "viridis": Viridis palette
        - "crest": Crest palette
        - "Paired": Paired categorical palette
        - "rocket", "rocket_r": Rocket palette and its reversed version
        - "mako": Mako palette
        - "RdYlGn": Redâ€“Yellowâ€“Green palette
        - "modern": Custom modern-style color set
        - "custom": Custom pastel/bright mixed color set

    Returns
    -------
    list of tuple
        A list of RGB color tuples in the range [0, 1].

    Notes
    -----
    - For continuous palettes (e.g., diverging, viridis, rocket), colors are sampled
      evenly across the palette using `numpy.linspace`.
    - For categorical palettes, the function simply slices the first `n_colors` items.

    Examples
    --------
    >>> color(3, tone="pastel")
    [(...RGB...), (...), (...)]

    >>> color(5, tone="diverging")
    [(0...., 0...., 0....), ...]

    >>> color(4, tone="modern")
    [(0.902, 0.223, 0.275), ...]
    """
    if tone == "diverging":
        cmap = sns.diverging_palette(0, 230, as_cmap=True)
    elif tone == "pastel":
        cmap = sns.color_palette("pastel")
    elif tone == "muted":
        cmap = sns.color_palette("muted")
    elif tone == "husl":
        cmap = sns.color_palette("husl")
    elif tone == "Dark2":
        cmap = sns.color_palette("Dark2")
    elif tone == "viridis":
        cmap = sns.color_palette("viridis")
    elif tone == "crest":
        cmap = sns.color_palette("crest")
    elif tone == "Paired":
        cmap = sns.color_palette("Paired")
    elif tone == "rocket":
        cmap = sns.color_palette("rocket")
    elif tone == "rocket_r":
        cmap = sns.color_palette("rocket_r")
    elif tone == "mako":
        cmap = sns.color_palette("mako")
    elif tone == "RdYlGn":
        cmap = sns.color_palette("RdYlGn")
    elif tone == "modern":
        cmap = sns.color_palette(["#E63946","#F1FAEE","#A8DADC","#457B9D","#1D3557"])
    elif tone == "custom":
        cmap = sns.color_palette(["#A077FF","#D6BBFF","#FFCAF8","#FE86C1","#40CBEA", "#9CE8EE"])

    positions = np.linspace(0, 1, n_colors)
    return [cmap(p) for p in positions] if callable(cmap) else cmap[:n_colors]



def cal_ChiSquare(cat_feature, target_feature, df, show_expected=False, show_residuals=False):
    """
    Perform a Chi-Square test of independence to evaluate whether two categorical variables
    are statistically associated (i.e., dependent) or independent from each other.

    This function tests the null hypothesis that the two categorical variables are independent.
    It prints the test statistic, degrees of freedom, p-value, and an interpretation based on the p-value.
    Optionally, it displays the expected frequency table under independence, and standardized residuals
    (including a heatmap) which help to identify specific group-level deviations.

    Parameters
    ----------
    cat_feature : str
        Name of the first categorical variable (typically the feature).

    target_feature : str
        Name of the second categorical variable (typically the target label).

    df : pd.DataFrame
        The input DataFrame containing the data.

    show_expected : bool, default=False
        If True, prints the expected frequencies under the assumption of independence.

    show_residuals : bool, default=False
        If True, prints the standardized residuals and shows them as a heatmap
        to identify where the strongest associations/deviations occur.

    Returns
    -------
    None
        Prints the Chi-Square test result, including statistical significance interpretation.
        Optionally prints expected values and standardized residuals.

    Notes
    -----
    - Hypotheses:
        Hâ‚€ (Null):     The two variables are independent (no association).
        Hâ‚� (Alt.):      There is a dependency or association between the variables.

    - Interpretation:
        If p-value < 0.05 â†’ Reject Hâ‚€ â†’ Conclude that the variables are significantly associated.
        If p-value â‰¥ 0.05 â†’ Fail to reject Hâ‚€ â†’ No statistically significant association found.

    - Standardized residuals:
        - Values > +2 or < -2 indicate strong deviation from expected frequency (local dependency).
        - Useful for identifying specific group-level contributions to the overall Chi-Square result.

    References
    ----------
    - https://en.wikipedia.org/wiki/Chi-squared_test
    - https://www.scribbr.com/statistics/chi-square-test-of-independence/
    """
    print(f"\n[i] Chi-Square Test of Independence: '{cat_feature}' vs. '{target_feature}'...")

    # Contingency table
    crosstab = pd.crosstab(df[cat_feature], df[target_feature])
    chi2, p, dof, expected = chi2_contingency(crosstab)

    print(f"[i] Chi-squared statistic: {chi2:.3f}")
    print(f"[i] Degrees of freedom: {dof}")
    print(f"[i] p-value: {p:.6f}")

    if p < 0.05:
        print("[âœ“] Result: p-value < 0.05 â†’ Reject Hâ‚€")
        print(f"[âœ“] Variables '{cat_feature}' and '{target_feature}' are statistically associated.")
    else:
        print("[!] Result: p-value â‰¥ 0.05 â†’ Fail to reject H0")
        print(f"[!] No statistically significant association detected between '{cat_feature}' and '{target_feature}'.")

    # Optional: show expected frequencies
    if show_expected:
        print("\n[i] Expected Frequencies:")
        print(pd.DataFrame(expected, index=crosstab.index, columns=crosstab.columns))

    # Optional: show standardized residuals
    if show_residuals:
        residuals = (crosstab - expected) / np.sqrt(expected)
        print("\n[i] Standardized Residuals:")
        print(round(residuals, 2))

        # Heatmap of residuals
        plt.figure(figsize=(10, 7))
        sns.heatmap(residuals, annot=True, cmap=sns.diverging_palette(0, 230, 90, 60, as_cmap=True), center=0, fmt=".2f", linewidths=0.5)
        plt.title(f"Standardized Residuals Heatmap: {cat_feature} vs {target_feature}", weight="bold", fontsize=12, pad=15)
        plt.ylabel(cat_feature)
        plt.xlabel(target_feature)
        plt.tight_layout()
        plt.show()

def perform_kruskal_test(df, categorical_feature, numeric_feature):
    """
    Perform the Kruskal-Wallis H-test to determine whether there are statistically
    significant differences in the distribution of a numeric variable across
    three or more independent groups.

    If the result is significant (p < 0.05), Dunn's post-hoc test with Bonferroni correction
    is performed to identify which group pairs differ.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing the categorical and numerical variables.

    categorical_feature : str
        The name of the categorical feature that defines the groups.

    numeric_feature : str
        The name of the numeric feature to be compared across groups.

    Returns
    -------
    None
        Prints the Kruskal-Wallis H-statistic, p-value, interpretation, and
        optionally the results of Dunn's post-hoc test.

    Notes
    -----
    - Hâ‚€ (null hypothesis): The distribution of the numeric variable is the same across all groups.
    - Hâ‚� (alternative hypothesis): At least one group has a different distribution.
    - If p < 0.05 â†’ reject Hâ‚€ â†’ use Dunnâ€™s test to explore specific group differences.
    - Kruskal-Wallis is a non-parametric alternative to one-way ANOVA.
    - It does not assume normality, but assumes:
        1. Independent samples
        2. Ordinal or continuous response variable
        3. Similar shapes of distributions

    Requirements
    ------------
    - `scipy.stats.kruskal`
    - `scikit-posthocs` package for Dunnâ€™s test (`import scikit_posthocs as sp`)

    References
    ----------
    - https://www.geeksforgeeks.org/kruskal-wallis-test/
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    - https://scikit-posthocs.readthedocs.io/en/latest/index.html
    """

    # Extract values
    groups = df[categorical_feature].dropna().unique()
    if len(groups) < 3:
        print(f"[x] Error: Kruskal-Wallis H-test requires 3 or more groups.")
        return
    else:
        print(f"\n[i] Kruskal-Wallis Test: {numeric_feature} ~ {categorical_feature}...")
        data_groups = [df[df[categorical_feature] == g][numeric_feature].dropna() for g in groups]

        # Perform kruskal
        stat, p = kruskal(*data_groups)

        print(f"[i] Kruskal-Wallis H-statistic: {stat:.3f}")
        print(f"[i] p-value: {p}")

        if p < 0.05:
            print("[âœ“] Significant difference detected among groups (p < 0.05).")
            print("[i] Running Dunn's Post-Hoc Test (Bonferroni corrected)...")
            dunn_result = sp.posthoc_dunn(df, val_col=numeric_feature, group_col=categorical_feature, p_adjust="bonferroni")
            print(dunn_result)
        else:
            print("[!] No statistically significant difference detected between groups (p â‰¥ 0.05).")

def check_normality_with_plots(df, feature, target_feature, threshold_skew_1=0.5, threshold_skew_2=1.0,
                               threshold_kurt=1.5, ncols=2):
    """
    Check the normality of numerical features *within each group* of a categorical feature,
    using Skewness, Kurtosis, and Qâ€“Q plots. 
    If non-normality is detected in any group, automatically perform Kruskalâ€“Wallis test.

    ---
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing both numeric and categorical features.

    feature : numeric
        Numerical columns to test (e.g. ["Temparature"]).

    target_feature : str
        Categorical variable name (e.g. "Fertilizer_Name").

    threshold_skew_1 : float, default = 0.5
        Threshold for approximately symmetric (|skew| â‰¤ 0.5).

    threshold_skew_2 : float, default = 1.0
        Threshold for moderate skewness (0.5 < |skew| â‰¤ 1.0).

    threshold_kurt : float, default = 1.5
        Absolute kurtosis threshold for approximate normality.

    ncols : int, default = 2
        Number of Qâ€“Q plots per row.
    """

    results = []
    non_normal_detected = False

    print(f"\n[i] Checking normality of feature '{feature}' by groups of '{target_feature}'...")

    # ===  Evaluate normality within each group ===
    print(f"[i] Evaluating distribution characteristics...")

    for grp, subset in df.groupby(target_feature):
        data = subset[feature].dropna()
        sk = skew(data)
        kt = kurtosis(data)
        abs_sk = abs(sk)
        abs_kt = abs(kt)

        # Skewness interpretation
        if abs_sk <= threshold_skew_1:
            skew_remark = "Approximately symmetric"
        elif abs_sk <= threshold_skew_2:
            skew_remark = "Moderately skewed"
        else:
            skew_remark = "Highly skewed"

        # Kurtosis interpretation
        if abs_kt < threshold_kurt:
            kurt_remark = "Normal tails"
        else:
            kurt_remark = "Heavy/light tails"

        remark = f"{skew_remark}, {kurt_remark}"
        results.append({
            "Feature": feature,
            "Group": grp,
            "Skewness": f"{sk:.4f}",
            "Kurtosis": f"{kt:.4f}",
            "Remark": remark
        })

        # Flag if any group is not approximately normal
        if not (abs_sk <= threshold_skew_1 and abs_kt <= threshold_kurt):
            non_normal_detected = True

    # === Visual Qâ€“Q plots ===
    print("[i] Generating Qâ€“Q plots for visual assessment...")

    n_groups = df[target_feature].nunique()
    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, grp in enumerate(df[target_feature].unique()):
        ax = axes[i]
        data = df.loc[df[target_feature] == grp, feature].dropna()
        probplot(data, dist="norm", plot=ax)
        ax.set_title(f"{feature} â€” {grp}", fontsize=12, weight="bold")
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Qâ€“Q Plots of {feature} by {target_feature}", fontsize=12, weight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

    # === Display results table ===
    df_result = pd.DataFrame(results)
    cm = sns.light_palette("blue", as_cmap=True)
    styled = (
        df_result.style
        .background_gradient(subset=["Skewness"], cmap=cm, vmin=-1, vmax=1)
        .background_gradient(subset=["Kurtosis"], cmap=cm, vmin=-1.5, vmax=1.5)
        .set_caption(
            f'<b><span style="font-size:14px; text-align:center; display:block;">'
            f'Skewness & Kurtosis of {feature} by {target_feature}'
            f'</span></b>'
        )
        .set_table_attributes('style="width:80%; margin:auto;"')
    )
    display(styled)

    # === Final evaluation ===
    if non_normal_detected:
        print("[!] At least one group deviates from normality.")
        print("[i] Non-normal distribution detected â†’ Consider Kruskalâ€“Wallis or Mannâ€“Whitney U test.")
    else:
        print("[âœ“] All groups approximately follow a normal distribution.")

    return non_normal_detected

def perform_anova_with_tukey(df, numeric_feature, categorical_feature, typ=2):
    """
    Perform a One-Way ANOVA test to determine whether there are statistically
    significant differences between the means of three or more independent groups.

    If the ANOVA test is significant (p < 0.05), Tukey's HSD post-hoc test is performed
    to identify which specific pairs of groups differ from each other.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing the numeric and categorical features.

    numeric_feature : str
        The name of the numerical (continuous) response variable.

    categorical_feature : str
        The name of the categorical (independent) variable used to group the data.

    typ : int, optional (default=2)
        The type of sum of squares to use in the ANOVA test:
        - Type I (1): Sequential.
        - Type II (2): Default and commonly used for balanced designs.
        - Type III (3): Use when model includes interaction terms or unbalanced data.

    Returns
    -------
    None
        Prints the ANOVA table, p-value, interpretation, and (if significant) the Tukey HSD test summary.

    Notes
    -----
    - Hâ‚€ (null hypothesis): All group means are equal.
    - Hâ‚� (alternative hypothesis): At least one group mean is different.
    - If p < 0.05 â†’ reject Hâ‚€ â†’ perform Tukeyâ€™s HSD to find which groups differ.
    - Assumptions:
        1. Independence of observations
        2. Normally distributed groups (Shapiro or Anderson test can check this)
        3. Homogeneity of variances (Levene's test)

    References
    ----------
    - https://www.scribbr.com/statistics/one-way-anova/
    - https://en.wikipedia.org/wiki/Analysis_of_variance
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html
    """

    # Extract unique groups
    groups = df[categorical_feature].dropna().unique()

    if len(groups) < 3:
        print("[x] Error: ANOVA requires at least 3 groups.")
        return
    else:
        print(f"\n[i] Running ANOVA Test: {numeric_feature} ~ {categorical_feature} (Type {typ})...")

        # Fit OLS model
        model = ols(f"{numeric_feature} ~ C({categorical_feature})", data=df).fit()

        # Perform ANOVA
        anova_table = anova_lm(model, typ=typ)
        print("\n[i] ANOVA Table:")
        print(anova_table)

        # Extract p-value
        p_value = anova_table["PR(>F)"].iloc[0]

        if p_value < 0.05:
            print("\n[âœ“] Significant difference detected (p < 0.05).")
            print("[i] Running Tukey's HSD post-hoc test...")

            tukey = pairwise_tukeyhsd(df[numeric_feature], df[categorical_feature])
            print(tukey.summary())
        else:
            print("\n[!] No statistically significant difference detected (p â‰¥ 0.05).")

def perform_welch_anova(df, numeric_feature, categorical_feature):
    """
    Perform Welchâ€™s ANOVA test to compare group means when the assumption of equal variances
    is violated but normality approximately holds.

    This version of ANOVA adjusts for unequal variances and sample sizes across groups.
    If the Welchâ€™s ANOVA is significant (p < 0.05), a Gamesâ€“Howell post-hoc test is performed
    to identify which specific group pairs differ significantly.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing both numeric and categorical variables.

    numeric_feature : str
        The name of the continuous (dependent) variable.

    categorical_feature : str
        The name of the categorical (independent) variable representing group membership.

    Returns
    -------
    None
        Prints Welchâ€™s ANOVA summary, p-value interpretation, and Gamesâ€“Howell post-hoc results.

    Notes
    -----
    - Hâ‚€ (null hypothesis): All group means are equal.
    - Hâ‚� (alternative hypothesis): At least one group mean differs.
    - If p < 0.05 â†’ reject Hâ‚€ â†’ perform Gamesâ€“Howell test.
    - Assumptions:
        1. Groups are independent.
        2. Data within each group are approximately normal.
        3. Variances are not necessarily equal (heteroscedasticity allowed).

    Key Differences vs Classical ANOVA
    ----------------------------------
    - Welchâ€™s ANOVA does **not assume equal variances**.
    - More robust when sample sizes and variances differ across groups.
    - Use **Gamesâ€“Howell post-hoc test** instead of Tukey HSD.

    References
    ----------
    - Welch, B. L. (1951). "On the comparison of several mean values: an alternative approach."
      Biometrika, 38(3/4), 330â€“336.
    - Games, P. A., & Howell, J. F. (1976). "Pairwise multiple comparison procedures with unequal Nâ€™s and/or variances."
      Journal of Educational Statistics, 1(2), 113â€“125.
    """

    # Drop NaN rows
    df = df[[numeric_feature, categorical_feature]].dropna()

    # Extract group values
    groups = [df.loc[df[categorical_feature] == g, numeric_feature] for g in df[categorical_feature].unique()]

    if len(groups) < 3:
        print("[x] Error: Welchâ€™s ANOVA requires 3 or more groups.")
        return

    print(f"\n[i] Running Welchâ€™s ANOVA Test: {numeric_feature} ~ {categorical_feature}")
    print("[i] Assessing mean differences under heteroscedastic conditions...")

    # Perform Welch's ANOVA (scipy.stats)
    welch_result = stats.f_oneway(*groups)
    print("\n[i] Welchâ€™s ANOVA Result:")
    print(f"[i] F-statistic: {welch_result.statistic:.4f}")
    print(f"[i] p-value: {welch_result.pvalue:.6f}")

    # Interpret result
    if welch_result.pvalue < 0.05:
        print("\n[âœ“] Significant difference detected (p < 0.05).")
        print("[i] Performing Gamesâ€“Howell post-hoc test...\n")

        # Perform Gamesâ€“Howell post-hoc test (robust for unequal variances)
        gh_result = pg.pairwise_gameshowell(dv=numeric_feature, between=categorical_feature, data=df)

        display(HTML("<b>Gamesâ€“Howell Post-hoc Test (adjusted p-values)</b>"))
        display(gh_result.style.background_gradient(cmap=cm).format(precision=4).set_table_attributes('style="width:80%; margin:auto;"'))
    else:
        print("\n[!] No statistically significant difference detected (p â‰¥ 0.05).")

def check_homogeneity_of_variance(df, feature, target_feature, alpha=0.05, ratio_threshold=2.0):
    """
    Check homogeneity of variances across groups using Leveneâ€™s test (median-centered).
    Also computes variance ratios and provides practical interpretation.

    ---
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing numeric and categorical features.
    feature : str
        Numeric variable to test (e.g. "Temparature").
    target_feature : str
        Categorical grouping variable (e.g. "Fertilizer_Name").
    alpha : float, default = 0.05
        Significance level for hypothesis testing.
    ratio_threshold : float, default = 2.0
        Threshold for maximum acceptable variance ratio (max(var)/min(var)).
        If ratio > threshold â†’ indicates heteroscedasticity in practice.

    ---
    Returns
    -------
    dict
        Dictionary with test statistic, p-value, variance ratio, and recommendation.

    ---
    Interpretation Logic
    ---------------------
    Step 1: Statistical Test
        - Hâ‚€: All group variances are equal.
        - Hâ‚�: At least one group has a different variance.
        - Leveneâ€™s Test (center='median') is robust to non-normality.

    Step 2: Practical Variance Ratio
        - ratio = max(var_i) / min(var_i)
        - < 2 â†’ practically equal
        - 2â€“4 â†’ moderate difference
        - > 4 â†’ strong heterogeneity

    Step 3: Recommendation
        - If p > 0.05 AND ratio < 2 â†’ Use One-Way ANOVA or Independent Two-Sample T-Test
        - If p < 0.05 BUT ratio < 2 â†’ Statistical diff, but practically negligible â†’ ANOVA or T-Test are acceptable, but Welchâ€™s ANOVA or Welchâ€™s T-Test is recommended.
        - If ratio â‰¥ 2 OR p < 0.05  â†’  Use Kruskalâ€“Wallis or Mannâ€“Whitney U test.
    """

    # Group data by category
    groups = [df.loc[df[target_feature] == g, feature].dropna() for g in df[target_feature].unique()]

    # Perform Leveneâ€™s Test (robust version)
    stat, p = levene(*groups, center="mean")

    # Compute variance ratio (max/min)
    variances = [np.var(g, ddof=1) for g in groups]
    ratio = max(variances) / min(variances)
    anova_use = False
    is_homogeneous_variances = False
    # Determine interpretation
    if p > alpha and ratio < ratio_threshold:
        status = "[âœ“] Variances are statistically and practically homogeneous."
        recommendation = "Use One-Way ANOVA or Independent Two-Sample T-Test."
        is_homogeneous_variances = True
        anova_use = True
    elif p < alpha and ratio < ratio_threshold:
        status = "[!] Statistically different variances, but practical difference is small."
        recommendation = "ANOVA or T-Test are acceptable, but Welchâ€™s ANOVA or Welchâ€™s T-Test is recommended."
        anova_use = True
    else:
        status = "[x] Strong variance heterogeneity detected."
        recommendation = "Use Kruskalâ€“Wallis or Mannâ€“Whitney U test."

    # Display summary table
    summary_df = pd.DataFrame({
        "Metric": ["Leveneâ€™s Statistic", "p-value", "Max/Min Variance Ratio"],
        "Value": [f"{stat:.4f}", f"{p:.6f}", f"{ratio:.2f}"]
    })
    display(summary_df.style
            .background_gradient(subset=["Value"], cmap="Blues")
            .set_caption(
        f'<b><span style="font-size:14px; text-align:center; display:block;">'
        f'Homogeneity of Variance â€” {feature} by {target_feature}</span></b>'
    ).set_table_attributes('style="width:70%; margin:auto;"'))

    # Print interpretation
    print("\n[i] Interpretation:")
    print(f"   {status}")
    print(f"   Recommendation â†’ {recommendation}")

    return anova_use, is_homogeneous_variances

def cal_mannwhitneyu(dataframe, categorical_feature, num_feature):
    """
    Perform the Mannâ€“Whitney U test (Wilcoxon rank-sum test) to assess whether there 
    is a statistically significant difference in the distribution of a numerical feature 
    between two independent groups defined by a binary categorical feature.

    The function also compares medians, calculates the effect size (r), provides interpretation,

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing the data.

    categorical_feature : str
        Column name of the categorical feature (must contain exactly 2 unique values).

    num_feature : str
        Column name of the numerical feature to compare.

    Returns
    -------
    None
        Prints the U statistic, p-value, medians, Z-score, effect size r, and interpretation.

    Notes
    -----
    - Hâ‚€ (Null Hypothesis): The two groups have the same distribution.
    - Hâ‚� (Alternative Hypothesis): The distributions are different.
    - If p â‰¤ 0.05 â†’ reject Hâ‚€ â†’ significant difference.
    - Effect size r helps interpret how strong the difference is:
        * Small ~0.1, Medium ~0.3, Large â‰¥0.5
    """
    groups = dataframe[categorical_feature].dropna().unique()

    # Validation check
    if len(groups) != 2:
        print(f"[x] Error: Mannâ€“Whitney U test requires exactly 2 groups, but found {len(groups)}.")
        return

    print(f"\n[i] Running Mannâ€“Whitney U Test: '{num_feature}' by '{categorical_feature}'...")

    group1 = dataframe[dataframe[categorical_feature] == groups[0]][num_feature].dropna()
    group2 = dataframe[dataframe[categorical_feature] == groups[1]][num_feature].dropna()

    stat, p = mannwhitneyu(group1, group2, alternative="two-sided")

    print(f"[i] U statistic: {stat}")
    print(f"[i] p-value    : {p:.6f}")

    # Interpretation
    if p <= 0.05:
        print("\n[âœ“] Significant difference detected between the two groups (Reject H0).")
        median1 = group1.median()
        median2 = group2.median()

        print(f"[i] Median of group '{groups[0]}': {median1:.4f}")
        print(f"[i] Median of group '{groups[1]}': {median2:.4f}")

        if median1 > median2:
            print(f"[i] Interpretation: Group '{groups[0]}' has a higher median '{num_feature}'.")
        elif median1 < median2:
            print(f"[i] Interpretation: Group '{groups[1]}' has a higher median '{num_feature}'.")
        else:
            print("[i] Interpretation: Medians are equal, although distributions may still differ.")
    else:
        print("\n[!] No statistically significant difference detected (Fail to reject H0).")

def t_test_with_cohens_d(data, categorical_feature, num_feature, equal_var=False):
    """
    Perform an Independent Two-Sample T-Test and compute Cohen's d to evaluate 
    the difference between two independent groups on a numeric variable.

    Supports both:
    - Studentâ€™s T-Test (equal variances)
    - Welchâ€™s T-Test (unequal variances, default)

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the categorical and numerical features.

    categorical_feature : str
        The name of the categorical column used to define the two groups (must have exactly 2 unique values).

    num_feature : str
        The name of the numerical feature to compare between the two groups.

    equal_var : bool, optional (default=False)
        If True â†’ Studentâ€™s t-test (equal variances).
        If False â†’ Welchâ€™s t-test (unequal variances).

    Returns
    -------
    None
        Prints the t-statistic, p-value, Cohenâ€™s d, and interpretation of the effect size.

    Notes
    -----
    - Hâ‚€ (null hypothesis): The two groups have equal means.
    - Hâ‚� (alternative): The group means differ significantly.
    - Cohen's d interpretation:
        - 0.2 â†’ small effect
        - 0.5 â†’ medium effect
        - 0.8+ â†’ large effect
    - Welchâ€™s t-test is recommended when group variances are unequal (default setting).

    References
    ----------
    - https://www.scribbr.com/statistics/t-test/
    - https://en.wikipedia.org/wiki/Welch%27s_t-test
    - https://en.wikipedia.org/wiki/Cohen%27s_d
    """

    # Extract unique groups
    groups = data[categorical_feature].dropna().unique()

    if len(groups) != 2:
        print("[x] Error: Independent T-Test requires exactly 2 groups.")
        return

    test_type = (
        "Studentâ€™s T-Test (equal variances)" 
        if equal_var 
        else "Welchâ€™s T-Test (unequal variances)"
    )

    print(f"\n[i] Running Independent Two-Sample T-Test: '{num_feature}' ~ '{categorical_feature}'")
    print(f"[i] Test Type: {test_type}")

    # Extract values
    x1 = data[data[categorical_feature] == groups[0]][num_feature].dropna()
    x2 = data[data[categorical_feature] == groups[1]][num_feature].dropna()

    # Run T-Test
    t_stat, p_value = ttest_ind(x1, x2, equal_var=equal_var)

    # Calculate Cohen's d (different formulas depending on variance assumption)
    nx1, nx2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

    if equal_var:
        # --- Studentâ€™s T-Test version (pooled variance)
        pooled_std = np.sqrt(((nx1 - 1) * s1 + (nx2 - 1) * s2) / (nx1 + nx2 - 2))
        cohens_d = (np.mean(x1) - np.mean(x2)) / pooled_std
    else:
        # --- Welchâ€™s T-Test version (average variance)
        s_pooled = np.sqrt((s1 + s2) / 2)
        cohens_d = (np.mean(x1) - np.mean(x2)) / s_pooled

    # Display results
    print(f"\n[i] Comparing groups: '{groups[0]}' vs '{groups[1]}'")
    print(f"[i] t-statistic: {t_stat:.3f}")
    print(f"[i] p-value: {p_value:.6f}")
    print(f"[i] Cohen's d: {cohens_d:.3f}")

    # Significance interpretation
    if p_value < 0.05:
        print("[âœ“] Significant difference detected (p < 0.05).")
    else:
        print("[!] No statistically significant difference detected (p â‰¥ 0.05).")

    # Effect size interpretation
    abs_d = abs(cohens_d)

    if abs_d < 0.2:
        effect_label = "small"
    elif abs_d < 0.5:
        effect_label = "medium"
    else:
        effect_label = "large"

    print(f"[i] Effect size interpretation: {effect_label} effect (|d| = {abs_d:.3f})")


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
datasets = [("Train Data", df_train), ("Original Data", df_origin)]

for i, (title, data) in enumerate(datasets):
    ax = axes[i, 0]
    sns.countplot(x=Config.target_feature, data=data, ax=ax, palette=color(n_colors=df_train[Config.target_feature].nunique()))
    ax.set_title(f"Count Plot of Exited in {title}", pad=15, weight="bold", fontsize=12)
    ax.set_ylabel("Number of Customer")
    ax.set_xlabel("Exited")
    ax.set_xticks([0, 1], ["Not churned", "churned"])

    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", label_type="edge", fontsize=10)

    loan_counts = data[Config.target_feature].value_counts().sort_index()
    wedges, texts, autotexts = axes[i, 1].pie(
        loan_counts,
        labels = ["Not Churned", "Churned"],
        autopct="%1.1f%%",
        startangle=90,
        colors=color(n_colors=df_train[Config.target_feature].nunique()),
        wedgeprops=dict(width=0.4, edgecolor="w"),
        radius=1.2,
        explode = (0, 0.08)
    )
    
    for text in texts + autotexts:
        text.set_fontsize(10)
    
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    axes[i, 1].add_artist(centre_circle)
    axes[i, 1].set_title(f"Exited in {title}", pad=15, weight="bold", fontsize=12)
    axes[i, 1].axis("equal") 

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.show()


def plot_numerical_features(df_train, df_test, df_origin, num_features):
    colors = color(n_colors=3)
    n = len(num_features)

    fig, ax = plt.subplots(n, 2, figsize=(12, n * 4))
    ax = np.array(ax).reshape(n, 2)

    for i, feature in enumerate(num_features):
        sns.histplot(data=df_train[feature], color=colors[0], bins=20, kde=True, ax=ax[i, 0], label="Train data")
        sns.histplot(data=df_origin[feature], color=colors[1], bins=20, kde=True, ax=ax[i, 0], label="Train data")
        sns.histplot(data=df_test[feature], color=colors[2], bins=20, kde=True, ax=ax[i, 0], label="Test data")
        ax[i, 0].set_title(f"Histogram of {feature}", pad=15, weight="bold", fontsize=12)
        ax[i, 0].legend()
        ax[i, 0].set_ylabel("")
        sns.despine(left=False, bottom=False, ax=ax[i, 0])

        df_plot = pd.concat([
            pd.DataFrame({"Dataset": "Train data", feature: df_train[feature]}),
            pd.DataFrame({"Dataset": "Origin data", feature: df_origin[feature]}),
            pd.DataFrame({"Dataset": "Test data", feature: df_test[feature]})
        ]).reset_index(drop=True)

        sns.boxplot(
            data=df_plot,
            x=feature,
            y="Dataset",
            palette=colors,
            orient="h",
            ax=ax[i, 1]
        )
        ax[i, 1].set_title(f"Horizontal Box plot of {feature}", pad=15, weight="bold", fontsize=12)
        sns.despine(left=False, bottom=False, ax=ax[i, 1])

    plt.tight_layout()
    plt.show()

plot_numerical_features(df_train = df_train, df_test = df_test, df_origin = df_origin, num_features=num_features)


def check_skewness(data, dataset_name, numerical_features = num_features, highlight=True, sort=True):
    skewness_dict = {}
    skew_feature = []
    for feature in numerical_features:
        skew = data[feature].skew(skipna=True)
        skewness_dict[feature] = skew

    skew_df = pd.DataFrame.from_dict(skewness_dict, orient="index", columns=["Skewness"])
    if sort:
        skew_df = skew_df.reindex(skew_df["Skewness"].abs().sort_values(ascending=False).index)

    print(f"\n[i] Skewness for {dataset_name}...")
    print("-"*70)
    print(f"{'Feature':<30} | {'Skewness':<9} | {'Remark'}")
    print("-"*70)
    for feature, row in skew_df.iterrows():
        skew = row["Skewness"]
        abs_skew = abs(skew)
        if abs_skew > 1:
            remark = "Highly skewed"
            color = "\033[91m"
        elif abs_skew > 0.5:
            remark = "Moderately skewed"
            color = "\033[93m"
        else:
            remark = "Approximately symmetric"
            color = ""
        endc = "\033[0m" if color else ""
        if highlight and color:
            print(f"{color}{feature:<30} | {skew:>+9.4f} | {remark}{endc}")
            skew_feature.append(feature)
        else:
            print(f"{feature:<30} | {skew:>+9.4f} | {remark}")
    print("-"*70)
    return skew_feature, skew_df

skew_feature_origin, skew_origin_df = check_skewness(df_origin, "Original Data")
skew_feature_train, skew_train_df = check_skewness(df_train, "Train Data")
skew_feature_test, skew_test_df = check_skewness(df_test, "Test Data")


def plot_correlation(df_train, df_origin, df_test, origin_name="Origin Data", train_name="Train Data", test_name="Test Data"):
    corr_train = df_train.corr(numeric_only=True)
    corr_origin = df_origin.corr(numeric_only=True)
    corr_test = df_test.corr(numeric_only=True)

    mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
    adjusted_mask_train = mask_train[1:, :-1]
    adjusted_cereal_corr_train = corr_train.iloc[1:, :-1]

    mask_origin = np.triu(np.ones_like(corr_origin, dtype=bool))
    adjusted_mask_origin = mask_origin[1:, :-1]
    adjusted_cereal_corr_origin = corr_origin.iloc[1:, :-1]

    mask_test = np.triu(np.ones_like(corr_test, dtype=bool))
    adjusted_mask_test = mask_test[1:, :-1]
    adjusted_cereal_corr_test = corr_test.iloc[1:, :-1]

    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    fig, ax = plt.subplots(1, 3, figsize=(25, 10))

    sns.heatmap(data=adjusted_cereal_corr_train, mask=adjusted_mask_train,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[0])
    ax[0].set_title(f"Correlation Heatmap of {train_name}", fontsize=16, weight="bold")

    sns.heatmap(data=adjusted_cereal_corr_test, mask=adjusted_mask_test,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[1])
    ax[1].set_title(f"Correlation Heatmap of {test_name}", fontsize=16, weight="bold")

    sns.heatmap(data=adjusted_cereal_corr_origin, mask=adjusted_mask_origin,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[2])
    ax[2].set_title(f"Correlation Heatmap of {origin_name}", fontsize=12, weight="bold", pad=15)

    plt.tight_layout()
    plt.show()

plot_correlation(df_train=df_train.drop(columns="Exited", axis=1),
                 df_origin=df_origin.drop(columns="Exited", axis=1),
                 df_test=df_test)


def plot_categorical_distribution_across_datasets(
    df_train: pd.DataFrame = df_train,
    df_origin: pd.DataFrame = df_origin,
    df_test: pd.DataFrame = df_test,
    feature="gender"
):
    # --- Summarize categorical feature in training set ---
    train_summary = (
        df_train[feature]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Train_Count")
    )

    # Calculate percentage distribution in training data
    train_summary["Train_%"] = round((train_summary["Train_Count"] / train_summary["Train_Count"].sum() * 100), 2)

    # --- Summarize categorical feature in training set ---
    original_summary = (
        df_origin[feature]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Origin_Count")
    )

    # Calculate percentage distribution in training data
    original_summary["Origin_%"] = round((original_summary["Origin_Count"] / original_summary["Origin_Count"].sum() * 100), 2)

    # --- Summarize categorical feature in test set ---
    test_summary = (
        df_test[feature]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Test_Count")
    )

    # Calculate percentage distribution in test data
    test_summary["Test_%"] = round((test_summary["Test_Count"] / test_summary["Test_Count"].sum() * 100), 2)

    # --- Merge train and test summaries for comparison ---
    summary = pd.merge(train_summary, original_summary, on="Category", how="inner")
    summary = pd.merge(summary, test_summary, on="Category", how="inner")

    # Ensure Category column is of string type before handling missing values
    summary["Category"] = summary["Category"].astype(str)

    # Fill missing values only for numeric columns
    cat_cols = ["Train_Count", "Train_%", "Origin_Count", "Origin_%", "Test_Count", "Test_%"]
    summary[cat_cols] = summary[cat_cols].fillna(0)

    # Convert summary table to HTML for better visualization in notebook
    html = summary.to_html(
        index=False,
        justify="center",
        classes="table table-striped table-hover",
    )

    # Display formatted HTML table with title
    display(HTML(f"[âœ“] Categorical Feature Distribution: <b>{feature}</b>" + html))
    print("*" * 80)


# ----- Run the function for all categorical features -----
for feature in cat_features:
    plot_categorical_distribution_across_datasets(df_train=df_train, df_origin=df_origin, df_test=df_test, feature=feature)


def top_ratio(df_test = df_test, df_train = df_train, df_origin=df_origin, cat_features = cat_features):
    dataset_names = ["Train", "Test", "Origin"]
    datasets = [df_train, df_origin, df_test]
    for i, (data, name) in enumerate(zip(datasets, dataset_names)):
        print(f"{name} Data")
        flagged = False
        for feature in cat_features:
            freq = data[feature].value_counts(normalize=True)
            top_ratio = freq.iloc[0]
            if top_ratio > 0.99:
                flagged = True
                print(f"[!]  {feature}: {top_ratio:.1%} lÃ  '{freq.index[0]}'")
        if not flagged:
            print("[âœ“] No feature has a category that makes up more than 99% of its values.")
        print("*" * 50)
top_ratio()


df_train_combined = pd.concat([df_train, df_origin], axis=0, ignore_index=True)
# Re-check duplicate

datasets = {
    "Training Data": df_train_combined,
    "Test Data": df_test
}

duplicate_summary = {}
for name, data in datasets.items():
    check_duplicates_report(data, name)
    duplicate_summary[name] = {
        "duplicates": data.duplicated().sum(),
        "total_rows": len(data)
    }
    print()


def perform_statical_testing(feature: str, df: pd.DataFrame = df_train_combined,  target_feature: str = Config.target_feature) -> None:
    """
    Perform statistical tests (normality and Kruskal-Wallis) 
    to evaluate whether there are significant differences 
    in the distribution of a numerical feature across categories 
    of the target variable.

    Args:
        feature (str): Name of the numerical feature to be tested.
        df (pd.DataFrame): Dataset containing both numerical and target columns.
        target_feature (str): Name of the target categorical feature.

    Returns:
        None: Prints or displays statistical test results.
    """
    # Perform normality test (e.g., Shapiro-Wilk or Dâ€™Agostino test) for feature distribution
    non_normal_detected = check_normality_with_plots(df=df, feature=feature, target_feature=target_feature)
    total_categories = df[target_feature].nunique()
    if total_categories > 2:
        if non_normal_detected == True:
            perform_kruskal_test(df=df, categorical_feature=target_feature,
                                numeric_feature=feature)
        else:
            anove_use, is_homogeneous_variances = check_homogeneity_of_variance(df=df, feature=feature,
                                                                                target_feature=target_feature)
            if anove_use and is_homogeneous_variances:
                perform_anova_with_tukey(df=df, numeric_feature=feature,
                                        categorical_feature=target_feature)
            elif anove_use and is_homogeneous_variances == False:
                perform_welch_anova(df=df, numeric_feature=feature, categorical_feature=target_feature)
            else:
                perform_kruskal_test(df=df, categorical_feature=target_feature,
                        numeric_feature=feature)
    else:
        if non_normal_detected == True:
            cal_mannwhitneyu(dataframe=df, categorical_feature=target_feature, num_feature=feature)
        else:
            anove_use, is_homogeneous_variances = check_homogeneity_of_variance(df=df, feature=feature,
                                                                                target_feature=target_feature)
            if anove_use and is_homogeneous_variances:
                t_test_with_cohens_d(data=df, categorical_feature=target_feature, num_feature=feature, equal_var=True)
            elif anove_use and is_homogeneous_variances == False:
                t_test_with_cohens_d(data=df, categorical_feature=target_feature, num_feature=feature, equal_var=False)
            else:
                cal_mannwhitneyu(dataframe=df, categorical_feature=target_feature, num_feature=feature)

def plot_numerical_distribution(feature: str, df: pd.DataFrame = df_train_combined,
                                target_feature: str = Config.target_feature, order: list = None) -> None:
    """
    Perform statistical testing and visualize the distribution of a numerical feature 
    across different classes of the target variable using violin plots and summary statistics.

    The function executes:
      1. Statistical tests.
      2. Summary table with mean, median, std per category.
      3. Violin plot for visualizing feature distributions across classes.

    Args:
        feature (str): The name of the numerical feature to analyze.
        df (pd.DataFrame): Input dataframe containing numerical & target features.
        target_feature (str): Target variable name (categorical feature).
        order (list, optional): Custom ordering for category display in the plot.

    Returns:
        None: Displays statistical summaries and plots directly.
    """

    # Compute summary statistics for each Fertilizer category
    df_summary_feature = (
        df.groupby(by=target_feature, as_index=False)
        .agg(
            Count=(feature, "count"),
            Mean=(feature, "mean"),
            Median=(feature, "median"),
            Std=(feature, "std")
        )
        .sort_values(by="Mean", ascending=False).reset_index(drop=True)
    )

    # Compute global statistics for the entire feature
    summary_data = [
        ("Overall Mean", f"{df[feature].mean():.2f}"),
        ("Overall Median", f"{df[feature].median()}"),
        ("Overall Std", f"{df[feature].std():.2f}")
    ]

    # Display overall statistics in HTML format for better notebook visualization
    summary_html = "<ul>" + "".join([
        f"<li><b>{k}:</b> {v}</li>" for k, v in summary_data
    ]) + "</ul>"
    display(HTML(summary_html))

    # Display detailed summary per category as styled dataframe
    display(
        df_summary_feature.style.background_gradient(cmap=cm)
        .set_table_attributes('style="width:75%; margin:auto;"')
    )

    # Run statistical significance testing
    perform_statical_testing(feature=feature, target_feature=target_feature)

    # Visualize distribution via violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=target_feature, y=feature, data=df, hue=target_feature, order=order,
                   palette=color(n_colors=df[target_feature].nunique()))
    
    plt.title(f"Violin plot of {feature} distribution by {target_feature}", pad=15, weight="bold", fontsize=12)
    plt.xlabel(f"{Config.target_feature}", labelpad=10)
    plt.ylabel(feature, labelpad=10)
    plt.legend().remove()
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.show()

for feature in num_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {feature} by {Config.target_feature}</b></h2>"))
    plot_numerical_distribution(feature=feature, df = df_train_combined)


# defining function for plotting
def bivariate_percent_plot(cat, df, figsize=(15, 6), order = None, rot = 0):
    
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {cat} by {Config.target_feature}</b></h2>"))
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=figsize)
    # Plot 1
    # Calculate the total number of each "cat" by Exited
    grouped = df.groupby([cat, Config.target_feature]).size().unstack(fill_value=0)
    # Calculate the percentages
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    if order is not None:
        percentages = percentages.loc[order]
    
    # That method uses HUSL colors, so you need hue, saturation, and lightness. 
    # I used hsluv.org to select the colors of this chart.
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    # Draw stacked bar plot
    ax[0] = percentages.plot(kind="bar", stacked=True, cmap=cmap, ax = ax[0], use_index=True)
    for container in ax[0].containers:
        ax[0].bar_label(container, fmt='%1.2f%%', label_type="center", fontsize=10)

    ax[0].set_title(f"Percentage of {Config.target_feature} by {cat}", fontsize=12, weight="bold", pad=15)
    ax[0].set_xlabel(f"{cat}", fontsize=12)
    ax[0].set_ylabel(f"% {Config.target_feature} Rate", fontsize=12)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=rot)
    ax[0].legend_.remove()
    sns.despine(left=False, bottom=False, ax=ax[0])

    # Plot 2
    sns.countplot(data=df, hue = Config.target_feature, x = cat,
                palette=color(n_colors=2), ax=ax[1], order=percentages.index, hue_order = [0, 1])
    # Show value for each bar.
    for container in ax[1].containers:
        ax[1].bar_label(container, fmt='%d', label_type="edge", fontsize=10)

    ax[1].set_title(f"{Config.target_feature} by {cat}", fontsize=12, weight="bold", pad=15)
    ax[1].set_xlabel(f"{cat}", fontsize=12)
    ax[1].set_ylabel("Number of Customer", fontsize=12)
    ax[1].legend(title=f"{Config.target_feature}", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=rot)
    sns.despine(left=False, bottom=False, ax=ax[1])
    plt.tight_layout()
    plt.show()

    cal_ChiSquare(cat_feature=cat, target_feature=Config.target_feature, df=df, show_residuals=True)

for feature in cat_features:
    bivariate_percent_plot(cat=feature, df= df_train_combined)



df_customer_churnma = df_train_combined.copy()


def geography_distribution(df=df_train):
    import geopandas as gpd
    # 1. Calculate churn rate by Geography: mean() = % of customers who exited
    churn_rate_by_geo = df.groupby("Geography")["Exited"].mean()

    # 2. Load geojson file (country names must match the "admin" column)
    world = gpd.read_file("https://raw.githubusercontent.com/daominhthuan42/kaggle-portfolio/refs/heads/main/14-bank-churn-customer/custom.geo.json")

    # 3. Map churn rate to each country
    world["churn_rate"] = world["admin"].map(churn_rate_by_geo)

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw base map (entire Europe in grey)
    world.plot(ax=ax, color="#D3D3D3", edgecolor="black", linewidth=0.3)

    # Highlight countries that have churn data
    highlighted = world[world["churn_rate"].notnull()]
    highlighted.plot(ax=ax, column="churn_rate", cmap="Reds", edgecolor="black", linewidth=0.8, legend=True)

    # Annotate country names and churn rates
    for idx, row in highlighted.iterrows():
        point = row["geometry"].representative_point()
        plt.annotate(
            text=f"{row['admin']} ({row['churn_rate'] * 100:.2f}%)",
            xy=(point.x, point.y),
            ha="center",
            fontsize=10,
            color="black",
            weight="bold"
        )

    ax.set_xlim(-10, 30)
    ax.set_ylim(35, 65)
    ax.axis("off")
    ax.set_title("Churn Rate by Geography (%)", fontsize=12, pad=15, weight="bold")

    plt.tight_layout()
    plt.show()

geography_distribution()


fig, ax = plt.subplots(2, 2, figsize=(16, 12))

# A. Age Distribution by Geography & Churn Status
sns.boxplot(x="Geography", y="Age", data=df_customer_churnma, ax=ax[0, 0], hue=Config.target_feature, palette=color(n_colors=2),
            order=["France", "Germany", "Spain"],)
ax[0, 0].set_title("Age Distribution by Geography & Churn Status", weight="bold", pad=15, fontsize=12)
ax[0, 0].set_xlabel("Geography")
ax[0, 0].set_ylabel("Age")
sns.despine(left=False, bottom=False, ax=ax[0, 0])

# B. Balance Distribution by Geography & Churn Status
sns.boxplot(x="Geography", y="Balance", data=df_customer_churnma, ax=ax[0, 1], hue=Config.target_feature, palette=color(n_colors=2))
ax[0, 1].set_title("Balance Distribution by Geography & Churn Status",weight="bold", pad=15, fontsize=12)
ax[0, 1].set_xlabel("Geography")
ax[0, 1].set_ylabel("Balance")
sns.despine(left=False, bottom=False, ax=ax[0, 1])

# C. Product Usage by Geography
sns.countplot(x="Geography", hue="NumOfProducts", data=df_customer_churnma, ax=ax[1, 0], palette=color(n_colors=4))
ax[1, 0].set_title("Product Usage by Geography", weight="bold", pad=15, fontsize=12)
ax[1, 0].set_xlabel("Geography")
ax[1, 0].set_ylabel("")
sns.despine(left=False, bottom=False, ax=ax[1, 0])

# D. Activity Status by Geography
sns.countplot(x="Geography", hue="IsActiveMember", data=df_customer_churnma, ax=ax[1, 1], palette=color(n_colors=2))
ax[1, 1].set_title("Activity Status by Geography", weight="bold", pad=15, fontsize=12)
ax[1, 1].set_xlabel("Geography")
ax[1, 1].set_ylabel("")
sns.despine(left=False, bottom=False, ax=ax[1, 1])

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(10, 5))

# A. Age Distribution by Geography & Churn Status
sns.boxplot(x="Geography", y="Balance", data=df_customer_churnma, ax=ax, hue=Config.target_feature, palette=color(n_colors=2))
ax.set_title("Balance Distribution by Geography & Churn Status", weight="bold", pad=15, fontsize=12)
ax.set_xlabel("Geography")
ax.set_ylabel("Balance")
sns.despine(left=False, bottom=False, ax=ax)

plt.tight_layout()
plt.show()


print("[i] Fitting logistic regression model...")
model_two = smf.logit(
    formula=f"{Config.target_feature} ~ Geography",
    data=df_customer_churnma
).fit()

print(model_two.summary())

# Compute Odds Ratios (OR)
coef = model_two.params
OR = np.exp(coef)

OR_table = pd.concat([coef, OR], axis=1)
OR_table.columns = ["coef", "OR"]

print("\n[i] Odds Ratios:")
print(OR_table)


# Create summary table by gender
def gender_churn_summary(df):
    summary = df.groupby("Gender").agg(
        ChurnRate=("Exited", "mean"),
        Avg_Balance=("Balance", "mean"),
        Percent_Inactive=("IsActiveMember", lambda x: (x == 0).mean()),
        Avg_Age=("Age", "mean"),
        Avg_CreditScore = ("CreditScore", "mean"),
        Avg_EstimatedSalary = ("EstimatedSalary", "mean"),
        Count=("Exited", "count")
    ).reset_index()

    # Format percentages for readability
    summary["ChurnRate"] = (summary["ChurnRate"] * 100).round(2)
    summary["Percent_Inactive"] = (summary["Percent_Inactive"] * 100).round(2)
    summary["Avg_Balance"] = summary["Avg_Balance"].round(0)
    summary["Avg_CreditScore"] = summary["Avg_CreditScore"].round(0)
    summary["Avg_EstimatedSalary"] = summary["Avg_EstimatedSalary"].round(0)
    summary["Avg_Age"] = summary["Avg_Age"].round(1)

    return summary

# Apply to the entire dataset
summary_all = gender_churn_summary(df_customer_churnma)

from tabulate import tabulate
print(tabulate(summary_all, headers="keys", tablefmt="github", showindex=False))


# 1. Create Balance & Income segments (high/low based on median)
balance_median = df_customer_churnma["Balance"].median()
income_median = df_customer_churnma["EstimatedSalary"].median()

df_customer_churnma["BalanceSegment"] = df_customer_churnma["Balance"].apply(lambda x: "High Balance" if x >= balance_median else "Low Balance")
df_customer_churnma["IncomeSegment"] = df_customer_churnma["EstimatedSalary"].apply(lambda x: "High Income" if x >= income_median else "Low Income")

# 2. Churn rate by activity level & gender
churn_activity_gender = (
    df_customer_churnma.groupby(["IsActiveMember", "Gender"])["Exited"]
    .mean()
    .reset_index()
)

# 3. Churn rate by customer value (Balance) & gender
churn_balance_gender = (
    df_customer_churnma.groupby(["BalanceSegment", "Gender"])["Exited"]
    .mean()
    .reset_index()
)

# 4. Churn rate by customer value (Income) & gender
churn_income_gender = (
    df_customer_churnma.groupby(["IncomeSegment", "Gender"])["Exited"]
    .mean()
    .reset_index()
)

# 5. Plot the charts
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# A. Activity vs Gender
sns.barplot(data=churn_activity_gender, x="IsActiveMember", y="Exited", hue="Gender", ax=axes[0], palette=color(n_colors=2))
axes[0].set_title("Churn Rate by Activity Level & Gender", weight="bold", pad=15, fontsize=12)
axes[0].set_ylabel("Churn Rate%")
sns.despine(ax=axes[0])

# B. Balance vs Gender
sns.barplot(data=churn_balance_gender, x="BalanceSegment", y="Exited", hue="Gender", ax=axes[1], palette=color(n_colors=2))
axes[1].set_title("Churn Rate by Balance Segment & Gender", weight="bold", pad=15, fontsize=12)
axes[1].set_ylabel("Churn Rate%")
sns.despine(ax=axes[1])

# C. Income vs Gender
sns.barplot(data=churn_income_gender, x="IncomeSegment", y="Exited", hue="Gender", ax=axes[2], palette=color(n_colors=2))
axes[2].set_title("Churn Rate by Income Segment & Gender", weight="bold", pad=15, fontsize=12)
axes[2].set_ylabel("Churn Rate%")
sns.despine(ax=axes[2])

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# 1. Balance by HasCrCard
sns.boxplot(
    data=df_customer_churnma,
    x="HasCrCard",
    y="Balance",
    palette=color(n_colors=2),
    hue="Exited",
    ax=ax[0, 0]
)
sns.despine(ax=ax[0, 0])
ax[0, 0].set_title("Balance by HasCrCard", weight="bold", pad=15)
ax[0, 0].set_xlabel("HasCrCard")
ax[0, 0].set_ylabel("Balance")

# 2. Age by HasCrCard
sns.boxplot(
    data=df_customer_churnma,
    x="HasCrCard",
    y="Age",
    palette=color(n_colors=2),
    hue="Exited",
    ax=ax[0, 1]
)
sns.despine(ax=ax[0, 1])
ax[0, 1].set_title("Age by HasCrCard", weight="bold", pad=15)
ax[0, 1].set_xlabel("HasCrCard")
ax[0, 1].set_ylabel("Age")

# 3. NumOfProducts distribution by HasCrCard
num_prod_dist = df_customer_churnma.groupby(
    ["HasCrCard", "NumOfProducts"]
).size().reset_index(name="Count")

sns.barplot(
    data=num_prod_dist,
    x="NumOfProducts",
    y="Count",
    hue="HasCrCard",
    palette=color(n_colors=2),
    ax=ax[1, 0]
)
sns.despine(ax=ax[1, 0])
ax[1, 0].set_title("NumOfProducts Distribution by HasCrCard", weight="bold", pad=15)
ax[1, 0].set_xlabel("NumOfProducts")
ax[1, 0].set_ylabel("Count")


# 4. IsActiveMember distribution by HasCrCard
active_dist = df_customer_churnma.groupby(
    ["HasCrCard", "IsActiveMember"]
).size().reset_index(name="Count")

sns.barplot(
    data=active_dist,
    x="IsActiveMember",
    y="Count",
    hue="HasCrCard",
    palette=color(n_colors=2),
    ax=ax[1, 1]
)
sns.despine(ax=ax[1, 1])
ax[1, 1].set_title("IsActiveMember Distribution by HasCrCard", weight="bold", pad=15)
ax[1, 1].set_xlabel("IsActiveMember")
ax[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# 1. CreditScore by IsActiveMember
sns.boxplot(data=df_customer_churnma, x="IsActiveMember", y="CreditScore", palette=color(n_colors=2), ax=ax[0,0])
sns.despine(ax=ax[0, 0])
ax[0,0].set_title("CreditScore by IsActiveMember", weight="bold", pad=15)
ax[0,0].set_xlabel("IsActiveMember")
ax[0,0].set_ylabel("CreditScore")

# 2. Balance by IsActiveMember
sns.boxplot(data=df_customer_churnma, x="IsActiveMember", y="Balance", palette=color(n_colors=2), ax=ax[0,1])
sns.despine(ax=ax[0, 1])
ax[0,1].set_title("Balance by IsActiveMember", weight="bold", pad=15)
ax[0,1].set_xlabel("IsActiveMember")
ax[0,1].set_ylabel("Balance")

# 3. Age by IsActiveMember
sns.boxplot(data=df_customer_churnma, x="IsActiveMember", y="Age", palette=color(n_colors=2), ax=ax[1,0])
sns.despine(ax=ax[1, 0])
ax[1,0].set_title("Age by IsActiveMember", weight="bold", pad=15)
ax[1,0].set_xlabel("IsActiveMember")
ax[1,0].set_ylabel("Age")

# 4. NumOfProducts distribution by IsActiveMember
num_prod_dist = df_customer_churnma.groupby(["IsActiveMember", "NumOfProducts"]).size().reset_index(name="Count")
sns.barplot(data=num_prod_dist, x="NumOfProducts", y="Count", hue="IsActiveMember", palette=color(n_colors=2), ax=ax[1,1])
sns.despine(ax=ax[1, 1])
ax[1,1].set_title("NumOfProducts Distribution by IsActiveMember", weight="bold", pad=15)
ax[1,1].set_xlabel("NumOfProducts")
ax[1,1].set_ylabel("Count")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 1. CreditScore by NumOfProducts
sns.boxplot(
    data=df_customer_churnma,
    x="NumOfProducts",
    y="CreditScore",
    hue="Exited",
    palette=color(n_colors=2),
    ax=ax[0]
)
ax[0].set_title("CreditScore by NumOfProducts", weight="bold", pad=15)
ax[0].set_xlabel("NumOfProducts")
ax[0].set_ylabel("CreditScore")
sns.despine(ax=ax[0])

# 2. Balance by NumOfProducts
sns.boxplot(
    data=df_customer_churnma,
    x="NumOfProducts",
    y="Balance",
    hue="Exited",
    palette=color(n_colors=2),
    ax=ax[1]
)
ax[1].set_title("Balance by NumOfProducts", weight="bold", pad=15)
ax[1].set_xlabel("NumOfProducts")
ax[1].set_ylabel("Balance")
sns.despine(ax=ax[1])

plt.tight_layout()
plt.show()


# Filter only the 2-product group and the 3â€“4 product group
df_compare = df_customer_churnma[df_customer_churnma["NumOfProducts"].isin([2, 3, 4])].copy()
df_compare["ProductGroup"] = df_compare["NumOfProducts"].apply(lambda x: "3-4 Products" if x >= 3 else "2 Products")

fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# 1. CreditScore
sns.boxplot(data=df_compare, x="ProductGroup", y="CreditScore", palette=color(n_colors=2), hue="Exited", ax=ax[0,0])
sns.despine(ax=ax[0, 0])
ax[0,0].set_title("CreditScore by Product Group", weight="bold", pad=15)

# 2. Balance
sns.boxplot(data=df_compare, x="ProductGroup", y="Balance", palette=color(n_colors=2), hue="Exited", ax=ax[0,1])
sns.despine(ax=ax[0, 1])
ax[0,1].set_title("Balance by Product Group", weight="bold", pad=15)

# 3. Age
sns.boxplot(data=df_compare, x="ProductGroup", y="Age", palette=color(n_colors=2), hue="Exited", ax=ax[1,0])
sns.despine(ax=ax[1, 0])
ax[1,0].set_title("Age by Product Group", weight="bold", pad=15)

# 4. IsActiveMember distribution
active_dist = df_compare.groupby(["ProductGroup", "IsActiveMember"]).size().reset_index(name="Count")
sns.barplot(data=active_dist, x="IsActiveMember", y="Count", hue="ProductGroup", palette=color(n_colors=2), ax=ax[1,1])
sns.despine(ax=ax[1, 1])
ax[1,1].set_title("IsActiveMember Distribution by Product Group", weight="bold", pad=15)

plt.tight_layout()
plt.show()


# Divide CreditScore into ranges (bins)
bins = [300, 500, 600, 700, 800, 900]  # adjust as needed
labels = ["300-499", "500-599", "600-699", "700-799", "800-899"]
df_customer_churnma["CS_group"] = pd.cut(df_customer_churnma["CreditScore"], bins=bins, labels=labels, right=False)

# Calculate churn rate for each group
churn_by_group = df_customer_churnma.groupby("CS_group")["Exited"].mean().reset_index()
churn_by_group["Exited"] *= 100  # convert to percentage

# Plot barplot
plt.figure(figsize=(10,5))
sns.barplot(data=churn_by_group, x="CS_group", y="Exited",palette=color(n_colors=5, tone="RdYlGn"))
sns.despine()
plt.title("Churn Rate by CreditScore Range", weight="bold", pad=15)
plt.ylabel("Churn Rate (%)")
plt.xlabel("CreditScore Range")

# Display percentage values
for i, v in enumerate(churn_by_group["Exited"]):
    plt.text(i, v, f"{v:.1f}%", ha="center", va="bottom")

plt.tight_layout()
plt.show()


def credit_score_group(x):
    if 350 <= x < 500:
        return "very_poor"
    elif 500 <= x < 600:
        return "poor"
    elif 600 <= x < 700:
        return "fair"
    elif 700 <= x < 800:
        return "good"
    elif 800 <= x <= 850:
        return "excellent"
    else:
        return "unknown"
df_customer_churnma["credit_score_group"] = df_customer_churnma["CreditScore"].apply(credit_score_group)

def age_group(x):
    if 18 <= x <= 25:
        return "young_adult"
    elif 26 <= x <= 35:
        return "adult"
    elif 36 <= x <= 45:
        return "mid_age"
    elif 46 <= x <= 55:
        return "mature"
    elif 56 <= x <= 65:
        return "senior"
    elif 66 <= x <= 92:
        return "elderly"
    else:
        return "unknown"
df_customer_churnma["age_group"] = df_customer_churnma["Age"].apply(age_group)

print("[i] Computing percentage table by Credit Score Group Ã— Age Group Ã— Geography...")
df_group = pd.crosstab(
    [df_customer_churnma["Geography"], df_customer_churnma["credit_score_group"], df_customer_churnma["age_group"]],
    df_customer_churnma["Exited"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Credit Score Group Ã— Age Group Ã— Geography) vs Exited...")
contingency = pd.crosstab(
    [df_customer_churnma["Geography"], df_customer_churnma["credit_score_group"], df_customer_churnma["age_group"]],
    df_customer_churnma["Exited"]
)

# Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
print(f"[i] Chi2: {chi2:.4f}")
print(f"[i] dof: {dof}")
print(f"[i] p-value: {p:.6f}")

# Standardized residuals (adjusted)
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom
std_resid_df = pd.DataFrame(std_resid, index=contingency.index,
                            columns=contingency.columns)

# Heatmap â€” display nicely the multiindex
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
plt.figure(figsize=(15,20))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Credit Score Group or Age Group or Geography vs Exited", 
          weight="bold", fontsize=12, pad=15)
plt.ylabel("Credit Score Group - Age Group - Geography | Exited")
plt.xlabel("")
plt.text(2.2,93, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


# Calculate churn rate for each group
churn_by_group = df_customer_churnma.groupby("age_group")["Exited"].mean().reset_index()
churn_by_group["Exited"] *= 100  # convert to percentage

# Plot barplot
plt.figure(figsize=(10,5))
sns.barplot(data=churn_by_group, x="age_group", y="Exited",palette=color(n_colors=4))
sns.despine()
plt.title("Churn Rate by Age Group", weight="bold", pad=15)
plt.ylabel("Churn Rate (%)")
plt.xlabel("Age Group")

# Display percentage values
for i, v in enumerate(churn_by_group["Exited"]):
    plt.text(i, v, f"{v:.1f}%", ha="center", va="bottom")

plt.tight_layout()
plt.show()


# Split Balance into High/Low based on the overall median
def balance_group(x):
    if x == 0:
        return "zero_balance"
    elif 0 < x <= 50000:
        return "low_balance"
    elif 50000 < x <= 120000:
        return "mid_balance"
    elif 120000 < x <= 200000:
        return "high_balance"
    elif 200000 < x <= 250898.1:
        return "very_high_balance"
    else:
        return "unknown"
df_customer_churnma["balance_group"] = df_customer_churnma["Balance"].apply(balance_group)

# Calculate churn rate
churn_rate = (
    df_customer_churnma.groupby(["Geography", "balance_group"])["Exited"]
    .mean()
    .reset_index()
)
churn_rate["Exited"] *= 100  # convert to percentage

# Plot bar chart
plt.figure(figsize=(13,6))
sns.barplot(data=churn_rate, x="Geography", y="Exited", hue="balance_group", 
            palette=color(n_colors=df_customer_churnma["balance_group"].nunique(), tone="RdYlGn"))
sns.despine()
plt.title("Churn Rate by Geography and Balance Segment", weight="bold", pad=15)
plt.ylabel("Churn Rate (%)")
plt.xlabel("")

# Display percentage values
for p in plt.gca().patches:
    if p.get_height() > 0:
        plt.text(p.get_x() + p.get_width()/2, p.get_height(),
                 f'{p.get_height():.1f}%',
                 ha='center', va='bottom')

plt.legend(title="Balance Group", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()



print("[i] Computing percentage table by Balance Group Ã— Active Member...")
df_group = pd.crosstab(
    [df_customer_churnma["balance_group"], df_customer_churnma["IsActiveMember"]],
    df_customer_churnma["Exited"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Balance Group Ã— Active Member) vs Exited...")
contingency = pd.crosstab(
    [df_customer_churnma["balance_group"], df_customer_churnma["IsActiveMember"]],
    df_customer_churnma["Exited"]
)

# Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
print(f"[i] Chi2: {chi2:.4f}")
print(f"[i] dof: {dof}")
print(f"[i] p-value: {p:.6f}")

# Standardized residuals (adjusted)
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom
std_resid_df = pd.DataFrame(std_resid, index=contingency.index,
                            columns=contingency.columns)

# Heatmap â€” display nicely the multiindex
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
plt.figure(figsize=(15,15))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Balance Group or Active Member vs Exited", 
          weight="bold", fontsize=12, pad=15)
plt.ylabel("Balance Group - Active Member | Exited")
plt.xlabel("")
plt.text(2.2,10.5, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


def tenure_group(x):
    if 0 <= x <= 2:
        return "Very New"
    elif 3 <= x <= 4:
        return "Early"
    elif 5 <= x <= 6:
        return "Mid Term"
    elif 7 <= x <= 10:
        return "Long Term"
    else:
        return "Unknown"

df_customer_churnma["tenure_group"] = df_customer_churnma["Tenure"].apply(tenure_group)

# Churn rate by TenureGroup Ã— Geography
churn_geo = (
    df_customer_churnma.groupby(["tenure_group", "Geography"])["Exited"]
    .mean()
    .reset_index()
)
churn_geo["Exited"] *= 100

# Churn rate by TenureGroup Ã— AgeGroup
churn_age = (
    df_customer_churnma.groupby(["tenure_group", "age_group"])["Exited"]
    .mean()
    .reset_index()
)
churn_age["Exited"] *= 100

fig, ax = plt.subplots(1, 2, figsize=(20, 8))
# Barplot: Tenure Group Ã— Geography
sns.barplot(
    data=churn_geo,
    x="tenure_group",
    y="Exited",
    hue="Geography",
    palette=color(n_colors=4, tone="RdYlGn"),
    ax=ax[0], order=["Very New", "Early", "Mid Term", "Long Term"]
)
sns.despine(ax=ax[0])
ax[0].set_title("Churn Rate by Tenure Group and Geography", weight="bold", pad=15)
ax[0].set_ylabel("Churn Rate (%)")
ax[0].set_xlabel("")

# Annotate
for p in ax[0].patches:
    height = p.get_height()
    if height > 0:
        ax[0].text(
            p.get_x() + p.get_width()/2,
            height,
            f"{height:.1f}%",
            ha="center", va="bottom"
        )

# Barplot: Tenure Group Ã— Age Group
sns.barplot(
    data=churn_age,
    x="tenure_group",
    y="Exited",
    hue="age_group",
    palette=color(n_colors=6, tone="RdYlGn"),
    ax=ax[1], order=["Very New", "Early", "Mid Term", "Long Term"]
)
sns.despine(ax=ax[1])
ax[1].set_title("Churn Rate by Tenure Group and Age Group", weight="bold", pad=15)
ax[1].set_ylabel("Churn Rate (%)")
ax[1].set_xlabel("")
ax[1].legend(title="Age Group", bbox_to_anchor=(0.9, 0.75), loc="upper left")

# Annotate
for p in ax[1].patches:
    height = p.get_height()
    if height > 0:
        ax[1].text(
            p.get_x() + p.get_width()/2,
            height,
            f"{height:.1f}%",
            ha="center", va="bottom"
        )

plt.tight_layout()
plt.show()


def salary_group(x):
    if 11.58 <= x <= 50000:
        return "low_income"
    elif 50000 < x <= 100000:
        return "mid_income"
    elif 100000 < x <= 150000:
        return "high_income"
    elif 150000 < x <= 199992.48:
        return "very_high_income"
    else:
        return "unknown"
df_customer_churnma["salary_group"] = df_customer_churnma["EstimatedSalary"].apply(salary_group)

print("[i] Computing percentage table by Credit Score Group Ã— Balance Group Ã— Salary Group...")
df_group = pd.crosstab(
    [df_customer_churnma["salary_group"], df_customer_churnma["credit_score_group"], df_customer_churnma["balance_group"]],
    df_customer_churnma["Exited"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Credit Score Group Ã— Balance Group Ã— Salary Group) vs Exited...")
contingency = pd.crosstab(
    [df_customer_churnma["salary_group"], df_customer_churnma["credit_score_group"], df_customer_churnma["balance_group"]],
    df_customer_churnma["Exited"]
)

# Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
print(f"[i] Chi2: {chi2:.4f}")
print(f"[i] dof: {dof}")
print(f"[i] p-value: {p:.6f}")

# Standardized residuals (adjusted)
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom
std_resid_df = pd.DataFrame(std_resid, index=contingency.index,
                            columns=contingency.columns)

# Heatmap â€” display nicely the multiindex
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
plt.figure(figsize=(15,23))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Credit Score Group or Balance Group or Salary Group vs Exited", 
          weight="bold", fontsize=12, pad=15)
plt.ylabel("Credit Score Group - Balance Group - Salary Group | Exited")
plt.xlabel("")
plt.text(2.2,110, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


# Credit Utilization
df_train_combined["CreditUtilization"] = df_train_combined["Balance"] / df_train_combined["EstimatedSalary"]
df_test["CreditUtilization"] = df_test["Balance"] / df_test["EstimatedSalary"]

# CreditScore per Age
df_train_combined["CreditScorePerAge"] = df_train_combined["CreditScore"] / df_train_combined["Age"]
df_test["CreditScorePerAge"] = df_test["CreditScore"] / df_test["Age"]

# HighBalance_Inactive
bal_median_train = df_train_combined["Balance"].median()
bal_median_test = df_test["Balance"].median()
df_train_combined["HighBalance_Inactive"] = ((df_train_combined["Balance"] >= bal_median_train) & (df_train_combined["IsActiveMember"] == 0)).astype(int)
df_test["HighBalance_Inactive"] = ((df_test["Balance"] >= bal_median_test) & (df_test["IsActiveMember"] == 0)).astype(int)


display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of CreditUtilization by Exited</b></h2>"))
plot_numerical_distribution(feature="CreditUtilization", df=df_train_combined)


display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of CreditScorePerAge by Exited</b></h2>"))
plot_numerical_distribution(feature="CreditScorePerAge", df=df_train_combined)


bivariate_percent_plot(cat="HighBalance_Inactive", df=df_train_combined)


num_features = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary", "CreditUtilization", "CreditScorePerAge"]
skew_feature_train, skew_train_df = check_skewness(data=df_train_combined, dataset_name="Train Data",
                                                   numerical_features=num_features)


skew_feature_test, skew_test_df = check_skewness(data=df_test, dataset_name="Test Data",
                                                 numerical_features=num_features)


from sklearn.preprocessing import PowerTransformer

def handle_skewed_features(
    df,
    zero_threshold=0.9,
    skew_threshold=0.5,
    num_features=None,
    exclude_cols=None
):
    """
    Handle skewed numerical features by applying appropriate transformations.

    Parameters:
    - df: pandas.DataFrame
    - zero_threshold: float (default=0.9)
    - skew_threshold: float (default=0.5)
    - num_features: list of numerical columns to consider
    - exclude_cols: list of columns to skip entirely

    Returns:
    - df: transformed DataFrame
    - transformed_cols: list of new feature names
    - high_zero_cols: list of sparse features (> zero_threshold)
    - skewed_cols: list of autoâ€‘detected skewed features
    """
    df = df.copy()
    if num_features is None:
        raise ValueError("`num_features` must be provided")
    if exclude_cols is None:
        exclude_cols = []

    # 1) pick the numeric cols to scan
    numerical_cols = [c for c in num_features if c not in exclude_cols]

    # 2) detect ultraâ€‘sparse
    zero_ratios = (df[numerical_cols] == 0).sum() / len(df)
    high_zero_cols = zero_ratios[zero_ratios > zero_threshold].index.tolist()

    # 3) compute skew
    skew_vals = df[numerical_cols].apply(lambda s: skew(s.dropna()))
    auto_skewed = skew_vals[abs(skew_vals) > skew_threshold].index.tolist()

    # 4) union these with your forced list
    to_transform = list(set(auto_skewed))

    transformed_cols = []
    dropped_cols     = []

    for col in to_transform:
        # if it's sparse â†’ binary+log
        if col in high_zero_cols:
            df[f"Has_{col}"] = (df[col] > 0).astype(int)
            df[f"Log_{col}"] = df[col].map(lambda x: np.log1p(x) if x > 0 else 0)
            transformed_cols += [f"Has_{col}", f"Log_{col}"]
            dropped_cols.append(col)
        # if it's discrete smallâ€‘cardinality, skip transform but keep
        elif df[col].nunique() <= 5:
            # do nothing (we still keep raw col in df)
            continue
        # otherwise apply Yeoâ€‘Johnson
        else:
            pt = PowerTransformer(method="yeo-johnson")
            arr = df[[col]].values  # shape (n,1)
            df[f"PT_{col}"] = pt.fit_transform(arr)
            transformed_cols.append(f"PT_{col}")
            dropped_cols.append(col)

    # drop originals for any column we did transform
    df.drop(columns=dropped_cols, inplace=True)

    return df, transformed_cols, high_zero_cols, auto_skewed


processed_train_df, transformed_columns, sparse_columns, skewed_columns = handle_skewed_features(df=df_train_combined, num_features=skew_feature_train)
num_features = ["CreditScore", "PT_Age", "Tenure", "Balance", "EstimatedSalary", "PT_CreditUtilization", "PT_CreditScorePerAge"]
skew_feature_train, skew_train_df = check_skewness(data=processed_train_df, dataset_name="Train Data", numerical_features=num_features)


processed_test_df, transformed_columns_test, sparse_columns_test, skewed_columns_test = handle_skewed_features(df=df_test, num_features=skew_feature_test)
skew_feature_test, skew_test_df = check_skewness(data=processed_test_df, numerical_features=num_features, dataset_name= "Test data")


checking_outlier(list_feature=num_features, df=processed_train_df, dataset_name="Training data")


checking_outlier(list_feature=num_features, df=processed_test_df, dataset_name="Test data")


processed_train_df["EstimatedSalary_Cat"] = pd.qcut(processed_train_df["EstimatedSalary"],
                                              q=5,
                                              labels=[1, 2, 3, 4, 5])

plt.figure(figsize=(8, 5))
sns.histplot(data=processed_train_df, x="EstimatedSalary_Cat", color="lightblue", edgecolor="black")

plt.title("Distribution of EstimatedSalary_Cat", fontsize=12, pad=15, weight="bold")
plt.xlabel("EstimatedSalary_Cat")
plt.ylabel("")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(processed_train_df, processed_train_df["EstimatedSalary_Cat"]):
    start_train_set = processed_train_df.iloc[train_index]
    start_test_set = processed_train_df.iloc[test_index]\

# Now we should remove the EstimatedSalary_Cat attribute so the data is back to its original state:
for set_ in (start_train_set, start_test_set): 
    set_.drop("EstimatedSalary_Cat", axis=1, inplace=True)

df_train_new = start_train_set.drop("Exited", axis=1)
df_train_label = start_train_set["Exited"].copy()


# Define feature groups
# Numerical features with outliers â†’ use RobustScaler
list_feature_num_robust = ["CreditScore", "PT_Age", "PT_CreditScorePerAge"]
# Numerical features with standard distribution â†’ use StandardScaler
list_feature_num_stand = ["Tenure", "Balance", "EstimatedSalary", "PT_CreditUtilization"]
# Categorical features that need One-Hot Encoding
list_feature_cat_onehot = ["Geography", "Gender", "NumOfProducts"]
# Binary/low-cardinality categorical features â†’ keep as-is (no one-hot)
list_feature_cat_keep = ["HasCrCard", "IsActiveMember", "HighBalance_Inactive"]

# Define transformers
# Robust scaling for outlier-prone numeric features
num_robust_transformer = Pipeline(steps=[
    ("scaler", RobustScaler()),
    ("imputer", SimpleImputer(strategy="median"))
])
# Standard scaling for normally distributed numeric features
num_stand_transformer = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("imputer", SimpleImputer(strategy="median"))
])
# One-hot encoding for multi-category categorical features
cat_onehot_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ("imputer", SimpleImputer(strategy="most_frequent"))
])
# Keep binary/boolean categorical features (1/0) without encoding
cat_keep_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Build ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num_robust", num_robust_transformer, list_feature_num_robust),
    ("num_standard", num_stand_transformer, list_feature_num_stand),
    ("cat_onehot", cat_onehot_transformer, list_feature_cat_onehot),
    ("cat_keep", cat_keep_transformer, list_feature_cat_keep),
])

# Fit the preprocessing pipeline
preprocessor.fit(df_train_new)
# Transform training data
df_train_new_prepared = preprocessor.transform(df_train_new)
# Extract clean feature names
list_feature_prepared = preprocessor.get_feature_names_out().tolist()
# Clean prefix names for readability
clean_features = [
    col.replace("num_standard__", "")
       .replace("num_robust__", "")
       .replace("cat_onehot__", "")
       .replace("cat_keep__", "")
       .replace("PT_", "") 
    for col in list_feature_prepared
]

clean_features


class_1 = df_train_label.sum()
class_0 = len(df_train_label) - class_1
scale_pos_weight = class_0 / class_1


def shap_plot(model, X_test, list_feature, type = None):
     # https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137/
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    X_test_sample = pd.DataFrame(X_test, columns=list_feature)
    explainer = shap.Explainer(model.predict, X_test_sample)
    shap_values = explainer(X_test_sample)
    if type =="bar":
        shap_importance = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": X_test_sample.columns, "importance": shap_importance})
        shap_df = shap_df.sort_values("importance", ascending=False).head(20)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=shap_df["importance"], y=shap_df["feature"], palette="viridis", order=shap_df["feature"])
        plt.xlabel("mean(|SHAP value|)")
        plt.title("SHAP Feature Importance", fontsize=14, weight="bold", pad=20)
        plt.tight_layout()
        plt.show()
    else:
        shap.summary_plot(shap_values, X_test_sample)


def run_oof(model, X, y, X_test, n_splits=Config.n_split_kfold):
    print(f"[i] Starting OOF training with {model.__class__.__name__}...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_models = []
    fold_auc = []

    # For plotting
    roc_curves = []
    pr_curves = []
    auc_values = []
    ap_values = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"\n[i] Fold {fold+1}/{n_splits}...")

        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if "LGBMClassifier" == model.__class__.__name__:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)]
                # verbose=False,
                # early_stopping_rounds=300
            )
        elif "HistGradientBoostingClassifier" == model.__class__.__name__:
            model.fit(X_train, y_train)
        else:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
                early_stopping_rounds=300
            )        

        fold_models.append(model)

        # Predict fold
        if "CatBoostClassifier" == model.__class__.__name__:
            fold_pred = model.predict_proba(
                X_valid,
                ntree_end=model.best_iteration_
            )[:, 1]
        elif "LGBMClassifier" == model.__class__.__name__:
            fold_pred = model.predict_proba(
                X_valid,
                num_iteration=model.best_iteration_
            )[:, 1]
        elif "XGBClassifier" == model.__class__.__name__:
            fold_pred = model.predict_proba(
                X_valid,
                iteration_range=(0, model.best_iteration + 1)
            )[:, 1]
        else:
            # HistGradientBoostingClassifier & all sklearn models
            fold_pred = model.predict_proba(X_valid)[:, 1]

        oof_pred[valid_idx] = fold_pred

        # Test prediction
        if "CatBoostClassifier" == model.__class__.__name__:
            test_pred += model.predict_proba(
                X_test,
                ntree_end=model.best_iteration_
            )[:, 1] / n_splits
        elif "LGBMClassifier" == model.__class__.__name__:
            test_pred += model.predict_proba(
                X_test,
                num_iteration=model.best_iteration_
            )[:, 1] / n_splits
        elif "XGBClassifier" == model.__class__.__name__:
            test_pred += model.predict_proba(
                X_test,
                iteration_range=(0, model.best_iteration + 1)
            )[:, 1] / n_splits
        else:
            test_pred += model.predict_proba(X_test)[:, 1] / n_splits

        # === Compute AUC for the fold ===
        auc_fold = roc_auc_score(y_valid, fold_pred)
        fold_auc.append(auc_fold)
        print(f"[âœ“] Fold {fold+1} - AUC: {auc_fold:.5f}")

        # === ROC curve by fold ===
        fpr, tpr, _ = roc_curve(y_valid, fold_pred)
        roc_curves.append((fpr, tpr))

        # === Precisionâ€“Recall curve by fold ===
        precision, recall, _ = precision_recall_curve(y_valid, fold_pred)
        pr_curves.append((recall, precision))

        # Average Precision for fold
        ap = average_precision_score(y_valid, fold_pred)
        ap_values.append(ap)

    # Final OOF AUC
    oof_auc = roc_auc_score(y, oof_pred)
    print("\n==========================")
    print(f"[i] OOF AUC: {oof_auc:.5f}")
    print(f"[âœ“] Fold AUCs: {fold_auc}")
    print("==========================\n")

    # 3 SUBPLOTS: ROC, PR, CONF MATRIX

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ------------------------ ROC Curve ------------------------
    ax = axes[0, 0]
    for i, (fpr, tpr) in enumerate(roc_curves):
        ax.plot(fpr, tpr, label=f"Fold {i+1} AUC = {fold_auc[i]:.4f}")

    ax.plot([0,1], [0,1], "--", color="gray")
    ax.set_title("ROC Curve (Each Fold)", pad=15, weight="bold", fontsize=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True)

    # --------------------- PR Curve ---------------------------
    ax = axes[0, 1]
    for i, (recall, precision) in enumerate(pr_curves):
        ax.plot(recall, precision, label=f"Fold {i+1} AP = {ap_values[i]:.4f}")

    ax.set_title("Precisionâ€“Recall Curve (Each Fold)", pad=15, weight="bold", fontsize=12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True)

    # --------------------- Confusion Matrix --------------------
    ax = axes[1, 0]
    threshold = 0.5
    y_pred_label = (oof_pred >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred_label)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix (Threshold = 0.5)", pad=15, weight="bold", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax = axes[1, 1]
    ax.axis("off")

    plt.tight_layout()
    plt.show()

    return oof_pred, test_pred, fold_models


X_val = start_test_set.drop("Exited", axis=1)
y_val = start_test_set["Exited"].copy()
X_val_prepared = preprocessor.transform(X_val)


from catboost import CatBoostClassifier
param_cb = {
"bootstrap_type": "Bernoulli",
"colsample_bylevel": 0.7658926065030167,
"iterations": 1984,
"depth": 5,
"learning_rate": 0.015171349611571652,
"l2_leaf_reg": 1.337578073889382,
"min_data_in_leaf": 76,
"subsample": 0.7280698256173038,
"eval_metric": "AUC",
"class_weights": [1, scale_pos_weight],
"verbose": False,
"random_seed": Config.seed,
"loss_function": "Logloss",
"allow_writing_files": False,
"thread_count": -1,
"grow_policy": "Lossguide",
"task_type": "CPU"
}

model_cb = CatBoostClassifier(**param_cb)
df_test_prepared = preprocessor.transform(processed_test_df)
oof_pred_cb, test_pred_cb, fold_models_cb = run_oof(model = model_cb, X = df_train_new_prepared, y = df_train_label.values, 
                                                    X_test = df_test_prepared)


from xgboost import XGBClassifier
param_xgb = {
"n_estimators": 2000,
"learning_rate": 0.013448370696239944,
"max_depth": 5,
"min_child_weight": 0.03159892617906292,
"subsample": 0.7185805327584004,
"colsample_bytree": 0.6611176553575923,
"gamma": 5.256647828620208,
"reg_alpha": 1.8881089265623683e-07,
"reg_lambda": 3.3879565314300123e-07,
"n_jobs": -1,
"random_state": Config.seed,
"scale_pos_weight": scale_pos_weight,
"objective": "binary:logistic",
"eval_metric": "auc",
"grow_policy": "lossguide",
"tree_method": "gpu_hist",
"predictor": "gpu_predictor",
"use_label_encoder": False,
"booster": "gbtree" 
}

model_xgb = XGBClassifier(**param_xgb)
oof_pred_xgb, test_pred_xgb, fold_models_xgb = run_oof(model = model_xgb, X = df_train_new_prepared, y = df_train_label.values, 
                                                       X_test = df_test_prepared)


from lightgbm import LGBMClassifier

param_lgbm = {
"n_estimators": 1530,
"learning_rate": 0.0030804807162494864,
"num_leaves": 345,
"max_depth": 5,
"min_child_samples": 62,
"min_split_gain": 0.6369464169580782,
"feature_fraction": 0.7846216887053435,
"bagging_fraction": 0.7647361784813704,
"reg_alpha": 0.3212093236745268,
"reg_lambda": 0.0006563052961663804,
"max_bin": 243,
"bagging_freq": 7,
"objective": "binary",
"metric": "AUC",
"random_state": Config.seed,
"n_jobs": -1,
"verbosity": -1,
"class_weight": {0: 1.0, 1: float(scale_pos_weight)},
"max_bin": 255
}

model_lgbm = LGBMClassifier(**param_lgbm)
oof_pred_lgbm, test_pred_lgbm, fold_models_lgbm = run_oof(model = model_lgbm, X = df_train_new_prepared, y = df_train_label.values, 
                                                          X_test = df_test_prepared)


from sklearn.ensemble import HistGradientBoostingClassifier

param_hgb = {
"learning_rate": 0.0323356329876002,
"max_iter": Config.max_iter,
"max_leaf_nodes": 15,
"min_samples_leaf": 254,
"l2_regularization": 0.00011009865934208523,
"max_bins": 117,
"early_stopping": True,
"validation_fraction": 0.1,
"random_state": Config.seed,
"verbose": 0,
"class_weight": "balanced",
"loss": "log_loss"
}

model_hgb = HistGradientBoostingClassifier(**param_hgb)
oof_pred_hgb, test_pred_hgb, fold_models_hgb = run_oof(model = model_hgb, X = df_train_new_prepared, y = df_train_label.values, 
                                                       X_test = df_test_prepared)


# Collect predictions (probabilities instead of labels) ---
ests = [("cb", model_cb), ("xgb", model_xgb), ("lgbm", model_lgbm), ("hgb", model_hgb)]
preds = {name: m.predict_proba(X_val_prepared)[:, 1] for name, m in ests}

auc_each = {name: roc_auc_score(y_val, preds[name]) for name,_ in ests}
display(auc_each)

A = np.column_stack([preds[name] for name,_ in ests])  # shape (n_val, n_models)
def obj_w(trial):
    w = np.array([trial.suggest_float(f"w_{i}", 0.0, 5.0) for i in range(A.shape[1])])
    if w.sum() == 0: 
        return 1e6
    y_hat = A.dot(w / w.sum())  # weighted average probs
    return roc_auc_score(y_val, y_hat)

study_w = optuna.create_study(direction="maximize")  
study_w.optimize(obj_w, n_trials=1000, show_progress_bar=True)

w = np.array([study_w.best_params[f"w_{i}"] for i in range(A.shape[1])])
weights = (w / w.sum()).tolist()
print("Best weights (normalized):", weights)
print("Best AUC:", study_w.best_value)


oof_pred = weights[0] * oof_pred_cb + weights[1] * oof_pred_xgb + weights[2] * oof_pred_lgbm + weights[3] * oof_pred_hgb
exited = weights[0] * test_pred_cb + weights[1] * test_pred_xgb + weights[2] * test_pred_lgbm + weights[3] * test_pred_hgb

# Prepare submission file
submission = pd.DataFrame({
    "id": list_test_id,
    "Exited": exited
})

submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved!")
submission.head()


# Plot distribution of predicted probabilities
plt.figure(figsize=(10, 6))
sns.kdeplot(oof_pred, fill=True, linewidth=1.5, alpha=0.2, label="OOF Predictions (Train)")
sns.kdeplot(exited, fill=True, linewidth=1.5, alpha=0.2, label="Test Predictions")
plt.title("KDE Distribution of Predicted Exited Probabilities", weight="bold", pad=15, fontsize=12)
plt.xlabel("Predicted Probability of Exited Probabilities")
sns.despine(left=False, bottom=False, right=False)
plt.ylabel("Frequency")
plt.xlabel("")
plt.xlim(0, 1)  # Limit x-axis to [0, 1]
plt.legend()
plt.tight_layout()
plt.show()


# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
binary_predictions = (exited > 0.5).astype(int)

# Plot distribution of binary predictions
plt.figure(figsize=(10, 6))
ax = sns.countplot(x=binary_predictions.flatten(), palette= color(n_colors=2))
plt.title("Distribution of Predicted Exited", weight="bold", pad=15, fontsize=12)
plt.xlabel("Exited Status (0: No, 1: Yes)")
plt.ylabel("")
sns.despine(left=False, bottom=False)
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])
for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(
        f"{count:,}", 
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="baseline",
        fontsize=10,
        xytext=(0, 5),
        textcoords="offset points"
    )
plt.tight_layout()
plt.show()


shap_plot(model=model_xgb, X_test=df_test_prepared[:1500], list_feature=clean_features, type="bar")


shap_plot(model=model_xgb, X_test=df_test_prepared[:1500], list_feature=clean_features)

