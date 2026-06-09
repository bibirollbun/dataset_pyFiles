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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import xgboost as xgb

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
    train_csv = "/kaggle/input/playground-series-s5e8/train.csv"
    test_csv = "/kaggle/input/playground-series-s5e8/test.csv"
    original_csv = "/kaggle/input/bank-marketing-dataset-full/bank-full.csv"
    target_feature = "y"   


# Load the datasets
df_train = pd.read_csv(Config.train_csv)
df_origin = pd.read_csv(Config.original_csv, sep=";")
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


df_train.drop("id", axis=1, inplace=True)
list_test_id = df_test["id"].copy().to_list()
df_test.drop("id", axis=1, inplace=True)


print("[i] Train Data describe:")
cm = sns.light_palette("blue", as_cmap=True)
display(df_train.drop(columns="y", axis=1).describe().T.style.background_gradient(cmap=cm))

print("\n[i] Origin Data describe:")
display(df_origin.drop(columns="y", axis=1).describe().T.style.background_gradient(cmap=cm))

print("\n[i] Test Data describe:")
display(df_test.describe().T.style.background_gradient(cmap=cm))


num_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
cat_features = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

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

# Display information about the DataFrames
print("[i] Displaying Train Data Info:")
df_train.info()

# Display information about the DataFrames
print("\n[i] Displaying Origin Data Info:")
df_origin.info()

print("\n[i] Displaying Test Data Info:")
df_test.info()


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


df_origin["y"] = df_origin["y"].map({"no": 0, "yes": 1})


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
datasets = [("Train Data", df_train), ("Original Data", df_origin)]

for i, (title, data) in enumerate(datasets):
    ax = axes[i, 0]

    # Vertical barplot
    sns.countplot(x=Config.target_feature, data=data, ax=ax, palette=color(n_colors=2))
    ax.set_title(f"Term Deposit Subscription Distribution â€” {title}", pad=15, weight="bold", fontsize=12)
    ax.set_xlabel("Subscription Status")
    ax.set_ylabel("")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    # Add count labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        ax.text(x, height + max(data[Config.target_feature].value_counts()) * 0.01,
                f"{int(height)}",
                ha="center", va="bottom", fontsize=10, color="black")

    # Pie chart
    y_counts = data[Config.target_feature].value_counts().sort_index()
    wedges, texts, autotexts = axes[i, 1].pie(
        y_counts,
        labels=["No", "Yes"],
        autopct="%1.1f%%",
        startangle=90,
        colors=color(n_colors=2),
        wedgeprops=dict(width=0.4, edgecolor="w"),
        radius=1.2,
        explode=(0, 0.08)
    )

    for text in texts + autotexts:
        text.set_fontsize(10)

    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    axes[i, 1].add_artist(centre_circle)
    axes[i, 1].set_title(f"Subscription Rate Breakdown â€” {title}", pad=15, weight="bold", fontsize=12)
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
    fig, ax = plt.subplots(1, 3, figsize=(24, 10))

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
    ax[2].set_title(f"Correlation Heatmap of {origin_name}", fontsize=16, weight="bold")

    plt.tight_layout()
    plt.show()

plot_correlation(df_train=df_train.drop(columns="y", axis=1),
                 df_origin=df_origin.drop(columns="y", axis=1),
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
                   palette=color(n_colors=df[target_feature].nunique(), tone="diverging"), inner="quartile")
    
    plt.title(f"Violin plot of {feature} distribution by Subscription ({target_feature})", pad=15, weight="bold", fontsize=12)
    plt.xlabel(f"Subscription ({target_feature})", labelpad=10)
    plt.ylabel(feature, labelpad=10)
    plt.legend().remove()
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.show()

for feature in num_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {feature} by Subscription (y)</b></h2>"))
    plot_numerical_distribution(feature=feature, df = df_train_combined)


from IPython.core.display import HTML
# defining function for plotting
def bivariate_percent_plot(cat, df, figsize=(15, 6), order = None, rot = 0):

    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {cat} by Subscription (y)</b></h2>"))
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=figsize)
    # Plot 1
    # Calculate the total number of each "cat" by Subscription
    grouped = df.groupby([cat, "y"]).size().unstack(fill_value=0)
    # Calculate the percentages
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    if order is not None:
        percentages = percentages.loc[order]
        labels = order
    else:
        labels = percentages.index

    # That method uses HUSL colors, so you need hue, saturation, and lightness.
    # I used hsluv.org to select the colors of this chart.
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    # Draw stacked bar plot
    ax[0] = percentages.plot(kind="bar", stacked=True, cmap=cmap, ax = ax[0], use_index=True)
    if feature in ["job", "month"]:
        pass
    else:
        for container in ax[0].containers:
            ax[0].bar_label(container, fmt="%1.0f%%", label_type="center", fontsize=10)

    ax[0].set_title(f"Percentage of Subscription by {cat}", fontsize=12, weight="bold", pad=15)
    ax[0].set_xlabel(f"{cat}", fontsize=12)
    ax[0].set_ylabel("% Subscription Rate", fontsize=12)
    ax[0].set_xticklabels(labels = labels, rotation = 45)
    ax[0].legend_.remove()
    sns.despine(left=False, bottom=False, ax=ax[0])

    # Plot 2
    sns.countplot(data=df, hue = "y", x = cat,
                palette=color(n_colors=2), ax=ax[1], order=order)
    # Show value for each bar.
    if feature in ["job", "month"]:
        pass
    else:
        for container in ax[1].containers:
            ax[1].bar_label(container, fmt="%d", label_type="edge", fontsize=10)

    ax[1].set_title(f"Subscription by {cat}", fontsize=12, weight="bold", pad=15)
    ax[1].set_xlabel(f"{cat}", fontsize=12)
    ax[1].set_ylabel("Number of Customer", fontsize=12)
    ax[1].legend(title="Subscription", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax[1].set_xticklabels(labels = ax[1].get_xticklabels(), rotation = 45)

    sns.despine(left=False, bottom=False, ax=ax[1])
    plt.tight_layout()
    plt.show()

    cal_ChiSquare(cat_feature=cat, target_feature=Config.target_feature, df=df, show_residuals=True)

for feature in cat_features:
    bivariate_percent_plot(cat=feature, df= df_train_combined)


df_bq = df_train_combined.copy()


print("[i] Computing percentage table by Job Ã— Marital Status Ã— Education Level...")
df_group = pd.crosstab(
    [df_bq["job"], df_bq["marital"], df_bq["education"]],
    df_bq["y"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Job Ã— Marital Status Ã— Education Level) vs Subscription (y)...")
contingency = pd.crosstab(
    [df_bq["job"], df_bq["marital"], df_bq["education"]],
    df_bq["y"]
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
plt.figure(figsize=(15,30))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Job or Marital Status or Education Level vs Subscription (y)", 
          weight="bold", fontsize=12, pad=15)
plt.ylabel("Job - Marital Status - Education Level | Subscription (y)")
plt.xlabel("")
plt.text(2.2,145, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


print("[i] Computing percentage table by Default Ã— Loan Ã— Housing...")
df_group = pd.crosstab(
    [df_bq["default"], df_bq["loan"], df_bq["housing"]],
    df_bq["y"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Default Ã— Loan Ã— Housing) vs Subscription (y)...")
contingency = pd.crosstab(
    [df_bq["default"], df_bq["loan"], df_bq["housing"]],
    df_bq["y"]
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
plt.title("Standardized Residuals Heatmap: Default or Loan or Housing vs Subscription (y)", 
          weight="bold", fontsize=12, pad=15)
plt.ylabel("Default - Loan - Housing | Subscription (y)")
plt.xlabel("")
plt.text(2.2,8.3, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


def bin_campaign(x):
    if x == 1:
        return "one"
    elif x == 2:
        return "two"
    elif x == 3:
        return "three"
    elif 4 <= x <= 5:
        return "high"
    elif 6 <= x <= 10:
        return "very_high"
    else:
        return "extreme"

df_bq["campaign_group"] = df_bq["campaign"].apply(bin_campaign)

print("[i] Computing percentage table by Campaign Group...")
df_group = pd.crosstab(
    [df_bq["campaign_group"]],
    df_bq["y"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Campaign Group) vs Subscription (y)...")
contingency = pd.crosstab(
    [df_bq["campaign_group"]],
    df_bq["y"]
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
plt.title("Standardized Residuals Heatmap: Campaign Group vs Subscription (y)", 
          weight="bold", fontsize=12, pad=15)
plt.ylabel("Campaign Group | Subscription (y)")
plt.xlabel("")
plt.text(2.2,6.2, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()



def bin_pdays(x):
    if x == -1:
        return "never_contacted"
    elif 1 <= x <= 7:
        return "recently_contacted"
    elif 8 <= x <= 30:
        return "contacted_last_month"
    elif 31 <= x <= 90:
        return "contacted_last_quarter"
    elif 91 <= x <= 180:
        return "contacted_3to6_months"
    elif 181 <= x <= 365:
        return "contacted_6to12_months"
    else:  # > 365
        return "contacted_over_1year"

df_bq["pdays_group"] = df_bq["pdays"].apply(bin_pdays)

def bin_previous(x):
    if x == 0:
        return "never_contacted_before"
    elif 1 <= x <= 2:
        return "lightly_contacted"
    elif 3 <= x <= 5:
        return "moderately_contacted"
    elif 6 <= x <= 10:
        return "heavily_contacted"
    elif 11 <= x <= 30:
        return "very_heavily_contacted"
    else:  # > 30
        return "extreme"

df_bq["previous_group"] = df_bq["previous"].apply(bin_previous)


print("[i] Fitting logistic regression model...")
model_two = smf.logit(
    formula="y ~ pdays_group + previous_group",
    data=df_bq
).fit()

print(model_two.summary())

# Compute Odds Ratios (OR)
coef = model_two.params
OR = np.exp(coef)

OR_table = pd.concat([coef, OR], axis=1)
OR_table.columns = ["coef", "OR"]

print("\n[i] Odds Ratios:")
print(OR_table)


def bin_duration(x):
    if x == 0:
        return "zero"
    elif 1 <= x <= 60:
        return "very_short"
    elif 61 <= x <= 180:
        return "short"
    elif 181 <= x <= 300:
        return "medium"
    elif 301 <= x <= 600:
        return "long"
    else:  # >600
        return "very_long"

df_bq["duration_group"] = df_bq["duration"].apply(bin_duration)

print("[i] Computing percentage table by Duration Group...")
df_group = pd.crosstab(
    [df_bq["duration_group"]],
    df_bq["y"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Duration Group) vs Subscription (y)...")
contingency = pd.crosstab(
    [df_bq["duration_group"]],
    df_bq["y"]
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
plt.title("Standardized Residuals Heatmap: Duration Group vs Subscription (y)", 
          weight="bold", fontsize=12, pad=15)
plt.ylabel("Duration Group | Subscription (y)")
plt.xlabel("")
plt.text(2.2,6.2, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


df_bq.groupby("y")["duration"].describe()


cal_ChiSquare(cat_feature="poutcome", target_feature="y", df=df_bq, show_residuals=True)


df_bq["poutcome_grouped"] = df_bq["poutcome"].replace({
    "success": "known_positive",
    "other": "known_positive",
    "failure": "known_positive",
    "unknown": "unknown"
})


categorical_cols = ["job", "marital", "education"]

for col in categorical_cols:
    dist = df_bq.groupby(["poutcome_grouped", col]).size().unstack().fillna(0)
    dist_ratio = dist.div(dist.sum(axis=1), axis=0) * 100
    print(f"\n Distribution of {col} by poutcome grouped (in %):\n")
    display(dist_ratio.T.round(2))


plt.figure(figsize=(10, 5))
sns.kdeplot(data=df_bq, x="age", hue="poutcome_grouped", common_norm=False,
            fill=True, palette=color(n_colors=2))
sns.despine(left=False, bottom=False)
plt.title("Age Distribution by Previous Campaign Outcome", weight="bold")
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()


financial_cols = ["balance", "housing", "loan", "default"]

for col in financial_cols:
    if df_bq[col].dtype == "object" or df_bq[col].dtype == "bool":
        # Categorical
        dist = df_bq.groupby(["poutcome_grouped", col]).size().unstack().fillna(0)
        dist_ratio = dist.div(dist.sum(axis=1), axis=0) * 100
        print(f"\n Distribution of {col} by poutcome_grouped (in %):\n")
        display(dist_ratio.T.round(2))
    else:
        # Numerical (e.g., balance)
        summary = df_bq.groupby("poutcome_grouped")[col].describe()
        print(f"\n Summary of {col} by poutcome_grouped:\n")
        display(summary.round(2))



contact_cols = ["contact", "duration", "month", "campaign"]

for col in contact_cols:
    if df_bq[col].dtype == "object":
        dist = df_bq.groupby(["poutcome_grouped", col]).size().unstack().fillna(0)
        dist_ratio = dist.div(dist.sum(axis=1), axis=0) * 100
        print(f"\n Distribution of {col} by poutcome_grouped (in %):\n")
        display(dist_ratio.T.round(2))
    else:
        summary = df_bq.groupby("poutcome_grouped")[col].describe()
        print(f"\n Summary of {col} by poutcome_grouped:\n")
        display(summary.round(2))


engagement_cols = ["previous", "pdays"]

for col in engagement_cols:
    summary = df_bq.groupby("poutcome_grouped")[col].describe()
    print(f"\n Summary of {col} by poutcome_grouped:\n")
    display(summary.round(2))


conversion = df_bq.groupby("poutcome_grouped")["y"].value_counts(normalize=True).unstack().fillna(0)
conversion["Conversion Rate (%)"] = conversion[1] * 100
display(conversion.round(2))


df_bq.groupby("y")["balance"].describe()


housing_conv = df_bq.groupby("housing")["y"].value_counts(normalize=True).unstack().fillna(0)
housing_conv["Conversion Rate (%)"] = housing_conv[1] * 100
print(housing_conv)
print("\n")
loan_conv = df_bq.groupby("loan")["y"].value_counts(normalize=True).unstack().fillna(0)
loan_conv["Conversion Rate (%)"] = loan_conv[1] * 100
print(loan_conv)


# Create condition columns for segmentation
df_bq["young"] = df_bq["age"] < df_bq["age"].median()
df_bq["high_balance"] = df_bq["balance"] > df_bq["balance"].median()
df_bq["no_loans"] = (df_bq["housing"] == "no") & (df_bq["loan"] == "no")
df_bq["single"] = df_bq["marital"] == "single"

# Combine conditions to define the target group
df_bq["target_group"] = (
    df_bq["young"] &
    df_bq["single"] &
    df_bq["high_balance"] &
    df_bq["no_loans"]
)

# Calculate conversion rate
grouped = df_bq.groupby("target_group")["y"].value_counts(normalize=True).unstack().fillna(0)
grouped.columns = ["No", "Yes"]
grouped["Conversion Rate (%)"] = grouped["Yes"] * 100
grouped = grouped.reset_index()

# Plot the bar chart
plt.figure(figsize=(6, 4))
sns.barplot(data=grouped, x="target_group", y="Conversion Rate (%)", palette=color(n_colors=2))
sns.despine(left=False, bottom=False)
plt.xticks([0, 1], ["Others", "Young, Single,\nHigh Balance,\nNo Loans"])
plt.ylabel("Subscription Rate (%)")
plt.xlabel("")
plt.title("Conversion Rate: Target Group vs Others", weight="bold")
plt.tight_layout()
plt.show()


# Create condition columns for segmentation
df_bq["tertiary"] = df_bq["education"] == "tertiary"
df_bq["no_default"] = df_bq["default"] == "no"
df_bq["high_duration"] = df_bq["duration"] >= df_bq["duration"].median()
df_bq["cellular"] = df_bq["contact"] == "cellular"

# Combine conditions to define the target group
df_bq["Combine_group"] = (
    df_bq["tertiary"] &
    df_bq["no_default"] &
    df_bq["high_duration"] &
    df_bq["cellular"]
)

# Calculate conversion rate
grouped = df_bq.groupby("Combine_group")["y"].value_counts(normalize=True).unstack().fillna(0)
grouped.columns = ["No", "Yes"]
grouped["Conversion Rate (%)"] = grouped["Yes"] * 100
grouped = grouped.reset_index()

# Plot the bar chart
plt.figure(figsize=(6, 4))
sns.barplot(data=grouped, x="Combine_group", y="Conversion Rate (%)", palette=color(n_colors=2))
sns.despine(left=False, bottom=False)
plt.xticks([0, 1], ["Others", "Tertiary, No Default,\nHigh Duration,\ncellular"])
plt.ylabel("Subscription Rate (%)")
plt.xlabel("")
plt.title("Conversion Rate: Combine Group vs Others", weight="bold")
plt.tight_layout()
plt.show()


# Calculate the conversion rate for each (job, month) combination
job_month_conv = (
    df_bq.groupby(["job", "month"])["y"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
    .reset_index()
)

# Rename columns for clarity
job_month_conv.columns.name = None
job_month_conv.rename(columns={0: "No", 1: "Yes"}, inplace=True)
job_month_conv["Conversion Rate (%)"] = job_month_conv["Yes"] * 100

# Sort by highest conversion rate
job_month_conv_sorted = job_month_conv.sort_values("Conversion Rate (%)", ascending=False)

# Display the top 10 combinations with highest subscription rates
top10 = job_month_conv_sorted.head(10)
top10["job_month"] = top10["job"].astype(str) + " | " + top10["month"].astype(str)
# Plot the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=top10, x="Conversion Rate (%)", y="job_month", palette="viridis")
plt.title("Top 10 Job-Month Combinations by Subscription Rate", weight="bold")
plt.xlabel("Subscription Rate (%)")
plt.ylabel("Job | Month")
plt.grid(axis="x")
plt.tight_layout()
plt.show()


# Create a flag for high-value customers
df_bq["high_balance"] = df_bq["balance"] > df_bq["balance"].median()
df_bq["no_default"] = df_bq["default"] == "no"
df_bq["high_value"] = df_bq["high_balance"] & df_bq["no_default"]

# Create a column for call duration groups (low vs. high)
df_bq["long_call"] = df_bq["duration"] >= df_bq["duration"].median()

# Analyze by contact strategy: channel, day, and call length
strategy_cols = ["contact", "day", "long_call"]

# Group by high_value + contact strategy
conversion = (
    df_bq.groupby(["high_value"] + strategy_cols)["y"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
    .reset_index()
)

# Rename columns and calculate conversion rate
conversion.columns.name = None
conversion.rename(columns={0: "No", 1: "Yes"}, inplace=True)
conversion["Conversion Rate (%)"] = conversion["Yes"] * 100

# Filter only high-value customers to examine the best-performing strategies
conversion_high_value = conversion[conversion["high_value"] == True].sort_values("Conversion Rate (%)", ascending=False)

# Display top 10 most effective contact strategies for high-value customers
top10 = conversion_high_value.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top10,
    x="Conversion Rate (%)",
    y=top10["contact"].astype(str) + " | Day " + top10["day"].astype(str) + " | " + top10["long_call"].map({True: "Long", False: "Short"}),
    palette="viridis"
)
plt.title("Top 10 Contact Strategies for High-Value Customers", weight="bold")
plt.xlabel("Subscription Rate (%)")
plt.ylabel("Contact Strategy (Channel | Day | Call Length)")
plt.grid(axis="x")
plt.tight_layout()
plt.show()


# Tag whether the customer has been contacted before
df_bq["was_contacted_before"] = df_bq["previous"] > 0

# Group by (job, marital, was_contacted_before) and calculate conversion rate
followup_effectiveness = (
    df_bq.groupby(["job", "marital", "was_contacted_before"])["y"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
    .reset_index()
)

# Rename columns
followup_effectiveness.columns.name = None
followup_effectiveness.rename(columns={0: "No", 1: "Yes"}, inplace=True)
followup_effectiveness["Conversion Rate (%)"] = followup_effectiveness["Yes"] * 100

# Show top 15 job + marital combinations with highest conversion rate when contacted before
top15_followups = followup_effectiveness[followup_effectiveness["was_contacted_before"] == True]\
    .sort_values("Conversion Rate (%)", ascending=False)\
    .head(15)

# Plot the chart
plt.figure(figsize=(12, 6))
sns.barplot(
    data=top15_followups,
    x="Conversion Rate (%)",
    y=top15_followups["job"].astype(str) + " | " + top15_followups["marital"].astype(str),
    palette="viridis"
)
plt.title("Top 15 Job + Marital Combinations with Prior Follow-up (previous > 0)", weight="bold")
plt.xlabel("Subscription Rate (%)")
plt.ylabel("Job | Marital Status")
plt.grid(axis="x")
plt.tight_layout()
plt.show()


# Label customers who have any type of loan (personal or housing)
df_bq["has_loan"] = (df_bq["loan"] == "yes") | (df_bq["housing"] == "yes")

# Group by relevant columns and compute total count and conversion rate
loan_campaign_timing = (
    df_bq[df_bq["has_loan"]]
    .groupby(["month", "day", "campaign"])
    .agg(count=("y", "size"),
         conversion_rate=("y", lambda x: (x == 1).mean() * 100))
    .reset_index()
)

# Keep only combinations with at least 30 customers to avoid small-sample bias
loan_campaign_timing_filtered = loan_campaign_timing[loan_campaign_timing["count"] >= 30]

# Select the top 15 strategies with the highest conversion rates
top15_loans = loan_campaign_timing_filtered.sort_values("conversion_rate", ascending=False).head(15)

# Plot the results
plt.figure(figsize=(12, 6))
sns.barplot(
    data=top15_loans,
    x="conversion_rate",
    y=top15_loans["month"].astype(str) + " | Day " + top15_loans["day"].astype(str) + " | #" + top15_loans["campaign"].astype(str),
    palette="viridis"
)
plt.title("Top 15 Contact Strategies for Customers with Loans (count â‰¥ 30)", weight="bold")
plt.xlabel("Subscription Rate (%)")
plt.ylabel("Month | Day | #Contacts")
plt.grid(axis="x")
plt.tight_layout()
plt.show()


cal_ChiSquare(cat_feature="education", target_feature="poutcome", df=df_bq, show_residuals=True)


age_stats = df_bq.groupby("poutcome")["age"].agg(["mean", "median"]).round(2).reset_index()
age_stats.columns = ["poutcome", "Mean Age", "Median Age"]
display(age_stats)

perform_statical_testing(feature="age", df=df_bq, target_feature="poutcome")

fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(x="poutcome", y="age", data=df_bq, hue="poutcome", palette=color(n_colors=4), inner="quartile", ax=ax)
ax.set_title(f"Violin plot of age distribution by poutcome", pad=15, weight = "bold")
ax.set_xlabel("poutcome", labelpad=10)
ax.set_ylabel("age", labelpad=10)
sns.despine(left=False, bottom=False, ax=ax)
plt.tight_layout()
plt.show()


# Flag customers with low balance
df_bq["low_balance"] = df_bq["balance"] < df_bq["balance"].median()
df_bq["no_loan"] = df_bq["loan"] == "no"
df_bq["no_default"] = df_bq["default"] == "no"

# Define early in the month (e.g., day â‰¤ 10)
df_bq["early_month"] = df_bq["day"] <= 10

# Define long call (greater than median duration)
df_bq["long_call"] = df_bq["duration"] > df_bq["duration"].median()

# Define the target group according to the business question
df_bq["target_group"] = (
    df_bq["low_balance"] &
    df_bq["no_loan"] &
    df_bq["no_default"] &
    df_bq["early_month"] &
    df_bq["long_call"]
)

# Calculate conversion rate
grouped = (
    df_bq.groupby("target_group")["y"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
    .reset_index()
)

# Rename columns
grouped.columns.name = None
grouped.rename(columns={0: "No", 1: "Yes"}, inplace=True)
grouped["Conversion Rate (%)"] = grouped["Yes"] * 100

# Plot the result
plt.figure(figsize=(8, 5))
sns.barplot(data=grouped, x="target_group", y="Conversion Rate (%)", palette=color(n_colors=2))
sns.despine(left=False, bottom=False)
plt.xticks([0, 1], ["Others", "Low Balance,\nNo Loan/Default,\nEarly & Long Call"])
plt.ylabel("Subscription Rate (%)")
plt.xlabel("")
plt.title("Conversion Rate: Target Group vs Others", weight="bold")
plt.tight_layout()
plt.show()


# Assume df_bq is your DataFrame
# Step 1: Assign age groups
bins = [0, 30, 40, 50, 60, 100]
labels = ["<30", "30-40", "40-50", "50-60", "60+"]
df_bq["age_group"] = pd.cut(df_bq["age"], bins=bins, labels=labels)

# Step 2: Group by (age_group, job, contact) and calculate conversion rate
grouped = (
    df_bq.groupby(["age_group", "job", "contact"])["y"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
    .reset_index()
)

# Rename columns and compute conversion rate
grouped.columns.name = None
grouped.rename(columns={0: "No", 1: "Yes"}, inplace=True)
grouped["Conversion Rate (%)"] = grouped["Yes"] * 100

# Sort by highest conversion rate
top10 = grouped.sort_values("Conversion Rate (%)", ascending=False).head(10)

# Step 3: Plot the chart
plt.figure(figsize=(12, 6))
sns.barplot(
    data=top10,
    x="Conversion Rate (%)",
    y=top10["age_group"].astype(str) + " | " + top10["job"].astype(str) + " | " + top10["contact"].astype(str),
    palette="viridis"
)
plt.title("Top 10 Combinations of Age Group, Job, and Contact Channel by Subscription Rate", weight="bold")
plt.xlabel("Subscription Rate (%)")
plt.ylabel("Age Group | Job | Contact")
plt.grid(axis="x")
plt.tight_layout()
plt.show()


# Create groupings for pdays and duration
df_bq["pdays_group"] = pd.cut(
    df_bq["pdays"],
    bins=[-1, 5, 10, 999],
    labels=["Early (â‰¤5 days)", "Mid (6â€“10 days)", "Late (>10 days)"]
)

df_bq["duration_group"] = pd.cut(
    df_bq["duration"],
    bins=[0, 100, 300, df_bq["duration"].max()],
    labels=["Short (â‰¤100s)", "Medium (101â€“300s)", "Long (>300s)"]
)

# Filter customers with poutcome = failure
df_failure = df_bq[df_bq["poutcome"] == "failure"]

# Group by pdays_group and duration_group
grouped = (
    df_failure
    .groupby(["pdays_group", "duration_group"])["y"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
    .reset_index()
)

# Calculate subscription rate (%)
grouped.columns.name = None
grouped.rename(columns={0: "No", 1: "Yes"}, inplace=True)
grouped["Conversion Rate (%)"] = grouped["Yes"] * 100

# Plot heatmap
pivot = grouped.pivot(index="pdays_group", columns="duration_group", values="Conversion Rate (%)")

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap=cm)
plt.title("Conversion Rate for Customers with Failed Previous Campaigns", weight="bold")
plt.xlabel("Call Duration Group")
plt.ylabel("Follow-up Timing (pdays Group)")
plt.tight_layout()
plt.show()



# age_group
def age_group(age):
    if age < 25:
        return "young"
    elif 25 <= age < 40:
        return "adult"
    elif 40 <= age < 60:
        return "mid-age"
    else:
        return "senior"

df_train_combined["age_group"] = df_train_combined["age"].apply(age_group)
df_test["age_group"] = df_test["age"].apply(age_group)

# has_credit_risk
df_train_combined["has_credit_risk"] = ((df_train_combined["default"] == "yes") |
                         (df_train_combined["housing"] == "yes") |
                         (df_train_combined["loan"] == "yes")).astype(int)
df_test["has_credit_risk"] = ((df_test["default"] == "yes") |
                         (df_test["housing"] == "yes") |
                         (df_test["loan"] == "yes")).astype(int)

# balance_category
def balance_category(bal):
    if bal < 0:
        return "negative"
    elif bal < 1000:
        return "low"
    elif bal < 5000:
        return "medium"
    else:
        return "high"

df_train_combined["balance_category"] = df_train_combined["balance"].apply(balance_category)
df_test["balance_category"] = df_test["balance"].apply(balance_category)

# contact_month_spring_summer
# Reflects the effectiveness of seasonal campaigns during the year (usually spring - summer).
spring_summer_months = ["mar", "apr", "may", "jun", "jul", "aug"]
df_train_combined["contact_month_spring_summer"] = df_train_combined["month"].isin(spring_summer_months).astype(int)
df_test["contact_month_spring_summer"] = df_test["month"].isin(spring_summer_months).astype(int)

# contact_effective
# Combine duration and previous campaign results.
df_train_combined["contact_effective"] = ((df_train_combined["duration"] > 120) & (df_train_combined["poutcome"] == "success")).astype(int)
df_test["contact_effective"] = ((df_test["duration"] > 120) & (df_test["poutcome"] == "success")).astype(int)

# is_high_value_customer
df_train_combined["is_high_value_customer"] = (
    (df_train_combined["balance"] > 5000) &
    (df_train_combined["education"].isin(["tertiary"])) &
    (df_train_combined["housing"] == "no") &
    (df_train_combined["loan"] == "no")
).astype(int)
df_test["is_high_value_customer"] = (
    (df_test["balance"] > 5000) &
    (df_test["education"].isin(["tertiary"])) &
    (df_test["housing"] == "no") &
    (df_test["loan"] == "no")
).astype(int)

# ultra_potential_customer
df_train_combined["ultra_potential_customer"] = ((df_train_combined["is_high_value_customer"] == 1) & (df_train_combined["contact_effective"] == 1)).astype(int)
df_test["ultra_potential_customer"] = ((df_test["is_high_value_customer"] == 1) & (df_test["contact_effective"] == 1)).astype(int)


new_features = ["age_group", "has_credit_risk", "balance_category", "contact_month_spring_summer", "contact_effective", 
                "is_high_value_customer", "ultra_potential_customer"]
for feature in new_features:
    bivariate_percent_plot(cat=feature, df= df_train_combined)


skew_feature_train, skew_train_df = check_skewness(df_train_combined, "Train Data")
skew_feature_test, skew_test_df = check_skewness(df_test, "Test Data")


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


processed_combined_df, transformed_columns, sparse_columns, skewed_columns = handle_skewed_features(df=df_train_combined, num_features=skew_feature_train)
num_features = ["PT_age", "PT_balance", "day", "PT_duration", "PT_campaign", "PT_pdays", "PT_previous"]
skew_feature_combined, skew_combined_df = check_skewness(data=processed_combined_df, numerical_features=num_features,
                                                   dataset_name= "Combined data")


processed_test_df, transformed_columns_test, sparse_columns_test, skewed_columns_test = handle_skewed_features(df=df_test, num_features=skew_feature_test)
skew_feature_test, skew_test_df = check_skewness(data=processed_test_df, numerical_features=num_features,
                                                   dataset_name= "Test data")


for col in ["default", "housing", "loan"]:
    processed_combined_df[col] = (
        processed_combined_df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"no": 0, "yes": 1})
    )

    processed_test_df[col] = (
        processed_test_df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"no": 0, "yes": 1})
    )


plt.figure(figsize=(8, 5))
sns.histplot(data=processed_combined_df, x="balance_category", color="lightblue", edgecolor="black")

plt.title("Distribution of balance_category", fontsize=12, pad=15, weight="bold")
plt.xlabel("balance_category")
plt.ylabel("")
plt.tight_layout()
plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(processed_combined_df, processed_combined_df["balance_category"]):
    start_train_set = processed_combined_df.iloc[train_index]
    start_test_set = processed_combined_df.iloc[test_index]


df_train_new = start_train_set.drop("y", axis=1)
df_train_label = start_train_set["y"].copy()


list_feature_num_robust = ["PT_age", "PT_balance", "PT_duration", "PT_pdays", "PT_previous"]
list_feature_num_stand = ["day", "PT_campaign"]
list_feature_cat_onehot = ["job", "marital", "education", "contact", "month", "poutcome", "age_group", "balance_category"]
list_feature_cat_keep = ["default", "housing", "loan", "has_credit_risk", "contact_month_spring_summer",
                         "contact_effective", "is_high_value_customer", "ultra_potential_customer"]


num_robust_transformer = Pipeline(steps=[
    ("scaler", RobustScaler())
])

num_stand_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

cat_onehot_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

cat_keep_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_robust", num_robust_transformer, list_feature_num_robust),
    ("num_standard", num_stand_transformer, list_feature_num_stand),
    ("cat_onehot", cat_onehot_transformer, list_feature_cat_onehot),
    ("cat_keep", cat_keep_transformer, list_feature_cat_keep),
])

preprocessor.fit(df_train_new)
df_train_new_prepared = preprocessor.transform(df_train_new)
list_feature_prepared = preprocessor.get_feature_names_out().tolist()
clean_features = [col.replace("num_standard__", "").replace("num_robust__", "").replace("cat__", "").replace("PT_", "") for col in list_feature_prepared]
clean_features


class_1 = df_train_label.sum()
class_0 = len(df_train_label) - class_1
scale_pos_weight = class_0 / class_1


X_val = start_test_set.drop("y", axis=1)
y_val = start_test_set["y"].copy()
X_val_prepared = preprocessor.transform(X_val)


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
        shap_df = shap_df.sort_values("importance", ascending=False).head(15)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=shap_df["importance"], y=shap_df["feature"], palette="viridis", order=shap_df["feature"])
        plt.xlabel("mean(|SHAP value|)")
        plt.title("SHAP Feature Importance", fontsize=12, weight="bold", pad=15)
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
                eval_set=[(X_valid, y_valid)],
                # verbose=False,
                # early_stopping_rounds=300
            )
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
            fold_pred = model.predict_proba(X_test)[:, 1]

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


from catboost import CatBoostClassifier

param_cb = {
  "iterations": 4928,
  "learning_rate": 0.02965344305433702,
  "depth": 6,
  "l2_leaf_reg": 0.0339907630913825,
  "border_count": 121,
  "random_strength": 12.100132443605029,
  "bootstrap_type": "Bernoulli",
  "subsample": 0.8795300421091226,
  "random_seed": Config.seed,
  "eval_metric": "Logloss",
  "verbose": 0,
  "class_weights": [1, scale_pos_weight],
  "thread_count": -1,
  "grow_policy": "Lossguide",
  "loss_function": "Logloss",
  "task_type": "GPU",
  "devices": "0"
}

model_cb = CatBoostClassifier(**param_cb)
df_test_prepared = preprocessor.transform(processed_test_df)
oof_pred_cb, test_pred_cb, fold_models_cb = run_oof(model = model_cb, X = df_train_new_prepared, y = df_train_label.values, 
                                                    X_test = df_test_prepared)


from lightgbm import LGBMClassifier

param_lgbm = {
"random_state": Config.seed,
"verbose": -1,
"n_estimators": 10000,
"metric": "AUC",
"objective": "binary",
"max_depth": 16,
"learning_rate": 0.007366917567300051,
"min_child_samples": 164,
"subsample": 0.9022880020285295,
"colsample_bytree": 0.4213201532077694,
"num_leaves": 122, 
"reg_alpha": 1.083996192298843,
"reg_lambda": 0.0700057221912873,
"class_weight": {0: 1.0, 1: float(scale_pos_weight)},
"n_jobs": -1,
"early_stopping_round": 300,
"max_bin": 255
}

model_lgbm = LGBMClassifier(**param_lgbm)
oof_pred_lgbm, test_pred_lgbm, fold_models_lgbm = run_oof(model = model_lgbm, X = df_train_new_prepared, y = df_train_label.values, 
                                                          X_test = df_test_prepared)


param_lgbm2 = {
"random_state": Config.seed,
"verbose": -1,
"n_estimators": 5000,
"metric": "AUC",
"objective": "binary",
"max_depth": 18,
"boosting_type": "goss",
"learning_rate": 0.013632406163139255,
"min_child_samples": 76,
"subsample": 0.8008906838837987,
"colsample_bytree": 0.22001761604503337,
"num_leaves": 345, 
"reg_alpha": 1.616390930105809,
"reg_lambda": 0.6118370655549995,
"class_weight": {0: 1.0, 1: float(scale_pos_weight)},
"n_jobs": -1,
"early_stopping_round": 300,
"max_bin": 255
}

model_lgbm2 = LGBMClassifier(**param_lgbm2)
oof_pred_lgbm2, test_pred_lgbm2, fold_models_lgbm2 = run_oof(model = model_lgbm2, X = df_train_new_prepared, y = df_train_label.values, 
                                                             X_test = df_test_prepared)


from xgboost import XGBClassifier

param_xgb = {
"n_estimators": 2477,
"learning_rate": 0.07329764338577169,
"max_depth": 10,
"min_child_weight": 12.836741271468108,
"gamma": 5.213583909601075,
"subsample": 0.7267867980610363,
"colsample_bytree": 0.6677560232735622,
"reg_alpha": 9.121274651805239e-06,
"reg_lambda": 6.652188929868966,
"max_bin": 452,
"random_state": Config.seed,
"n_jobs": -1,
"verbosity": 0,
"objective": "binary:logistic",
"eval_metric": "auc",
"scale_pos_weight": scale_pos_weight,
"grow_policy": "lossguide",
"tree_method": "gpu_hist",
"predictor": "gpu_predictor",
"use_label_encoder": False,
"booster": "gbtree"
}

model_xgb = XGBClassifier(**param_xgb)
oof_pred_xgb, test_pred_xgb, fold_models_xgb = run_oof(model = model_xgb, X = df_train_new_prepared, y = df_train_label.values, 
                                                       X_test = df_test_prepared)


# Collect predictions (probabilities instead of labels) ---
ests = [("cb", model_cb), ("xgb", model_xgb), ("lgbm1", model_lgbm), ("lgbm2", model_lgbm2)]
preds = {name: m.predict_proba(X_val_prepared)[:, 1] for name, m in ests}

# auc_each = {name: roc_auc_score(y_val, preds[name]) for name,_ in ests}
# display(auc_each)

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


oof_pred = weights[0] * oof_pred_cb + weights[1] * oof_pred_xgb + weights[2] * oof_pred_lgbm + weights[3] * oof_pred_lgbm2
subscription = weights[0] * test_pred_cb + weights[1] * test_pred_xgb + weights[2] * test_pred_lgbm + weights[3] * test_pred_lgbm2

# Prepare submission file
submission = pd.DataFrame({
    "id": list_test_id,
    "y": subscription
})

submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved!")
submission.head()


# Plot distribution of predicted probabilities
plt.figure(figsize=(10, 6))
sns.kdeplot(oof_pred, fill=True, linewidth=1.5, alpha=0.2, label="OOF Predictions (Train)")
sns.kdeplot(subscription, fill=True, linewidth=1.5, alpha=0.2, label="Test Predictions")
plt.title("KDE Distribution of Predicted Subscription Probabilities", weight="bold", pad=15, fontsize=12)
plt.xlabel("Predicted Probability of Subscription Probabilities")
sns.despine(left=False, bottom=False, right=False)
plt.ylabel("Frequency")
plt.xlabel("")
plt.xlim(0, 1)  # Limit x-axis to [0, 1]
plt.legend()
plt.tight_layout()
plt.show()


# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
binary_predictions = (subscription > 0.5).astype(int)

# Plot distribution of binary predictions
plt.figure(figsize=(10, 6))
ax = sns.countplot(x=binary_predictions.flatten(), palette= color(n_colors=2))
plt.title("Distribution of Predicted Subscription", weight="bold", pad=15, fontsize=12)
plt.xlabel("Subscription Status (0: No, 1: Yes)")
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


shap_plot(model=model_xgb, X_test=df_test_prepared[:3000], list_feature=clean_features, type="bar")


shap_plot(model=model_xgb, X_test=df_test_prepared[:3000], list_feature=clean_features)

