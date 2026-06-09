!pip install statsmodels > pip_log_statsmodels.txt 2>&1
!pip install scikit_posthocs > pip_log_scikit_posthocs.txt 2>&1
!pip install pingouin > pip_log_pingouin.txt 2>&1


# Core data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Statistical functions
from scipy.stats import skew, kurtosis, probplot

# Display utilities for Jupyter notebooks
from IPython.display import display, HTML

# Machine learning preprocessing and modeling
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Metrics
from sklearn.metrics import (roc_curve, roc_auc_score, classification_report, confusion_matrix,
                             precision_recall_curve, auc, average_precision_score, log_loss)

# Statistical
from scipy.stats import chi2_contingency
from scipy.stats import probplot
from scipy.stats import kruskal
import scikit_posthocs as sp
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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
    SEED = 42
    MAX_ITER = 50000
    N_SPLIT = 1
    TEST_SIZE = 0.2


# Load the datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

# Verify shapes
print("Train Data Shape:", df_train.shape)
print("\nTest Data Shape:", df_test.shape)


# Display few rows of each dataset
print("Train Data Preview:")
display(df_train.head())

print("\nTest Data Preview:")
display(df_test.head())


# Display information about the DataFrames
print("Train Data Info:")
df_train.info()

print("\nTest Data Info:")
df_test.info()


df_train.drop("id", axis=1, inplace=True)
list_test_id = df_test["id"].copy().to_list()
df_test.drop("id", axis=1, inplace=True)


num_features = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]
cat_features = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]
print("Train Data describe:")
cm = sns.light_palette("green", as_cmap=True)
display(df_train[num_features].describe().T.style.background_gradient(cmap=cm))

print("\nTest Data describe:")
display(df_test[num_features].describe().T.style.background_gradient(cmap=cm))


def convert_cat(features, df):
    for feature in features:
        if feature in df.columns:
            df[feature] = df[feature].astype("category")
convert_cat(cat_features, df=df_train)
convert_cat(cat_features, df=df_test)

print("Train Data describe:")
display(df_train[cat_features].describe().T.style.background_gradient(cmap="Greens", subset=["unique", "freq"]))

print("\nTest Data describe:")
display(df_test[cat_features].describe().T.style.background_gradient(cmap="Greens", subset=["unique", "freq"]))


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

    print("=" * 40)
    if dataset_name:
        print(f"ğŸ”� Missing Value Summary for: {dataset_name}")
    else:
        print("ğŸ”� Missing Value Summary:")
    print("=" * 40)
    
    if total_missing == 0:
        print(f"âœ… No missing values detected in {total_rows:,} rows.")
    else:
        try:
            from tabulate import tabulate
            print(tabulate(missing_df, headers="keys", tablefmt="pretty", showindex=False, colalign=("left", "left", "left")))
        except ImportError:
            print(missing_df.to_string(index=False))
        
        print(f"\nâš ï¸�  Total missing values: {total_missing:,} out of {total_rows:,} rows.")

print("Missing value train dataset: ")
displayNULL(df_train, dataset_name="Train Set")

print("\nMissing value test dataset: ")
displayNULL(df_test, dataset_name="Test Set")


def check_duplicates_report(df, dataset_name):
    duplicates_count = df.duplicated().sum()
    total_rows = len(df)

    print("=" * 40)
    print(f"ğŸ”� {dataset_name} Duplicate Analysis")
    print("=" * 40)

    if duplicates_count == 0:
        print(f"âœ… No duplicates found in {total_rows:,} rows")
    else:
        print(f"âš ï¸�  {duplicates_count} duplicates found ({duplicates_count/total_rows:.2%})")
        print(f"    Total rows affected: {duplicates_count:,}/{total_rows:,}")

datasets = {
    "Training Data": df_train,
    "Test Data": df_test
}

duplicate_summary = {}
for name, data in datasets.items():
    check_duplicates_report(data, name)
    duplicate_summary[name] = {
        "duplicates": data.duplicated().sum(),
        "total_rows": len(data)
    }


def checking_outlier(list_feature, df, dataset_name):
    print("=" * 40)
    print(f"ğŸ”� {dataset_name} Checking outlier")
    print("=" * 40)
    outlier_info = []
    for feature in list_feature:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]
        if len(outliers) == 0:
            pass
        else:
            outlier_info.append({
            "Feature": feature,
            "Outlier Count": len(outliers),
            # "Outlier Detail": outliers.tolist()
            })
    return pd.DataFrame(outlier_info)

checking_outlier(list_feature=num_features, df=df_train, dataset_name="Training data")


checking_outlier(list_feature=num_features, df=df_test, dataset_name="Test data")


def color(n_colors=2, tone="diverging"):
    stop = 1
    if tone == "diverging":
        cmap = sns.diverging_palette(0, 230, as_cmap=True)
        stop = 0.9
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
    positions = np.linspace(0, stop, n_colors)
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
    print(f"\nğŸ”� Chi-Square Test of Independence: '{cat_feature}' vs. '{target_feature}'")

    # Contingency table
    crosstab = pd.crosstab(df[cat_feature], df[target_feature])
    chi2, p, dof, expected = chi2_contingency(crosstab)

    print(f"Chi-squared statistic: {chi2:.3f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p:.6f}")

    if p < 0.05:
        print("âœ… Result: p-value < 0.05 â†’ Reject Hâ‚€")
        print(f"â†’ There is a **statistically significant association** between '{cat_feature}' and '{target_feature}'.")
    else:
        print("â�� Result: p-value â‰¥ 0.05 â†’ Fail to reject Hâ‚€")
        print(f"â†’ No statistically significant association between '{cat_feature}' and '{target_feature}'.")

    # Optional: show expected frequencies
    if show_expected:
        print("\nğŸ“Š Expected Frequencies:")
        print(pd.DataFrame(expected, index=crosstab.index, columns=crosstab.columns))
    else:
        pass

    # Optional: show standardized residuals
    if show_residuals:
        # cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
        residuals = (crosstab - expected) / np.sqrt(expected)
        print("\nğŸ“ˆ Standardized Residuals:")
        print(round(residuals, 2))

        # Heatmap of residuals
        plt.figure(figsize=(10, 7))
        sns.heatmap(residuals, annot=True, cmap="RdYlGn", center=0, fmt=".2f", linewidths=0.5)
        plt.title(f"Standardized Residuals Heatmap: {cat_feature} vs {target_feature}", weight="bold", fontsize=13, pad=25)
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
        print(f"â�Œ Error: Kruskal-Wallis H-test requires 3 or more groups.")
        return
    else:
        print(f"\nğŸ”� Kruskal-Wallis Test: {numeric_feature} ~ {categorical_feature}")
        data_groups = [df[df[categorical_feature] == g][numeric_feature].dropna() for g in groups]

        # Perform kruskal
        stat, p = kruskal(*data_groups)

        print(f"Kruskal-Wallis H-statistic: {stat:.3f}")
        print(f"p-value: {p}")

        if p < 0.05:
            print("ğŸŸ¢ Significant difference found. Running Dunn's Post-Hoc Test...")
            dunn_result = sp.posthoc_dunn(df, val_col=numeric_feature, group_col=categorical_feature, p_adjust="bonferroni")
            print(dunn_result)
        else:
            print("\nâ„¹ï¸� No significant difference found (p >= 0.05)")

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

    print(f"\n Checking normality of numeric feature(s) by target feature: '{target_feature}'")

    # ===  Evaluate normality within each group ===
    print(f"\nğŸ”¹ Feature: {feature}")

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
    n_groups = df[target_feature].nunique()
    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, grp in enumerate(df[target_feature].unique()):
        ax = axes[i]
        data = df.loc[df[target_feature] == grp, feature].dropna()
        probplot(data, dist="norm", plot=ax)
        ax.set_title(f"{feature} â€” {grp}", fontsize=11, weight="bold")
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Qâ€“Q Plots of {feature} by {target_feature}", fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

    # === Display results table ===
    df_result = pd.DataFrame(results)
    cm = sns.light_palette("green", as_cmap=True)
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

    if non_normal_detected == True:
        print("\nâš ï¸� At least one group deviates from normality â†’ Running Kruskalâ€“Wallis test or Mannâ€“Whitney U test...")
    else:
        print("\nâœ… All groups approximately follow normal distribution.")

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
        print(f"â�Œ Error: ANOVA requires 3 or more groups.")
        return
    else:
        print(f"\nğŸ”� ANOVA Test: {numeric_feature} ~ {categorical_feature} (Type {typ})")

        # Fit OLS model
        model = ols(f"{numeric_feature} ~ C({categorical_feature})", data=df).fit()

        # Perform ANOVA
        anova_table = anova_lm(model, typ=typ)
        print("\nğŸ“Š ANOVA Table:")
        print(anova_table)

        # Extract p-value
        p_value = anova_table["PR(>F)"].iloc[0]

        if p_value < 0.05:
            print("\nâœ… Significant difference found (p < 0.05)")
            print("â�¡ï¸� Performing Tukey's HSD post-hoc test:")

            tukey = pairwise_tukeyhsd(df[numeric_feature], df[categorical_feature])
            print(tukey.summary())
        else:
            print("\nâ„¹ï¸� No significant difference found (p >= 0.05)")

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
        print("â�Œ Error: Welchâ€™s ANOVA requires 3 or more groups.")
        return

    print(f"\nğŸ”� Welchâ€™s ANOVA Test: {numeric_feature} ~ {categorical_feature}")
    print("Testing mean differences under heteroscedasticity assumption...")

    # Perform Welch's ANOVA (scipy.stats)
    welch_result = stats.f_oneway(*groups)
    print("\nWelchâ€™s ANOVA Result:")
    print(f"F-statistic = {welch_result.statistic:.4f},  p-value = {welch_result.pvalue:.6f}")

    # Interpret result
    if welch_result.pvalue < 0.05:
        print("\nâœ… Significant difference found (p < 0.05)")
        print("â�¡ï¸� Performing Gamesâ€“Howell post-hoc test:\n")

        # Perform Gamesâ€“Howell post-hoc test (robust for unequal variances)
        gh_result = pg.pairwise_gameshowell(dv=numeric_feature, between=categorical_feature, data=df)
        # display(gh_result)

        display(HTML("<b>Gamesâ€“Howell Post-hoc Test (adjusted p-values)</b>"))
        display(gh_result.style.background_gradient(cmap=cm).format(precision=4).set_table_attributes('style="width:80%; margin:auto;"'))
    else:
        print("\nâ„¹ï¸� No significant difference found (p â‰¥ 0.05)")


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
        - If p > 0.05 AND ratio < 2 â†’ ANOVA suitable
        - If p < 0.05 BUT ratio < 2 â†’ Statistical diff, but practically negligible â†’ still OK for ANOVA
        - If ratio â‰¥ 2 OR p < 0.05  â†’  Use Welchâ€™s ANOVA or Kruskalâ€“Wallis
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
        status = "âœ… Homogeneous variances."
        recommendation = "Use One-Way ANOVA or Independent Two-Sample T-Test."
        is_homogeneous_variances = True
        anova_use = True
    elif p < alpha and ratio < ratio_threshold:
        status = "âš ï¸� Statistically significant difference, but practically small â€” ANOVA or T-Test still acceptable."
        recommendation = "Use Welchâ€™s ANOVA or Welchâ€™s T-Test."
        anova_use = True
    else:
        status = "ğŸš¨ Variances differ substantially â€” Use non-parametric test."
        recommendation = "Use Kruskalâ€“Wallis or Mannâ€“Whitney U test (Wilcoxon rank-sum test)."

    # Display summary table
    summary_df = pd.DataFrame({
        "Metric": ["Leveneâ€™s Statistic", "p-value", "Max/Min Variance Ratio"],
        "Value": [f"{stat:.4f}", f"{p:.6f}", f"{ratio:.2f}"]
    })
    display(summary_df.style
            .background_gradient(subset=["Value"], cmap="Greens")
            .set_caption(
        f'<b><span style="font-size:14px; text-align:center; display:block;">'
        f'Homogeneity of Variance â€” {feature} by {target_feature}</span></b>'
    ).set_table_attributes('style="width:70%; margin:auto;"'))

    # Print interpretation
    print("\nğŸ”� Interpretation:")
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

    if len(groups) != 2:
        print(f"â�Œ Error: Mann-Whitney U test requires exactly 2 groups, but found {len(groups)}.")
        return

    print(f"ğŸ”� Mannâ€“Whitney U Test for '{num_feature}' by '{categorical_feature}'\n")

    group1 = dataframe[dataframe[categorical_feature] == groups[0]][num_feature].dropna()
    group2 = dataframe[dataframe[categorical_feature] == groups[1]][num_feature].dropna()

    stat, p = mannwhitneyu(group1, group2, alternative="two-sided")

    print(f"U statistic : {stat}")
    print(f"p-value     : {p}")

    # Interpretation
    if p <= 0.05:
        print("\nâœ… Result: Statistically significant difference between the two groups (Reject Hâ‚€).")
        median1 = group1.median()
        median2 = group2.median()
        if median1 > median2:
            print(f" Interpretation: Group '{groups[0]}' has a higher median '{num_feature}' than Group '{groups[1]}'.")
        elif median1 < median2:
            print(f" Interpretation: Group '{groups[1]}' has a higher median '{num_feature}' than Group '{groups[0]}'.")
        else:
            print(" Interpretation: The medians are equal, but distributions may still differ.")
    else:
        print("\nâšª Result: No statistically significant difference between the two groups (Fail to reject Hâ‚€).")

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
        print(f"â�Œ Error: Independent T-Test requires exactly 2 groups.")
        return

    print(f"ğŸ”� Independent Two-Sample T-Test: {num_feature} ~ {categorical_feature}")
    print(f"â†’ Test Type: {'Studentâ€™s T-Test (equal variances)' if equal_var else 'Welchâ€™s T-Test (unequal variances)'}")

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

    # Output
    print(f"\nComparing groups: '{groups[0]}' vs. '{groups[1]}'")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    # Significance interpretation
    if p_value < 0.05:
        print("\nâœ… Significant difference found (p < 0.05)")
    else:
        print("\nâ„¹ï¸� No significant difference found (p â‰¥ 0.05)")

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        size = "small"
    elif abs(cohens_d) < 0.5:
        size = "medium"
    else:
        size = "large"

    print(f"Effect size interpretation: {size} effect ({abs(cohens_d)})")


df_train["loan_paid_back"] = df_train["loan_paid_back"].map({0: "Not paid", 1: "Paid"})

# Prepare data and colors
status_counts = df_train["loan_paid_back"].value_counts().sort_index()
order = status_counts.index.tolist()
colors = color(n_colors=len(order), tone="RdYlGn")
palette = dict(zip(order, colors))

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# --- Pie chart ---
ax[0].pie(status_counts, labels=order, colors=colors, autopct="%1.2f%%", startangle=150, shadow=True)
ax[0].set_title("Proportion of Paid vs Not Paid Loans", fontweight="bold", fontsize=14, pad=20)

# --- Count plot ---
sns.countplot(data=df_train, x="loan_paid_back", order=order, palette=palette, ax=ax[1])
ax[1].set_title("Count of Paid vs Not Paid Loans", fontweight="bold", fontsize=14, pad=20)
for container in ax[1].containers:
    ax[1].bar_label(container, fmt="%d", label_type="edge", fontsize=10)
ax[1].set(xlabel="Loan Status", ylabel="Frequency")
sns.despine(ax=ax[1])

plt.tight_layout()
plt.show()


def plot_numerical_features(df_train, df_test, num_features):
    colors = color(n_colors=2, tone="RdYlGn")
    n = len(num_features)

    fig, axes = plt.subplots(n, 2, figsize=(12, n * 4))
    axes = np.array(axes).reshape(n, 2)

    for i, feature in enumerate(num_features):
        sns.histplot(data=df_train[feature], color=colors[0], bins=20, kde=True, ax=axes[i, 0], label="Train data")
        sns.histplot(data=df_test[feature], color=colors[1], bins=20, kde=True, ax=axes[i, 0], label="Test data")
        axes[i, 0].set_title(f"Histogram of {feature}", pad=15, weight="bold", fontsize=14)
        axes[i, 0].legend()
        axes[i, 0].set_ylabel("")
        sns.despine(left=False, bottom=False, ax=axes[i, 0])

        df_plot = pd.concat([
            pd.DataFrame({"Dataset": "Train data", feature: df_train[feature]}),
            pd.DataFrame({"Dataset": "Test data", feature: df_test[feature]})
        ]).reset_index(drop=True)

        sns.boxplot(data=df_plot, x=feature, y="Dataset", palette=colors, orient="h", ax=axes[i, 1])
        axes[i, 1].set_title(f"Horizontal Box plot of {feature}", pad=15, weight="bold", fontsize=14)
        sns.despine(left=False, bottom=False, ax=axes[i, 1])

    plt.tight_layout()
    plt.show()

plot_numerical_features(df_train = df_train, df_test = df_test, num_features=num_features)


def check_skewness(data, dataset_name, numerical_features = num_features, highlight=True, sort=True):
    skewness_dict = {}
    skew_feature = []
    for feature in numerical_features:
        skew = data[feature].skew(skipna=True)
        skewness_dict[feature] = skew

    skew_df = pd.DataFrame.from_dict(skewness_dict, orient="index", columns=["Skewness"])
    if sort:
        skew_df = skew_df.reindex(skew_df["Skewness"].abs().sort_values(ascending=False).index)
    else:
        pass

    print(f"\nğŸ”� Skewness for {dataset_name}:")
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

skew_feature_train, skew_train_df = check_skewness(df_train, "Train Data")
skew_feature_test, skew_test_df = check_skewness(df_test, "Test Data")


def plot_correlation(df_train, df_test, train_name="Train Data", test_name="Test Data", figsize=(24, 10)):
    corr_train = df_train.corr(numeric_only=True)
    corr_test = df_test.corr(numeric_only=True)

    mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
    adjusted_mask_train = mask_train[1:, :-1]
    adjusted_cereal_corr_train = corr_train.iloc[1:, :-1]

    mask_test = np.triu(np.ones_like(corr_test, dtype=bool))
    adjusted_mask_test = mask_test[1:, :-1]
    adjusted_cereal_corr_test = corr_test.iloc[1:, :-1]

    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(data=adjusted_cereal_corr_train, mask=adjusted_mask_train,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[0])
    ax[0].set_title(f"Correlation Heatmap of {train_name}", fontsize=16, weight="bold", loc="center", pad=15)

    sns.heatmap(data=adjusted_cereal_corr_test, mask=adjusted_mask_test,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[1])
    ax[1].set_title(f"Correlation Heatmap of {test_name}", fontsize=16, weight="bold", loc="center", pad=15)

    plt.tight_layout()
    plt.show()

plot_correlation(df_train=df_train.drop(columns="loan_paid_back", axis=1),
                 df_test=df_test)


def plot_categorical_distribution_across_datasets(
    df_train: pd.DataFrame = df_train,
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
    summary = pd.merge(train_summary, test_summary, on="Category", how="inner")

    # Ensure Category column is of string type before handling missing values
    summary["Category"] = summary["Category"].astype(str)

    # Fill missing values only for numeric columns
    numeric_cols = ["Train_Count", "Train_%", "Test_Count", "Test_%"]
    summary[numeric_cols] = summary[numeric_cols].fillna(0)

    # Convert summary table to HTML for better visualization in notebook
    html = summary.to_html(
        index=False,
        justify="center",
        classes="table table-striped table-hover",
    )

    # Display formatted HTML table with title
    display(HTML(f"<h4>ğŸ”� Categorical Feature Distribution: <b>{feature}</b> ğŸ�¦</h4>" + html))
    print("*" * 120)


# ----- Run the function for all categorical features -----
cat_features = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]

for feature in cat_features:
    plot_categorical_distribution_across_datasets(df_train, df_test, feature)


def perform_statical_testing(feature: str, df: pd.DataFrame = df_train,  target_feature: str = "loan_paid_back") -> None:
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

def plot_numerical_distribution(feature: str, df: pd.DataFrame = df_train,
                                target_feature: str = "loan_paid_back", order: list = None) -> None:
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
                   palette=color(n_colors=df[target_feature].nunique(), tone="RdYlGn"))
    
    plt.title(f"Violin plot of {feature} distribution by {target_feature}", pad=15, weight="bold")
    plt.xlabel(target_feature, labelpad=10)
    plt.ylabel(feature, labelpad=10)
    plt.legend().remove()
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.show()

for feature in num_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:green;'><b>Distribution of {feature} by Loan Status</b></h2>"))
    plot_numerical_distribution(feature=feature, df = df_train)


def bivariate_percent_plot(cat, target_feature, df, figsize=(15, 6), order=None):
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:green;'><b>Distribution of {cat} by {target_feature}</b></h2>"))
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=figsize)

    # === Data processing ===
    grouped = df.groupby([cat, target_feature]).size().unstack(fill_value=0)

    # 1) Define a fixed hue order (adjust if needed)
    target_order = [c for c in ["Paid", "Not paid"] if c in grouped.columns]
    # Fallback if the labels are 0/1 or have different names
    if not target_order:
        target_order = list(grouped.columns)

    # 2) Calculate percentages row-wise and reorder columns by target_order
    percentages = grouped.div(grouped.sum(axis=1), axis=0)[target_order] * 100

    # 3) Define X-axis category order
    if order is not None:
        percentages = percentages.loc[order]
        labels = order
    else:
        labels = percentages.index.tolist()

    # === Consistent color palette for both charts ===
    base_colors = color(n_colors=len(target_order), tone="RdYlGn")
    color_map = dict(zip(target_order, base_colors))

    # === Plot 1: Stacked bar chart (percentage) ===
    bottom = np.zeros(len(percentages))
    for cls in target_order:
        ax[0].bar(percentages.index, percentages[cls].values, bottom=bottom,
                  label=cls, color=color_map[cls])
        bottom += percentages[cls].values

    # Add percentage labels
    if cat != "grade_subgrade":
        for container in ax[0].containers:
            ax[0].bar_label(container, fmt="%1.0f%%", label_type="center",
                            fontsize=9, color="black")

    ax[0].set_title(f"Percentage of {target_feature} by {cat}", fontsize=14, weight="bold")
    ax[0].set_xlabel(f"{cat}", fontsize=12)
    ax[0].set_ylabel(f"% {target_feature} Rate", fontsize=12)
    ax[0].set_xticklabels(labels=labels, rotation=45)
    sns.despine(left=False, bottom=False, ax=ax[0])
    ax[0].legend().remove()

    # === Plot 2: Count plot (using same color_map + hue_order) ===
    sns.countplot(data=df, hue=target_feature, x=cat,
                  order=labels, hue_order=target_order,
                  palette=color_map, ax=ax[1])

    if cat != "grade_subgrade":
        for container in ax[1].containers:
            ax[1].bar_label(container, fmt="%d", label_type="edge",
                            fontsize=9, color="black")

    ax[1].set_title(f"{target_feature} by {cat}", fontsize=14, weight="bold")
    ax[1].set_xlabel(f"{cat}", fontsize=12)
    ax[1].set_ylabel("Number of Customers", fontsize=12)
    ax[1].legend(title=target_feature, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax[1].set_xticklabels(labels=ax[1].get_xticklabels(), rotation=45)
    sns.despine(left=False, bottom=False, ax=ax[1])

    plt.tight_layout()
    plt.show()

    # === Chi-Square Test ===
    cal_ChiSquare(cat_feature=cat, target_feature=target_feature, df=df, show_residuals=True)

# === Run for all categorical features ===
for feature in cat_features:
    bivariate_percent_plot(cat=feature, target_feature="loan_paid_back", df=df_train)


df_bq = df_train.copy()

def map_grade_category(grade):
    if isinstance(grade, str):
        if grade.startswith(("A", "B")):
            return "High"
        elif grade.startswith(("C", "D")):
            return "Medium"
        elif grade.startswith(("E", "F")):
            return "Low"
    return "Unknown"

df_bq["grade_category"] = df_bq["grade_subgrade"].map(map_grade_category)
df_bq.drop(columns="grade_subgrade", axis=1, inplace=True)


bins = [0, 40000, 80000, 150000, 400000]
labels = ["Low", "Lower-Middle", "Upper-Middle", "High"]
df_bq["annual_income_group"] = pd.cut(df_bq["annual_income"], bins=bins, labels=labels, include_lowest=True)

bins = [395, 580, 670, 750, 850]
labels = ["Low", "Fair", "Good", "Excellent"]
df_bq["credit_score_group"] = pd.cut(df_bq["credit_score"], bins=bins, labels=labels, include_lowest=True)

df_bq_income_group_credit_score_group = pd.crosstab(
    [df_bq["annual_income_group"], df_bq["credit_score_group"]],
    df_bq["loan_paid_back"],
    normalize="index"
) * 100

display(df_bq_income_group_credit_score_group)

contingency = pd.crosstab(
    [df_bq["annual_income_group"], df_bq["credit_score_group"]],
    df_bq["loan_paid_back"]
)
chi2, p, dof, ex = chi2_contingency(contingency)
print("Chi2:", chi2, "p-value:", p)

# 1) Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

print(f"Chi2: {chi2:.4f} | dof: {dof} | p-value: {p:.6f}")

# 2) Standardized residuals (adjusted)
#    r_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - row_prob_i) * (1 - col_prob_j))
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom

std_resid_df = pd.DataFrame(std_resid, index=contingency.index,
                            columns=contingency.columns)

# 3) Heatmap â€” display nicely the multiindex
plt.figure(figsize=(10,6))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap="RdYlGn", center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Income Group or Credit Score Group vs Loan Status", weight="bold", fontsize=14, pad=20)
plt.ylabel("Income Group or Credit Score Group | Credit Score")
plt.xlabel("Loan Status")
plt.tight_layout()
plt.show()


bins_dti = [0, 0.15, 0.30, 0.45, 1]
labels_dti = ["Low", "Medium", "High", "Very High"]
df_bq["DTI_group"] = pd.cut(df_bq["debt_to_income_ratio"], bins=bins_dti, labels=labels_dti, include_lowest=True)

bins_rate = [3.2, 8.0, 12.0, 16.0, 21.0]
labels_rate = ["Low", "Moderate", "High", "Very High"]
df_bq["Rate_group"] = pd.cut(df_bq["interest_rate"], bins=bins_rate, labels=labels_rate, include_lowest=True)

df_bq["DTI_Rate_combo"] = df_bq["DTI_group"].astype(str) + " | " + df_bq["Rate_group"].astype(str)

combo_summary = (df_bq.groupby("DTI_Rate_combo")["loan_paid_back"].value_counts(normalize=True).unstack(fill_value=0)* 100)

plt.figure(figsize=(10,6))
sns.heatmap(
    combo_summary[["Not paid"]],
    annot=True, fmt=".1f", cmap="Reds", cbar_kws={'label': '% Default'},
)
plt.title("Default Rate (%) by DTI and Interest Rate Combination", fontsize=14, weight="bold")
plt.xlabel("")
plt.ylabel("DTI_Rate_combo")
plt.show()


df_bq_education_employment = pd.crosstab(
    [df_bq["education_level"], df_bq["employment_status"]],
    df_bq["loan_paid_back"],
    normalize="index"
) * 100

display(df_bq_education_employment)

contingency = pd.crosstab(
    [df_bq["education_level"], df_bq["employment_status"]],
    df_bq["loan_paid_back"]
)
chi2, p, dof, ex = chi2_contingency(contingency)
print("Chi2:", chi2, "p-value:", p)

# 1) Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

print(f"Chi2: {chi2:.4f} | dof: {dof} | p-value: {p:.6f}")

# 2) Standardized residuals (adjusted)
#    r_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - row_prob_i) * (1 - col_prob_j))
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom

std_resid_df = pd.DataFrame(std_resid, index=contingency.index, columns=contingency.columns)

# 3) Heatmap â€” display nicely the multiindex
plt.figure(figsize=(11,7))
sns.heatmap(std_resid_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"label": "Standardized Residual"})
plt.title("Standardized Residuals Heatmap: Education or Employment vs Loan Status", weight="bold", fontsize=14, pad=20)
plt.ylabel("Education or Employment | Loan Status")
plt.xlabel("Loan Status")
plt.tight_layout()
plt.show()


df_bq_loan_purpose_rate_group = pd.crosstab(
    [df_bq["loan_purpose"], df_bq["Rate_group"]],
    df_bq["loan_paid_back"],
    normalize="index"
) * 100

display(df_bq_loan_purpose_rate_group)

contingency = pd.crosstab(
    [df_bq["loan_purpose"], df_bq["Rate_group"]],
    df_bq["loan_paid_back"]
)
chi2, p, dof, ex = chi2_contingency(contingency)
print("Chi2:", chi2, "p-value:", p)

# 1) Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

print(f"Chi2: {chi2:.4f} | dof: {dof} | p-value: {p:.6f}")

# 2) Standardized residuals (adjusted)
#    r_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - row_prob_i) * (1 - col_prob_j))
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom

std_resid_df = pd.DataFrame(std_resid, index=contingency.index, columns=contingency.columns)

# 3) Heatmap â€” display nicely the multiindex
plt.figure(figsize=(11,8))
sns.heatmap(std_resid_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"label": "Standardized Residual"})
plt.title("Standardized Residuals Heatmap: Loan Purpose or Rate Group vs Loan Status", weight="bold", fontsize=14, pad=20)
plt.ylabel("Loan Purpose or Rate Group | Loan Status")
plt.xlabel("Loan Status")
plt.tight_layout()
plt.show()


df_bq_rate_credit_score_group = pd.crosstab(
    [df_bq["Rate_group"], df_bq["credit_score_group"]],
    df_bq["loan_paid_back"],
    normalize="index"
) * 100

display(df_bq_rate_credit_score_group)

contingency = pd.crosstab(
    [df_bq["Rate_group"], df_bq["credit_score_group"]],
    df_bq["loan_paid_back"]
)
chi2, p, dof, ex = chi2_contingency(contingency)
print("Chi2:", chi2, "p-value:", p)

# 1) Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

print(f"Chi2: {chi2:.4f} | dof: {dof} | p-value: {p:.6f}")

# 2) Standardized residuals (adjusted)
#    r_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - row_prob_i) * (1 - col_prob_j))
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom

std_resid_df = pd.DataFrame(std_resid, index=contingency.index, columns=contingency.columns)

# 3) Heatmap â€” display nicely the multiindex
plt.figure(figsize=(11,8))
sns.heatmap(std_resid_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"label": "Standardized Residual"})
plt.title("Standardized Residuals Heatmap: Rate Group or Credit Score Group vs Loan Status", weight="bold", fontsize=14, pad=20)
plt.ylabel("Rate Group or Credit Score Group | Loan Status")
plt.xlabel("Loan Status")
plt.tight_layout()
plt.show()


df_bq_rate_dti_credit_group = pd.crosstab(
    [df_bq["Rate_group"], df_bq["DTI_group"], df_bq["credit_score_group"]],
    df_bq["loan_paid_back"],
    normalize="index"
) * 100

display(df_bq_rate_dti_credit_group)

contingency = pd.crosstab(
    [df_bq["Rate_group"], df_bq["DTI_group"], df_bq["credit_score_group"]],
    df_bq["loan_paid_back"]
)
chi2, p, dof, ex = chi2_contingency(contingency)
print("Chi2:", chi2, "p-value:", p)

# 1) Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

print(f"Chi2: {chi2:.4f} | dof: {dof} | p-value: {p:.6f}")

# 2) Standardized residuals (adjusted)
#    r_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - row_prob_i) * (1 - col_prob_j))
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom

std_resid_df = pd.DataFrame(std_resid, index=contingency.index, columns=contingency.columns)

# 3) Heatmap â€” display nicely the multiindex
plt.figure(figsize=(15,15))
sns.heatmap(std_resid_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"label": "Standardized Residual"})
plt.title("Standardized Residuals Heatmap: Rate Group or DTI Group or Credit Score Group vs Loan Status", weight="bold", fontsize=14, pad=20)
plt.ylabel("Rate Group or DTI Group or Credit Score Group | Loan Status")
plt.xlabel("Loan Status")
plt.tight_layout()
plt.show()


df_bq_gender_marital_status = pd.crosstab(
    [df_bq["gender"], df_bq["marital_status"]],
    df_bq["loan_paid_back"],
    normalize="index"
) * 100

display(df_bq_gender_marital_status)

contingency = pd.crosstab(
    [df_bq["gender"], df_bq["marital_status"]],
    df_bq["loan_paid_back"]
)
chi2, p, dof, ex = chi2_contingency(contingency)
print("Chi2:", chi2, "p-value:", p)

# 1) Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

print(f"Chi2: {chi2:.4f} | dof: {dof} | p-value: {p:.6f}")

# 2) Standardized residuals (adjusted)
#    r_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - row_prob_i) * (1 - col_prob_j))
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom

std_resid_df = pd.DataFrame(std_resid, index=contingency.index, columns=contingency.columns)

# 3) Heatmap â€” display nicely the multiindex
plt.figure(figsize=(11,8))
sns.heatmap(std_resid_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"label": "Standardized Residual"})
plt.title("Standardized Residuals Heatmap: Gender or Marital Status vs Loan Status", weight="bold", fontsize=14, pad=20)
plt.ylabel("Gender or Marital Status | Loan Status")
plt.xlabel("Loan Status")
plt.tight_layout()
plt.show()


df_bq_rate_dti_credit_group = pd.crosstab(
    [df_bq["education_level"], df_bq["employment_status"]],
    df_bq["grade_category"],
    normalize="index"
) * 100

display(df_bq_rate_dti_credit_group)

contingency = pd.crosstab(
    [df_bq["education_level"], df_bq["employment_status"]],
    df_bq["grade_category"]
)
chi2, p, dof, ex = chi2_contingency(contingency)
print("Chi2:", chi2, "p-value:", p)

# 1) Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

print(f"Chi2: {chi2:.4f} | dof: {dof} | p-value: {p:.6f}")

# 2) Standardized residuals (adjusted)
#    r_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - row_prob_i) * (1 - col_prob_j))
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom

std_resid_df = pd.DataFrame(std_resid, index=contingency.index, columns=contingency.columns)

# 3) Heatmap â€” display nicely the multiindex
plt.figure(figsize=(15,15))
sns.heatmap(std_resid_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"label": "Standardized Residual"})
plt.title("Standardized Residuals Heatmap: Education or Employment vs Grade Category", weight="bold", fontsize=14, pad=20)
plt.ylabel("Education or Employment | Grade Category")
plt.xlabel("Grade Category")
plt.tight_layout()
plt.show()


df_bq_group = pd.crosstab(
    [df_bq["education_level"], df_bq["annual_income_group"], df_bq["loan_purpose"]],
    df_bq["Rate_group"],
    normalize="index"
) * 100

display(df_bq_group)

contingency = pd.crosstab(
    [df_bq["education_level"], df_bq["annual_income_group"], df_bq["loan_purpose"]],
    df_bq["Rate_group"]
)
chi2, p, dof, ex = chi2_contingency(contingency)
print("Chi2:", chi2, "p-value:", p)

# 1) Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

print(f"Chi2: {chi2:.4f} | dof: {dof} | p-value: {p:.6f}")

# 2) Standardized residuals (adjusted)
#    r_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - row_prob_i) * (1 - col_prob_j))
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom

std_resid_df = pd.DataFrame(std_resid, index=contingency.index, columns=contingency.columns)

# 3) Heatmap â€” display nicely the multiindex
plt.figure(figsize=(15,35))
sns.heatmap(std_resid_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"label": "Standardized Residual"})
plt.title("Standardized Residuals Heatmap: Education or Annual Income or Loan Purpose vs Rate Group", weight="bold", fontsize=14, pad=20)
plt.ylabel("Education or Annual Income or Loan Purpose | Rate Group")
plt.xlabel("Rate Group")
plt.tight_layout()
plt.show()


df_bq_group = pd.crosstab(
    [df_bq["grade_category"], df_bq["loan_purpose"], df_bq["employment_status"]],
    df_bq["loan_paid_back"],
    normalize="index"
) * 100

display(df_bq_group)

contingency = pd.crosstab(
    [df_bq["grade_category"], df_bq["loan_purpose"], df_bq["employment_status"]],
    df_bq["loan_paid_back"]
)
chi2, p, dof, ex = chi2_contingency(contingency)
print("Chi2:", chi2, "p-value:", p)

# 1) Chi-square test + expected counts
chi2, p, dof, expected = chi2_contingency(contingency.values)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

print(f"Chi2: {chi2:.4f} | dof: {dof} | p-value: {p:.6f}")

# 2) Standardized residuals (adjusted)
#    r_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - row_prob_i) * (1 - col_prob_j))
row_sums = contingency.sum(axis=1).values[:, None]        # shape (R,1)
col_sums = contingency.sum(axis=0).values[None, :]        # shape (1,C)
grand_total = contingency.values.sum()

row_prob = row_sums / grand_total               # R x 1
col_prob = col_sums / grand_total               # 1 x C

denom = np.sqrt(expected * (1 - row_prob) * (1 - col_prob))
std_resid = (contingency.values - expected) / denom

std_resid_df = pd.DataFrame(std_resid, index=contingency.index, columns=contingency.columns)

# 3) Heatmap â€” display nicely the multiindex
plt.figure(figsize=(15,35))
sns.heatmap(std_resid_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"label": "Standardized Residual"})
plt.title("Standardized Residuals Heatmap: Grade Category or Loan Purpose or Employment Status vs Loan Status", weight="bold", fontsize=14, pad=20)
plt.ylabel("Grade Category or Loan Purpose or Employment Status | Loan Status")
plt.xlabel("Loan Status")
plt.tight_layout()
plt.show()


# Ability to afford the loan
df_train["income_to_loan_ratio"] = df_train["annual_income"] / (df_train["loan_amount"] + 1)
df_test["income_to_loan_ratio"] = df_test["annual_income"] / (df_test["loan_amount"] + 1)

# Loan burden level
df_train["loan_burden_score"] = df_train["loan_amount"] / (df_train["annual_income"] + 1)
df_test["loan_burden_score"] = df_test["loan_amount"] / (df_test["annual_income"] + 1)

# Credit score normalized by income
df_train["credit_to_income"] = df_train["credit_score"] / (df_train["annual_income"] + 1)
df_test["credit_to_income"] = df_test["credit_score"] / (df_test["annual_income"] + 1)

# Additional interest cost
df_train["interest_burden"] = (df_train["interest_rate"]) * df_train["loan_amount"]
df_test["interest_burden"] = (df_test["interest_rate"]) * df_test["loan_amount"]

# Rate adjusted by credit strength
df_train["normalized_interest"] = (df_train["interest_rate"]) * df_train["credit_score"]
df_test["normalized_interest"] = (df_test["interest_rate"]) * df_test["credit_score"]

# Combined DTI Ã— credit risk
df_train["dti_credit_ratio"] = df_train["debt_to_income_ratio"] * (850 - df_train["credit_score"])
df_test["dti_credit_ratio"] = df_test["debt_to_income_ratio"] * (850 - df_test["credit_score"])

# Income relative to credit
df_train["income_credit_ratio"] = df_train["annual_income"] / (df_train["credit_score"] + 1)
df_test["income_credit_ratio"] = df_test["annual_income"] / (df_test["credit_score"] + 1)

# Loan size relative to rating
df_train["loan_credit_ratio"] = df_train["loan_amount"] / (df_train["credit_score"] + 1)
df_test["loan_credit_ratio"] = df_test["loan_amount"] / (df_test["credit_score"] + 1)

# Estimated disposable income
df_train["free_income_est"] = df_train["annual_income"] * (1 - df_train["debt_to_income_ratio"])
df_test["free_income_est"] = df_test["annual_income"] * (1 - df_test["debt_to_income_ratio"])

# Payment stress level
df_train["stress_score"] = (df_train["loan_amount"] / (df_train["annual_income"] + 1)) * df_train["debt_to_income_ratio"]
df_test["stress_score"] = (df_test["loan_amount"] / (df_test["annual_income"] + 1)) * df_test["debt_to_income_ratio"]

# Stress inflated by interest rate
df_train["risk_pressure"] = df_train["stress_score"] * df_train["interest_rate"]
df_test["risk_pressure"] = df_test["stress_score"] * df_test["interest_rate"]

# Credit score adjusted for leverage
df_train["adjusted_credit"] = df_train["credit_score"] - (df_train["debt_to_income_ratio"] * 200)
df_test["adjusted_credit"] = df_test["credit_score"] - (df_test["debt_to_income_ratio"] * 200)

# Credit quality per loan size
df_train["credit_to_loan"] = df_train["credit_score"] / (df_train["loan_amount"] + 1)
df_test["credit_to_loan"] = df_test["credit_score"] / (df_test["loan_amount"] + 1)


num_features = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate", "income_to_loan_ratio", 
                "loan_burden_score", "credit_to_income", "interest_burden", "normalized_interest", "dti_credit_ratio", 
                "income_credit_ratio", "loan_credit_ratio", "free_income_est", "stress_score", "risk_pressure", "adjusted_credit", "credit_to_loan"]
new_num_features = ["income_to_loan_ratio", "loan_burden_score", "credit_to_income", "interest_burden", "normalized_interest", "dti_credit_ratio", 
                "income_credit_ratio", "loan_credit_ratio", "free_income_est", "stress_score", "risk_pressure", "adjusted_credit", "credit_to_loan"]
cat_features = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]

for feature in new_num_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:green;'><b>Distribution of {feature} by Loan Status</b></h2>"))
    plot_numerical_distribution(feature=feature, df = df_train)


plot_correlation(df_train=df_train.drop(columns="loan_paid_back", axis=1),
                 df_test=df_test)


skew_feature_train, skew_train_df = check_skewness(df_train, "Train Data", numerical_features=num_features)


skew_feature_test, skew_test_df = check_skewness(df_test, "Test Data", numerical_features=num_features)


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


processed_train_df, transformed_columns, sparse_columns, skewed_columns = handle_skewed_features(df=df_train, num_features=skew_feature_train)
num_features = ["PT_annual_income", "PT_debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate", "PT_income_to_loan_ratio", 
                "PT_loan_burden_score", "PT_credit_to_income", "interest_burden", "normalized_interest", "PT_dti_credit_ratio", 
                "PT_income_credit_ratio", "loan_credit_ratio", "PT_free_income_est", "PT_stress_score", "PT_risk_pressure", "adjusted_credit", "PT_credit_to_loan"]
skew_feature_train, skew_train_df = check_skewness(processed_train_df, "Train Data", numerical_features=num_features)


processed_test_df, transformed_columns_test, sparse_columns_test, skewed_columns_test = handle_skewed_features(df=df_test, num_features=skew_feature_test)
skew_feature_test, skew_test_df = check_skewness(data=processed_test_df, numerical_features=num_features,
                                                   dataset_name= "Test data")


checking_outlier(list_feature=num_features, df=processed_train_df, dataset_name="Data")


checking_outlier(list_feature=num_features, df=processed_test_df, dataset_name="Data")


# Display information about the DataFrames
print("Train Data Info:")
processed_train_df.info()

print("\nTest Data Info:")
processed_test_df.info()


processed_train_df["loan_paid_back"] = processed_train_df["loan_paid_back"].map({"Not paid": 0, "Paid": 1}).astype(int)
# We need to update the data for the columns, this helps to reduce memory.
processed_train_df = processed_train_df.astype({
    "credit_score": "int16",
    "loan_amount": "float16",
    "interest_rate": "float16",
    "loan_paid_back": "int8",
    "interest_burden": "float32",
    "normalized_interest": "float16",
    "loan_credit_ratio": "float16",
    "adjusted_credit": "float16",
    "PT_loan_burden_score": "float16",
    "PT_dti_credit_ratio": "float16",
    "PT_credit_to_income": "float16",
    "PT_free_income_est": "float16",
    "PT_annual_income": "float16",
    "PT_credit_to_loan": "float16",
    "PT_stress_score": "float16",
    "PT_income_credit_ratio": "float16",
    "PT_income_to_loan_ratio": "float16",
    "PT_risk_pressure": "float16"
})

processed_test_df = processed_test_df.astype({
    "credit_score": "int16",
    "loan_amount": "float16",
    "interest_rate": "float16",
    "interest_burden": "float32",
    "normalized_interest": "float16",
    "loan_credit_ratio": "float16",
    "adjusted_credit": "float16",
    "PT_loan_burden_score": "float16",
    "PT_dti_credit_ratio": "float16",
    "PT_credit_to_income": "float16",
    "PT_free_income_est": "float16",
    "PT_annual_income": "float16",
    "PT_credit_to_loan": "float16",
    "PT_stress_score": "float16",
    "PT_income_credit_ratio": "float16",
    "PT_income_to_loan_ratio": "float16",
    "PT_risk_pressure": "float16"
})

# Display information about the DataFrames
print("Train Data Info:")
processed_train_df.info()

print("\nTest Data Info:")
processed_test_df.info()


processed_train_df["credit_score_cat"] = pd.qcut(processed_train_df["credit_score"],
                                              q=4,
                                              labels=[1, 2, 3, 4])

plt.figure(figsize=(8, 5))
sns.histplot(data=processed_train_df, x="credit_score_cat", color="lightblue", edgecolor="black")
sns.despine(top=True, right=True, left=False, bottom=False)
plt.title("Distribution of credit_score_cat", fontsize=14, weight="bold",pad=20)
plt.xlabel("credit_score_cat", fontsize=12)
plt.ylabel("")
plt.tight_layout()
plt.show()


split = StratifiedShuffleSplit(n_splits=Config.N_SPLIT, test_size=Config.TEST_SIZE, 
                               random_state=Config.SEED)
for train_index, val_index in split.split(processed_train_df, processed_train_df["credit_score_cat"]):
    start_train_set = processed_train_df.loc[train_index]
    start_val_set = processed_train_df.loc[val_index]

# Now we should remove the credit_score_cat attribute so the data is back to its original state:
for set_ in (start_train_set, start_val_set): 
    set_.drop("credit_score_cat", axis=1, inplace=True)

df_train_new = start_train_set.drop("loan_paid_back", axis=1)
df_train_label = start_train_set["loan_paid_back"].copy()


list_standard = ["PT_debt_to_income_ratio", "credit_score", "PT_loan_burden_score", "PT_credit_to_income", "PT_stress_score", "PT_risk_pressure", 
                "PT_credit_to_loan"]

list_robust = ["loan_amount", "interest_rate", "PT_income_to_loan_ratio", "interest_burden", "PT_annual_income",
               "normalized_interest", "PT_dti_credit_ratio", "PT_income_credit_ratio", "loan_credit_ratio", "PT_free_income_est", "adjusted_credit"]

standard_transfomer = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("imputer", SimpleImputer(strategy="median"))
])

robust_transfomer = Pipeline(steps=[
    ("scaler", RobustScaler()),
    ("imputer", SimpleImputer(strategy="median"))
])

cat_transfomer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num_standard", standard_transfomer, list_standard),
        ("num_robust", robust_transfomer, list_robust),
        ("cat", cat_transfomer, cat_features),
    ]
)

preprocessor.fit(df_train_new)

df_train_new_prepared = preprocessor.transform(df_train_new)
list_feature_prepared = preprocessor.get_feature_names_out().tolist()
clean_features = [col.replace("num_standard__", "").replace("num_robust__", "").replace("cat__", "").replace("PT_", "") for col in list_feature_prepared]


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


def plot_ROC_confusionMatrix(estimator, X_val, y_val, figsize):
    y_pred_prob = estimator.predict_proba(X_val)[:, 1]  # Probability of positive class
    y_pred = estimator.predict(X_val)

    fig, ax = plt.subplots(nrows=2, ncols=2, sharey=False, figsize=figsize)

    # Plot 1
    # Calculate ROC
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    rocScore = roc_auc_score(y_val, y_pred_prob)

    ax[0, 0].plot(fpr, tpr, label=f"{estimator.__class__.__name__} (AUC = {rocScore:.2f})")
    ax[0, 0].plot([0, 1], [0, 1], "b--")
    ax[0, 0].set_xlabel("False Positive Rate")
    ax[0, 0].set_ylabel("True Positive Rate")
    ax[0, 0].set_title(f"ROC ({estimator.__class__.__name__})", fontsize=14, weight="bold", pad=20)
    ax[0, 0].legend()

    # Plot 2
    confusionMatrix = confusion_matrix(y_val, y_pred)
    sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="Blues", ax=ax[0, 1])
    ax[0, 1].set_title(f"Confusion Matrix ({estimator.__class__.__name__})", fontsize=14, weight="bold", pad=20)
    ax[0, 1].set_xlabel("Prediction")
    ax[0, 1].set_ylabel("Actual")

    # plot 3
    avg_prec = average_precision_score(y_val, y_pred_prob)   
    precision, recall, thresholds_pr = precision_recall_curve(y_val, y_pred_prob)
    ax[1, 0].plot(recall, precision, label=f"PR Curve (AP = {avg_prec:.3f})")
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, weight="bold", pad=20)
    ax[1, 0].legend()

    ax.flat[-1].set_visible(False)

    plt.tight_layout()
    plt.show()

    print(classification_report(y_val, y_pred))


# Function to evaluate models
def evaluate_model(model, X_train, X_val, y_train, y_val, figsize = (15, 6), show_shap_plot = False):
    print(f"Evaluating {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    plot_ROC_confusionMatrix(estimator = model, X_val = X_val, y_val = y_val, figsize = figsize)


X_val = start_val_set.drop("loan_paid_back", axis=1)
y_val = start_val_set["loan_paid_back"].copy()
X_val_prepared = preprocessor.transform(X_val)


import xgboost as xgb

param_xgb = {
"lambda": 0.0028334499645967606, 
"alpha": 6.173470071867061, 
"max_depth": 3, 
"eta": 0.11987979274427926, 
"subsample": 0.9194731846804935, 
"colsample_bytree": 0.6174129520077346, 
"min_child_weight": 5, 
"gamma": 0.9449756382275054, 
"n_estimators": 1800,
"n_jobs": -1,
"verbosity": 0,
"random_state": Config.SEED,
"use_label_encoder": False,
"objective": "binary:logistic",
"eval_metric": "auc",
"tree_method": "hist",
"booster": "gbtree"
}

model_xgb = xgb.XGBClassifier(**param_xgb)

evaluate_model(model = model_xgb, X_train=df_train_new_prepared, X_val=X_val_prepared,
               y_train=df_train_label, y_val=y_val, figsize=(15, 10))



from catboost import CatBoostClassifier

# After running optuna
param_cb = {
"iterations": 1383, 
"learning_rate": 0.22775461488679877, 
"depth": 5, 
"l2_leaf_reg": 7.46314929623761, 
"random_strength": 1.5904542174434636e-05, 
"bagging_temperature": 0.03502831981387006, 
"border_count": 252,
"loss_function": "Logloss",
"eval_metric": "AUC",
"verbose": 0,
"random_seed": Config.SEED,
"bootstrap_type": "Bayesian",
"thread_count": -1,
"grow_policy": "Lossguide"
}

model_cb = CatBoostClassifier(**param_cb)
evaluate_model(model = model_cb, X_train=df_train_new_prepared, X_val=X_val_prepared,
               y_train=df_train_label, y_val=y_val, figsize=(15, 10))


from lightgbm import LGBMClassifier

params_lgbm = {
"objective": "binary",
"metric": "binary_logloss",
"boosting_type": "gbdt",
"num_leaves": 31,
"learning_rate": 0.0322942967545754,
"feature_fraction": 0.6236144085285287,
"bagging_fraction": 0.9596685778433888,
"bagging_freq": 3,
"max_depth": 15,
"min_child_samples": 20,
"subsample": 0.782964614940435,
"colsample_bytree": 0.7330716143099598,
"reg_alpha": 0.24890188410341635,
"reg_lambda": 0.004657445631362826,
"random_state": Config.SEED,
"verbose": -1,
"n_jobs": -1,
"n_estimators": 3000
}
model_lgbm = LGBMClassifier(**params_lgbm)

evaluate_model(model = model_lgbm, X_train=df_train_new_prepared, X_val=X_val_prepared,
               y_train=df_train_label, y_val=y_val, figsize=(15, 10))


# Collect predictions (probabilities instead of labels) ---
ests = [("cb", model_cb), ("xgb", model_xgb), ("lgbm", model_lgbm)]

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


from sklearn.ensemble import VotingClassifier

voting_clf_soft = VotingClassifier(estimators=[("cb", model_cb), ("lgbm", model_lgbm), ("xgb", model_xgb)], weights=weights, voting="soft", n_jobs=-1)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

cv_scores = cross_val_score(voting_clf_soft, X=df_train_new_prepared, y=df_train_label, cv=kfold, scoring="roc_auc",  n_jobs=-1)
print(f"Cross-validated ROC-AUC (mean Â± std): {cv_scores.mean():.4f} Â± {cv_scores.std():.4f}")

evaluate_model(model = voting_clf_soft, X_train=df_train_new_prepared, X_val=X_val_prepared,
               y_train=df_train_label, y_val=y_val, figsize=(15, 10))


df_test_prepared = preprocessor.transform(processed_test_df)

# Generate predicted probabilities for the test set
y_pred_test_prob_cat = voting_clf_soft.predict_proba(df_test_prepared)
loan_status = y_pred_test_prob_cat[:, 1]

# Prepare submission file
submission = pd.DataFrame({
    "id": list_test_id,
    "loan_paid_back": loan_status
})

submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved!")
submission.head()


# Plot distribution of predicted probabilities
plt.figure(figsize=(10, 6))
sns.histplot(loan_status, bins=30, kde=True)
plt.title("Distribution of Predicted Loan Paid Back Probabilities", weight="bold", pad=15, fontsize=12)
plt.xlabel("Predicted Probability of Loan Paid Back")
sns.despine(left=False, bottom=False, right=False)
plt.ylabel("Frequency")
plt.xlim(0, 1)  # Limit x-axis to [0, 1]
plt.show()


# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
binary_predictions = (loan_status > 0.5).astype(int)

# Plot distribution of binary predictions
plt.figure(figsize=(8, 5))
sns.countplot(x=binary_predictions.flatten(), palette= "RdYlGn")
plt.title("Distribution of Predicted Loan Paid Back", weight="bold", pad=15, fontsize=12)
plt.xlabel("Loan Paid Back (0: Not paid, 1: Paid)")
plt.ylabel("")
sns.despine(left=False, bottom=False)
plt.xticks(ticks=[0, 1], labels=["Not paid", "Paid"])
plt.show()


shap_plot(model=voting_clf_soft.named_estimators_["xgb"], X_test=df_test_prepared[:3000], list_feature=clean_features, type="bar")


shap_plot(model=voting_clf_soft.named_estimators_["cb"], X_test=df_test_prepared[:3000], list_feature=clean_features)

