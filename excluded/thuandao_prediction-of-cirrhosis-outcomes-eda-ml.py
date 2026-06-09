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


# Load the datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s3e26/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s3e26/test.csv")

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


num_features = ["N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides",
                "Platelets", "Prothrombin"]
print("Train Data describe:")
cm = sns.light_palette("green", as_cmap=True)
display(df_train[num_features].describe().T.style.background_gradient(cmap=cm))

print("\nTest Data describe:")
display(df_test[num_features].describe().T.style.background_gradient(cmap=cm))


cat_features = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
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
        cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
        residuals = (crosstab - expected) / np.sqrt(expected)
        print("\nğŸ“ˆ Standardized Residuals:")
        print(round(residuals, 2))

        # Heatmap of residuals
        plt.figure(figsize=(10, 5))
        sns.heatmap(residuals, annot=True, cmap=cmap, center=0, fmt=".2f", linewidths=0.5)
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
            .background_gradient(subset=["Value"], cmap="Blues")
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


# Prepare data and colors
status_counts = df_train["Status"].value_counts().sort_index()
order = status_counts.index.tolist()
colors = color(n_colors=len(order))
palette = dict(zip(order, colors))

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Highlight a specific label if needed
explode = [0.1 if lbl == "CL" else 0 for lbl in order]

# --- Pie chart ---
ax[0].pie(status_counts, labels=order, colors=colors, autopct="%1.2f%%", startangle=150, explode=explode, shadow=True)
ax[0].set_title("Status Distribution", fontweight="bold", fontsize=14, pad=20)

# --- Count plot ---
sns.countplot(data=df_train, x="Status", order=order, palette=palette, ax=ax[1])
ax[1].set_title("Count plot of Status", fontweight="bold", fontsize=14, pad=20)
for container in ax[1].containers:
    ax[1].bar_label(container, fmt="%d", label_type="edge", fontsize=10)
ax[1].set(xlabel="Status", ylabel="Frequency")
sns.despine(ax=ax[1])

plt.tight_layout()
plt.show()


def plot_numerical_features(df_train, df_test, num_features):
    colors = color(n_colors=2)
    n = len(num_features)

    fig, axes = plt.subplots(n, 2, figsize=(12, n * 4))
    axes = np.array(axes).reshape(n, 2)

    for i, feature in enumerate(num_features):
        sns.histplot(data=df_train[feature], color=colors[0], bins=20, kde=True, ax=axes[i, 0], label="Train data")
        sns.histplot(data=df_test[feature], color=colors[1], bins=20, kde=True, ax=axes[i, 0], label="Test data")
        axes[i, 0].set_title(f"Histogram of {feature}", pad=14, weight="bold")
        axes[i, 0].legend()
        axes[i, 0].set_ylabel("")
        # axes[i, 0].axvline(df_train[feature].median(), color="green", linestyle="--", label="Median Train")
        # axes[i, 0].axvline(df_test[feature].median(), color="orange", linestyle="--", label="Median Test")
        sns.despine(left=False, bottom=False, ax=axes[i, 0])

        df_plot = pd.concat([
            pd.DataFrame({"Dataset": "Train data", feature: df_train[feature]}),
            pd.DataFrame({"Dataset": "Test data", feature: df_test[feature]})
        ]).reset_index(drop=True)

        sns.boxplot(data=df_plot, x=feature, y="Dataset", palette=colors, orient="h", ax=axes[i, 1])
        axes[i, 1].set_title(f"Horizontal Box plot of {feature}", pad=14, weight="bold")
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


corr_matrix = df_train[num_features].corr(numeric_only=True)
# one_like can build a matrix of boolean(True, False) with the same shape as our data
ones_corr = np.ones_like(corr_matrix, dtype=bool)
mask = np.triu(ones_corr)
adjusted_mask = mask[1:, :-1]
adjusted_cereal_corr = corr_matrix.iloc[1:, :-1]

fig, ax = plt.subplots(figsize = (15, 10))
# That method uses HUSL colors, so you need hue, saturation, and lightness. 
# I used hsluv.org to select the colors of this chart.
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

sns.heatmap(data=adjusted_cereal_corr, mask=adjusted_mask,
            annot=True, fmt=".2f", cmap=cmap,
            vmin=-1, vmax=1, linecolor="white", linewidths=0.5)

title = "Correlation Matrix Composition\n"
ax.set_title(title, loc="center", fontsize=18, weight="bold")
plt.tight_layout()
plt.show()


def plot_categorical_distribution_across_datasets(train_data, test_data, feature, tone="diverging"):
    # ----- Unified order across both datasets -----
    order = (
        pd.Index(train_data[feature].dropna().unique())
        .union(pd.Index(test_data[feature].dropna().unique()))
        .tolist()
    )
    order = list(map(str, order))  # trÃ¡nh lá»—i khi cÃ³ mixed types
    # Map vá»� string cho váº½
    tdf = train_data.copy()
    vdf = test_data.copy()
    tdf[feature] = tdf[feature].astype(str)
    vdf[feature] = vdf[feature].astype(str)

    # ----- Consistent colors -----
    colors = color(n_colors=len(order), tone=tone)
    palette = dict(zip(order, colors))

    fig, ax = plt.subplots(2, 2, figsize=(18, 10))
    datasets = [(tdf, "Train"), (vdf, "Test")]

    # ----- Bar charts -----
    for i, (data, name) in enumerate(datasets):
        sns.countplot(
            data=data, x=feature, order=order, palette=palette, ax=ax[i, 0]
        )
        ax[i, 0].set_title(f"{name} Data: {feature} Counts", fontsize=12, pad=15, weight="bold")
        ax[i, 0].set_xlabel(feature)
        ax[i, 0].set_ylabel("Count")
        ax[i, 0].set_axisbelow(True)
        sns.despine(ax=ax[i, 0])

        # annotate Ä‘Ãºng cho cá»™t dá»�c
        for p in ax[i, 0].patches:
            height = int(p.get_height())
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax[i, 0].annotate(
                f"{height}", (x, y), ha="center", va="bottom", fontsize=10
            )

    # ----- Donut (pie) charts -----
    for i, (data, name) in enumerate(datasets):
        counts = data[feature].value_counts().reindex(order, fill_value=0)  # match order
        ax[i, 1].pie(
            counts.values,
            labels=order,
            autopct="%1.1f%%",
            startangle=90,
            colors=[palette[lbl] for lbl in order],   # match colors
            textprops={"fontsize": 12},
            radius=1.2,
            shadow=True,
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        ax[i, 1].add_artist(centre_circle)
        ax[i, 1].set_title(f"{name} Data: {feature} Distribution (%)",
                           fontsize=12, pad=15, weight="bold")
        ax[i, 1].axis("equal")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()

for feature in cat_features:
    plot_categorical_distribution_across_datasets(df_train, df_test, feature)


def perform_statical_testing(feature: str, df: pd.DataFrame = df_train,  target_feature: str = "Status") -> None:
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
                                target_feature: str = "Status", order: list = None) -> None:
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
    
    plt.title(f"Violin plot of {feature} distribution by {target_feature}", pad=15, weight="bold")
    plt.xlabel(target_feature, labelpad=10)
    plt.ylabel(feature, labelpad=10)
    plt.legend().remove()
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.show()

for feature in num_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:green;'><b>Distribution of {feature} by Status</b></h2>"))
    plot_numerical_distribution(feature=feature, df = df_train)


def bivariate_percent_plot(cat, target_feature, df, figsize=(15, 6), order=None):
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:green;'><b>Distribution of {cat} by {target_feature}</b></h2>"))
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=figsize)

    # === Data processing ===
    grouped = df.groupby([cat, target_feature]).size().unstack(fill_value=0)
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    if order is not None:
        percentages = percentages.loc[order]
        labels = order
    else:
        labels = percentages.index

    # === Use the same color palette as right chart ===
    palette = color(n_colors=df[target_feature].nunique())

    # Convert palette to dict for stacked bar color mapping
    color_map = dict(zip(percentages.columns, palette))

    # === Plot 1: Stacked bar chart (percentage) ===
    bottom = np.zeros(len(percentages))
    for fert in percentages.columns:
        ax[0].bar(percentages.index, percentages[fert], bottom=bottom, label=fert, color=color_map[fert])
        bottom += percentages[fert].values

    # Add percentage labels
    for container in ax[0].containers:
        ax[0].bar_label(container, fmt="%1.0f%%", label_type="center", fontsize=9, color="black", weight="bold")

    ax[0].set_title(f"Percentage of {target_feature} by {cat}", fontsize=14, weight="bold")
    ax[0].set_xlabel(f"{cat}", fontsize=12)
    ax[0].set_ylabel(f"% {target_feature} Rate", fontsize=12)
    ax[0].set_xticklabels(labels=labels)
    sns.despine(left=False, bottom=False, ax=ax[0])
    ax[0].legend().remove()

    # === Plot 2: Count plot ===
    sns.countplot(data=df, hue=target_feature, x=cat, palette=palette, ax=ax[1], order=labels)
    for container in ax[1].containers:
        ax[1].bar_label(container, fmt="%d", label_type="edge", fontsize=10, weight="bold")
    ax[1].set_title(f"{target_feature} by {cat}", fontsize=14, weight="bold")
    ax[1].set_xlabel(f"{cat}", fontsize=12)
    ax[1].set_ylabel("Number of Customer", fontsize=12)
    ax[1].legend(title=target_feature, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax[1].set_xticklabels(labels=ax[1].get_xticklabels())
    sns.despine(left=False, bottom=False, ax=ax[1])
    plt.tight_layout()
    plt.show()

    cal_ChiSquare(cat_feature=cat, target_feature=target_feature, df=df, show_residuals=True)

for feature in cat_features:
    bivariate_percent_plot(cat=feature, target_feature= "Status",df= df_train)


skew_feature_train, skew_train_df = check_skewness(df_train, "Train Data")
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


processed_train_df, transformed_columns, sparse_columns, skewed_columns = handle_skewed_features(df=df_train, num_features=skew_feature_train)
num_features = ["N_Days", "Age", "PT_Bilirubin", "PT_Cholesterol", "PT_Albumin", "PT_Copper", "PT_Alk_Phos", "PT_SGOT", "PT_Tryglicerides",
                "Platelets", "PT_Prothrombin"]
skew_feature_train, skew_train_df = check_skewness(processed_train_df, "Train Data", numerical_features=num_features)


processed_test_df, transformed_columns_test, sparse_columns_test, skewed_columns_test = handle_skewed_features(df=df_test, num_features=skew_feature_test)
skew_feature_test, skew_test_df = check_skewness(data=processed_test_df, numerical_features=num_features,
                                                   dataset_name= "Test data")


checking_outlier(list_feature=num_features, df=processed_train_df, dataset_name="Data")


checking_outlier(list_feature=num_features, df=processed_test_df, dataset_name="Data")


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

processed_train_df["Status"] = le.fit_transform(processed_train_df["Status"])

label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("ğŸ”¹ Status â†’ Status mapping:")
for name, code in label_mapping.items():
    print(f"{name:>10}  â†’  {code}")


processed_train_df["PT_Albumin_cat"] = pd.qcut(processed_train_df["PT_Albumin"],
                                              q=4,
                                              labels=[1, 2, 3, 4])

plt.figure(figsize=(8, 5))
sns.histplot(data=processed_train_df, x="PT_Albumin_cat", color="lightblue", edgecolor="black")

plt.title("Distribution of PT_Albumin_cat", fontsize=14, pad=15, weight="bold")
plt.xlabel("PT_Albumin_cat", fontsize=12)
plt.ylabel("")
plt.tight_layout()
plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, val_index in split.split(processed_train_df, processed_train_df["PT_Albumin_cat"]):
    start_train_set = processed_train_df.loc[train_index]
    start_val_set = processed_train_df.loc[val_index]

# Now we should remove the PT_Albumin_cat attribute so the data is back to its original state:
for set_ in (start_train_set, start_val_set): 
    set_.drop("PT_Albumin_cat", axis=1, inplace=True)

df_train_new = start_train_set.drop("Status", axis=1)
df_train_label = start_train_set["Status"].copy()


# There are no **missing values** into dataset. But we will still handle missing values â€‹â€‹to check the data in the future.
num_transfomer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transfomer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("encoder", OneHotEncoder(handle_unknown="ignore")) # Handling Text and Categorical Attributes
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transfomer, num_features),
        ("cat", cat_transfomer, cat_features),
    ]
)

preprocessor.fit(df_train_new)

df_train_new_prepared = preprocessor.transform(df_train_new)
list_feature_prepared = preprocessor.get_feature_names_out().tolist()
clean_features = [col.replace("num__", "").replace("cat__", "").replace("PT_", "") for col in list_feature_prepared]
clean_features


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier)
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = [
    KNeighborsClassifier(),
    LogisticRegression(random_state=42, class_weight="balanced", max_iter=50000),
    DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    ExtraTreesClassifier(random_state=42, class_weight="balanced"),
    AdaBoostClassifier(random_state=42),
    XGBClassifier(random_state=42),
    MLPClassifier(random_state=42, max_iter=50000),
    GradientBoostingClassifier(random_state=42),
    CatBoostClassifier(verbose=0, random_seed=42, auto_class_weights="Balanced"),
    LGBMClassifier(random_state=42, verbosity=-1, class_weight="balanced"),
    HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
]


def generate_baseline_results(models=models, X=df_train_new_prepared, y=df_train_label,
                              metric="neg_log_loss", cv=kfold, plot_result=False):
    entries = []
    for model in models:
        model_name = getattr(model, "name", model.__class__.__name__)
        scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
        scores = -scores
        for fold_idx, s in enumerate(scores, start=1):
            entries.append((model_name, fold_idx, s))

    cv_df = pd.DataFrame(entries, columns=["model_name", "fold_id", "score"])

    summary = (cv_df.groupby("model_name")["score"]
                    .agg(Mean="mean", Std="std", N="size")
                    .sort_values("Mean", ascending=True))

    if plot_result:
        order = summary.index.tolist()
        plt.figure(figsize=(18, 8))
        sns.barplot(data=cv_df, x="model_name", y="score", order=order, errorbar=("sd"), palette="viridis")
        title_metric = metric.upper() if isinstance(metric, str) else "Score"
        nfolds = getattr(cv, "n_splits", "CV")
        plt.title(f"Baseline {title_metric} using {nfolds}-fold cross-validation", fontsize=14, weight="bold", pad=20)
        plt.xlabel("Model"); plt.ylabel(title_metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return summary

generate_baseline_results(plot_result = True)


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


from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score

def _get_scores(estimator, X):
    """Return y_score for plotting curves."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    elif hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    else:
        raise ValueError("Estimator does not support predict_proba or decision_function.")


def plot_ROC_confusionMatrix(estimator, X_val, y_val, figsize=(15, 6)):
    """
    Visualization layout:
    - [0,0]: ROC (Binary or Multiclass OvR)
    - [0,1]: Confusion Matrix
    - [1,0]: Precision-Recall (Binary or Multiclass OvR)
    - [1,1]: hidden
    """
    y_pred = estimator.predict(X_val)
    classes = np.array(sorted(np.unique(y_val)))
    n_classes = len(classes)

    y_score = _get_scores(estimator, X_val)

    # ===== Normalize binary shape =====
    if n_classes == 2:
        if y_score.ndim == 1:
            pos_scores = y_score
        elif y_score.ndim == 2 and y_score.shape[1] == 2:
            pos_scores = y_score[:, 1]
        elif y_score.ndim == 2 and y_score.shape[1] == 1:
            pos_scores = y_score[:, 0]
        else:
            pos_scores = y_score[:, -1]
    else:
        if y_score.ndim == 1:
            raise ValueError("Multiclass requires 2D y_score (n_samples, n_classes).")
        if hasattr(estimator, "classes_"):
            order = [np.where(estimator.classes_ == c)[0][0] for c in classes]
            y_score = y_score[:, order]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    # ===== (1) ROC curve =====
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_val, pos_scores, pos_label=classes.max())
        roc_auc = roc_auc_score(y_val, pos_scores)
        ax[0, 0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax[0, 0].plot([0, 1], [0, 1], linestyle="--")
        ax[0, 0].set_title("ROC Curve (Binary)", fontsize=13, weight="bold", pad=14)
        roc_auc_macro = float(roc_auc)
    else:
        y_val_bin = label_binarize(y_val, classes=classes)
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        roc_auc_macro = float(np.mean([roc_auc[i] for i in range(n_classes)]))

        order = np.argsort([roc_auc[i] for i in range(n_classes)])
        picks = [order[0], order[n_classes // 2], order[-1]] if n_classes >= 3 else list(range(n_classes))
        for i in picks:
            ax[0, 0].plot(fpr[i], tpr[i], label=f"class {classes[i]} AUC = {roc_auc[i]:.3f}")

        ax[0, 0].plot([0, 1], [0, 1], linestyle="--")
        ax[0, 0].set_title(f"ROC Curve (Multiclass, macro AUC = {roc_auc_macro:.3f})", fontsize=13, weight="bold", pad=14)
    ax[0, 0].set_xlabel("False Positive Rate")
    ax[0, 0].set_ylabel("True Positive Rate")
    ax[0, 0].legend(loc="lower right")

    # ===== (2) Confusion Matrix =====
    cm = confusion_matrix(y_val, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0, 1],
                xticklabels=classes, yticklabels=classes)
    ax[0, 1].set_title("Confusion Matrix", fontsize=13, weight="bold", pad=14)
    ax[0, 1].set_xlabel("Predicted")
    ax[0, 1].set_ylabel("Actual")

    # ===== (3) Precisionâ€“Recall curve =====
    if n_classes == 2:
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, pos_scores, pos_label=classes.max())
        pr_ap = average_precision_score(y_val, pos_scores)
        ax[1, 0].plot(recall_curve, precision_curve, label=f"AP = {pr_ap:.3f}")
        pr_auc_macro = float(pr_ap)
        ax[1, 0].set_title("Precisionâ€“Recall (Binary)", fontsize=13, weight="bold", pad=14)
    else:
        y_val_bin = label_binarize(y_val, classes=classes)
        precision, recall, pr_ap = {}, {}, {}
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_val_bin[:, i], y_score[:, i])
            pr_ap[i] = average_precision_score(y_val_bin[:, i], y_score[:, i])

        pr_auc_macro = float(np.mean([pr_ap[i] for i in range(n_classes)]))

        order_pr = np.argsort([pr_ap[i] for i in range(n_classes)])
        picks_pr = [order_pr[0], order_pr[n_classes // 2], order_pr[-1]] if n_classes >= 3 else list(range(n_classes))
        for i in picks_pr:
            ax[1, 0].plot(recall[i], precision[i], label=f"class {classes[i]} AP = {pr_ap[i]:.3f}")

        ax[1, 0].set_title(f"Precisionâ€“Recall (Multiclass, macro AP = {pr_auc_macro:.3f})", fontsize=13, weight="bold", pad=14)

    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 0].legend(loc="lower left")

    # ===== (4) Hide empty cell =====
    ax.flat[-1].set_visible(False)
    plt.tight_layout()
    plt.show()

    # ===== Classification Report =====
    print(classification_report(y_val, y_pred, digits=3))

    metrics = {
        "roc_auc_macro": float(roc_auc_macro),
        "pr_auc_macro": float(pr_auc_macro),
        "accuracy": float(accuracy_score(y_val, y_pred))
    }
    return metrics


# ===== evaluate_model keeps the same API, adds return metrics =====
def evaluate_model(model, X_train, X_val, y_train, y_val, figsize=(15, 6), show_shap_plot=False):
    print(f"Evaluating {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    metrics = plot_ROC_confusionMatrix(estimator=model, X_val=X_val, y_val=y_val, figsize=figsize)

    if show_shap_plot:
        try:
            # Limit to 200 samples for faster SHAP computation
            shap_sample = X_val.iloc[:200] if hasattr(X_val, "iloc") else X_val[:200]
            shap_plot(model=model, X_test=shap_sample, list_feature=list_feature_prepared)
        except Exception as e:
            print("SHAP plot skipped:", e)

    return metrics


X_val = start_val_set.drop("Status", axis=1)
y_val = start_val_set["Status"].copy()
X_val_prepared = preprocessor.transform(X_val)


param_gbc = {
"n_estimators": 650, 
"learning_rate": 0.04884874232147335, 
"max_depth": 2, 
"min_samples_split": 6, 
"min_samples_leaf": 9, 
"subsample": 0.7400733676883955, 
"max_features": None,
"random_state": 42,
}

model_gbc = GradientBoostingClassifier(**param_gbc)
evaluate_model(model = model_gbc, X_train=df_train_new_prepared, X_val=X_val_prepared,
                               y_train=df_train_label, y_val=y_val, figsize=(15, 10))


param_lgbm = {
"n_estimators": 700, 
"learning_rate": 0.00584509334736526, 
"num_leaves": 185, 
"max_depth": 16, 
"min_child_samples": 8,
"subsample": 0.5319608915318002, 
"colsample_bytree": 0.5744567525383822, 
"reg_alpha": 2.948061606600429e-08, 
"reg_lambda": 1.8275933105098061,
"random_state": 42,
"objective": "multiclass",
"metric": "multi_logloss",
"n_jobs": -1,
"class_weight": "balanced",
"verbosity": -1
}

model_lgbm = LGBMClassifier(**param_lgbm)
evaluate_model(model = model_lgbm, X_train=df_train_new_prepared, X_val=X_val_prepared,
                               y_train=df_train_label, y_val=y_val, figsize=(15, 10))


param_cb = {
"bootstrap_type": "Bayesian", 
"iterations": 500, 
"depth": 8, 
"learning_rate": 0.04319007286588319, 
"l2_leaf_reg": 1.6775308558511295, 
"bagging_temperature": 0.6909789638528715, 
"random_strength": 1.4900742848944053, 
"border_count": 254, 
"grow_policy": "Depthwise",
"verbose": 0,
"random_seed": 42,
"eval_metric": "MultiClass",
"loss_function": "MultiClass",
"auto_class_weights":"Balanced",
"task_type": "CPU",
"thread_count": -1
}

model_cb = CatBoostClassifier(**param_cb)

evaluate_model(model = model_cb, X_train=df_train_new_prepared, X_val=X_val_prepared,
                               y_train=df_train_label, y_val=y_val, figsize=(15, 10))


# --- Collect predictions (probabilities for all classes) ---
ests = [("cb", model_cb), ("lgbm", model_lgbm), ("gbc", model_gbc)]

# Get full probability outputs (n_val, n_classes)
preds = {name: m.predict_proba(X_val_prepared) for name, m in ests}

# Compute log loss for each model (multiclass)
logloss_each = {name: log_loss(y_val, preds[name]) for name, _ in ests}
display(logloss_each)

# --- Stack along the 2nd axis (model dimension) ---
# A has shape (n_val, n_classes * n_models)
A = np.hstack([preds[name] for name, _ in ests])

n_models = len(ests)
n_classes = preds[ests[0][0]].shape[1]

# --- Objective function ---
def obj_w(trial):
    w = np.array([trial.suggest_float(f"w_{i}", 0.0, 5.0) for i in range(n_models)])
    if w.sum() == 0:
        return 1e6
    
    # Reshape into (n_val, n_models, n_classes)
    A_reshaped = A.reshape(len(y_val), n_models, n_classes)
    
    # Weighted average across the model axis
    y_hat = np.tensordot(A_reshaped, w / w.sum(), axes=(1, 0))
    
    return log_loss(y_val, y_hat)

# --- Optimize ---
study_w = optuna.create_study(direction="minimize")  # minimize log loss
study_w.optimize(obj_w, n_trials=1000, show_progress_bar=True)

# --- Result ---
w = np.array([study_w.best_params[f"w_{i}"] for i in range(n_models)])
weights = (w / w.sum()).tolist()

print("Best weights (normalized):", weights)
print("Best Log Loss:", study_w.best_value)


from sklearn.ensemble import VotingClassifier
voting_clf_soft = VotingClassifier(
    estimators=[("cb", model_cb), ("lgbm", model_lgbm), ("gbc", model_gbc)],
    voting="soft", weights=weights, n_jobs=-1)

cv_scores = cross_val_score(
    voting_clf_soft,
    X=df_train_new_prepared,
    y=df_train_label,
    cv=kfold,
    scoring="neg_log_loss",
    n_jobs=-1
)
print(f"Cross-validated Log Loss (mean Â± std): {-cv_scores.mean():.4f} Â± {-cv_scores.std():.4f}")


evaluate_model(model = voting_clf_soft, X_train=df_train_new_prepared, X_val=X_val_prepared,
                               y_train=df_train_label, y_val=y_val, figsize=(15, 10))


df_test_prepared = preprocessor.transform(processed_test_df)


# Generate predicted probabilities for the test set
y_pred_test_prob_cat = voting_clf_soft.predict_proba(df_test_prepared)
submission = pd.DataFrame(y_pred_test_prob_cat, columns=["Status_C", "Status_CL", "Status_D"])
submission["id"] = list_test_id
submission = submission[["id", "Status_C", "Status_CL", "Status_D"]]
submission.to_csv("submission.csv", index=False)
display(submission.head())


cols = ["Status_C", "Status_D", "Status_CL"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, col in zip(axes, cols):
    sns.histplot(submission[col], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}", weight="bold", pad=15, fontsize=12)
    ax.set_xlabel("Predicted Probability")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Frequency")

sns.despine(left=False, bottom=False, right=False)
plt.tight_layout()
plt.show()


shap_plot(model=voting_clf_soft.named_estimators_["cb"], X_test=df_test_prepared[:500], list_feature=clean_features, type="bar")

