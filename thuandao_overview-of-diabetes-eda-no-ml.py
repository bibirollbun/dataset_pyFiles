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
from sklearn.preprocessing import OneHotEncoder, RobustScaler
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
                             precision_recall_curve, average_precision_score)

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
    train_csv = "/kaggle/input/playground-series-s5e12/train.csv"
    test_csv = "/kaggle/input/playground-series-s5e12/test.csv"
    target_feature = "diagnosed_diabetes"   


# Load the datasets
df_train = pd.read_csv(Config.train_csv)
df_test = pd.read_csv(Config.test_csv)

# Verify shapes
print("[i] Train Data Shape:", df_train.shape)
print("\n[i] Test Data Shape:", df_test.shape)


# Display few rows of each dataset
print("[i] Train Data Preview:")
display(df_train.head())

print("\n[i] Test Data Preview:")
display(df_test.head())


# Display information about the DataFrames
print("[i] Displaying Train Data Info:")
df_train.info()

print("\n[i] Displaying Test Data Info:")
df_test.info()

print("\n[âœ“] Completed displaying DataFrame info.")


df_train.drop("id", axis=1, inplace=True)
list_test_id = df_test["id"].copy().to_list()
df_test.drop("id", axis=1, inplace=True)


num_features = ["age", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day", 
                "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol_total", "hdl_cholesterol", 
                "ldl_cholesterol", "triglycerides"]
cat_features = ["alcohol_consumption_per_week", "gender", "ethnicity", "education_level", "income_level", "smoking_status", 
                "employment_status", "family_history_diabetes", "hypertension_history", "cardiovascular_history"]
print("[i] Train Data describe:")
cm = sns.light_palette("blue", as_cmap=True)
display(df_train[num_features].describe().T.style.background_gradient(cmap=cm))

print("\n[i] Test Data describe:")
display(df_test[num_features].describe().T.style.background_gradient(cmap=cm))


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
convert_cat(cat_features, df=df_test)

# Display information about the DataFrames
print("[i] Displaying Train Data Info:")
df_train.info()

print("\n[i] Displaying Test Data Info:")
df_test.info()


print("[i] Train Data describe:")
display(df_train[cat_features].describe().T.style.background_gradient(cmap="Blues", subset=["unique", "freq"]))

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

print("[i] Checking missing values for Train Set...")
displayNULL(df_train, dataset_name="Train Set")

print("\n[i] Checking missing values for Test Set...")
displayNULL(df_test, dataset_name="Test Set")

print("\n[âœ“] Completed checking for NULL values across datasets.")


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


df_plot = df_train.copy()
df_plot["diagnosed_diabetes"] = df_plot["diagnosed_diabetes"].map({
    0: "No Diabetes",
    1: "Diagnosed with Diabetes"
})

# Prepare data and colors
status_counts = df_plot["diagnosed_diabetes"].value_counts().sort_index()
order = status_counts.index.tolist()
colors = color(n_colors=len(order), tone="custom")
palette = dict(zip(order, colors))

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# --- Pie chart ---
ax[0].pie(status_counts, labels=order, colors=colors, autopct="%1.2f%%", startangle=150, shadow=True)
ax[0].set_title("Diabetes Diagnosis Proportion", fontweight="bold", fontsize=12, pad=15)

# --- Count plot ---
sns.countplot(data=df_plot, x="diagnosed_diabetes", order=order, palette=palette, ax=ax[1])
ax[1].set_title("Diabetes Diagnosis Count Distribution", fontweight="bold", fontsize=12, pad=15)
for container in ax[1].containers:
    ax[1].bar_label(container, fmt="%d", label_type="edge", fontsize=10)
ax[1].set(xlabel="Diabetes Status", ylabel="")
sns.despine(ax=ax[1])

plt.tight_layout()
plt.show()


def rgba_from_tuple(c):
    """Convert a tuple (r,g,b,a) floats (0â€“1) into rgba string."""
    r, g, b, a = c
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"

def plot_numerical_features_plotly(df_train, df_test, num_features):
    n = len(num_features)

    fig = make_subplots(
        rows=n,
        cols=2,
        subplot_titles=[
            f"<b>Histogram of {num_features[i//2]}</b>" if i % 2 == 0 else f"<b>Box Plot of {num_features[i//2]}</b>"
            for i in range(n * 2)
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.015
    )

    raw_colors = color(n_colors=2, tone="diverging")
    train_color = rgba_from_tuple(raw_colors[0])
    test_color  = rgba_from_tuple(raw_colors[1])

    for idx, feature in enumerate(num_features, start=1):

        # Histogram train
        fig.add_trace(
            go.Histogram(
                x=df_train[feature],
                name="Train data",
                opacity=0.6,
                marker_color=train_color,
                nbinsx=20
            ),
            row=idx, col=1
        )

        # Histogram test
        fig.add_trace(
            go.Histogram(
                x=df_test[feature],
                name="Test data",
                opacity=0.6,
                marker_color=test_color,
                nbinsx=20
            ),
            row=idx, col=1
        )

        # Box train
        fig.add_trace(
            go.Box(
                x=df_train[feature],
                name="Train data",
                marker_color=train_color,
                orientation="h"
            ),
            row=idx, col=2
        )

        # Box test
        fig.add_trace(
            go.Box(
                x=df_test[feature],
                name="Test data",
                marker_color=test_color,
                orientation="h"
            ),
            row=idx, col=2
        )
    fig.update_annotations(font=dict(family="Segoe UI", size=12, color="white"))
    fig.update_layout(
        height=350 * n,
        width=1200,
        title=dict(
            text="<b>Numerical Feature Comparison: Train vs Test<b>",
            font=dict(size=12, family="Segoe UI", color="white")
        ),
        showlegend=False,
        hovermode="x unified",
        template="plotly_dark"
    )

    # Remove gridlines for clean look
    fig.update_xaxes(showgrid=False, title_font=dict(size=12, color="white", family="Segoe UI"))
    fig.update_yaxes(showgrid=False, title_font=dict(size=12, color="white", family="Segoe UI"))
    fig.add_annotation(
        text="Created By Thuan Dao.",
        xref="paper", yref="paper",
        x=0.95, y=-0.015,
        showarrow=False,
        font=dict(size=10, color="white", family="Segoe UI")
    )

    fig.show()
# plot_numerical_features_plotly(df_train = df_train, df_test = df_test, num_features=num_features)
# Due to the extremely large output generated by this cell, the notebook exceeded Kaggleâ€™s IOPub output limits and triggered a timeout.
# To prevent this issue, the visualization has been simplified (or switched to a lighter rendering method) to reduce output size and ensure smooth execution.


def plot_numerical_features(df_train, df_test, num_features):
    colors = color(n_colors=2, tone="diverging")
    n = len(num_features)

    fig, ax = plt.subplots(n, 2, figsize=(12, n * 4))
    ax = np.array(ax).reshape(n, 2)

    for i, feature in enumerate(num_features):
        sns.histplot(data=df_train[feature], color=colors[0], bins=20, kde=True, ax=ax[i, 0], label="Train data")
        sns.histplot(data=df_test[feature], color=colors[1], bins=20, kde=True, ax=ax[i, 0], label="Test data")
        ax[i, 0].set_title(f"Histogram of {feature}", pad=15, weight="bold", fontsize=12)
        ax[i, 0].legend()
        ax[i, 0].set_ylabel("")
        sns.despine(left=False, bottom=False, ax=ax[i, 0])

        df_plot = pd.concat([
            pd.DataFrame({"Dataset": "Train data", feature: df_train[feature]}),
            pd.DataFrame({"Dataset": "Test data", feature: df_test[feature]})
        ]).reset_index(drop=True)

        sns.boxplot(data=df_plot, x=feature, y="Dataset", palette=colors, orient="h", ax=ax[i, 1])
        ax[i, 1].set_title(f"Horizontal Box plot of {feature}", pad=15, weight="bold", fontsize=12)
        ax[i, 1].set_ylabel("")
        sns.despine(left=False, bottom=False, ax=ax[i, 1])

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

    print(f"\n[i] Skewness for {dataset_name}...")
    print("-"*80)
    print(f"{'Feature':<40} | {'Skewness':<9} | {'Remark'}")
    print("-"*80)
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
            print(f"{color}{feature:<40} | {skew:>+9.4f} | {remark}{endc}")
            skew_feature.append(feature)
        else:
            print(f"{feature:<40} | {skew:>+9.4f} | {remark}")
    print("-"*80)
    return skew_feature, skew_df

skew_feature_train, skew_train_df = check_skewness(df_train, "Train Data")
skew_feature_test, skew_test_df = check_skewness(df_test, "Test Data")


def plot_correlation(df_train, df_test, train_name="Train Data", test_name="Test Data", figsize=(25, 15)):
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
    ax[0].set_title(f"Correlation Heatmap of {train_name}", fontsize=12, weight="bold", loc="center", pad=15)

    sns.heatmap(data=adjusted_cereal_corr_test, mask=adjusted_mask_test,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[1])
    ax[1].set_title(f"Correlation Heatmap of {test_name}", fontsize=12, weight="bold", loc="center", pad=15)
    plt.text(13,16, "Created Thuan Dao.", fontsize=10, color="black", fontweight="bold", style="italic")
    plt.tight_layout()
    plt.show()

plot_correlation(df_train=df_train.drop(columns="diagnosed_diabetes", axis=1),
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
    display(HTML(f"[âœ“] Categorical Feature Distribution: <b>{feature}</b>" + html))
    print("*" * 80)


# ----- Run the function for all categorical features -----
cat_features = ["alcohol_consumption_per_week", "gender", "ethnicity", "education_level", "income_level", "smoking_status", 
                "employment_status", "family_history_diabetes", "hypertension_history", "cardiovascular_history"]

for feature in cat_features:
    plot_categorical_distribution_across_datasets(df_train, df_test, feature)


def perform_statical_testing(feature: str, df: pd.DataFrame = df_plot,  target_feature: str = "diagnosed_diabetes") -> None:
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

def plot_numerical_distribution(feature: str, df: pd.DataFrame = df_plot,
                                target_feature: str = "diagnosed_diabetes", order: list = None) -> None:
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
                   palette=color(n_colors=df[target_feature].nunique(), tone="diverging"))
    
    plt.title(f"Violin plot of {feature} distribution by {target_feature}", pad=15, weight="bold")
    # plt.xlabel(target_feature, labelpad=10)
    plt.ylabel(feature, labelpad=10)
    plt.legend().remove()
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.show()

for feature in num_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {feature} by Diagnosed Diabetes</b></h2>"))
    plot_numerical_distribution(feature=feature, df = df_plot)


def alcohol_bin(x):
    if x == 0:
        return "0"
    if x == 1:
        return "1"
    elif x == 2:
        return "2"
    elif x == 3:
        return "3"
    elif x == 4:
        return "4"
    else:
        return ">=5"
        
df_train["alcohol_consumption_per_week"] = df_train["alcohol_consumption_per_week"].apply(alcohol_bin)
df_test["alcohol_consumption_per_week"] = df_test["alcohol_consumption_per_week"].apply(alcohol_bin)
df_plot["alcohol_consumption_per_week"] = df_plot["alcohol_consumption_per_week"].apply(alcohol_bin)

def bivariate_percent_plot(cat, target_feature, df, figsize=(15, 6), order=None):
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {cat} by {target_feature}</b></h2>"))
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=figsize)

    # Data processing
    grouped = df.groupby([cat, target_feature]).size().unstack(fill_value=0)

    # Define a fixed hue order (adjust if needed)
    target_order = [c for c in ["Diagnosed with Diabetes", "No Diabetes"] if c in grouped.columns]

    # Fallback if the labels are 0/1 or have different names
    if not target_order:
        target_order = list(grouped.columns)

    # Calculate percentages row-wise and reorder columns by target_order
    percentages = grouped.div(grouped.sum(axis=1), axis=0)[target_order] * 100

    # Define X-axis category order
    if order is not None:
        percentages = percentages.loc[order]
        labels = order
    else:
        labels = percentages.index.tolist()

    # Consistent color palette for both charts
    base_colors = color(n_colors=len(target_order), tone="diverging")
    color_map = dict(zip(target_order, base_colors))

    # Plot 1: Stacked bar chart (percentage)
    bottom = np.zeros(len(percentages))
    for cls in target_order:
        ax[0].bar(percentages.index, percentages[cls].values, bottom=bottom,
                  label=cls, color=color_map[cls])
        bottom += percentages[cls].values

    # Add percentage labels
    for container in ax[0].containers:
        ax[0].bar_label(container, fmt="%1.0f%%", label_type="center",
                        fontsize=10, color="black")

    ax[0].set_title(f"Diabetes Rate Across {cat.replace('_', ' ').title()}", fontsize=12, weight="bold", pad=15)
    ax[0].set_xlabel(f"{cat}", fontsize=10)
    ax[0].set_ylabel(f"% {target_feature} Rate", fontsize=10)
    sns.despine(left=False, bottom=False, ax=ax[0])
    ax[0].legend().remove()

    # Plot 2: Count plot (using same color_map + hue_order)
    sns.countplot(data=df, hue=target_feature, x=cat,
                  order=labels, hue_order=target_order,
                  palette=color_map, ax=ax[1])

    for container in ax[1].containers:
        ax[1].bar_label(container, fmt="%d", label_type="edge",
                        fontsize=10, color="black")

    ax[1].set_title(f"Customer Count by {cat.replace('_', ' ').title()} and Diabetes Status", fontsize=12, weight="bold", pad=15)
    ax[1].set_xlabel(f"{cat}", fontsize=10)
    ax[1].set_ylabel("Number of Customers", fontsize=10)
    ax[1].legend(title=target_feature, bbox_to_anchor=(1.05, 1), loc="upper left")
    sns.despine(left=False, bottom=False, ax=ax[1])

    plt.tight_layout()
    plt.show()

    # Chi-Square Test
    cal_ChiSquare(cat_feature=cat, target_feature=target_feature, df=df, show_residuals=True)

# Run for all categorical features
for feature in cat_features:
    bivariate_percent_plot(cat=feature, target_feature="diagnosed_diabetes", df=df_plot)


df_plot["diagnosed_diabetes_binary"] = df_plot["diagnosed_diabetes"].map({
    "No Diabetes": 0,
    "Diagnosed with Diabetes": 1
})

def bmi_bin(x):
    if x < 18.5:
        return "Underweight"
    if x >= 18.5 and x < 25:
        return "Normal"
    elif x >= 25 and x < 30:
        return "Overweight"
    else:
        return "Obese"
df_plot["bmi_group"] = df_plot["bmi"].apply(bmi_bin)

print("[i] Fitting logistic regression model...")
model_two = smf.logit(
    formula="diagnosed_diabetes_binary ~ bmi_group + family_history_diabetes",
    data=df_plot
).fit()

print(model_two.summary())

# Compute Odds Ratios (OR)
coef = model_two.params
OR = np.exp(coef)

OR_table = pd.concat([coef, OR], axis=1)
OR_table.columns = ["coef", "OR"]

print("\n[i] Odds Ratios:")
print(OR_table)


print("[i] Computing percentage table by BMI Group Ã— Family History Diabetes...")
df_group = pd.crosstab(
    [df_plot["bmi_group"], df_plot["family_history_diabetes"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (BMI Group Ã— Family History Diabetes) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["bmi_group"], df_plot["family_history_diabetes"]],
    df_plot["diagnosed_diabetes"]
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
plt.figure(figsize=(13,7))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: BMI Group or Family History Diabetes vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("BMI Group or Family History Diabetes | Diagnosed Diabetes")
plt.xlabel("")
plt.text(2.5, 10, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


def whr_group(row):
    whr = row["waist_to_hip_ratio"]
    sex = row["gender"]  # 'Male' or 'Female'

    if sex == "Female":
        if whr < 0.80:
            return "Low Risk"
        elif whr < 0.85:
            return "Moderate Risk"
        else:
            return "High Risk"

    else:  # Male
        if whr < 0.90:
            return "Low Risk"
        elif whr < 1.00:
            return "Moderate Risk"
        else:
            return "High Risk"

df_plot["whr_group"] = df_plot.apply(whr_group, axis=1)

def sbp_group(x):
    if x < 120:
        return "Normal"
    elif x < 130:
        return "Elevated"
    elif x < 140:
        return "Hypertension Stage 1"
    else:
        return "Hypertension Stage 2"

df_plot["sbp_group"] = df_plot["systolic_bp"].apply(sbp_group)

def tg_group(x):
    if x < 150:
        return "Normal"
    elif x < 200:
        return "Borderline High"
    elif x < 500:
        return "High"
    else:
        return "Very High"

df_plot["tg_group"] = df_plot["triglycerides"].apply(tg_group)

print("[i] Fitting logistic regression model...")
model_two = smf.logit(
    formula="diagnosed_diabetes_binary ~ whr_group + sbp_group + tg_group",
    data=df_plot
).fit()

print(model_two.summary())

# Compute Odds Ratios (OR)
coef = model_two.params
OR = np.exp(coef)

OR_table = pd.concat([coef, OR], axis=1)
OR_table.columns = ["coef", "OR"]

print("\n[i] Odds Ratios:")
print(OR_table)


print("[i] Computing percentage table by WHR Group Ã— Systolic BP Group Ã— Triglycerides Group...")
df_group = pd.crosstab(
    [df_plot["whr_group"], df_plot["sbp_group"], df_plot["tg_group"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (WHR Group Ã— Systolic BP Group Ã— Triglycerides Group) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["whr_group"], df_plot["sbp_group"], df_plot["tg_group"]],
    df_plot["diagnosed_diabetes"],
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
plt.figure(figsize=(15,13))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: WHR Group or Systolic BP Group or Triglycerides Group vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("WHR Group - Systolic BP Group - Triglycerides Group | Diagnosed Diabetes")
plt.xlabel("")
plt.text(2.5,40, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


def age_group(x):
    if x < 35:
        return "Young Adult"
    elif x < 50:
        return "Middle Adult"
    elif x < 65:
        return "Older Adult"
    else:
        return "Senior"

df_plot["age_group"] = df_plot["age"].apply(age_group)

print("[i] Computing percentage table by Age Group Ã— Hypertension History Ã— Cardiovascular History...")
df_group = pd.crosstab(
    [df_plot["age_group"], df_plot["hypertension_history"], df_plot["cardiovascular_history"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Age Group Ã— Hypertension History Ã— Cardiovascular History) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["age_group"], df_plot["hypertension_history"], df_plot["cardiovascular_history"]],
    df_plot["diagnosed_diabetes"],
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
plt.figure(figsize=(15,13))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Age Group or Hypertension History or Cardiovascular History vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Age Group - Hypertension History - Cardiovascular History | Diagnosed Diabetes")
plt.xlabel("")
plt.text(2.2,17, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


def ldl_group(x):
    if x < 100:
        return "Optimal"
    elif x < 130:
        return "Near Optimal"
    elif x < 160:
        return "Borderline High"
    elif x < 190:
        return "High"
    else:
        return "Very High"

df_plot["ldl_group"] = df_plot["ldl_cholesterol"].apply(ldl_group)

print("[i] Fitting logistic regression model...")
model_two = smf.logit(
    formula="diagnosed_diabetes_binary ~ ldl_group + whr_group",
    data=df_plot
).fit()

print(model_two.summary())

# Compute Odds Ratios (OR)
coef = model_two.params
OR = np.exp(coef)

OR_table = pd.concat([coef, OR], axis=1)
OR_table.columns = ["coef", "OR"]

print("\n[i] Odds Ratios:")
print(OR_table)


print("[i] Computing percentage table by BMI Group Ã— WHR Group...")
df_group = pd.crosstab(
    [df_plot["bmi_group"], df_plot["whr_group"]],
    df_plot["ldl_group"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (BMI Group Ã— WHR Group) vs LDL...")
contingency = pd.crosstab(
    [df_plot["bmi_group"], df_plot["whr_group"]],
    df_plot["ldl_group"]
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
plt.figure(figsize=(15,13))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: BMI Group or WHR Group vs LDL", weight="bold", fontsize=12, pad=15)
plt.ylabel("BMI Group - WHR Group | LDL")
plt.xlabel("")
plt.text(7,13, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


def pa_group(x):
    if x < 30:
        return "Inactive"
    elif x < 150:
        return "Insufficiently Active"
    elif x < 300:
        return "Active"
    else:
        return "Highly Active"

df_plot["pa_group"] = df_plot["physical_activity_minutes_per_week"].apply(pa_group)

def screen_time_group(x):
    if x < 2:
        return "Low"
    elif x < 4:
        return "Moderate"
    elif x < 8:
        return "High"
    else:
        return "Very High"

df_plot["screen_time_group"] = df_plot["screen_time_hours_per_day"].apply(screen_time_group)

print("[i] Computing percentage table by Physical Activity Group Ã— Screen Time Group...")
df_group = pd.crosstab(
    [df_plot["pa_group"], df_plot["screen_time_group"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Physical Activity Group Ã— Screen Time Group) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["pa_group"], df_plot["screen_time_group"]],
    df_plot["diagnosed_diabetes"]
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
plt.figure(figsize=(12,10))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Physical Activity Group or Screen Time Group vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Physical Activity Group - Screen Time Group | Diagnosed Diabetes")
plt.xlabel("")
plt.text(2,17, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


def diet_group(x):
    if x < 3:
        return "Poor Diet"
    elif x < 6:
        return "Fair Diet"
    elif x < 8:
        return "Good Diet"
    else:
        return "Excellent Diet"

df_plot["diet_group"] = df_plot["diet_score"].apply(diet_group)

print("[i] Fitting logistic regression model...")
model_two = smf.logit(
    formula="diagnosed_diabetes_binary ~ diet_group + pa_group + bmi_group",
    data=df_plot
).fit()

print(model_two.summary())

# Compute Odds Ratios (OR)
coef = model_two.params
OR = np.exp(coef)

OR_table = pd.concat([coef, OR], axis=1)
OR_table.columns = ["coef", "OR"]

print("\n[i] Odds Ratios:")
print(OR_table)


print("[i] Computing percentage table by Diet Group Ã— Physical Activity Group Ã— BMI Group...")
df_group = pd.crosstab(
    [df_plot["diet_group"], df_plot["pa_group"], df_plot["bmi_group"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Diet Group Ã— Physical Activity Group Ã— BMI Group) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["diet_group"], df_plot["pa_group"], df_plot["bmi_group"]],
    df_plot["diagnosed_diabetes"]
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
plt.figure(figsize=(15,18))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Diet Group or Physical Activity Group or BMI Group vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Diet Group - Physical Activity Group - BMI Group | Diagnosed Diabetes")
plt.xlabel("")
plt.text(2.2,65.5, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


print("[i] Computing percentage table by Alcohol Consumption per Week Ã— BMI Group...")
df_group = pd.crosstab(
    [df_plot["alcohol_consumption_per_week"], df_plot["bmi_group"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Alcohol Consumption per Week Ã— BMI Group) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["alcohol_consumption_per_week"], df_plot["bmi_group"]],
    df_plot["diagnosed_diabetes"]
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
plt.figure(figsize=(15,10))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Alcohol Consumption per Week or BMI Group vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Alcohol Consumption per Week - BMI Group | Diagnosed Diabetes")
plt.xlabel("")
plt.text(0, 21, "Analysis:", fontsize=10, fontweight="bold", color="#333")
plt.text(0, 21.5, "BMI overwhelmingly determines diabetes risk, and alcohol consumption does not notably strengthen or weaken this relationship. The risk pattern stays the same at every alcohol level.", 
         fontsize=10, color="#666", wrap=True)
plt.text(2.2,22.5, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


def hr_group(x):
    if x < 60:
        return "Bradycardia"
    elif x < 80:
        return "Normal"
    elif x < 100:
        return "Elevated"
    else:
        return "Tachycardia"

df_plot["hr_group"] = df_plot["heart_rate"].apply(hr_group)

def sleep_group(x):
    if x < 6:
        return "Short Sleep"
    elif x <= 8:
        return "Recommended Sleep"
    else:
        return "Long Sleep"

df_plot["sleep_group"] = df_plot["sleep_hours_per_day"].apply(sleep_group)


print("[i] Computing percentage table by Sleep Group Ã— HR Group Ã— Employment Status...")
df_group = pd.crosstab(
    [df_plot["sleep_group"], df_plot["hr_group"], df_plot["employment_status"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Sleep Group Ã— HR Group Ã— Employment Status) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["sleep_group"], df_plot["hr_group"], df_plot["employment_status"]],
    df_plot["diagnosed_diabetes"]
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
plt.figure(figsize=(15,10))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Sleep Group or HR Group or Employment Status vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Sleep Group - HR Group - Employment Status | Diagnosed Diabetes")
plt.xlabel("")
plt.text(0, 40, "Analysis:", fontsize=10, fontweight="bold", color="#333")
plt.text(0, 41.5, "Sleep duration remains a weak risk factor: even when combined with heart rate and employment status, the main diabetes patterns are driven by HR and work status, while sleep only slightly amplifies or softens those effects.", 
         fontsize=10, color="#666", wrap=True)
plt.text(2.2,42.5, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


print("[i] Computing percentage table by Income Level Ã— Ethnicity...")
df_group = pd.crosstab(
    [df_plot["income_level"], df_plot["ethnicity"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Income Level Ã— Ethnicity) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["income_level"], df_plot["ethnicity"]],
    df_plot["diagnosed_diabetes"]
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
plt.figure(figsize=(15,10))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Income Level or Ethnicity vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Income Level - Ethnicity | Diagnosed Diabetes")
plt.xlabel("")
plt.text(0, 26.5, "Analysis:", fontsize=10, fontweight="bold", color="#333")
plt.text(0, 27.5, "Income level reshapes ethnic diabetes risk: low income magnifies risk for White and Asian groups, while middle income substantially protects Hispanicsâ€”showing that socioeconomic context is a key driver behind ethnic disparities.", 
         fontsize=10, color="#666", wrap=True)
plt.text(2.2,28.5, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


print("[i] Fitting logistic regression model...")
model_two = smf.logit(
    formula="diagnosed_diabetes_binary ~ education_level + income_level",
    data=df_plot
).fit()

print(model_two.summary())

# Compute Odds Ratios (OR)
coef = model_two.params
OR = np.exp(coef)

OR_table = pd.concat([coef, OR], axis=1)
OR_table.columns = ["coef", "OR"]

print("\n[i] Odds Ratios:")
print(OR_table)


print("[i] Computing percentage table by Education Level Ã— Income Level...")
df_group = pd.crosstab(
    [df_plot["education_level"], df_plot["income_level"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Education Level Ã— Income Level) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["education_level"], df_plot["income_level"]],
    df_plot["diagnosed_diabetes"]
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
plt.figure(figsize=(15,10))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Education Level or Income Level vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Education Level - Income Level | Diagnosed Diabetes")
plt.xlabel("")
plt.text(2.2,21.5, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


print("[i] Fitting logistic regression model...")
model_two = smf.logit(
    formula="diagnosed_diabetes_binary ~ employment_status + income_level",
    data=df_plot
).fit()

print(model_two.summary())

# Compute Odds Ratios (OR)
coef = model_two.params
OR = np.exp(coef)

OR_table = pd.concat([coef, OR], axis=1)
OR_table.columns = ["coef", "OR"]

print("\n[i] Odds Ratios:")
print(OR_table)


print("[i] Computing percentage table by Employment Level Ã— Income Level...")
df_group = pd.crosstab(
    [df_plot["employment_status"], df_plot["income_level"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Employment Level Ã— Income Level) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["employment_status"], df_plot["income_level"]],
    df_plot["diagnosed_diabetes"]
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
plt.figure(figsize=(15,10))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Employment Level or Income Level vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Employment Level - Income Level | Diagnosed Diabetes")
plt.xlabel("")
plt.text(2.2,22.5, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


print("[i] Computing percentage table by Alcohol Consumption per Week Ã— Education Level Ã— Income Level...")
df_group = pd.crosstab(
    [df_plot["alcohol_consumption_per_week"], df_plot["education_level"], df_plot["income_level"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Alcohol Consumption per Week Ã— Education Level Ã— Income Level) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["alcohol_consumption_per_week"], df_plot["education_level"], df_plot["income_level"]],
    df_plot["diagnosed_diabetes"]
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
plt.figure(figsize=(15,25))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Alcohol Consumption per Week or Education Level or Income Level vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Alcohol Consumption per Week - Education Level - Income Level | Diagnosed Diabetes")
plt.xlabel("")
plt.text(0, 102.5, "Analysis:", fontsize=10, fontweight="bold", color="#333")
plt.text(0, 104, "Alcohol consumption differs slightly across education and income groups, but these variations do not translate into meaningful differences in diabetes prevalence. Alcohol does not mediate the relationship between socioeconomic status and diabetes risk in this dataset.", 
         fontsize=10, color="#666", wrap=True)
plt.text(2.2,105.5, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


df_hypertension = df_plot[df_plot["hypertension_history"] == 1]
print("[i] Fitting logistic regression model...")
model_two = smf.logit(
    formula="diagnosed_diabetes_binary ~ bmi_group + ldl_group + tg_group",
    data=df_hypertension
).fit()

print(model_two.summary())

# Compute Odds Ratios (OR)
coef = model_two.params
OR = np.exp(coef)

OR_table = pd.concat([coef, OR], axis=1)
OR_table.columns = ["coef", "OR"]

print("\n[i] Odds Ratios:")
print(OR_table)


df_high_ldl = df_plot[df_plot["ldl_group"].isin(["Borderline High", "High", "Very High"])]
print("[i] Computing percentage table by Cardiovascular History Ã— High LDL...")
df_group = pd.crosstab(
    [df_high_ldl["hypertension_history"], df_high_ldl["ldl_group"]],
    df_high_ldl["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Cardiovascular History Ã— High LDL) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_high_ldl["hypertension_history"], df_high_ldl["ldl_group"]],
    df_high_ldl["diagnosed_diabetes"]
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
plt.figure(figsize=(12,10))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Cardiovascular History or High LDL vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Cardiovascular History - High LDL | Diagnosed Diabetes")
plt.xlabel("")
plt.text(0, 6.5, "Analysis:", fontsize=10, fontweight="bold", color="#333")
plt.text(0, 6.8, "Cardiovascular history and high LDL interact to amplify diabetes risk. Together, they produce significantly higher-than-expected diabetes rates, as shown by large positive residuals.", 
         fontsize=10, color="#666", wrap=True)
plt.text(2.2, 7.2, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


df_older = df_plot[df_plot["age_group"].isin(["Older Adult", "Senior"])]
print("[i] Computing percentage table by Age Group Ã— Family History Diabetes...")
df_group = pd.crosstab(
    [df_older["age_group"], df_older["family_history_diabetes"]],
    df_older["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Age Group Ã— Family History Diabetes) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_older["age_group"], df_older["family_history_diabetes"]],
    df_older["diagnosed_diabetes"]
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
plt.figure(figsize=(12,10))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Age Group or Family History Diabetes vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Age Group - Family History Diabetes | Diagnosed Diabetes")
plt.xlabel("")
plt.text(0, 4.3, "Analysis:", fontsize=10, fontweight="bold", color="#333")
plt.text(0, 4.4, "Family history of diabetes has a much stronger impact among Older Adults, but this impact becomes weaker among Seniors.", 
         fontsize=10, color="#666", wrap=True)
plt.text(2.2, 4.5, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


df_plot["young_high_bmi"] = (
    (df_plot["age_group"] == "Young Adult") &
    (df_plot["bmi_group"].isin(["Overweight", "Obese"]))
).astype(int)

df_plot["older_normal_bmi"] = (
    (df_plot["age_group"].isin(["Older Adult", "Senior"])) &
    (df_plot["bmi_group"] == "Normal")
).astype(int)

print("[i] Fitting logistic regression model...")
model_two = smf.logit(
    formula="diagnosed_diabetes_binary ~ young_high_bmi + older_normal_bmi",
    data=df_plot
).fit()

print(model_two.summary())

# Compute Odds Ratios (OR)
coef = model_two.params
OR = np.exp(coef)

OR_table = pd.concat([coef, OR], axis=1)
OR_table.columns = ["coef", "OR"]

print("\n[i] Odds Ratios:")
print(OR_table)


df_plot["healthy_lifestyle"] = (
    (df_plot["pa_group"].isin(["Active", "Highly Active"])) &
    (df_plot["diet_group"].isin(["Good Diet", "Excellent Diet"]))
).astype(int)

df_plot["unhealthy_lifestyle"] = (
    (df_plot["pa_group"].isin(["Insufficiently Active", "Inactive"])) &
    (df_plot["diet_group"].isin(["Poor Diet", "Fair Diet"]))
).astype(int)

print("[i] Fitting logistic regression model...")
model_two = smf.logit(
    formula="diagnosed_diabetes_binary ~ healthy_lifestyle + unhealthy_lifestyle",
    data=df_plot
).fit()

print(model_two.summary())

# Compute Odds Ratios (OR)
coef = model_two.params
OR = np.exp(coef)

OR_table = pd.concat([coef, OR], axis=1)
OR_table.columns = ["coef", "OR"]

print("\n[i] Odds Ratios:")
print(OR_table)


print("[i] Computing percentage table by Ethnicity Ã— PA Group Ã— Diet Group...")
df_group = pd.crosstab(
    [df_plot["ethnicity"], df_plot["pa_group"], df_plot["diet_group"]],
    df_plot["diagnosed_diabetes"],
    normalize="index"
) * 100
display(df_group)

print("[i] Running Chi-Square Test for combined factors (Ethnicity Ã— PA Group Ã— Diet Group) vs Diagnosed Diabetes...")
contingency = pd.crosstab(
    [df_plot["ethnicity"], df_plot["pa_group"], df_plot["diet_group"]],
    df_plot["diagnosed_diabetes"]
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
plt.figure(figsize=(12,15))
sns.heatmap(
    std_resid_df,
    annot=True, fmt=".2f", cmap=cmap, center=0,
    cbar_kws={"label": "Standardized Residual"}
)
plt.title("Standardized Residuals Heatmap: Ethnicity or PA Group or Diet Group vs Diagnosed Diabetes", weight="bold", fontsize=12, pad=15)
plt.ylabel("Ethnicity - PA Group - Diet Group | Diagnosed Diabetes")
plt.xlabel("")
plt.text(2.2, 83, "Created Thuan Dao.", fontsize=8, color="black", fontweight="bold", style="italic")
plt.tight_layout()
plt.show()


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


num_features = ["age", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day", 
                "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol_total", "hdl_cholesterol", 
                "ldl_cholesterol", "triglycerides"]
cat_features = ["alcohol_consumption_per_week", "gender", "ethnicity", "education_level", "income_level", "smoking_status", 
                "employment_status", "family_history_diabetes", "hypertension_history", "cardiovascular_history"]


processed_train_df, transformed_columns, sparse_columns, skewed_columns = handle_skewed_features(df=df_train, num_features=skew_feature_train)
num_features = ["age", "PT_physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day", 
                "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol_total", "hdl_cholesterol", 
                "ldl_cholesterol", "triglycerides"]
skew_feature_train, skew_train_df = check_skewness(processed_train_df, "Train Data", numerical_features=num_features)


processed_test_df, transformed_columns_test, sparse_columns_test, skewed_columns_test = handle_skewed_features(df=df_test, num_features=skew_feature_test)
skew_feature_test, skew_test_df = check_skewness(data=processed_test_df, numerical_features=num_features,
                                                   dataset_name= "Test data")


checking_outlier(list_feature=num_features, df=processed_train_df, dataset_name="Train Data")


checking_outlier(list_feature=num_features, df=processed_test_df, dataset_name="Test Data")


processed_train_df["bmi_cat"] = pd.qcut(processed_train_df["bmi"],
                                              q=5,
                                              labels=[1, 2, 3, 4, 5])

plt.figure(figsize=(8, 5))
sns.histplot(data=processed_train_df, x="bmi_cat", color="lightblue", edgecolor="black")
sns.despine(top=True, right=True, left=False, bottom=False)
plt.title("Distribution of bmi_cat", fontsize=14, weight="bold",pad=20)
plt.xlabel("bmi_cat", fontsize=12)
plt.ylabel("")
plt.tight_layout()
plt.show()


split = StratifiedShuffleSplit(n_splits=Config.n_split_shuffle, test_size=Config.test_size, 
                               random_state=Config.seed)
for train_index, val_index in split.split(processed_train_df, processed_train_df["bmi_cat"]):
    start_train_set = processed_train_df.loc[train_index]
    start_val_set = processed_train_df.loc[val_index]

# Now we should remove the bmi_cat attribute so the data is back to its original state:
for set_ in (start_train_set, start_val_set): 
    set_.drop("bmi_cat", axis=1, inplace=True)

df_train_new = start_train_set.drop("diagnosed_diabetes", axis=1)
df_train_label = start_train_set["diagnosed_diabetes"].copy()


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
        ("num_robust", robust_transfomer, num_features),
        ("cat", cat_transfomer, cat_features),
    ]
)

preprocessor.fit(df_train_new)

df_train_new_prepared = preprocessor.transform(df_train_new)
list_feature_prepared = preprocessor.get_feature_names_out().tolist()
clean_features = [col.replace("num_standard__", "").replace("num_robust__", "").replace("cat__", "").replace("PT_", "") for col in list_feature_prepared]
clean_features


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
        plt.title("SHAP Feature Importance", fontsize=12, weight="bold", pad=15)
        plt.tight_layout()
        plt.show()
    else:
        shap.summary_plot(shap_values, X_test_sample)


X_val = start_val_set.drop("diagnosed_diabetes", axis=1)
y_val = start_val_set["diagnosed_diabetes"].copy()
X_val_prepared = preprocessor.transform(X_val)


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

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=300
        )

        fold_models.append(model)

        # Predict fold
        fold_pred = model.predict_proba(
            X_valid,
            iteration_range=(0, model.best_iteration + 1)
        )[:, 1]

        oof_pred[valid_idx] = fold_pred

        # Test prediction
        test_pred += model.predict_proba(
            X_test,
            iteration_range=(0, model.best_iteration + 1)
        )[:, 1] / n_splits

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


param_xgb = {
"learning_rate": 0.010101790233963715, 
"max_depth": 4, 
"min_child_weight": 7.875908100225339, 
"subsample": 0.7225393932188394, 
"colsample_bytree": 0.5325708121965714, 
"gamma": 1.2582788478340508,
"lambda": 0.016947240752074988, 
"alpha": 7.335937487680093, 
"max_leaves": 123,
"booster": "gbtree",
"random_state": Config.seed,
"use_label_encoder": False,
"verbosity": 0,
"tree_method": "gpu_hist",
"predictor": "gpu_predictor",
"grow_policy": "lossguide",
"n_jobs": -1,
"n_estimators": 20000,
"max_bin": 256,
"objective": "binary:logistic",
"eval_metric": "auc"    
}

model_xgb = xgb.XGBClassifier(**param_xgb)

df_test_prepared = preprocessor.transform(processed_test_df)
oof_pred, test_pred, fold_models = run_oof(model = model_xgb, X = df_train_new_prepared, y = df_train_label.values, 
           X_test = df_test_prepared)


# Prepare submission file
submission = pd.DataFrame({
    "id": list_test_id,
    "diagnosed_diabetes": test_pred
})

submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved!")
submission.head()


# Plot distribution of predicted probabilities
plt.figure(figsize=(10, 6))
sns.kdeplot(oof_pred, fill=True, linewidth=1.5, alpha=0.2, label="OOF Predictions (Train)")
sns.kdeplot(test_pred, fill=True, linewidth=1.5, alpha=0.2, label="Test Predictions")
plt.title("KDE Distribution of Predicted Diagnosed Diabetes", weight="bold", pad=15, fontsize=12)
plt.xlabel("Predicted Probability of Diagnosed Diabetes")
sns.despine(left=False, bottom=False, right=False)
plt.ylabel("Frequency")
plt.xlabel("")
plt.xlim(0, 1)  # Limit x-axis to [0, 1]
plt.legend()
plt.tight_layout()
plt.show()


# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
binary_predictions = (test_pred > 0.5).astype(int)

# Plot distribution of binary predictions
plt.figure(figsize=(10, 6))
ax = sns.countplot(x=binary_predictions.flatten(), palette= "RdYlGn")
plt.title("Distribution of Predicted Diagnosed Diabetes", weight="bold", pad=15, fontsize=12)
plt.xlabel("Diagnosed Diabetes (0: No Diabetes, 1: Diagnosed with Diabetes)")
plt.ylabel("")
plt.xlabel("")
sns.despine(left=False, bottom=False)
plt.xticks(ticks=[0, 1], labels=["No Diabetes", "Diagnosed with Diabetes"])
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


shap_plot(model=model_xgb, X_test=df_test_prepared[:1000], list_feature=clean_features, type="bar")


shap_plot(model=model_xgb, X_test=df_test_prepared[:1000], list_feature=clean_features)

