!pip install statsmodels > pip_log_statsmodels.txt 2>&1
!pip install scikit_posthocs > pip_log_scikit_posthocs.txt 2>&1
!pip install imbalanced-learn==0.11.0 > pip_log_imbalanced.txt 2>&1


# Core data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Statistical functions
from scipy.stats import skew

# Display utilities for Jupyter notebooks
from IPython.display import display

# Machine learning preprocessing and modeling
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Metrics
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, auc

# Statistical
from scipy.stats import chi2_contingency
from scipy.stats import shapiro, probplot
from scipy.stats import mannwhitneyu
from scipy.stats import levene
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import kruskal
from scipy.stats import anderson
from scipy.stats import normaltest
import scikit_posthocs as sp

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500) # To display all the columns of dataframe
pd.set_option('max_colwidth', None) # To set the width of the column to maximum


# Load the datasets
# train.csv: Features and target labels
# test.csv: Features only
# personality_datasert.csv: Supplemental dataset for imputation and enrichment

df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
df_original = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv")

# Verify shapes
print("Train Data Shape:", df_train.shape)
print("\nTest Data Shape:", df_test.shape)
print("\nOriginal Data Shape:", df_original.shape)


# Display few rows of each dataset
print("Train Data Preview:")
display(df_train.head())

print("\nTest Data Preview:")
display(df_test.head())

print("\nOriginal Data Preview:")
display(df_original.head())


# Display information about the DataFrames
print("Train Data Info:")
df_train.info()

print("\nTest Data Info:")
df_test.info()

print("\nOriginal Data Info:")
df_original.info()


df_train.drop("id", axis=1, inplace=True)
list_test_id = df_test["id"].copy().to_list()
df_test.drop("id", axis=1, inplace=True)


print("Train Data describe:")
cm = sns.light_palette("blue", as_cmap=True)
display(df_train.describe().T.style.background_gradient(cmap=cm))

print("\nTest Data describe:")
display(df_test.describe().T.style.background_gradient(cmap=cm))

print("\nOriginal Data describe:")
display(df_original.describe().T.style.background_gradient(cmap=cm))


print("Train Data describe:")
df_train.drop(columns="Personality", axis=1).describe(include=["category", "object"]).T


print("\nTest Data describe:")
df_test.describe(include=["category", "object"]).T


print("\nOriginal Data describe:")
df_original.drop(columns="Personality", axis=1).describe(include=["category", "object"]).T


def displayNULL(df, dataset_name=None):
    total_rows = len(df)

    missing_df = df.isnull().sum().reset_index()
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

print("\nMissing value test dataset: ")
displayNULL(df_original, dataset_name="Original Set")


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
    "Test Data": df_test,
    "Original Data": df_original
}

duplicate_summary = {}
for name, data in datasets.items():
    check_duplicates_report(data, name)
    duplicate_summary[name] = {
        "duplicates": data.duplicated().sum(),
        "total_rows": len(data)
    }
    print()


df_original = df_original.drop_duplicates(["Time_spent_Alone", "Stage_fear", "Social_event_attendance",
                      "Going_outside", "Drained_after_socializing", 
                      "Friends_circle_size", "Post_frequency"])

datasets = {
    "Training Data": df_train,
    "Test Data": df_test,
    "Original Data": df_original
}

for name, data in datasets.items():
    check_duplicates_report(data, name)
    duplicate_summary[name] = {
        "duplicates": data.duplicated().sum(),
        "total_rows": len(data)
    }
    print()


num_features = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]
cat_features = ["Stage_fear", "Drained_after_socializing"]


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


checking_outlier(list_feature=num_features, df=df_original, dataset_name="Original data")


df_original = (df_original.rename(columns={"Personality": "match_p"}))


merge_cols = [col for col in df_original.columns if col != "match_p"]
df_train = df_train.merge(df_original, how="left", on=merge_cols)
df_test = df_test.merge(df_original, how="left", on=merge_cols)


# After Merging
print("\nNull values after merge (train):")
display(df_train.isnull().sum().to_frame("Missing Values"))
print("\nNull values after merge (test):")
display(df_test.isnull().sum().to_frame("Missing Values"))


print("\ntrain_df info:")
df_train.info()
print("\ntest_df info:")
df_test.info()


def color(n_colors=2):
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    positions = np.linspace(0, 1, n_colors)
    colors = [cmap(p) for p in positions]
    return colors


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
        plt.figure(figsize=(6, 4))
        sns.heatmap(residuals, annot=True, cmap=cmap, center=0, fmt=".2f", linewidths=0.5)
        plt.title(f"Standardized Residuals Heatmap: {cat_feature} vs {target_feature}")
        plt.ylabel(cat_feature)
        plt.xlabel(target_feature)
        plt.tight_layout()
        plt.show()
    else:
        pass

def cal_shapiro(cat_feature, num_feature, df, plot_result=False):
    """
    Perform the Shapiroâ€“Wilk test to assess normality of a numerical feature 
    within each group defined by a categorical feature.

    This function iterates through all unique, non-null values of a categorical variable,
    and applies the Shapiroâ€“Wilk test to the corresponding subgroup of the numeric variable.
    Optionally, it displays a Q-Q plot to visually assess the distribution.

    Parameters
    ----------
    cat_feature : str
        The name of the categorical column that defines the groups.

    num_feature : str
        The name of the numerical column to test for normality.

    df : pd.DataFrame
        The input DataFrame containing the data.

    plot_result : bool, optional (default=False)
        If True, displays a Q-Q plot for each group to visually assess normality.

    Returns
    -------
    None
        Prints the Shapiroâ€“Wilk test statistic and p-value for each group,
        and optionally shows a Q-Q plot.

    Notes
    -----
    - Hâ‚€ (null hypothesis): The data is normally distributed.
    - Hâ‚� (alternative): The data is not normally distributed.
    - If p > 0.05 â†’ fail to reject Hâ‚€ â†’ data appears normal.
    - If p â‰¤ 0.05 â†’ reject Hâ‚€ â†’ data likely not normal.
    - The test is not reliable for n > 5000 (as per scipy recommendation).
    - Requires at least 3 non-null values per group.

    References
    ----------
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
    - https://www.statskingdom.com/shapiro-wilk-test.html
    """

    print(f"ğŸ”� Shapiro-Wilk Normality Test for {num_feature} across {cat_feature} groups")

    for group in df[cat_feature].dropna().unique():
        data = df[df[cat_feature] == group][num_feature].dropna()
        n = len(data)

        if n < 3:
            print(f"âš ï¸� Group {group} has too few values ({n}) to perform Shapiro-Wilk test.")
        elif n > 5000:
            print(f"âš ï¸� Group {group} has {n} samples. Shapiro-Wilk may not be reliable for n > 5000.")
        else:
            stat, p = shapiro(data)
            print(f"Group: {group}")
            print(f"  Shapiro-Wilk statistic: {stat:.3f}")
            print(f"  p-value: {p}")
            if p > 0.05:
                print(f" ğŸŸ¢ Group '{group}' appears to follow a normal distribution.\n")
            else:
                print(f" âšª Group '{group}' does not appear to follow a normal distribution.\n")

            if plot_result:
                probplot(data, dist="norm", plot=plt)
                plt.title(f"QQ Plot - {group}")
                plt.show()
            else:
                pass

def cal_levene(dataframe, categorical_feature, num_feature, center="mean"):
    """
    Perform Leveneâ€™s test to assess the equality (homogeneity) of variances 
    for a numeric feature across two or more groups defined by a categorical feature.

    Levene's test is used to verify the assumption of equal variances 
    (homoscedasticity), which is important for parametric tests such as the 
    independent t-test and ANOVA.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing the features to test.

    categorical_feature : str
        The name of the categorical column that defines the grouping.

    num_feature : str
        The name of the numerical column whose variance is being compared across groups.

    center : str, optional (default="mean")
        Specifies the measure of central tendency to use when calculating deviations:
        - "mean": classic Levene's test (sensitive to non-normal data)
        - "median": more robust to non-normal distributions (Brownâ€“Forsythe test)

    Returns
    -------
    None
        Prints the Levene test statistic, p-value, and an interpretation of whether 
        the variances are equal or significantly different.

    Notes
    -----
    - Hâ‚€ (Null Hypothesis): All groups have equal variances.
    - Hâ‚� (Alternative Hypothesis): At least one group has different variance.
    - If p > 0.05 â†’ Fail to reject Hâ‚€ â†’ Variances are approximately equal.
    - If p â‰¤ 0.05 â†’ Reject Hâ‚€ â†’ Variances are significantly different (heteroscedasticity).

    References
    ----------
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html
    - https://www.geeksforgeeks.org/levenes-test-in-python/
    """

    print(f"ğŸ”� Leveneâ€™s test: {num_feature} ~ {categorical_feature}")
    # Extract unique group labels
    groups = dataframe[categorical_feature].unique()    
    # Create a list of values for each group
    data_groups = [dataframe[dataframe[categorical_feature] == g][num_feature] for g in groups]    
    # Perform Leveneâ€™s test
    stat, p = levene(*data_groups, center=center)
    
    print(f"Levene statistic: {stat:.3f}")
    print(f"p-value: {p}")
    if p > 0.05:
        print("ğŸŸ¢ Variances are approximately equal across groups.")
    else:
        print("âšª Variances are significantly different across groups.")

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
    print("Hâ‚€: The distributions of the two groups are equal.")
    print("Hâ‚�: The distributions are different.\n")

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


def t_test_with_cohens_d(data, categorical_feature, num_feature, equal_var = False):
    """
    Perform an Independent Two-Sample T-Test and compute Cohen's d to evaluate 
    the difference between two independent groups on a numeric variable.

    This function tests whether the means of two independent groups are statistically different,
    and also calculates the magnitude of the difference (effect size) using Cohen's d.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the categorical and numerical features.

    categorical_feature : str
        The name of the categorical column used to define the two groups (must have exactly 2 unique values).

    num_feature : str
        The name of the numerical feature to compare between the two groups.

    equal_var : bool, optional (default=False)
        Assumes equal population variance if True (Studentâ€™s t-test). If False (default), performs Welchâ€™s t-test.

    Returns
    -------
    None
        Prints the t-statistic, p-value, Cohenâ€™s d, and interpretation of the effect size.

    Notes
    -----
    - Hâ‚€ (null hypothesis): The two groups have equal means.
    - Hâ‚� (alternative): The means are significantly different.
    - Cohen's d interpretation:
        - 0.2  â†’ small effect
        - 0.5  â†’ medium effect
        - 0.8+ â†’ large effect
    - Welchâ€™s t-test is recommended when group variances are unequal (default setting).

    References
    ----------
    - https://www.scribbr.com/statistics/t-test/
    - https://en.wikipedia.org/wiki/Cohen%27s_d
    """

    # Extract unique groups
    groups = data[categorical_feature].dropna().unique()

    if len(groups) > 2:
        print(f"â�Œ Error: Independent T-Test requires 2 groups.")
        return
    else:
        print(f"ğŸ”� Independent T-Test: {num_feature} ~ {categorical_feature}")
        # Extract values
        x1 = data[data[categorical_feature] == groups[0]][num_feature].dropna()
        x2 = data[data[categorical_feature] == groups[1]][num_feature].dropna()

        # T-test (independent)
        t_stat, p_value = ttest_ind(x1, x2, equal_var=equal_var)  # Welchâ€™s t-test if variances may differ

        # Calculate Cohenâ€™s d
        nx1, nx2 = len(x1), len(x2)
        pooled_std = np.sqrt(((nx1 - 1)*np.var(x1, ddof=1) + (nx2 - 1)*np.var(x2, ddof=1)) / (nx1 + nx2 - 2))
        cohens_d = (np.mean(x1) - np.mean(x2)) / pooled_std

        # Output
        print(f"\nğŸ”� T-Test between group'{groups[0]}' and group '{groups[1]}':")
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_value:.6f}")

        if p_value < 0.05:
            print("\nâœ… Significant difference found (p < 0.05)")
            print(f"\nğŸ“� Cohen's d: {cohens_d:.3f}")            
            # Interpretation of Cohen's d
            if abs(cohens_d) < 0.2:
                size = "small"
            elif abs(cohens_d) < 0.5:
                size = "medium"
            else:
                size = "large"
            print(f"ğŸ§  Effect size interpretation: {size} effect")
        else:
            print("\nâ„¹ï¸� No significant difference found (p >= 0.05)")

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

def cal_Anderson(df, numerical_col, group_col):
    """
    Perform the Andersonâ€“Darling test to assess whether the data within each group 
    follows a normal distribution.

    This function applies the Anderson-Darling normality test for each subgroup 
    defined by a categorical column, and optionally plots KDE distributions.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    numerical_col : str
        The name of the numeric column to test for normality.

    group_col : str
        The name of the categorical column defining the groups to be tested separately.

    Returns
    -------
    None
        Prints the Andersonâ€“Darling test statistic, critical values, and interpretation
        for each group. Optionally displays a KDE plot for visual comparison.

    Notes
    -----
    - Hâ‚€ (null hypothesis): The data follows a normal distribution.
    - If test statistic > critical value â†’ â�Œ Reject Hâ‚€ â†’ Data is not normally distributed.
    - If test statistic â‰¤ critical value â†’ âœ… Fail to reject Hâ‚€ â†’ Data may be normal.
    - The test is more sensitive to deviations in the tails of the distribution than other tests like Shapiro-Wilk.

    Limitations
    ----------
    - Not recommended for very small sample sizes (< 8).
    - Not reliable for very large sample sizes where even small deviations may be flagged.

    References
    ----------
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
    - https://www.itl.nist.gov/div898/handbook/eda/section3/eda35e.htm
    - https://en.wikipedia.org/wiki/Andersonâ€“Darling_test
    """

    groups = df[group_col].dropna().unique()
    
    print(f"ğŸ“Š Checking normality of '{numerical_col}' across groups of '{group_col}' using Anderson-Darling Test:\n")
    print("â„¹ï¸�  Hâ‚€: The data follows a normal distribution.")
    print("â„¹ï¸�  If test statistic > critical value â†’ â�Œ Reject Hâ‚€ â†’ Not normally distributed.\n")

    for group in groups:
        data = df[df[group_col] == group][numerical_col].dropna()
        result = anderson(data, dist="norm")
        
        print(f"Group = {group}")
        print(f"  - Sample size: {len(data)}")
        print(f"  - Test statistic: {result.statistic:.4f}")
        
        for sl, cv in zip(result.significance_level, result.critical_values):
            verdict = "â�Œ Reject Hâ‚€ â†’ Not normal" if result.statistic > cv else "âœ… Fail to reject Hâ‚€ â†’ Possibly normal"
            print(f"    - Î± = {sl}% | CV = {cv:.4f} â†’ {verdict}")
        print()


def cal_normaltest(cat_feature, num_feature, df):
    """
    Perform Dâ€™Agostino and Pearsonâ€™s normality test on a numerical feature 
    across groups defined by a categorical feature.

    Parameters
    ----------
    cat_feature : str
        The name of the categorical column that defines the groups.

    num_feature : str
        The name of the numerical column to test for normality.

    df : pd.DataFrame
        The input DataFrame containing the data.

    Returns
    -------
    None
        Prints the test statistic and p-value for each group.

    Notes
    -----
    - Hâ‚€ (null hypothesis): The data is normally distributed.
    - Hâ‚� (alternative): The data is not normally distributed.
    - If p > 0.05 â†’ fail to reject Hâ‚€ â†’ data appears normal.
    - If p â‰¤ 0.05 â†’ reject Hâ‚€ â†’ data likely not normal.
    - Recommended for n â‰¥ 20, especially reliable for n > 50.
    - Requires at least 8 non-null values per group (as per scipy recommendation).
    """
    
    print(f"ğŸ”� Dâ€™Agostino and Pearson Normality Test for '{num_feature}' across '{cat_feature}' groups\n")

    for group in df[cat_feature].dropna().unique():
        data = df[df[cat_feature] == group][num_feature].dropna()
        n = len(data)

        print(f" Group: {group} (n = {n})")
        
        if n < 8:
            print(f"âš ï¸� Too few observations (< 8) to perform the test.\n")
            continue

        stat, p = normaltest(data)

        print(f"  Statistic : {stat:.3f}")
        print(f"  p-value   : {p:.5f}")
        
        if p > 0.05:
            print(f"  ğŸŸ¢ Interpretation: Data appears to follow a normal distribution.\n")
        else:
            print(f"  ğŸ”´ Interpretation: Data does not appear to follow a normal distribution.\n")


personality_Distribution  = df_train["Personality"].value_counts().loc[["Extrovert", "Introvert"]]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(15, 5))
# ax[0]
ax[0].pie(
    personality_Distribution,
    labels = ["Extrovert", "Introvert"],
    colors = color(n_colors=2),
    autopct = "%1.2f%%",
    startangle = 150,
    explode = (0, 0.08),
    shadow= True
)
ax[0].set_title("Personality Distribution", weight="bold", fontsize=14, pad=25)
ax[0].grid(axis="x", color="gray", linestyle=":", linewidth=0.7)

# ax[1]
sns.countplot(data=df_train, x = "Personality", palette=color(n_colors=2), ax=ax[1])
ax[1].set_title("Count plot of Personality Distribution", weight="bold", fontsize=14, pad=25)
for container in ax[1].containers:
    ax[1].bar_label(container, fmt="%d", label_type="edge", fontsize=10, weight="bold")
ax[1].set_ylabel("Number of People")
sns.despine(ax=ax[1], top=True, right=True, left=False, bottom=False)

plt.tight_layout()
plt.show()


def plot_numerical_features(df_train, df_test, num_features):
    colors = color()
    n = len(num_features)

    fig, axes = plt.subplots(n, 2, figsize=(12, n * 4))
    axes = np.array(axes).reshape(n, 2)

    for i, feature in enumerate(num_features):
        sns.histplot(data=df_train[feature], color=colors[0], bins=20, kde=True, ax=axes[i, 0], label="Train data")
        sns.histplot(data=df_test[feature], color=colors[1], bins=20, kde=True, ax=axes[i, 0], label="Test data")
        axes[i, 0].set_title(f"Histogram of {feature}")
        axes[i, 0].legend()
        # axes[i, 0].set_facecolor("lightgray")
        axes[i, 0].set_ylabel("")
        axes[i, 0].grid(color="gray", linestyle=":", linewidth=0.7)
        axes[i, 0].axvline(df_train[feature].median(), color="green", linestyle="--", label="Median Train")
        axes[i, 0].axvline(df_test[feature].median(), color="orange", linestyle="--", label="Median Test")
        sns.despine(left=False, bottom=False, ax=axes[i, 0])

        df_plot = pd.concat([
            pd.DataFrame({"Dataset": "Train data", feature: df_train[feature]}),
            pd.DataFrame({"Dataset": "Test data", feature: df_test[feature]})
        ]).reset_index(drop=True)

        sns.violinplot(
            data=df_plot,
            x=feature,
            y="Dataset",
            palette=colors,
            orient="h",
            ax=axes[i, 1]
        )
        axes[i, 1].set_title(f"Horizontal Violin plot of {feature}")
        # axes[i, 1].set_facecolor("lightgray")
        axes[i, 1].grid(color="gray", linestyle=":", linewidth=0.7)
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

skew_feature_train, skew_train_df = check_skewness(df_train, "Train data")
skew_feature_test, skew_test_df = check_skewness(df_test, "Test data")


def plot_correlation(train_data, test_data, train_name="Train Data", test_name="Test Data"):
    corr_train = train_data.corr(numeric_only=True)
    corr_test = test_data.corr(numeric_only=True)

    mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
    adjusted_mask_train = mask_train[1:, :-1]
    adjusted_cereal_corr_train = corr_train.iloc[1:, :-1]

    mask_test = np.triu(np.ones_like(corr_test, dtype=bool))
    adjusted_mask_test = mask_test[1:, :-1]
    adjusted_cereal_corr_test = corr_test.iloc[1:, :-1]

    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

    fig, ax = plt.subplots(1, 2, figsize=(24, 10))

    sns.heatmap(data=adjusted_cereal_corr_train, mask=adjusted_mask_train,
                annot=True, fmt=".1f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[0])
    ax[0].set_title(f"Correlation Heatmap of {train_name}", fontsize=16, weight="bold")

    sns.heatmap(data=adjusted_cereal_corr_test, mask=adjusted_mask_test,
                annot=True, fmt=".1f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[1])
    ax[1].set_title(f"Correlation Heatmap of {test_name}", fontsize=16, weight="bold")

    plt.tight_layout()
    plt.show()


plot_correlation(train_data=df_train.drop(columns="Personality", axis=1), test_data=df_test)


def plot_categorical_distribution_both(cat_features, df_train, df_test, order=None):
    for feature in cat_features:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 9))

        # COUNT PLOT â€“ TRAIN
        sns.countplot(data=df_train, y=feature, ax=ax[0, 0],
                      palette=color(n_colors=len(df_train[feature].unique())), order=order)
        ax[0, 0].set_title(f"[Train] Count plot of {feature}", fontsize=13, pad=12)
        ax[0, 0].set_ylabel(feature)
        ax[0, 0].set_xlabel("")
        ax[0, 0].grid(axis="x", linestyle=":", linewidth=0.7)
        sns.despine(ax=ax[0, 0], left=False, bottom=False)

        for p in ax[0, 0].patches:
            width = p.get_width()
            y = p.get_y() + p.get_height() / 2
            ax[0, 0].text(width + 0.01 * df_train[feature].value_counts().max(), y,
                        f"{int(width)}", ha="left", va="center", fontsize=9, fontweight="bold")
            
        # COUNT PLOT â€“ TEST
        sns.countplot(data=df_test, y=feature, ax=ax[0, 1], 
                      palette=color(n_colors=len(df_test[feature].unique())), order=order)
        ax[0, 1].set_title(f"[Test] Count plot of {feature}", fontsize=13, pad=12)
        ax[0, 1].set_ylabel(feature)
        ax[0, 1].set_xlabel("")
        ax[0, 1].grid(axis="x", linestyle=":", linewidth=0.7)
        sns.despine(ax=ax[0, 1], left=False, bottom=False)

        for p in ax[0, 1].patches:
            width = p.get_width()
            y = p.get_y() + p.get_height() / 2
            ax[0, 1].text(width + 0.01 * df_test[feature].value_counts().max(), y,
                        f"{int(width)}", ha="left", va="center", fontsize=9, fontweight="bold")
            
        # PIE CHART â€“ TRAIN
        train_counts = df_train[feature].value_counts().sort_index()
        wedges, texts, autotexts = ax[1, 0].pie(
            train_counts,
            labels=train_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=color(n_colors=len(df_train[feature].unique())),
            wedgeprops=dict(width=0.4, edgecolor="w"),
            radius=1.1
        )
        for t in texts + autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")
        ax[1, 0].set_title(f"[Train] Percentage Distribution of {feature}", pad=10)
        ax[1, 0].add_artist(plt.Circle((0, 0), 0.7, color="white"))
        ax[1, 0].axis("equal")

        # PIE CHART â€“ TEST
        test_counts = df_test[feature].value_counts().sort_index()
        wedges, texts, autotexts = ax[1, 1].pie(
            test_counts,
            labels=test_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=color(n_colors=len(df_test[feature].unique())),
            wedgeprops=dict(width=0.4, edgecolor="w"),
            radius=1.1
        )
        for t in texts + autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")
        ax[1, 1].set_title(f"[Test] Percentage Distribution of {feature}", pad=10)
        ax[1, 1].add_artist(plt.Circle((0, 0), 0.7, color="white"))
        ax[1, 1].axis("equal")

    plt.tight_layout()
    plt.show()


plot_categorical_distribution_both(cat_features=cat_features, df_train = df_train, df_test = df_test)


def top_ratio(df_test = df_test, df_train = df_train, cat_features = cat_features):
    dataset_names = ["Train", "Test"]
    datasets = [df_train, df_test]
    for i, (data, name) in enumerate(zip(datasets, dataset_names)):
        print(f"{name} Data")
        flagged = False
        for feature in cat_features:
            freq = data[feature].value_counts(normalize=True)
            top_ratio = freq.iloc[0]
            if top_ratio > 0.99:
                flagged = True
                print(f"âš ï¸�  {feature}: {top_ratio:.1%} lÃ  '{freq.index[0]}'")
        if not flagged:
            print("âœ… No feature has a category that makes up more than 99% of its values.")
        print("*" * 50)
top_ratio()


from IPython.core.display import HTML
def perform_statical_testing(feature, df_train = df_train, total_categories = 2, target_feature = "Personality"):
    cal_normaltest(cat_feature=target_feature, num_feature=feature, df=df_train)
    if total_categories == 2:
        cal_mannwhitneyu(dataframe=df_train, categorical_feature=target_feature, num_feature=feature)
    else:
        pass

def plot_numerical_distribution_by_Personality(feature, df_train = df_train, target_feature = "Personality", order = None):
    """
    Performs statical testing for each groups (distribution by target_feature) by ANOVA, T-test, Mann-Whitney U test,... <br>
    Draw violinplot and histogram to display the distribution for each groups of feature.
    Parameters:
        feature (str): The name of the column representing the numerical variable.
        df_train (pd.DataFrame): The input dataset.
        target_feature (str): The name of the column representing the target feature.
        order (list): Order items in plot.

    Returns:
        None
    """

    # Summary information
    df_summary_feature = df_train.groupby(by = target_feature, as_index= False)\
    .agg (
        Count = (feature, "count"),
        Mean = (feature, "mean"),
        Median = (feature, "median"),
        Std = (feature, "std")
    )
    df_summary_feature = df_summary_feature.sort_values(by="Mean", ascending=False)    

    summary_data = [
        ("Overall Mean", f"{df_train[feature].mean():.2f}"),
        ("Overall Median", f"{df_train[feature].median()}"),
        ("Overall Std", f"{df_train[feature].std():.2f}")
    ]
    summary_html = "<ul>" + "".join([f"<li><b>{k}:</b> {v}</li>" for k, v in summary_data]) + "</ul>"
    display(HTML(summary_html))
    display(df_summary_feature.style.background_gradient(cmap=cm).set_table_attributes('style="width:75%; margin:auto;"'))

    perform_statical_testing(feature=feature, df_train = df_train, target_feature=target_feature)

    # Plot distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    colors = [cmap(0.0), cmap(1.0)]
    sns.violinplot(x=target_feature, y=feature, data=df_train, hue=target_feature, palette=colors, ax=ax)
    ax.set_title(f"Violin plot of {feature} distribution by {target_feature}", pad=15, weight = "bold")
    ax.set_xlabel(target_feature, labelpad=10)
    ax.set_ylabel(feature, labelpad=10)
    plt.grid(axis="y", color="gray", linestyle=":", alpha=0.7)
    sns.despine(left=False, bottom=False, ax=ax)
    plt.tight_layout()
    plt.show()

for feature in num_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {feature} by Personality</b></h2>"))
    plot_numerical_distribution_by_Personality(feature=feature)


# defining function for plotting
def bivariate_percent_plot(cat, df, figsize=(15, 6), order = None, rot = 0):
    
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {cat} by Personality</b></h2>"))
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=figsize)
    # Plot 1
    # Calculate the total number of each "cat" by Personality
    grouped = df.groupby([cat, "Personality"]).size().unstack(fill_value=0)
    # Calculate the percentages
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    if order is not None:
        percentages = percentages.loc[order]
        labels = order
    else:
        labels = percentages.index
    
    percentages = percentages.reindex(columns=["Introvert", "Extrovert"])

    # That method uses HUSL colors, so you need hue, saturation, and lightness. 
    # I used hsluv.org to select the colors of this chart.
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    # Draw stacked bar plot
    ax[0] = percentages.plot(kind="bar", stacked=True, cmap=cmap, ax = ax[0], use_index=True)
    for container in ax[0].containers:
        ax[0].bar_label(container, fmt='%1.2f%%', label_type="center", weight="bold", fontsize=10)

    ax[0].set_title(f"Percentage of Personality by {cat}", fontsize=14, weight="bold")
    ax[0].set_xlabel(f"{cat}", fontsize=12)
    ax[0].set_ylabel("% Personality Rate", fontsize=12)
    ax[0].set_xticklabels(labels = labels, rotation = 0)
    ax[0].legend_.remove()
    # ax[0].grid(color="gray", linestyle=":", linewidth=0.7)
    sns.despine(left=False, bottom=False, ax=ax[0])

    # Plot 2
    sns.countplot(data=df, hue = "Personality", x = cat,
                palette=color(n_colors=2), ax=ax[1], order=order, hue_order = ["Introvert", "Extrovert"])
    # Show value for each bar.
    for container in ax[1].containers:
        ax[1].bar_label(container, fmt='%d', label_type="edge", fontsize=10, weight="bold")

    ax[1].set_title(f"Personality by {cat}", fontsize=14, weight="bold")
    ax[1].set_xlabel(f"{cat}", fontsize=12)
    ax[1].set_ylabel("Number of Customer", fontsize=12)
    ax[1].legend(title="Personality", bbox_to_anchor=(1.05, 1), loc="upper left")
    # ax[1].grid(color="gray", linestyle=":", linewidth=0.7)
    sns.despine(left=False, bottom=False, ax=ax[1])
    plt.tight_layout()
    plt.show()

    cal_ChiSquare(cat_feature=cat, target_feature="Personality", df=df, show_residuals=True)


for feature in cat_features:
    bivariate_percent_plot(cat=feature, df= df_train)


y_train = df_train.pop("Personality").map({"Extrovert": 0, "Introvert": 1}).values
ntrain = df_train.shape[0]
all_data = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
all_data.drop(columns="Personality", inplace=True, errors="ignore")


def fill_missing_by_quantile_group(df, group_source_col, target_col, quantiles=[0, 0.25, 0.5, 0.75, 1.0], labels=None):
    if labels is None:
        labels = [f"Q{i+1}" for i in range(len(quantiles) - 1)]
    bin_col = f"{group_source_col}_bin"
    df[bin_col] = pd.qcut(df[group_source_col], q=quantiles, labels=labels)
    df[target_col] = df[target_col].fillna(df.groupby(bin_col)[target_col].transform("median"))
    df.drop(columns=[bin_col], inplace=True)
    return df


# Time_spent_Alone
for source in ["Social_event_attendance", "Going_outside"]:
    all_data = fill_missing_by_quantile_group(all_data, source, "Time_spent_Alone")
print("Filled Time_spent_Alone missing values")
print(all_data["Time_spent_Alone"].isnull().value_counts(), "\n")

# Social_event_attendance
for source in ["Going_outside", "Friends_circle_size", "Post_frequency"]:
    all_data = fill_missing_by_quantile_group(all_data, source, "Social_event_attendance")
print("Filled Social_event_attendance missing values")
print(all_data["Social_event_attendance"].isnull().value_counts(), "\n")

# Friends_circle_size
for source in ["Post_frequency", "Going_outside", "Social_event_attendance"]:
    all_data = fill_missing_by_quantile_group(all_data, source, "Friends_circle_size")
print("Filled Friends_circle_size missing values")
print(all_data["Friends_circle_size"].isnull().value_counts(), "\n")

# Post_frequency
all_data = fill_missing_by_quantile_group(all_data, "Friends_circle_size", "Post_frequency")
all_data = fill_missing_by_quantile_group(all_data, "Time_spent_Alone", "Post_frequency")
print("Filled Post_frequency missing values")
print(all_data["Post_frequency"].isnull().value_counts(), "\n")

# Going_outside (final pass)
for source in ["Friends_circle_size", "Post_frequency"]:
    all_data = fill_missing_by_quantile_group(all_data, source, "Going_outside")
print("Final pass on Going_outside")
print(all_data["Going_outside"].isnull().value_counts(), "\n")

# Final fill for categorical columns
all_data.fillna({
    "Stage_fear": "Unknown",
    "Drained_after_socializing": "Unknown"
}, inplace=True)
print("Filled missing categorical values")
print(all_data[["Stage_fear", "Drained_after_socializing"]].isnull().sum(), "\n")

# One hot - encoding match_p
all_data = pd.get_dummies(all_data, columns=["match_p"], prefix=["match"])

# Verify There Are No Missing Values
print("Data Overview after Imputation:")
all_data.info()


def final_data_summary(df):
    total_rows = df.shape[0]
    summary = pd.DataFrame({
        "Feature": df.columns,
        "Count": df.count().values,
        "Missing Count": df.isnull().sum().values,
        "Missing %": (df.isnull().sum() / total_rows * 100).round(2).values,
        "Dtype": [str(dtype) for dtype in df.dtypes]
    })
    summary = summary[["Feature", "Count", "Missing Count", "Missing %", "Dtype"]]
    return summary

print("\nFinal Data Summary Check:")
final_summary_df = final_data_summary(all_data)
display(final_summary_df)

# Assert no missing values remain
assert all_data.isnull().sum().sum() == 0, "There are still missing values!"

print("\nNo missing values detected. Data is ready for modeling.")


#  Refer: https://www.kaggle.com/code/yeonseokcho/introvert-extrovert-knn-imputer/notebook#5.-Generation-features
# 1. Social media activity score
all_data["Social_Activity_Score"] = (all_data["Post_frequency"] * 
                                             all_data["Social_event_attendance"])

# 2. Product of social event attendance and going outside 
all_data["Event_Outside_Product"] = (all_data["Social_event_attendance"] * 
                                             all_data["Going_outside"])

# 3. Whether the person has many friends 
all_data["Many_Friends"] = (all_data["Friends_circle_size"] > 
                                    all_data["Friends_circle_size"].median()).astype(int)

# 4. Whether the person spends a lot of time alone
all_data["Much_Alone"] = (all_data["Time_spent_Alone"] > 
                                  all_data["Time_spent_Alone"].median()).astype(int)

# 5. Whether the person is an active poster on social media 
all_data["Active_Poster"] = (all_data["Post_frequency"] > 
                                    all_data["Post_frequency"].median()).astype(int)

# 6. Extroversion score 
all_data["Extroversion_Score"] = (all_data["Social_event_attendance"] + 
                                          all_data["Going_outside"] + 
                                          all_data["Friends_circle_size"] + 
                                          all_data["Post_frequency"] - 
                                          all_data["Time_spent_Alone"])


cat_features = ["Stage_fear", "Drained_after_socializing"]
def convert_cat(df, cat_features= cat_features):
    for feature in cat_features:
        if feature in df.columns:
            df[feature] = df[feature].astype("category")
        else:
            pass

convert_cat(df=all_data)
print("\nFinal Data Summary Check:")
final_summary_df = final_data_summary(all_data)
display(final_summary_df)


df_train = all_data[:ntrain]
df_test = all_data[ntrain:]
# X = X_train
# y = y_train
df_train["Personality"] = pd.Series(y_train).map({0: "Extrovert", 1: "Introvert"}).values


print("\nFinal Data Summary Check:")
final_summary_df = final_data_summary(df_train)
display(final_summary_df)


df_train.info()


list_new_features = ["Social_Activity_Score", "Event_Outside_Product", "Extroversion_Score"]
for feature in list_new_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {feature} by Personality</b></h2>"))
    plot_numerical_distribution_by_Personality(feature=feature, df_train=df_train)


list_new_cat_features = ["Many_Friends", "Much_Alone", "Active_Poster"]
for feature in list_new_cat_features:
    bivariate_percent_plot(cat=feature, df= df_train)


num_features.extend(list_new_features)


skew_feature_train, skew_train_df = check_skewness(data=df_train, dataset_name="Train Data", numerical_features=num_features)


skew_feature_test, skew_test_df = check_skewness(data=df_test, dataset_name="Test Data", numerical_features=num_features)


from sklearn.preprocessing import PowerTransformer

def handle_skewed_features(
    df,
    zero_threshold=0.9,
    skew_threshold=0.5,
    num_features=None,
    exclude_cols=None,
    dataset="Train data"
    
):
    """
    Handle skewed numerical features by applying appropriate transformations,
    *forcing* certain columns to be transformed even if they don't exceed skew_threshold.

    Parameters:
    - df: pandas.DataFrame
    - zero_threshold: float (default=0.9)
    - skew_threshold: float (default=0.5)
    - num_features: list of numerical columns to consider
    - exclude_cols: list of columns to skip entirely
    - dataset: Name of dataset

    Returns:
    - df: transformed DataFrame
    - transformed_cols: list of new feature names
    - high_zero_cols: list of sparse features (> zero_threshold)
    - skewed_cols: list of autoâ€‘detected skewed features
    - pt_dict: dict mapping each YJâ€‘transformed col â†’ its PowerTransformer
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


num_features = ["Social_event_attendance", "Going_outside", "Friends_circle_size",  "Post_frequency", 
                "PT_Time_spent_Alone",  "Social_Activity_Score", "Event_Outside_Product",  "PT_Extroversion_Score"]


processed_train_df, transformed_columns, sparse_columns, skewed_columns = handle_skewed_features(df=df_train, num_features=skew_feature_train)

skew_feature_train, skew_train_df = check_skewness(data=processed_train_df, numerical_features=num_features,
                                                   dataset_name= "Train data")


processed_test_df, transformed_columns_test, sparse_columns_test, skewed_columns_test = handle_skewed_features(df=df_test, num_features=skew_feature_test, dataset="Test data")

skew_feature_test, skew_test_df = check_skewness(data=processed_test_df, numerical_features=num_features,
                                                   dataset_name= "Test data")


checking_outlier(list_feature=num_features, df=processed_train_df, dataset_name="Training data")


checking_outlier(list_feature=num_features, df=processed_test_df, dataset_name="Test data")


processed_train_df["Social_event_attendance_Cat"] = pd.qcut(processed_train_df["Social_event_attendance"],
                                              q=4,
                                              labels=[1, 2, 3, 4])

plt.figure(figsize=(8, 5))
sns.histplot(data=processed_train_df, x="Social_event_attendance_Cat", color="lightblue", edgecolor="black")

plt.title("Distribution of Social_event_attendance_Cat", fontsize=14)
plt.xlabel("Social_event_attendance_Cat", fontsize=12)
plt.ylabel("")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(processed_train_df, processed_train_df["Social_event_attendance_Cat"]):
    start_train_set = processed_train_df.iloc[train_index]
    start_test_set = processed_train_df.iloc[test_index]


# Now we should remove the Social_event_attendance_Cat attribute so the data is back to its original state:
for set_ in (start_train_set, start_test_set): 
    set_.drop("Social_event_attendance_Cat", axis=1, inplace=True)


df_train_new = start_train_set.drop("Personality", axis=1)
df_train_label_new = start_train_set["Personality"].copy()


df_train_new["Many_Friends"].value_counts()


list_feature_num_robust = ["Social_Activity_Score"]
list_feature_num_stand = ["Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency",
                          "PT_Time_spent_Alone", "PT_Extroversion_Score", "Event_Outside_Product"]
list_feature_cat_onehot = ["Stage_fear", "Drained_after_socializing"]
list_feature_cat_keep = ["Many_Friends", "Much_Alone", "Active_Poster", "match_Extrovert", "match_Introvert"]


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
list_feature_prepared


label = LabelEncoder()
df_train_label_new = label.fit_transform(df_train_label_new)

# Preview label encoding
print("Label Encoding Mapping:", dict(zip(label.classes_, label.transform(label.classes_))))


# Using SMOTE to handling imbalance data.
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(df_train_new_prepared, df_train_label_new)


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

seed = 42
max_iter = 50000

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
models = [
    LinearSVC(max_iter=max_iter, random_state=42),
    SVC(kernel="rbf", random_state=seed),
    KNeighborsClassifier(metric = "minkowski", p = 2, n_neighbors=5),
    GaussianNB(),
    LogisticRegression(solver="liblinear", max_iter=max_iter, random_state=seed),
    DecisionTreeClassifier(max_depth=5, random_state=seed),
    RandomForestClassifier(n_estimators=100, random_state=seed),
    ExtraTreesClassifier(random_state=seed),
    AdaBoostClassifier(random_state=seed),
    XGBClassifier(n_estimators= 2000, max_depth= 4, eval_metric = "logloss",
                  random_state=seed, min_child_weight= 2, gamma=0.9,
                  subsample=0.8, colsample_bytree=0.8, objective= "binary:logistic",
                  nthread= -1),
    MLPClassifier(max_iter=max_iter, random_state=seed),
    GradientBoostingClassifier(random_state=seed),
    RidgeClassifier(alpha=1.0, random_state=seed, max_iter=max_iter),
    RidgeClassifierCV(alphas=[0.1, 0.5, 1.0], cv=kfold),
    CatBoostClassifier(verbose=0, random_seed=seed),
    BaggingClassifier(random_state=seed),
    LGBMClassifier(random_state=seed, verbosity=-1)
]


def generate_baseline_results(models = models, X = X_resampled, y = y_resampled,
                              metric = "accuracy", cv = kfold, plot_result = False):
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        model_scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
        for fold_idx, score in enumerate(model_scores):
            entries.append((model_name, fold_idx, score))
        cv_df = pd.DataFrame(entries, columns=["model_name", "fold_id", "accuracy_score"])

    # Summary
    mean = cv_df.groupby("model_name")["accuracy_score"].mean()
    std = cv_df.groupby("model_name")["accuracy_score"].std()

    baseline_result = pd.concat([mean, std], axis=1, ignore_index=True)
    baseline_result.columns = ["Mean", "Standard Deviation"]

    # Sort by accuracy
    baseline_result.sort_values(by=["Mean"], ascending=False, inplace=True)   

    if plot_result:
        plt.figure(figsize=(18, 8))
        sns.barplot(x="model_name", y="accuracy_score", data=cv_df, palette="viridis")
        plt.title("Base-Line Model Accuracy using 5-fold cross-validation", fontsize=14, weight="bold", pad=20)
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        return baseline_result
    else:
        return baseline_result


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
        plt.title("SHAP Feature Importance")
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
    ax[0, 0].set_title(f"ROC ({estimator.__class__.__name__})", weight="bold")
    ax[0, 0].legend()

    # Plot 2
    confusionMatrix = confusion_matrix(y_val, y_pred)
    sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="Blues", ax=ax[0, 1])
    ax[0, 1].set_title(f"Confusion Matrix ({estimator.__class__.__name__})", weight="bold")
    ax[0, 1].set_xlabel("Prediction")
    ax[0, 1].set_ylabel("Actual")

    # plot 3
    precision, recall, thresholds_pr = precision_recall_curve(y_val, y_pred_prob)
    pr_auc = auc(recall, precision)
    ax[1, 0].plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.3f})")
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 0].set_title("Precision-Recall Curve")
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
    if show_shap_plot:
        shap_sample = X_val.iloc[:200] if isinstance(X_val, pd.DataFrame) else X_val[:200]
        shap_plot(model=model, X_test=shap_sample, list_feature=list_feature_prepared)


X_val = start_test_set.drop("Personality", axis=1)
y_val = label.transform(start_test_set["Personality"].copy())
X_val_prepared = preprocessor.transform(X_val)


# After running optuna.
param = {"iterations": 938,
 "learning_rate": 0.0175895780773417,
 "depth": 8,
 "l2_leaf_reg": 0.0796249088905002,
 "random_strength": 0.006337957542773361,
 "bagging_temperature": 0.2704726166207223,
 "border_count": 226,
 "random_seed": seed,
 "eval_metric": "Accuracy",
 "verbose": 0
 }


best_model_catboost = CatBoostClassifier(**param)
best_model_catboost


evaluate_model(model = best_model_catboost, X_train=X_resampled, X_val=X_val_prepared,
               y_train=y_resampled, y_val=y_val, figsize=(15, 10))


# After running optuna.
param_xgb = {
    "colsample_bylevel": 0.8168489864941239,
    "colsample_bynode": 0.8850485490950061,
    "colsample_bytree": 0.8379339940113913,
    "gamma": 2.3977359439809276,
    "learning_rate": 0.0616974880921061,
    "max_depth": 344,
    "max_leaves": 89,
    "min_child_weight": 10,
    "n_estimators": 696,
    "n_jobs": -1,
    "random_state": seed,
    "reg_alpha": 1.849084818346014,
    "reg_lambda": 29.680324563362227,
    "subsample": 0.5902901569391961,
    "verbosity": 0,
    "eval_metric": "logloss",
    "objective": "binary:logistic"
 }


best_model_xgb = XGBClassifier(**param_xgb)
best_model_xgb


evaluate_model(model = best_model_xgb, X_train=X_resampled, X_val=X_val_prepared,
               y_train=y_resampled, y_val=y_val, figsize=(15, 10))


# After running optuna.
param_rf = {
 "n_estimators": 201,
 "max_depth": 26,
 "min_samples_split": 5,
 "min_samples_leaf": 1,
 "max_features": None,
 "bootstrap": True,
 "criterion": "entropy",
 "n_jobs": -1,
 "random_state": seed
 }


best_model_rf = RandomForestClassifier(**param_rf)
best_model_rf


evaluate_model(model = best_model_rf, X_train=X_resampled, X_val=X_val_prepared,
               y_train=y_resampled, y_val=y_val, figsize=(15, 10))


# After running optuna.
param_lgbm_gbdt = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.6467443250209886,
    "learning_rate": 0.06547186748153115,
    "min_child_samples": 34,
    "min_child_weight": 0.24399244943904663,
    "n_estimators": 498,
    "n_jobs": -1,
    "num_leaves": 158,
    "random_state": 42,
    "reg_alpha": 6.568921253574134,
    "reg_lambda": 62.66165355751099,
    "subsample": 0.0011019938618584968,
    "verbose": -1,
    "random_state": seed
 }


best_model_lgbm_gbdt = LGBMClassifier(**param_lgbm_gbdt)
best_model_lgbm_gbdt


evaluate_model(model = best_model_lgbm_gbdt, X_train=X_resampled, X_val=X_val_prepared,
               y_train=y_resampled, y_val=y_val, figsize=(15, 10))


# After running optuna.
param_lgbm_goss = {
    "boosting_type": "goss",
    "colsample_bytree": 0.8384834064170148,
    "learning_rate": 0.07006829797238343,
    "min_child_samples": 46,
    "min_child_weight": 0.7625394962666617,
    "n_estimators": 1887,
    "n_jobs": -1,
    "num_leaves": 341,
    "random_state": 42,
    "reg_alpha": 10.53082019937197,
    "reg_lambda": 67.44600065144685,
    "subsample": 0.4925008305336127,
    "verbose": -1,
    "random_state": seed
}


best_model_lgbm_goss = LGBMClassifier(**param_lgbm_goss)
best_model_lgbm_goss


evaluate_model(model = best_model_lgbm_goss, X_train=X_resampled, X_val=X_val_prepared,
               y_train=y_resampled, y_val=y_val, figsize=(15, 10))


from sklearn.ensemble import VotingClassifier
voting_clf_soft = VotingClassifier(
    estimators=[
        ("catboost", best_model_catboost),
        ("xgb", best_model_xgb),
        ("rf", best_model_rf),
        ("lgbm_gbdt", best_model_lgbm_gbdt),
        ("lgbm_goss", best_model_lgbm_goss)
    ],
    voting="soft",
    n_jobs=-1
)


cv_scores = cross_val_score(
    voting_clf_soft,
    X=X_resampled,
    y=y_resampled,
    cv=kfold,
    scoring="accuracy",
    n_jobs=-1
)


print(f"Cross-validated Accuracy (mean Â± std): {cv_scores.mean():.4f} Â± {cv_scores.std():.4f}")


evaluate_model(model = voting_clf_soft, X_train=X_resampled, X_val=X_val_prepared,
               y_train=y_resampled, y_val=y_val, figsize=(15, 10))


df_test_prepared = preprocessor.transform(processed_test_df)


from sklearn.metrics import accuracy_score
best_acc = 0
best_thresh = 0.5
for t in np.arange(0.1, 0.9, 0.01):
    preds = (voting_clf_soft.predict_proba(X_val_prepared)[:,1] >= t).astype(int)
    acc = accuracy_score(y_val, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print(f"Best threshold = {best_thresh:.2f}, accuracy = {best_acc:.4f}")


proba_test = voting_clf_soft.predict_proba(df_test_prepared)[:, 1]
y_pred_final = (proba_test >= best_thresh).astype(int)
y_pred_final = label.inverse_transform(y_pred_final)


submission_df = pd.DataFrame({
    "id": list_test_id,
    "Personality": y_pred_final
})

submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file saved!")
submission_df.head()


plt.figure(figsize=(8, 5))
sns.histplot(proba_test, bins=20, kde=True, color="skyblue")
plt.axvline(x=best_thresh, color="red", linestyle="--", label=f"Threshold = {best_thresh:.2f}")
plt.title("Distribution of Personality Probabilities (predict_proba)")
plt.xlabel("Personality Probability")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("\nTest Set Prediction Distribution:")
print(submission_df["Personality"].value_counts())
sns.countplot(data=submission_df, x="Personality", palette=color(n_colors=2))
plt.title(f"Test Set Personality Distribution)")
plt.show()


shap_plot(model=voting_clf_soft, X_test=df_test_prepared[:100], list_feature=list_feature_prepared, type="bar")


shap_plot(model=voting_clf_soft, X_test=df_test_prepared[:100], list_feature=list_feature_prepared)

