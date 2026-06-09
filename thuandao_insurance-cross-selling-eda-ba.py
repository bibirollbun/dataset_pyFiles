!pip install statsmodels > pip_log_statsmodels.txt 2>&1
!pip install scikit_posthocs > pip_log_scikit_posthocs.txt 2>&1


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
df_train = pd.read_csv("/kaggle/input/playground-series-s4e7/train.csv")
df_origin = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction-data/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e7/test.csv")

# Verify shapes
print("Train Data Shape:", df_train.shape)
print("\nOrigin Data Shape:", df_origin.shape)
print("\nTest Data Shape:", df_test.shape)


# Display few rows of each dataset
print("Train Data Preview:")
display(df_train.head())

print("\nOrigin Data Preview:")
display(df_origin.head())

print("\nTest Data Preview:")
display(df_test.head())


# Display information about the DataFrames
print("Train Data Info:")
df_train.info()

print("\nOrigin Data Info:")
df_origin.info()

print("\nTest Data Info:")
df_test.info()


num_features = ["Age", "Annual_Premium", "Vintage", "Region_Code", "Policy_Sales_Channel"]
cat_features = ["Gender", "Driving_License", "Previously_Insured", "Vehicle_Age", "Vehicle_Damage"]
feature_drop = ["Gender", "Driving_License", "Previously_Insured", "Vehicle_Age", "Vehicle_Damage", "Response", "id"]


print("Train Data describe:")
cm = sns.light_palette("blue", as_cmap=True)
display(df_train.drop(columns=feature_drop, axis=1).describe().T.style.background_gradient(cmap=cm))

print("\nOrigin Data describe:")
display(df_origin.drop(columns=feature_drop, axis=1).describe().T.style.background_gradient(cmap=cm))

print("\nTest Data describe:")
display(df_test.drop(columns=cat_features, axis=1).describe().T.style.background_gradient(cmap=cm))


def convert_cat(features, df):
    for feature in features:
        if feature in df.columns:
            df[feature] = df[feature].astype("category")

convert_cat(features=cat_features, df=df_train)
convert_cat(features=cat_features, df=df_origin)
convert_cat(features=cat_features, df=df_test)


# Display information about the DataFrames
print("Train Data Info:")
df_train.info()

print("\nOrigin Data Info:")
df_origin.info()

print("\nTest Data Info:")
df_test.info()


print("Train Data describe:")
display(df_train.drop(columns="Response", axis=1).describe(include=["category", "object"]).T)

print("Origin Data describe:")
display(df_origin.drop(columns="Response", axis=1).describe(include=["category", "object"]).T)

print("Test Data describe:")
display(df_test.describe(include=["category", "object"]).T)


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

print("\nMissing value Origin dataset: ")
displayNULL(df_origin, dataset_name="Origin Set")

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
    "Origin Data":  df_origin,
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


# Drop duplicate
for df in [df_origin]:
    df.drop_duplicates(inplace=True)

for name, data in datasets.items():
    check_duplicates_report(data, name)
    duplicate_summary[name] = {
        "duplicates": data.duplicated().sum(),
        "total_rows": len(data)
    }
    print()


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


checking_outlier(list_feature=num_features, df=df_origin, dataset_name="Origin data")


checking_outlier(list_feature=num_features, df=df_test, dataset_name="Test data")


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
        print("\nStandardized Residuals:")
        print(round(residuals, 2))

        # Heatmap of residuals
        plt.figure(figsize=(6, 4))
        sns.heatmap(residuals, annot=True, cmap=cmap, center=0, fmt=".2f", linewidths=0.5)
        plt.title(f"Standardized Residuals Heatmap: {cat_feature} vs {target_feature}", weight = "bold")
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


# Set target variable
target_variable = "Response"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
datasets = [("Train Data", df_train), ("Original Data", df_origin)]

for i, (title, data) in enumerate(datasets):
    ax = axes[i, 0]
    sns.countplot(y=target_variable, data=data, ax=ax, palette=color(n_colors=2))
    ax.set_title(f"Count Plot of Response in {title}", pad=20, weight="bold")
    ax.set_ylabel("Response")
    ax.set_xlabel("Count")
    ax.set_yticks([0, 1], ["No", "Yes"])
    ax.grid(axis="x", color="gray", linestyle=":", linewidth=0.7)

    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    for p in ax.patches:
        width = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(width + max(data[target_variable].value_counts())*0.01, y,
                f"{int(width)}", 
                ha="left", va="center", fontsize=10, fontweight="bold", color="black")

    loan_counts = data[target_variable].value_counts().sort_index()
    wedges, texts, autotexts = axes[i, 1].pie(
        loan_counts,
        labels=["No", "Yes"],
        autopct="%1.1f%%",
        startangle=90,
        colors=color(n_colors=2),
        wedgeprops=dict(width=0.4, edgecolor="w"),
        radius=1.2,
        explode = (0, 0.08)
    )
    
    for text in texts + autotexts:
        text.set_fontsize(10)
        text.set_fontweight("bold")
    
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    axes[i, 1].add_artist(centre_circle)
    axes[i, 1].set_title(f"Response in {title}", pad=20, weight="bold")
    axes[i, 1].axis("equal") 

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.show()


def plot_numerical_features(df_train, df_test, df_origin, num_features):
    colors = color(n_colors=3)  # The color function you defined earlier
    n = len(num_features)

    fig, ax = plt.subplots(n, 1, figsize=(10, n * 4))
    if n == 1:
        ax = [ax]  # Ensure ax is iterable when there is only one feature

    for i, feature in enumerate(num_features):
        # Combine data for violin plot
        df_plot = pd.concat([
            pd.DataFrame({"Dataset": "Train data", feature: df_train[feature]}),
            pd.DataFrame({"Dataset": "Origin data", feature: df_origin[feature]}),
            pd.DataFrame({"Dataset": "Test data", feature: df_test[feature]})
        ]).reset_index(drop=True)

        sns.violinplot(
            data=df_plot,
            x=feature,
            y="Dataset",
            palette=colors,
            ax=ax[i]
        )
        ax[i].set_title(f"Violin plot of {feature}", pad=20, weight="bold")
        ax[i].grid(color="gray", linestyle=":", linewidth=0.7)
        sns.despine(left=False, bottom=False, ax=ax[i])

    plt.tight_layout()
    plt.show()

# Call the function
plot_numerical_features(
    df_train=df_train,
    df_test=df_test,
    df_origin=df_origin,
    num_features=num_features
)


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
                annot=True, fmt=".1f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[0])
    ax[0].set_title(f"Correlation Heatmap of {train_name}", fontsize=16, weight="bold")

    sns.heatmap(data=adjusted_cereal_corr_test, mask=adjusted_mask_test,
                annot=True, fmt=".1f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[1])
    ax[1].set_title(f"Correlation Heatmap of {test_name}", fontsize=16, weight="bold")

    sns.heatmap(data=adjusted_cereal_corr_origin, mask=adjusted_mask_origin,
                annot=True, fmt=".1f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[2])
    ax[2].set_title(f"Correlation Heatmap of {origin_name}", fontsize=16, weight="bold")

    plt.tight_layout()
    plt.show()

plot_correlation(df_train=df_train.drop(columns="Response", axis=1),
                 df_origin=df_origin.drop(columns="Response", axis=1),
                 df_test=df_test)


def plot_categorical_distribution(cat_features, df_train, df_test, df_origin, order=None):
    for feature in cat_features:
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(25, 10))

        # Determine order dynamically if not provided
        if order is None:
            unique_vals = sorted(df_train[feature].dropna().unique())
        else:
            unique_vals = order

        # COUNT PLOT â€“ TRAIN
        sns.countplot(data=df_train, x=feature, ax=ax[0, 0],
                      palette=color(n_colors=len(unique_vals)), order=unique_vals)
        ax[0, 0].set_title(f"[Train] Count plot of {feature}", fontsize=13, pad=12, weight="bold")
        ax[0, 0].grid(axis="y", linestyle=":", linewidth=0.7)
        sns.despine(ax=ax[0, 0], left=False, bottom=False)
        for container in ax[0, 0].containers:
            ax[0, 0].bar_label(container, fmt='%d', label_type="edge", fontsize=10, weight="bold")

        # COUNT PLOT â€“ ORIGIN
        sns.countplot(data=df_origin, x=feature, ax=ax[0, 1],
                      palette=color(n_colors=len(unique_vals)), order=unique_vals)
        ax[0, 1].set_title(f"[Origin] Count plot of {feature}", fontsize=13, pad=12, weight="bold")
        ax[0, 1].grid(axis="y", linestyle=":", linewidth=0.7)
        sns.despine(ax=ax[0, 1], left=False, bottom=False)
        for container in ax[0, 1].containers:
            ax[0, 1].bar_label(container, fmt='%d', label_type="edge", fontsize=10, weight="bold")

        # COUNT PLOT â€“ TEST
        sns.countplot(data=df_test, x=feature, ax=ax[0, 2], 
                      palette=color(n_colors=len(unique_vals)), order=unique_vals)
        ax[0, 2].set_title(f"[Test] Count plot of {feature}", fontsize=13, pad=12, weight="bold")
        ax[0, 2].grid(axis="y", linestyle=":", linewidth=0.7)
        sns.despine(ax=ax[0, 2], left=False, bottom=False)
        for container in ax[0, 2].containers:
            ax[0, 2].bar_label(container, fmt='%d', label_type="edge", fontsize=10, weight="bold")

        # PERCENTAGE BARPLOT â€“ TRAIN
        train_percent = df_train[feature].value_counts(normalize=True) * 100
        train_percent = train_percent.reindex(unique_vals).fillna(0)
        sns.barplot(x=train_percent.index, y=train_percent.values, ax=ax[1, 0],
                    palette=color(n_colors=len(unique_vals)))
        ax[1, 0].set_title(f"[Train] Percentage Distribution of {feature}", pad=10, weight="bold")
        sns.despine(ax=ax[1, 0], left=False, bottom=False)
        ax[1, 0].set_ylabel("Percentage (%)")
        ax[1, 0].set_xlabel(feature)
        ax[1, 0].grid(axis="y", linestyle=":", linewidth=0.7)
        for i, v in enumerate(train_percent.values):
            ax[1, 0].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=10, weight="bold")

        # PERCENTAGE BARPLOT â€“ ORIGIN
        origin_percent = df_origin[feature].value_counts(normalize=True) * 100
        origin_percent = origin_percent.reindex(unique_vals).fillna(0)
        sns.barplot(x=origin_percent.index, y=origin_percent.values, ax=ax[1, 1],
                    palette=color(n_colors=len(unique_vals)))
        ax[1, 1].set_title(f"[Origin] Percentage Distribution of {feature}", pad=10, weight="bold")
        sns.despine(ax=ax[1, 1], left=False, bottom=False)
        ax[1, 1].set_ylabel("Percentage (%)")
        ax[1, 1].set_xlabel(feature)
        ax[1, 1].grid(axis="y", linestyle=":", linewidth=0.7)
        for i, v in enumerate(origin_percent.values):
            ax[1, 1].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=10, weight="bold")

        # PERCENTAGE BARPLOT â€“ TEST
        test_percent = df_test[feature].value_counts(normalize=True) * 100
        test_percent = test_percent.reindex(unique_vals).fillna(0)
        sns.barplot(x=test_percent.index, y=test_percent.values, ax=ax[1, 2],
                    palette=color(n_colors=len(unique_vals)))
        ax[1, 2].set_title(f"[Test] Percentage Distribution of {feature}", pad=10, weight="bold")
        sns.despine(ax=ax[1, 2], left=False, bottom=False)
        ax[1, 2].set_ylabel("Percentage (%)")
        ax[1, 2].set_xlabel(feature)
        ax[1, 2].grid(axis="y", linestyle=":", linewidth=0.7)
        for i, v in enumerate(test_percent.values):
            ax[1, 2].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=10, weight="bold")

        plt.tight_layout()
        plt.show()

plot_categorical_distribution(cat_features=cat_features, df_train = df_train, df_test = df_test, df_origin=df_origin)


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
                print(f"âš ï¸�  {feature}: {top_ratio:.1%} lÃ  '{freq.index[0]}'")
        if not flagged:
            print("âœ… No feature has a category that makes up more than 99% of its values.")
        print("*" * 50)
top_ratio()


df_train = pd.concat([df_train, df_origin], ignore_index=True)
print(df_train.shape)


from IPython.core.display import HTML
targer_feature = "Response"
def perform_statical_testing(feature, df_train = df_train, total_categories = 2, target_feature = targer_feature):
    cal_normaltest(cat_feature=target_feature, num_feature=feature, df=df_train)
    if total_categories == 2:
        cal_mannwhitneyu(dataframe=df_train, categorical_feature=target_feature, num_feature=feature)
    else:
        pass

def plot_numerical_distribution_by_targer_feature(feature, df_train = df_train, target_feature = targer_feature, order = None):
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

    perform_statical_testing(feature=feature, target_feature=target_feature)

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
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {feature} by {targer_feature}</b></h2>"))
    plot_numerical_distribution_by_targer_feature(df_train=df_train, feature=feature)


# defining function for plotting
def bivariate_percent_plot(cat, df, figsize=(15, 6), order = None, rot = 0, target_var = "Response"):
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {cat} by {target_var}</b></h2>"))
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=figsize)
    # Plot 1
    # Calculate the total number of each "cat" by "target_var"
    grouped = df.groupby([cat, target_var]).size().unstack(fill_value=0)
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
        ax[0].bar_label(container, fmt='%1.2f%%', label_type="center", weight="bold", fontsize=9)

    ax[0].set_title(f"Percentage of {target_var} by {cat}", fontsize=14, weight="bold")
    ax[0].set_xlabel(f"{cat}", fontsize=12)
    ax[0].set_ylabel(f"% {target_var} Rate", fontsize=12)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=rot)
    # ax[0].grid(axis="y", color="gray", linestyle=":", linewidth=0.7)
    ax[0].legend_.remove()
    sns.despine(left=False, bottom=False, ax=ax[0])

    # Plot 2
    sns.countplot(data=df, hue = target_var, x = cat,
                palette=color(n_colors=2), ax=ax[1], order=percentages.index, hue_order = [0, 1])
    # Show value for each bar.
    for container in ax[1].containers:
        ax[1].bar_label(container, fmt='%d', label_type="edge", fontsize=9, weight="bold")

    ax[1].set_title(f"{target_var} by {cat}", fontsize=14, weight="bold")
    ax[1].set_xlabel(f"{cat}", fontsize=12)
    ax[1].set_ylabel("Number of Customer", fontsize=12)
    ax[1].legend(title=target_var, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=rot)
    # ax[1].grid(axis="y", color="gray", linestyle=":", linewidth=0.7)
    sns.despine(left=False, bottom=False, ax=ax[1])
    plt.tight_layout()
    plt.show()

    cal_ChiSquare(cat_feature=cat, target_feature=target_var, df=df, show_residuals=True)


for feature in cat_features:
    bivariate_percent_plot(cat=feature, df= df_train)


df_train_ba = df_train.copy()


df_train.drop("id", axis=1, inplace=True)
df_origin.drop("id", axis=1, inplace=True)
list_test_id = df_test["id"].copy().to_list()
df_test.drop("id", axis=1, inplace=True)


df_train_ba["Male_PreviouslyInsured0_Vehicle1_2Year"] = (
    (df_train_ba["Gender"] == "Male") &
    (df_train_ba["Previously_Insured"] == 0) &
    (df_train_ba["Vehicle_Age"] == "1-2 Year")
)


bivariate_percent_plot(cat="Male_PreviouslyInsured0_Vehicle1_2Year", df=df_train_ba, figsize=(18, 6))


plt.figure(figsize=(10, 5))
sns.boxplot(data=df_train_ba, x="Vehicle_Damage", y="Age", hue="Response", palette=color(n_colors=2))

plt.title("Age Distribution by Vehicle Damage and Response", weight = "bold")
plt.xlabel("Vehicle Damage")
plt.ylabel("Age")
plt.tight_layout()
plt.show()


def compute_clv(
    data: pd.DataFrame,
    annual_premium_col: str = "Annual_Premium",
    vintage_col: str = "Vintage",
    gross_margin: float = 0.35,   # Gross profit margin on premium
    discount_rate: float = 0.10,  # Annual discount rate
    retention_rate: float = 0.80  # Annual renewal probability (assumption)
) -> pd.DataFrame:
    """
    Estimated CLV = realized_margin_to_date + expected_future_margin

      realized_margin_to_date = Annual_Premium * gross_margin * tenure_years
      expected_future_margin  = Annual_Premium * gross_margin * retention_rate / (1 + discount_rate - retention_rate)

    Vintage is assumed to be in DAYS â†’ converted to years.
    """
    dfc = data.copy()
    # Convert days of tenure to years
    dfc["tenure_years"] = dfc[vintage_col] / 365.0
    # Realized margin so far
    dfc["realized_margin"] = dfc[annual_premium_col] * gross_margin * dfc["tenure_years"]
    # Multiplier for expected future margin
    multiplier = retention_rate / (1.0 + discount_rate - retention_rate)
    # Expected future margin
    dfc["future_margin_expected"] = dfc[annual_premium_col] * gross_margin * multiplier
    # Estimated CLV
    dfc["CLV_est"] = dfc["realized_margin"] + dfc["future_margin_expected"]
    return dfc

compute_clv(data=df_train_ba).head()


# Compute response rate and volume by regionâ€“channel
grp = (df.groupby(["Region_Code","Policy_Sales_Channel"])
         .agg(resp_rate=("Response","mean"), volume=("Response","size"))
         .reset_index())

# Filter the top 6 channels by volume for each region (and keep only the top 6 regions by volume)
top_regions = df["Region_Code"].value_counts().head(6).index
grp = grp[grp["Region_Code"].isin(top_regions)].copy()

grp["rank_in_region"] = grp.groupby("Region_Code")["volume"].rank(method="first", ascending=False)
grp_top = grp[grp["rank_in_region"] <= 6]

# Plot facet bar chart
g = sns.FacetGrid(grp_top, col="Region_Code", col_wrap=3, height=3.2, sharex=False, sharey=True)
g.map_dataframe(sns.barplot, x="Policy_Sales_Channel", y="resp_rate", order=None)
g.set_titles("Region {col_name}")
g.set_axis_labels("Channel", "Response rate")
for ax in g.axes.flat:
    ax.tick_params(axis="x", rotation=90)
    ax.grid(alpha=0.3)
plt.suptitle("Top Channels by Response Rate within Each Region (Top 6 Regions)", y=1.02, weight="bold")
plt.tight_layout()
plt.show()


# High Premium threshold = top 25%
q75 = df_train_ba["Annual_Premium"].quantile(0.75)
df_train_ba["High_Premium"] = df_train_ba["Annual_Premium"] >= q75

# Response rate by Vehicle_Age x High_Premium
resp_rate = (
    df_train_ba.groupby(["Vehicle_Age", "High_Premium"])["Response"]
    .mean()
    .rename("response_rate")
    .reset_index()
)

# Include the median premium of each group for comparison
med_tbl = (
    df_train_ba.groupby(["Vehicle_Age", "High_Premium"])["Annual_Premium"]
    .median()
    .rename("median_premium")
    .reset_index()
)

result = resp_rate.merge(med_tbl, on=["Vehicle_Age", "High_Premium"])
print("\nResponse rate & median premium by Vehicle_Age x High_Premium:")
print(result.sort_values(["Vehicle_Age", "High_Premium"]))


plt.figure(figsize=(10, 5))
sns.violinplot(data=df_train_ba, x="Vehicle_Age", y="Annual_Premium", hue="Response", palette=color(n_colors=2))

plt.title("Annual Premium Distribution by Vehicle Age and Response", weight = "bold")
plt.xlabel("Vehicle Age")
plt.ylabel("Annual Premium")
plt.tight_layout()
plt.show()


# Create a combined column: Driving_License x Previously_Insured
df["DLxPI"] = df["Driving_License"].astype(str) + "_" + df["Previously_Insured"].astype(str)

# Calculate response rate for each combined group
grouped = (
    df.groupby("DLxPI")["Response"]
    .mean()
    .reset_index()
    .rename(columns={"Response": "response_rate"})
)

# Create a figure with 2 subplots side by side (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Subplot 1: Barplot (Response Rate) ---
sns.barplot(data=grouped, x="DLxPI", y="response_rate", ax=axes[0], palette="Blues")
axes[0].set_title("Response Rate by DL x PI", weight="bold")
axes[0].set_xlabel("Driving_License x Previously_Insured")
axes[0].set_ylabel("Response Rate")
axes[0].set_ylim(0, 1)

# --- Subplot 2: Countplot (Customer count by Response) ---
sns.countplot(data=df, x="DLxPI", hue="Response", ax=axes[1], palette="Set2")
axes[1].set_title("Response Count by DL x PI", weight="bold")
axes[1].set_xlabel("Driving_License x Previously_Insured")
axes[1].set_ylabel("Customer Count")

# Display the combined plots
plt.tight_layout()
plt.show()


def label_segment(row):
    if row["Age"] >= 40 and row["Vehicle_Age"] == "< 1 Year":
        return "Old_Person_New_Car"
    elif row["Age"] < 40 and row["Vehicle_Age"] == "< 2 Year":
        return "Young_Person_Old_Car"
    else:
        return "Other"

df_train_ba["PersonCarSegment"] = df_train_ba.apply(label_segment, axis=1)

segment_stats = (
    df_train_ba[df_train_ba["PersonCarSegment"] != "Other"]
    .groupby("PersonCarSegment")["Response"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "response_rate", "count": "num_customers"})
    .reset_index()
)
print(segment_stats)


df_train_ba["PersonCarSegment"].value_counts()


plt.figure(figsize=(10, 5))
sns.barplot(data=segment_stats, x="PersonCarSegment", y="response_rate", palette="muted")
plt.title("Response Rate: Old Person with New Car vs. Young Person with Old Car", weight="bold")
plt.xlabel("Customer Segment")
plt.ylabel("Response Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# ==== 1) Bin Age (same as your original) ====
df_train_ba["Age_Group"] = pd.cut(
    df_train_ba["Age"],
    bins=[0, 30, 40, 50, 60, np.inf],
    labels=["<30", "30-39", "40-49", "50-59", "60+"],
    right=False
)

# (Optional) Use qcut to split into age quartiles:
# df_train_ba["Age_Group"] = pd.qcut(df_train_ba["Age"], q=5, duplicates="drop")

# ==== 2) Bin Annual_Premium (Q1/Q3) ====
q25 = df_train_ba["Annual_Premium"].quantile(0.25)
q75 = df_train_ba["Annual_Premium"].quantile(0.75)

def premium_group(p):
    if p < q25:
        return "Low"
    elif p < q75:
        return "Medium"
    else:
        return "High"

df_train_ba["Premium_Group"] = df_train_ba["Annual_Premium"].apply(premium_group)

# ==== 3) Calculate response_rate by the triplet ====
grouped = (
    df_train_ba
    .groupby(["Age_Group", "Premium_Group", "Policy_Sales_Channel"], dropna=False)["Response"]
    .agg(response_rate="mean", num_customers="count")
    .reset_index()
)

# ==== 4) Filter out small groups + calculate lift ====
overall_rate = df_train_ba["Response"].mean()
min_n = 200        # adjust as needed
top_n = 10

ranked = (
    grouped.assign(lift=lambda d: d["response_rate"] / overall_rate)
           .query("num_customers >= @min_n")
           .sort_values(["response_rate", "num_customers"], ascending=[False, False])
)

print("Overall response rate:", round(overall_rate, 4))
print("\nTop groups by response rate (min_n = %d):" % min_n)
print(ranked.head(top_n))

# ==== 5) (Optional) Plot Top-N barplot ====
plot_df = ranked.head(top_n).copy()
plot_df["segment"] = (
    plot_df["Age_Group"].astype(str) + " | " +
    plot_df["Premium_Group"].astype(str) + " | " +
    plot_df["Policy_Sales_Channel"].astype(str)
)

plt.figure(figsize=(12, 5))
sns.barplot(data=plot_df, x="segment", y="response_rate")
plt.title("Top Segments by Response Rate\n(Age x Premium x Channel)", weight="bold")
plt.ylabel("Response Rate")
plt.xlabel("Age_Group | Premium_Group | Policy_Sales_Channel")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1)
# annotate number of customers & lift
for i, row in plot_df.reset_index().iterrows():
    plt.text(i, row["response_rate"] + 0.01,
             f"n={row['num_customers']}\nlift={row['lift']:.2f}",
             ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()


# Premium_per_Vintage
df_train["Premium_per_Vintage"] = df_train["Annual_Premium"] / df_train["Vintage"]
df_test["Premium_per_Vintage"] = df_test["Annual_Premium"] / df_test["Vintage"]

# VehicleAge_x_VehicleDamage
df_train["VehicleAge_x_VehicleDamage"] = (
    df_train["Vehicle_Age"].astype(str) + "_" + df_train["Vehicle_Damage"].astype(str)
)
df_test["VehicleAge_x_VehicleDamage"] = (
    df_test["Vehicle_Age"].astype(str) + "_" + df_test["Vehicle_Damage"].astype(str)
)

# DLxPI
df_train["DLxPI"] = (
    df_train["Driving_License"].astype(str) + "_" + df_train["Previously_Insured"].astype(str)
)
df_test["DLxPI"] = (
    df_test["Driving_License"].astype(str) + "_" + df_test["Previously_Insured"].astype(str)
)


cat_features = ["Gender", "Driving_License", "Previously_Insured", "Vehicle_Age", "Vehicle_Damage",
                "VehicleAge_x_VehicleDamage", "DLxPI"]  
convert_cat(features=cat_features, df=df_train)
convert_cat(features=cat_features, df=df_test)


df_train = df_train.astype({
    "Age": "int8",
    "Region_Code": "int8",
    "Vintage": "int16",
    "Response": "int8",
    "Premium_per_Vintage": "float32",
    "Policy_Sales_Channel": "int8",
    "Annual_Premium": "float32"
})

df_test = df_test.astype({
    "Age": "int8",
    "Region_Code": "int8",
    "Vintage": "int16",
    "Premium_per_Vintage": "float32",
    "Policy_Sales_Channel": "int8",
    "Annual_Premium": "float32"
})


# Display information about the DataFrames
print("Train Data Info:")
df_train.info()

print("\nTest Data Info:")
df_test.info()


new_cat_features = ["VehicleAge_x_VehicleDamage", "DLxPI"]
for feature in new_cat_features:
    bivariate_percent_plot(cat=feature, df=df_train)


display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of Premium_per_Vintage by Response</b></h2>"))
plot_numerical_distribution_by_targer_feature(feature="Premium_per_Vintage", df_train=df_train)

