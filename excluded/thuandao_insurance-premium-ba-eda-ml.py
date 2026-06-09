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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
import scikit_posthocs as sp
from scipy.stats import normaltest

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500) # To display all the columns of dataframe
pd.set_option("max_colwidth", None) # To set the width of the column to maximum


# Load the datasets
# train.csv: Features and target labels
# test.csv: Features only

df_train = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")

# Verify shapes
print("Train Data Shape:", df_train.shape)
print("\nTest Data Shape:", df_test.shape)


# Display few rows of each dataset
print("Train Data Preview:")
display(df_train.head())

print("\nTest Data Preview:")
display(df_test.head())


# Replace space to under score.
df_train.columns = df_train.columns.str.strip().str.replace(" ", "_")
df_test.columns = df_test.columns.str.strip().str.replace(" ", "_")


# Display information about the DataFrames
print("Train Data Info:")
df_train.info()

print("\nTest Data Info:")
df_test.info()


df_train["Customer_Feedback"].value_counts()


df_train.drop("id", axis=1, inplace=True)
list_test_id = df_test["id"].copy().to_list()
df_test.drop("id", axis=1, inplace=True)


# Convert Policy_Start_Date to to_datetime.
df_train["Policy_Start_Date"] = pd.to_datetime(df_train["Policy_Start_Date"], format="mixed")
df_test["Policy_Start_Date"] = pd.to_datetime(df_test["Policy_Start_Date"], format="mixed")

num_features = ["Age", "Annual_Income", "Health_Score", "Credit_Score", "Vehicle_Age"]
cat_features = ["Gender", "Marital_Status", "Education_Level", "Occupation", "Location", "Policy_Type", 
                "Smoking_Status", "Exercise_Frequency", "Property_Type", "Customer_Feedback", "Number_of_Dependents", "Previous_Claims", "Insurance_Duration"]

print("Train Data describe:")
cmap = sns.light_palette("blue", as_cmap=True)
display(df_train.drop(columns=["Number_of_Dependents", "Previous_Claims", "Insurance_Duration",
                               "Policy_Start_Date"], axis=1).describe().T.style.background_gradient(cmap=cmap))

print("\nTest Data describe:")
display(df_test.drop(columns=["Number_of_Dependents", "Previous_Claims", "Insurance_Duration",
                               "Policy_Start_Date"], axis=1).describe().T.style.background_gradient(cmap=cmap))


def convert_cat(df, cat_features= cat_features):
    for feature in cat_features:
        if feature in df.columns:
            df[feature] = df[feature].astype("category")
        else:
            pass

convert_cat(df=df_train)
convert_cat(df=df_test)

# Display information about the DataFrames
print("Train Data Info:")
df_train.info()

print("\nTest Data Info:")
df_test.info()


print("Train Data describe:")
df_train.describe(include=["category", "object"]).T


print("\nTest Data describe:")
df_test.describe(include=["category", "object"]).T


# Function to calculate missing values, percentages, and data types
def missing_values_table(df):
    missing_count = df.isnull().sum()
    missing_percentage = 100 * missing_count / len(df)
    return pd.DataFrame({
        "Missing Values": missing_count,
        "Percentage (%)": missing_percentage
    })

# Create tables for train and test datasets
train_missing_table = missing_values_table(df_train)
test_missing_table = missing_values_table(df_test)

# Display the tables
print("Missing Values Table - Training Dataset:\n")
display(train_missing_table[train_missing_table["Missing Values"] > 0])  # Display only features with missing values
print("\n")

print("Missing Values Table - Test Dataset:\n")
display(test_missing_table[test_missing_table["Missing Values"] > 0]) 


from matplotlib import cm

# Filter missing values for train and test datasets
train_missing = train_missing_table[train_missing_table["Missing Values"] > 0].sort_values(by="Percentage (%)", ascending=False)
test_missing = test_missing_table[test_missing_table["Missing Values"] > 0].sort_values(by="Percentage (%)", ascending=False)

# Set up the figure and subplots
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Bar plot for train dataset
train_colors = cm.get_cmap("viridis", len(train_missing))(range(len(train_missing)))
ax[0].barh(train_missing.index, train_missing["Percentage (%)"], color=train_colors)
ax[0].set_title("Percentage of Missing Values (Train Data)", fontsize=12)
ax[0].set_xlabel("Percentage (%)", fontsize=10)
ax[0].set_ylabel("Features", fontsize=10)
ax[0].grid(axis="x", linestyle="--", alpha=0.6)
ax[0].invert_yaxis()  

# Bar plot for test dataset
test_colors = cm.get_cmap("viridis", len(test_missing))(range(len(test_missing)))
ax[1].barh(test_missing.index, test_missing["Percentage (%)"], color=test_colors)
ax[1].set_title("Percentage of Missing Values (Test Data)", fontsize=12)
ax[1].set_xlabel("Percentage (%)", fontsize=10)
ax[1].grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


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
        print("\nğŸ“ˆ Standardized Residuals:")
        print(round(residuals, 2))

        # Heatmap of residuals
        plt.figure(figsize=(6, 4))
        sns.heatmap(residuals, annot=True, cmap=cmap, center=0, fmt=".2f", linewidths=0.5)
        plt.title(f"Standardized Residuals Heatmap: {cat_feature} vs {target_feature}", pad=15, weight = "bold")
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

    df_clean = df[[categorical_feature, numeric_feature]].dropna()

    # Extract values
    groups = df_clean[categorical_feature].dropna().unique()
    if len(groups) < 3:
        print(f"â�Œ Error: Kruskal-Wallis H-test requires 3 or more groups.")
        return
    else:
        print(f"\nğŸ”� Kruskal-Wallis Test: {numeric_feature} ~ {categorical_feature}")
        data_groups = [df_clean[df_clean[categorical_feature] == g][numeric_feature].dropna() for g in groups]

        # Perform kruskal
        stat, p = kruskal(*data_groups)

        print(f"Kruskal-Wallis H-statistic: {stat:.3f}")
        print(f"p-value: {p}")
        
        if p < 0.05:
            print("ğŸŸ¢ Significant difference found. Running Dunn's Post-Hoc Test...")
            dunn_result = sp.posthoc_dunn(df_clean, val_col=numeric_feature, group_col=categorical_feature, p_adjust="bonferroni")
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


from scipy.signal import find_peaks
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(15, 6))

sns.boxplot(data=df_train, y = "Premium_Amount", ax=ax[0], color="#00BFC4")
ax[0].set_title(f"Box plot of Premium Amount", fontsize=14, pad=20, weight="bold")
ax[0].grid(axis="x", color="gray", linestyle=":", linewidth=0.7)
ax[0].set_ylabel("Premium Amount")
sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)

sns.histplot(data=df_train, x = "Premium_Amount", ax=ax[1], color="#00BFC4", kde=True, bins=40)
ax[1].set_title(f"Histogram of Premium Amount", fontsize=14, pad=20, weight="bold")
ax[1].set_xlabel("Premium Amount")
ax[1].set_ylabel("Frequency")
ax[1].grid(axis="x", color="gray", linestyle=":", linewidth=0.7)
sns.despine(ax=ax[1], top=True, right=True, left=False, bottom=False)

# Extract KDE values to find peaks
kde = sns.kdeplot(df_train["Premium_Amount"], ax=ax[1], color="#00BFC4").lines[0].get_data()
kde_x, kde_y = kde[0], kde[1]
peaks, _ = find_peaks(kde_y)

# Highlight peaks
for peak_idx in peaks:
    plt.plot(kde_x[peak_idx], kde_y[peak_idx], "ro")  # Red dots on peaks

plt.tight_layout()
plt.show()


def plot_numerical_features(df_train, df_test, num_features):
    colors = color()
    n = len(num_features)

    fig, ax = plt.subplots(n, 2, figsize=(12, n * 4))
    ax = np.array(ax).reshape(n, 2)

    for i, feature in enumerate(num_features):
        sns.histplot(data=df_train[feature], color=colors[0], bins=20, kde=True, ax=ax[i, 0], label="Train data")
        sns.histplot(data=df_test[feature], color=colors[1], bins=20, kde=True, ax=ax[i, 0], label="Test data")
        ax[i, 0].set_title(f"Histogram of {feature}", fontsize=14, pad=20, weight="bold")
        ax[i, 0].legend()
        ax[i, 0].set_ylabel("")
        ax[i, 0].grid(color="gray", linestyle=":", linewidth=0.7)
        ax[i, 0].axvline(df_train[feature].median(), color="green", linestyle="--", label="Median Train")
        ax[i, 0].axvline(df_test[feature].median(), color="orange", linestyle="--", label="Median Test")
        sns.despine(left=False, bottom=False, ax=ax[i, 0])

        df_plot = pd.concat([
            pd.DataFrame({"Dataset": "Train data", feature: df_train[feature]}),
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
        ax[i, 1].set_title(f"Horizontal Box plot of {feature}", fontsize=14, pad=20, weight="bold")
        ax[i, 1].grid(color="gray", linestyle=":", linewidth=0.7)
        sns.despine(left=False, bottom=False, ax=ax[i, 1])

    plt.tight_layout()
    plt.show()

plot_numerical_features(df_train = df_train, df_test = df_test, num_features=num_features)


def check_skewness(data, dataset_name, numerical_features = num_features, highlight=True, sort=True):
    skewness_dict = {}
    skew_feature = []
    for feature in numerical_features:
        if feature == "Premium_Amount" and dataset_name == "Test data":
            pass # The feature Premium_Amount only exist in train data.
        else:
            skew = data[feature].skew(skipna=True)
            skewness_dict[feature] = skew

    skew_df = pd.DataFrame.from_dict(skewness_dict, orient="index", columns=["Skewness"])
    if sort:
        skew_df = skew_df.reindex(skew_df["Skewness"].abs().sort_values(ascending=False).index)
    else:
        pass
    
    print(f"\nğŸ”� Skewness for {dataset_name}:")
    print("-"*70)
    print(f"{'Feature':<25} | {'Skewness':<10} | {'Remark'}")
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
            print(f"{color}{feature:<25} | {skew:>+9.4f} | {remark}{endc}")
            skew_feature.append(feature)
        else:
            print(f"{feature:<25} | {skew:>+9.4f} | {remark}")
    print("-"*70)
    return skew_feature, skew_df

skew_feature_train, skew_train_df = check_skewness(df_train, "Train data")
skew_feature_test, skew_test_df = check_skewness(df_test, "Test data")


def plot_correlation(df_train, df_test, train_name="Train Data", test_name="Test Data"):
    corr_train = df_train.corr(numeric_only=True)
    corr_test = df_test.corr(numeric_only=True)

    mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
    adjusted_mask_train = mask_train[1:, :-1]
    adjusted_cereal_corr_train = corr_train.iloc[1:, :-1]

    mask_test = np.triu(np.ones_like(corr_test, dtype=bool))
    adjusted_mask_test = mask_test[1:, :-1]
    adjusted_cereal_corr_test = corr_test.iloc[1:, :-1]

    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))

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

plot_correlation(df_train=df_train, df_test=df_test)


def plot_categorical_distribution_across_datasets(cat_features, train_data=df_train, test_data=df_test):
    import matplotlib.pyplot as plt
    import seaborn as sns

    for feature in cat_features:
        dataset_names = ["Train", "Test"]
        datasets = [train_data, test_data]

        fig, ax = plt.subplots(2, 2, figsize=(16, 8))

        for i, (data, name) in enumerate(zip(datasets, dataset_names)):
            order = data[feature].value_counts(ascending=False).index
            sns.countplot(x=feature, data=data, ax=ax[0, i],
                          palette=color(n_colors=train_data[feature].nunique()), order=order)

            ax[0, i].set_title(f"{name} Data: {feature} Counts", fontsize=14, pad=20, weight="bold")
            ax[0, i].set_xlabel("")
            ax[0, i].set_ylabel("")
            for p in ax[0, i].patches:
                count = int(p.get_height())
                ax[0, i].annotate(f"{count:,}",
                                  (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha="center", va="bottom", fontsize=11, color="black")

            ax[0, i].set_axisbelow(True)
            ax[0, i].grid(axis="y", color="gray", linestyle=":", linewidth=0.7)
            sns.despine(left=False, bottom=False, ax=ax[0, i])

        # Barplot: Percentage
        for i, (data, name) in enumerate(zip(datasets, dataset_names)):
            percent = data[feature].value_counts(normalize=True) * 100
            percent = percent.sort_values(ascending=False).reset_index()
            percent.columns = [feature, "Percent"]

            sns.barplot(x=feature, y="Percent", data=percent, ax=ax[1, i],
                        palette=color(n_colors=train_data[feature].nunique()), order=percent[feature])

            ax[1, i].set_title(f"{name} Data: {feature} Distribution (%)", fontsize=14, pad=20, weight="bold")
            ax[1, i].set_ylabel("Percentage (%)")
            ax[1, i].set_xlabel("")
            ax[1, i].tick_params(axis="x", rotation=0)
            ax[1, i].grid(color="gray", linestyle=":", linewidth=0.7, axis="y")
            sns.despine(left=False, bottom=False, ax=ax[1, i])

            for p in ax[1, i].patches:
                percentage = p.get_height()
                ax[1, i].annotate(f"{percentage:.1f}%",
                                  (p.get_x() + p.get_width() / 2., percentage),
                                  ha="center", va="bottom", fontsize=11, color="black")

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.show()

plot_categorical_distribution_across_datasets(cat_features=cat_features)


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
def perform_statical_testing(total_categories, feature, df_train = df_train, target_feature = "Premium_Amount"):
    cal_normaltest(cat_feature=feature, num_feature=target_feature, df=df_train)
    if total_categories == 2:
        cal_mannwhitneyu(dataframe=df_train, categorical_feature=feature, num_feature=target_feature)
    else:
        perform_kruskal_test(df=df_train, categorical_feature=feature, numeric_feature=target_feature)

def plot_categorical_distribution_by_Premium_Amount(feature, df_train = df_train, target_feature = "Premium_Amount", order = None):
    """
    Performs statical testing for each groups (distribution by target_feature) by ANOVA, T-test, Mann-Whitney U test,... <br>
    Draw violin and histogram to display the distribution for each groups of feature.
    Parameters:
        feature (str): The name of the column representing the grouping variable (categorical).
        df_train (pd.DataFrame): The input dataset.
        target_feature (str): The name of the column representing the target feature.
        order (list): Order items in plot.

    Returns:
        None
    """

    # Summary information
    df_summary_feature = df_train.groupby(by = feature, as_index= False)\
    .agg (
        Count = (target_feature, "count"),
        Mean_Premium_Amount = (target_feature, "mean"),
        Median_Premium_Amount = (target_feature, "median"),
        Std_Premium_Amount = (target_feature, "std")
    )
    df_summary_feature = df_summary_feature.sort_values(by="Mean_Premium_Amount", ascending=False)    

    summary_data = [
        ("Total Categories", f"{df_summary_feature.shape[0]}"),
        ("Overall Target Mean", f"{df_train[target_feature].mean():.2f}")
    ]
    summary_html = "<ul>" + "".join([f"<li><b>{k}:</b> {v}</li>" for k, v in summary_data]) + "</ul>"
    display(HTML(summary_html))
    display(df_summary_feature.style.background_gradient(cmap=cmap).set_table_attributes('style="width:75%; margin:auto;"'))

    perform_statical_testing(total_categories=df_summary_feature.shape[0], 
                             feature=feature, df_train=df_train, target_feature=target_feature)

    # Plot distribution
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.violinplot(x=feature, y=target_feature, data=df_train, hue=feature, 
                palette=color(n_colors=df_train[feature].nunique()), ax=ax)
    ax.set_title(f"Violin plot of {target_feature} distribution by {feature}", pad=15, weight = "bold")
    ax.set_xlabel(feature, labelpad=10)
    ax.set_ylabel(target_feature, labelpad=10)
    # if feature in ["Neighborhood", "Exterior1st", "Exterior2nd"]:
    #     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.grid(axis="y", color="gray", linestyle=":", alpha=0.7)
    sns.despine(left=False, bottom=False, ax=ax)

    plt.tight_layout()
    plt.show()

for feature in cat_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of Premium Amount by {feature}</b></h2>"))
    plot_categorical_distribution_by_Premium_Amount(feature=feature)


df_train_ma = df_train.copy()


bins = [0, 25, 35, 45, 55, np.inf]
labels = ["<25", "25â€“34", "35â€“44", "45â€“54", ">55"]
df_train_ma["Age_Group"] = pd.cut(df_train_ma["Age"], bins=bins, labels=labels, right=False)


median_by_age_group = df_train_ma.groupby("Age_Group")["Premium_Amount"].median().reset_index()
median_by_age_group.columns = ["Age_Group", "Median_Premium"]
print(median_by_age_group)


plt.figure(figsize=(10, 5))
ax = sns.boxplot(data=df_train_ma, y = "Premium_Amount", x = "Age_Group",
                   palette=color(n_colors=df_train_ma["Age_Group"].nunique()))
plt.title("Distribution of Premium Amount by Age Group", pad=15, weight = "bold")
plt.xlabel("Age Group")
plt.ylabel("Premium Amount")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(data=df_train_ma, x="Number_of_Dependents", hue="Policy_Type",
              palette=color(n_colors=df_train_ma["Policy_Type"].nunique()))
plt.title("Distribution of Policy Type by Number of Dependents", pad=15, weight = "bold")
plt.xlabel("Number of Dependents")
plt.ylabel("")
plt.legend(title='Policy Type', loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
plt.show()


cal_ChiSquare(cat_feature="Number_of_Dependents", target_feature="Policy_Type", df=df_train_ma, show_residuals=True)


median_by_occupation = df_train_ma.groupby("Occupation")["Annual_Income"].median().reset_index()
median_by_occupation.columns = ["Occupation", "Median_Income"]
print(median_by_occupation)


fig, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.boxplot(
    data=df_train_ma,
    y="Annual_Income",
    x="Occupation",
    ax=ax[0],
    palette=sns.color_palette("viridis", n_colors=df_train_ma["Occupation"].nunique())
)
ax[0].set_xlabel("Occupation")
ax[0].set_ylabel("Annual Income")
ax[0].set_title("Annual Income Distribution by Occupation", pad=15, weight = "bold")

sns.countplot(
    data=df_train_ma,
    x="Occupation",
    hue="Previous_Claims",
    ax=ax[1],
    palette="viridis"
)
ax[1].set_xlabel("Occupation")
ax[1].set_ylabel("Count")
ax[1].set_title("Distribution of Previous Claims by Occupation", pad=15, weight = "bold")
ax[1].legend(title="Previous Claims", loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3)

plt.tight_layout()
plt.show()


cal_ChiSquare(cat_feature="Occupation", target_feature="Previous_Claims", df=df_train_ma, show_residuals=True)


median_by_policy_type = df_train_ma.groupby("Policy_Type")["Annual_Income"].median().reset_index()
median_by_policy_type.columns = ["Policy_Type", "Median_Income"]
print(median_by_policy_type)


plt.figure(figsize=(10, 5))
ax = sns.violinplot(data=df_train_ma, y = "Annual_Income", x = "Policy_Type",
                   palette=color(n_colors=df_train_ma["Policy_Type"].nunique()))
plt.title("Distribution of Annual Income by Policy Type", pad=15, weight = "bold")
plt.xlabel("Policy Type")
plt.ylabel("Annual Income")
plt.tight_layout()
plt.show()


perform_statical_testing(total_categories=3, feature="Policy_Type", df_train=df_train_ma, target_feature="Annual_Income")


df_train_ma["Credit_Score_Group"] = pd.qcut(df_train_ma["Credit_Score"], q=4, labels=["Low", "Medium", "High", "Very High"])


median_by_credit_score_group = df_train_ma.groupby("Credit_Score_Group")["Premium_Amount"].median().reset_index()
median_by_credit_score_group.columns = ["Credit_Score_Group", "Median_Amount"]
print(median_by_credit_score_group)


fig, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.boxplot(
    data=df_train_ma,
    y="Premium_Amount",
    x="Credit_Score_Group",
    ax=ax[0],
    palette=sns.color_palette("viridis", n_colors=df_train_ma["Credit_Score_Group"].nunique())
)
ax[0].set_xlabel("Credit Score Group")
ax[0].set_ylabel("Premium Amount")
ax[0].set_title("Premium Amount Distribution by Credit Score Group", pad=15, weight = "bold")

sns.countplot(
    data=df_train_ma,
    x="Credit_Score_Group",
    hue="Previous_Claims",
    ax=ax[1],
    palette="viridis"
)
ax[1].set_xlabel("Credit Score Group")
ax[1].set_ylabel("Count")
ax[1].set_title("Distribution of Previous Claims by Credit Score Group", pad=15, weight = "bold")
ax[1].legend(title="Previous Claims", loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3)

plt.tight_layout()
plt.show()


perform_statical_testing(total_categories=4, feature="Credit_Score_Group", df_train=df_train_ma, target_feature="Premium_Amount")


cal_ChiSquare(cat_feature="Credit_Score_Group", target_feature="Previous_Claims", df=df_train_ma, show_residuals=True)


avg_premium_by_age = df_train_ma.groupby("Vehicle_Age")["Premium_Amount"].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=avg_premium_by_age, x="Vehicle_Age", y="Premium_Amount", marker="o")
plt.title("Average Premium Amount by Vehicle Age", pad=15, weight = "bold")
plt.xlabel("Vehicle Age")
plt.ylabel("Average Premium Amount")
plt.tight_layout()
plt.show()


# 1. Calculate mean, median premium, and number of customers per policy type
premium_stats = df_train_ma.groupby("Policy_Type")["Premium_Amount"].agg(
    Mean_Premium="mean",
    Median_Premium="median",
    Customer_Count="count"
)

# 2. Convert "Previous_Claims" to numeric type for valid aggregation
df_train_ma["Previous_Claims"] = pd.to_numeric(df_train_ma["Previous_Claims"], errors="coerce")

# Then compute total number of claims per policy type
claims_by_policy = df_train_ma.groupby("Policy_Type")["Previous_Claims"].sum().rename("Total_Claims")

# 3. Merge premium and claims stats into one financial summary table
financial_perf = premium_stats.merge(claims_by_policy, on="Policy_Type")

# 4. Estimate average number of claims per customer
financial_perf["Avg_Claims_per_Customer"] = financial_perf["Total_Claims"] / financial_perf["Customer_Count"]

# 5. Display the financial performance table using tabulate
from tabulate import tabulate

print(tabulate(financial_perf.reset_index(), headers="keys", tablefmt="fancy_grid", showindex=False))


df_train_ma["Policy_Month"] = df_train_ma["Policy_Start_Date"].dt.month
df_train_ma["Policy_Weekday"] = df_train_ma["Policy_Start_Date"].dt.day_name()


fig, ax = plt.subplots(2, 2, figsize=(20, 13))
avg_premium_by_month = df_train_ma.groupby("Policy_Month")["Premium_Amount"].mean().reset_index()
sns.lineplot(
    data=avg_premium_by_month,
    y="Premium_Amount",
    x="Policy_Month",
    ax=ax[0, 0],
    marker="o"
)
ax[0, 0].set_xlabel("Occupation")
ax[0, 0].set_ylabel("Annual Income")
ax[0, 0].set_title("Average Premium Amount by Policy Month", pad=15, weight = "bold")

sns.countplot(
    data=df_train_ma,
    x="Policy_Month",
    hue="Policy_Type",
    ax=ax[0, 1],
    palette="viridis"
)
ax[0, 1].set_xlabel("Policy_Month")
ax[0, 1].set_ylabel("")
ax[0, 1].set_title("Distribution of Policy Type by Policy Month", pad=15, weight = "bold")
ax[0, 1].legend(title="Policy Type", loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3)

sns.countplot(
    data=df_train_ma,
    x="Policy_Month",
    hue="Previous_Claims",
    ax=ax[1, 0],
    palette="viridis"
)
ax[1, 0].set_xlabel("Policy_Month")
ax[1, 0].set_ylabel("")
ax[1, 0].set_title("Distribution of Previous Claim by Policy Month", pad=15, weight = "bold")
ax[1, 0].legend(title="Previous Claims", loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3)

sns.countplot(
    data=df_train_ma,
    x="Policy_Month",
    ax=ax[1, 1],
    palette="viridis"
)

monthly_counts = df_train_ma["Policy_Month"].value_counts().sort_index()
mean_count = monthly_counts.mean()

ax[1, 1].axhline(mean_count, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_count:.0f}")
ax[1, 1].legend(loc="upper right")

ax[1, 1].set_xlabel("Policy_Month")
ax[1, 1].set_ylabel("")
ax[1, 1].set_title("Total Policy Registrations by Month", pad=15, weight="bold")

plt.tight_layout()
plt.show()


df_train_ma["Health_Score_Group"] = pd.qcut(
    df_train_ma["Health_Score"],
    q=3,
    labels=["Low", "Medium", "High"]
)


plt.figure(figsize=(10, 6))
sns.countplot(data=df_train_ma, x="Health_Score_Group", hue="Policy_Type", palette="viridis")
plt.title("Distribution of Policy Type by Health Score Group", weight="bold")
plt.xlabel("Health Score Group")
plt.ylabel("Number of Customers")
plt.legend(title="Policy Type", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)
plt.tight_layout()
plt.show()


cal_ChiSquare(cat_feature="Health_Score_Group", target_feature="Policy_Type", df=df_train_ma, show_residuals=True)


plt.figure(figsize=(10, 6))
sns.countplot(data=df_train_ma, x="Exercise_Frequency", hue="Policy_Type", palette="viridis")
plt.title("Distribution of Policy Type by Exercise Frequency", weight="bold")
plt.xlabel("Exercise Frequency")
plt.ylabel("Number of Customers")
plt.legend(title="Exercise Frequency", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)
plt.tight_layout()
plt.show()


cal_ChiSquare(cat_feature="Exercise_Frequency", target_feature="Policy_Type", df=df_train_ma, show_residuals=True)


df_train_ma_poor = df_train_ma[df_train_ma["Customer_Feedback"] == "Poor"]
df_train_ma_poor["Gender"].value_counts(normalize=True) * 100


df_train_ma_poor["Occupation"].value_counts(normalize=True) * 100


df_train_ma_poor["Location"].value_counts(normalize=True) * 100


median_by_customer_feedback = df_train_ma.groupby("Customer_Feedback")["Credit_Score"].median().reset_index()
median_by_customer_feedback.columns = ["Customer_Feedback", "Median_Score"]
print(median_by_customer_feedback)

order = ["Poor", "Average", "Good"]
plt.figure(figsize=(10, 5))
ax = sns.boxplot(data=df_train_ma, y = "Credit_Score", x = "Customer_Feedback", order=order,
                   palette=color(n_colors=df_train_ma["Customer_Feedback"].nunique()))
plt.title("Distribution of Credit Score by Customer Feedback", pad=15, weight = "bold")
plt.xlabel("Customer Feedback")
plt.ylabel("Credit Score")
plt.tight_layout()
plt.show()

perform_statical_testing(total_categories=3, feature="Customer_Feedback", df_train=df_train_ma, target_feature="Credit_Score")


median_by_customer_feedback_hs = df_train_ma.groupby("Customer_Feedback")["Health_Score"].median().reset_index()
median_by_customer_feedback_hs.columns = ["Customer_Feedback", "Median_Score"]
print(median_by_customer_feedback_hs)

plt.figure(figsize=(10, 5))
ax = sns.boxplot(data=df_train_ma, y = "Health_Score", x = "Customer_Feedback", order=order,
                   palette=color(n_colors=df_train_ma["Customer_Feedback"].nunique()))
plt.title("Distribution of Health Score by Customer Feedback", pad=15, weight = "bold")
plt.xlabel("Customer Feedback")
plt.ylabel("Health Score")
plt.tight_layout()
plt.show()

perform_statical_testing(total_categories=3, feature="Customer_Feedback", df_train=df_train_ma, target_feature="Health_Score")


plt.figure(figsize=(10, 6))
sns.countplot(data=df_train_ma, x="Policy_Type", hue="Customer_Feedback", palette="viridis")
plt.title("Distribution of Customer Feedback by Policy Type", weight="bold")
plt.xlabel("Policy Type")
plt.ylabel("Number of Customers")
plt.legend(title="Customer Feedback", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)
plt.tight_layout()
plt.show()

cal_ChiSquare(cat_feature="Customer_Feedback", target_feature="Policy_Type", df=df_train_ma, show_residuals=True)


plt.figure(figsize=(10, 6))
sns.countplot(data=df_train_ma, x="Previous_Claims", hue="Customer_Feedback", palette="viridis")
plt.title("Distribution of Customer Feedback by Previous Claims", weight="bold")
plt.xlabel("Previous Claims")
plt.ylabel("Number of Customers")
plt.legend(title="Customer Feedback", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)
plt.tight_layout()
plt.show()

cal_ChiSquare(cat_feature="Customer_Feedback", target_feature="Previous_Claims", df=df_train_ma, show_residuals=True)


median_by_income = df_train_ma.groupby("Location")["Annual_Income"].median().reset_index()
median_by_income.columns = ["Location", "Median_Income"]
print(median_by_income)

plt.figure(figsize=(10, 5))
ax = sns.boxplot(data=df_train_ma, y = "Annual_Income", x = "Location",
                   palette=color(n_colors=df_train_ma["Location"].nunique()))
plt.title("Distribution of Annual Income by Location", pad=15, weight = "bold")
plt.xlabel("Location")
plt.ylabel("Annual Income")
plt.tight_layout()
plt.show()

perform_statical_testing(total_categories=3, feature="Location", df_train=df_train_ma, target_feature="Annual_Income")


for col in num_features:
    df_train[col] = df_train[col].fillna(df_train[col].median())
    df_test[col] = df_test[col].fillna(df_test[col].median())


# List of categorical columns with numeric values
cat_numeric_cols = ["Number_of_Dependents", "Previous_Claims", "Insurance_Duration"]

# Fill missing values with the mode for each column
for col in cat_numeric_cols:
    mode_val = df_train[col].mode(dropna=True)[0]  # Get the most frequent value (mode)
    df_train[col] = df_train[col].fillna(mode_val)
    mode_val = df_test[col].mode(dropna=True)[0]  # Get the most frequent value (mode)
    df_test[col] = df_test[col].fillna(mode_val)


cat_cols = ["Marital_Status", "Occupation", "Customer_Feedback"]

for col in cat_cols:
    # If the column is of Categorical type, add "Unknown" to the list of valid categories
    if pd.api.types.is_categorical_dtype(df_train[col]):
        df_train[col] = df_train[col].cat.add_categories("Unknown")
        df_test[col] = df_test[col].cat.add_categories("Unknown")
    
    # Then fill missing values with "Unknown"
    df_train[col] = df_train[col].fillna("Unknown")
    df_test[col] = df_test[col].fillna("Unknown")


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


# Convert some cat features to numerical to generate new features
df_train["Previous_Claims"] = pd.to_numeric(df_train["Previous_Claims"], errors="coerce")
df_test["Previous_Claims"] = pd.to_numeric(df_test["Previous_Claims"], errors="coerce")

df_train["Number_of_Dependents"] = pd.to_numeric(df_train["Number_of_Dependents"], errors="coerce")
df_test["Number_of_Dependents"] = pd.to_numeric(df_test["Number_of_Dependents"], errors="coerce")

df_train["Insurance_Duration"] = pd.to_numeric(df_train["Insurance_Duration"], errors="coerce")
df_test["Insurance_Duration"] = pd.to_numeric(df_test["Insurance_Duration"], errors="coerce")


# Income group
df_train["Income_Group"] = pd.qcut(df_train["Annual_Income"], q=4, labels=["Low Income", "Mid-Low", "Mid-High", "High Income"])
df_test["Income_Group"] = pd.qcut(df_test["Annual_Income"], q=4, labels=["Low Income", "Mid-Low", "Mid-High", "High Income"])

# Age Group
df_train["Age_Group"] = pd.cut(df_train["Age"], bins=[0, 25, 35, 45, 55, 100], labels=["<25", "25â€“35", "35â€“45", "45â€“55", ">55"])
df_test["Age_Group"] = pd.cut(df_test["Age"], bins=[0, 25, 35, 45, 55, 100], labels=["<25", "25â€“35", "35â€“45", "45â€“55", ">55"])

# Health_Risk_Level
df_train["Health_Risk_Level"] = pd.qcut(df_train["Health_Score"], q=3, labels=["High Risk", "Medium Risk", "Low Risk"])
df_test["Health_Risk_Level"] = pd.qcut(df_test["Health_Score"], q=3, labels=["High Risk", "Medium Risk", "Low Risk"])

# Has_Claimed
df_train["Has_Claimed"] = (df_train["Previous_Claims"] > 0).astype(int)
df_test["Has_Claimed"] = (df_test["Previous_Claims"] > 0).astype(int)

# Is_Smoker_At_Risk
df_train["Is_Smoker_At_Risk"] = ((df_train["Smoking_Status"] == "Yes") & (df_train["Health_Score"] < df_train["Health_Score"].median())).astype(int)
df_test["Is_Smoker_At_Risk"] = ((df_test["Smoking_Status"] == "Yes") & (df_test["Health_Score"] < df_test["Health_Score"].median())).astype(int)

# Policy_Season
def map_month_to_season(month):
    if month in [12, 1, 2]: return "Winter"
    if month in [3, 4, 5]: return "Spring"
    if month in [6, 7, 8]: return "Summer"
    return "Autumn"

df_train["Policy_Month"] = df_train["Policy_Start_Date"].dt.month
df_train["Policy_Season"] = df_train["Policy_Month"].apply(map_month_to_season)

df_test["Policy_Month"] = df_test["Policy_Start_Date"].dt.month
df_test["Policy_Season"] = df_test["Policy_Month"].apply(map_month_to_season)

# Dependents_per_Year
df_train["Dependents_per_Year"] = df_train["Number_of_Dependents"] / (df_train["Insurance_Duration"] + 1)
df_test["Dependents_per_Year"] = df_test["Number_of_Dependents"] / (df_test["Insurance_Duration"] + 1)

# Is_Negative_Feedback
df_train["Is_Negative_Feedback"] = df_train["Customer_Feedback"].apply(lambda x: x == "Poor")
df_test["Is_Negative_Feedback"] = df_test["Customer_Feedback"].apply(lambda x: x == "Poor")

# Age Ã— Health Score â†’ reflects health-related risk increasing with age
df_train["Age_Health"] = df_train["Age"] * df_train["Health_Score"]
df_test["Age_Health"] = df_test["Age"] * df_test["Health_Score"]

# Annual Income Ã— Credit Score â†’ suggests financial stability
df_train["Income_Credit"] = df_train["Annual_Income"] * df_train["Credit_Score"]
df_test["Income_Credit"] = df_test["Annual_Income"] * df_test["Credit_Score"]

# Number of Dependents Ã— Insurance Duration â†’ indicates long-term financial burden
df_train["Dependents_Duration"] = df_train["Number_of_Dependents"] * df_train["Insurance_Duration"]
df_test["Dependents_Duration"] = df_test["Number_of_Dependents"] * df_test["Insurance_Duration"]

# Vehicle Age Ã— Previous Claims â†’ highlights vehicle-related risk
df_train["VehicleAge_Claims"] = df_train["Vehicle_Age"] * df_train["Previous_Claims"]
df_test["VehicleAge_Claims"] = df_test["Vehicle_Age"] * df_test["Previous_Claims"]

# Health Score Ã— Previous Claims â†’ explores if health impacts claim frequency
df_train["Health_Claims"] = df_train["Health_Score"] * df_train["Previous_Claims"]
df_test["Health_Claims"] = df_test["Health_Score"] * df_test["Previous_Claims"]


list_feature_convert = ["Previous_Claims", "Number_of_Dependents", "Insurance_Duration", "Policy_Season"]
convert_cat(df=df_train, cat_features=list_feature_convert)
convert_cat(df=df_test, cat_features=list_feature_convert)

df_train.drop(columns="Policy_Start_Date", axis=1, inplace=True)
df_test.drop(columns="Policy_Start_Date", axis=1, inplace=True)


df_train["Smoking_Status"] = df_train["Smoking_Status"].map({"Yes": 1, "No": 0}).astype(bool)
df_test["Smoking_Status"] = df_test["Smoking_Status"].map({"Yes": 1, "No": 0}).astype(bool)

bool_cols = ["Has_Claimed", "Is_Smoker_At_Risk", "Is_Negative_Feedback"]

for col in bool_cols:
    df_train[col] = df_train[col].astype(bool)
    df_test[col] = df_test[col].astype(bool)

# Display information about the DataFrames
print("Train Data Info:")
df_train.info()

print("\nTest Data Info:")
df_test.info()


cat_features = df_train.select_dtypes(include=["category"]).columns.tolist()
cat_features.extend(["Is_Smoker_At_Risk", "Is_Negative_Feedback", "Has_Claimed"])
cat_features


num_features = df_train.select_dtypes(exclude=["category"]).columns.tolist()
num_features.remove("Is_Negative_Feedback")
num_features.remove("Is_Smoker_At_Risk")
num_features.remove("Has_Claimed")
num_features


new_features = ["Income_Group", "Age_Group", "Health_Risk_Level", "Has_Claimed", "Is_Smoker_At_Risk", "Policy_Season", "Is_Negative_Feedback"]
for feature in new_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of Premium Amount by {feature}</b></h2>"))
    plot_categorical_distribution_by_Premium_Amount(feature=feature)


def plot_correlation_new(features, df_train, df_test, train_name="Train Data", test_name="Test Data"):
    corr_train = df_train[features].corr()
    if "Premium_Amount" in features:
        features.remove("Premium_Amount")
    corr_test = df_test[features].corr()

    mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
    adjusted_mask_train = mask_train[1:, :-1]
    adjusted_cereal_corr_train = corr_train.iloc[1:, :-1]

    mask_test = np.triu(np.ones_like(corr_test, dtype=bool))
    adjusted_mask_test = mask_test[1:, :-1]
    adjusted_cereal_corr_test = corr_test.iloc[1:, :-1]

    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(data=adjusted_cereal_corr_train, mask=adjusted_mask_train,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[0])
    ax[0].set_title(f"Correlation Heatmap of {train_name}", fontsize=16, weight="bold")

    sns.heatmap(data=adjusted_cereal_corr_test, mask=adjusted_mask_test,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[1])
    ax[1].set_title(f"Correlation Heatmap of {test_name}", fontsize=16, weight="bold")

    plt.tight_layout()
    plt.show()

plot_correlation_new(features=num_features,df_train=df_train, df_test=df_test)


num_features.append("Premium_Amount")
skew_feature_train, skew_train_df = check_skewness(data=df_train, dataset_name="Train Data", numerical_features=num_features)


num_features.remove("Premium_Amount")
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
    pt_dict          = {}

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
            pt_dict[col] = pt
            transformed_cols.append(f"PT_{col}")
            dropped_cols.append(col)

    # drop originals for any column we did transform
    df.drop(columns=dropped_cols, inplace=True)

    return df, transformed_cols, high_zero_cols, auto_skewed, pt_dict


processed_train_df, transformed_columns, sparse_columns, skewed_columns, pt_dict_train = handle_skewed_features(df=df_train, num_features=skew_feature_train)
num_features_train = ["Age", "PT_Annual_Income", "Health_Score", "Vehicle_Age", "Credit_Score", "Policy_Month", "PT_Dependents_per_Year", "PT_Age_Health", "PT_Income_Credit",
                "PT_Dependents_Duration", "PT_VehicleAge_Claims", "PT_Health_Claims", "PT_Premium_Amount"]
skew_feature_train, skew_train_df = check_skewness(data=processed_train_df, numerical_features=num_features_train,
                                                   dataset_name= "Train data")


processed_test_df, transformed_columns_test, sparse_columns_test, skewed_columns_test, pt_dict_test = handle_skewed_features(df=df_test, num_features=skew_feature_test, dataset="Test data")
num_features_test = ["Age", "PT_Annual_Income", "Health_Score", "Vehicle_Age", "Credit_Score", "Policy_Month", "PT_Dependents_per_Year", "PT_Age_Health", "PT_Income_Credit",
                "PT_Dependents_Duration", "PT_VehicleAge_Claims", "PT_Health_Claims"]
skew_feature_test, skew_test_df = check_skewness(data=processed_test_df, numerical_features=num_features_test,
                                                   dataset_name= "Test data")


def plot_correlation_new(features, df_train, df_test, train_name="Train Data", test_name="Test Data"):
    corr_train = df_train[features].corr()
    if "PT_Premium_Amount" in features:
        features.remove("PT_Premium_Amount")
    corr_test = df_test[features].corr()

    mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
    adjusted_mask_train = mask_train[1:, :-1]
    adjusted_cereal_corr_train = corr_train.iloc[1:, :-1]

    mask_test = np.triu(np.ones_like(corr_test, dtype=bool))
    adjusted_mask_test = mask_test[1:, :-1]
    adjusted_cereal_corr_test = corr_test.iloc[1:, :-1]

    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(data=adjusted_cereal_corr_train, mask=adjusted_mask_train,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[0])
    ax[0].set_title(f"Correlation Heatmap of {train_name}", fontsize=16, weight="bold")

    sns.heatmap(data=adjusted_cereal_corr_test, mask=adjusted_mask_test,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[1])
    ax[1].set_title(f"Correlation Heatmap of {test_name}", fontsize=16, weight="bold")

    plt.tight_layout()
    plt.show()

num_features_cor = num_features_train
plot_correlation_new(features=num_features_cor,df_train=processed_train_df, df_test=processed_test_df)


# We need convert bool column to int8 to avoid to error "SimpleImputer does not support data with dtype bool".
list_feature_cat_int8 = ["Smoking_Status", "Is_Negative_Feedback", "Is_Smoker_At_Risk", "Has_Claimed"]
for col in list_feature_cat_int8:
    if processed_train_df[col].dtype == "bool":
        processed_train_df[col] = processed_train_df[col].astype("int8")
    if processed_test_df[col].dtype == "bool":
        processed_test_df[col] = processed_test_df[col].astype("int8")

# Display information about the DataFrames
print("Train Data Info:")
processed_train_df.info()

print("\nTest Data Info:")
processed_test_df.info()


plt.figure(figsize=(8, 5))
sns.histplot(data=processed_train_df, x="Income_Group", color="lightblue", edgecolor="black")

plt.title("Distribution of Income_Group", fontsize=14)
plt.xlabel("Income_Group", fontsize=12)
plt.ylabel("")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(processed_train_df, processed_train_df["Income_Group"]):
    start_train_set = processed_train_df.iloc[train_index]
    start_test_set = processed_train_df.iloc[test_index]


df_train_new = start_train_set.drop("PT_Premium_Amount", axis=1)
df_train_label_new = start_train_set["PT_Premium_Amount"].copy()


list_feature_num_robust = ["PT_Annual_Income", "PT_Age_Health", "PT_Income_Credit"]
list_feature_num_stand = ["Age", "Health_Score", "Vehicle_Age", "Credit_Score",
                          "Policy_Month", "PT_Dependents_per_Year", "PT_Dependents_Duration", "PT_VehicleAge_Claims", "PT_Health_Claims"]
list_feature_cat_onehot = ["Gender", "Marital_Status", "Number_of_Dependents", "Education_Level", "Occupation", "Location", "Policy_Type", "Previous_Claims", "Insurance_Duration",
                           "Customer_Feedback", "Exercise_Frequency", "Property_Type", "Income_Group", "Age_Group", "Health_Risk_Level", "Policy_Season"]
list_feature_cat_keep = ["Smoking_Status", "Is_Negative_Feedback", "Is_Smoker_At_Risk", "Has_Claimed"]


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


def safe_rmsle(y_true, y_pred):
    y_true = np.maximum(0, y_true) + 1
    y_pred = np.maximum(0, y_pred) + 1
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

rmsle_scorer = make_scorer(safe_rmsle, greater_is_better=False)


# We use some models to compare performance.
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

seed = 42
max_iter = 50000
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

models = [
    CatBoostRegressor(random_seed=seed, verbose=False),
    Lasso(alpha=0.0005, max_iter=max_iter, random_state=seed),
    Ridge(alpha=10, max_iter=max_iter, random_state=seed),
    GradientBoostingRegressor(random_state=seed),
    XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.1, random_state=seed, verbosity=0),
    LGBMRegressor(random_state=seed, verbosity=-1)
]


def generate_baseline_results(models=models, X=df_train_new_prepared, y=df_train_label_new,
                              metric=rmsle_scorer, cv=kfold, plot_result=False):
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        model_scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
        for fold_idx, score in enumerate(model_scores):
            entries.append((model_name, fold_idx, -score))  # negate the score here

    cv_df = pd.DataFrame(entries, columns=["model_name", "fold_id", "rmsle_score"])

    # Summary
    mean = cv_df.groupby("model_name")["rmsle_score"].mean()
    std = cv_df.groupby("model_name")["rmsle_score"].std()

    baseline_result = pd.concat([mean, std], axis=1, ignore_index=True)
    baseline_result.columns = ["Mean", "Standard Deviation"]

    # Sort by RMSLE (lower is better)
    baseline_result.sort_values(by="Mean", ascending=True, inplace=True)

    if plot_result:
        plt.figure(figsize=(18, 8))
        sns.barplot(x="model_name", y="rmsle_score", data=cv_df, palette="viridis")
        plt.title("Baseline Model RMSLE using Cross-Validation", fontsize=14, weight="bold", pad=20)
        plt.xlabel("Model")
        plt.ylabel("RMSLE")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    return baseline_result

generate_baseline_results(plot_result=True)


def shap_plot(model, X_test, list_feature):
     # https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137/
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    X_test_sample = pd.DataFrame(X_test, columns=list_feature)
    explainer = shap.Explainer(model.predict, X_test_sample)
    shap_values = explainer(X_test_sample)

    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": X_test_sample.columns, "importance": shap_importance})
    shap_df = shap_df.sort_values("importance", ascending=False).head(30)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=shap_df["importance"], y=shap_df["feature"], palette="viridis", order=shap_df["feature"])
    plt.xlabel("mean(|SHAP value|)")
    plt.title("SHAP Feature Importance", weight="bold", pad=20)
    plt.tight_layout()
    plt.show()


# Function to evaluate regression models
def evaluate_model(model, X_train, X_val, y_train, y_val, pt_premium_amount=None, show_shap_plot = False):
    RESET = "\033[0m"
    BLUE = "\033[94m"
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Back-transform predictions and ground truth
    if pt_premium_amount is not None:
        y_val_real = pt_premium_amount.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
        y_pred_real = pt_premium_amount.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_val_real = y_val
        y_pred_real = y_pred
    
    # Metrics: RMSLE
    rmsle = safe_rmsle(y_val_real, y_pred_real)
    if rmsle < 0:
        rmsle = -rmsle

    print(f"Model: {model.__class__.__name__}{RESET}")
    print(f"Root Mean Squared Logarithmic Error (RMSLE): {BLUE}{rmsle:.4f}{RESET}")
    print("-" * 80)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_val_real, y_pred_real, alpha=0.4, color="royalblue")
    plt.plot([y_val_real.min(), y_val_real.max()], [y_val_real.min(), y_val_real.max()], "r--", lw=2, label="Perfect Prediction (y=x)")
    plt.xlabel("Actual Values (SalePrice)")
    plt.ylabel("Predicted Values (SalePrice)")
    plt.title("Predicted vs. Actual Values (Validation Set)", weight="bold", pad=20)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    # Residuals plot
    residuals = y_val_real - y_pred_real
    plt.figure(figsize=(7, 7))
    plt.scatter(y_val_real, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--", lw=2)
    plt.xlabel("Actual Values (SalePrice)")
    plt.ylabel("Prediction Error (Residuals)")
    plt.title("Residual Plot", weight="bold", pad=20)
    plt.tight_layout()
    plt.show() 

    if show_shap_plot:
        shap_plot(model = model, X_test = X_val, list_feature = list_feature_prepared)

    return rmsle


X_val = start_test_set.drop("PT_Premium_Amount", axis=1)
y_val = start_test_set["PT_Premium_Amount"].copy()
X_val_prepared = preprocessor.transform(X_val)


# After running optuna
param_xgb = {
"n_estimators": 552,
"max_depth": 15,
"learning_rate": 0.07148357030575762,
"subsample": 0.6699865700828613,
"colsample_bytree": 0.9402353786703079,
"gamma": 0.8905466258539609,
"reg_alpha": 4.539188741638234,
"reg_lambda": 2.0218637418195575,
"min_child_weight": 16,
"random_state": seed,
"verbosity": 0    
}

best_model_xgb = XGBRegressor(**param_xgb)
best_model_xgb


weight_rmsle_xgb = evaluate_model(model = best_model_xgb, X_train = df_train_new_prepared, 
                   X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val, pt_premium_amount = pt_dict_train["Premium_Amount"])


param_cb = {
"iterations": 1833,
"depth": 10,
"learning_rate": 0.19464582738554204,
"l2_leaf_reg": 3.7646977567494093,
"bagging_temperature": 0.2052232889552441,
"random_strength": 0.05417731781709137,
"border_count": 187,
"loss_function": "RMSE",
"verbose": 0,
"random_seed": seed
}

best_model_cb = CatBoostRegressor(**param_cb)
best_model_cb


weight_rmsle_cb = evaluate_model(model = best_model_cb, X_train = df_train_new_prepared, 
                   X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val, pt_premium_amount = pt_dict_train["Premium_Amount"])


from sklearn.ensemble import VotingRegressor

voting_reg = VotingRegressor(estimators=[
    ("xgb", best_model_xgb),
    ("cat", best_model_cb)
], n_jobs=-1, weights=[weight_rmsle_xgb, weight_rmsle_cb])

cv_scores = cross_val_score(
    voting_reg,
    X=df_train_new_prepared,
    y=df_train_label_new,
    cv=kfold,
    scoring=rmsle_scorer,
    n_jobs=-1
)

mean_score = -cv_scores.mean()
std_score = cv_scores.std()

print(f"Cross-validated RMSLE (mean Â± std): {mean_score:.4f} Â± {std_score:.4f}")


evaluate_model(model = voting_reg, X_train = df_train_new_prepared, 
                   X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val, pt_premium_amount = pt_dict_train["Premium_Amount"])


df_test_prepared = preprocessor.transform(processed_test_df)
y_pred_test = voting_reg.predict(df_test_prepared)
y_pred_test_real = pt_dict_train["Premium_Amount"].inverse_transform(y_pred_test.reshape(-1, 1)).ravel()


submission_df = pd.DataFrame({
    "Id": list_test_id,
    "Premium Amount": y_pred_test_real
})

submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file saved!")
submission_df.head()


plt.figure(figsize=(10, 6))
sns.histplot(submission_df["Premium Amount"], bins=50, kde=True, color="skyblue")
plt.title("Histogram of Predicted Premium Amount", weight="bold", pad=20)
plt.xlabel("Premium Amount")
plt.ylabel("Frequency")
plt.show()


shap_plot(
    model = voting_reg,
    X_test = df_test_prepared[:200],
    list_feature = list_feature_prepared
)

