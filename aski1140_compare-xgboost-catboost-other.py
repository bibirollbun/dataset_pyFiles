# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import chi2_contingency 
from statsmodels.stats.proportion import proportions_ztest

from IPython.display import HTML
import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


class config:
    dir_train = "/kaggle/input/playground-series-s5e8/train.csv"
    dir_test = "/kaggle/input/playground-series-s5e8/test.csv"
    dir_sub = "/kaggle/input/playground-series-s5e8/sample_submission.csv"


# Create dataframe for train set
df_train = pd.read_csv(config.dir_train, index_col = "id")
# Create dataframe for test set
df_test = pd.read_csv(config.dir_test, index_col = "id")
# Create dataframe for submission set
df_sub = pd.read_csv(config.dir_sub, index_col = "id")

# Head of train set
print("Train Dataset")
print(f"Shape of Train Dataset: {df_train.shape}")
df_train.head()


def check_isna_sum(dataframe: pd.DataFrame):
    """
        This function checks nan values for all of features for given dataset
    """
    for column in dataframe.columns:
        isnan_ = dataframe[column].isna().sum()
        print(f"{column.capitalize()} feature has {isnan_} NaN values")

def check_unique_count(dataframe:pd.DataFrame):
    """
        This function checks the count of unique values for each features
    """
    for column in dataframe.columns:
        count_ = dataframe[column].nunique()
        print(f"{column.capitalize()} feature has {count_} unique values")

def first_eda(dataframe:pd.DataFrame):
    """
        This function performs initial analysis. 
    """
    print(f"This dataframe has {dataframe.shape[0]} rows and {dataframe.shape[1]} columns")
    print("*"*100)
    
    # Performs Count of NaN Values for each feature
    text_ = "Count of NaN Values for Each Feature"
    print(f"{text_}".center(50))
    print("*"*100)
    check_isna_sum(dataframe)
    print("*"*100)

    # Performs Count of Unique Values for each feature
    text_ = "Count of Unique Values for Each Feature"
    print(f"{text_}".center(50))
    print("*"*100)
    check_unique_count(dataframe)
    print("*"*100)

    # Printing head of dataframe
    text_ = "Head of Dataset"
    print(f"{text_}".center(50))
    print("*"*100)
    print(dataframe.head())
    print("*"*100)

    # Printing describe of dataframe
    text_ = "Describe of Dataset"
    print(f"{text_}".center(50))
    print("*"*100)
    print(dataframe.describe().T)
    print("*"*100)

    # Printing info of dataframe
    text_ = "Info of Dataset"
    print(f"{text_}".center(50))
    print("*"*100)
    print(dataframe.info())
    print("*"*100)


first_eda(df_train)


# Looking at the distribution of Target variable
value_counts_ = df_train.y.value_counts()
labels = ["No", "Yes"]
# declaring exploding pie
explode = [0.1, 0.1]

plt.figure(figsize = (8, 5))

plt.pie(value_counts_,
        labels = labels,
        startangle = 140,
        autopct = "%.1f%%",
        explode = explode,
        colors = sns.color_palette("pastel"))
plt.title("Distribution of Target Variable", fontsize = 12, fontweight = "bold", loc = "center")
plt.show()


categorical_features = ["job", "marital", "education", "default", 
                        "housing", "loan", "contact", "day", 
                        "month", "poutcome"]

numeric_features = [col for col in df_train.columns if not col in categorical_features and not col == "y"]




# Create plot for each feature
fig,axes = plt.subplots(nrows = 2, ncols = 3, figsize = (12, 8))
axes = axes.flatten()

# Plot barplot for each feature by Target
for idx, feature in enumerate(numeric_features):
    sns.barplot(data = df_train, x = "y", y = f"{feature}", palette = "husl", ax = axes[idx])
    axes[idx].set_title(f"{feature.upper()} by Target", fontsize = 10, fontweight = "bold", loc = "center")
    axes[idx].set_xlabel("Subscription (0=No, 1=Yes)")
    axes[idx].set_ylabel(f"{feature}".capitalize())

# Hide empty plots
for idx in range(len(numeric_features), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.show()


previous_no0 = df_train[df_train["previous"] != 0].copy()
pdays_no1 = df_train[df_train["pdays"] != -1].copy()

fig,axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 6))

sns.barplot(data = previous_no0, x = "y", y = "previous", palette = "husl", ax = axes[0])
sns.barplot(data = pdays_no1, x = "y", y = "pdays", palette = "husl", ax = axes[1])

axes[0].set_title("Previous not equal to 0")
axes[1].set_title("Pdays not equal to -1")
plt.show()


# Kdeplot for numerical features

fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (12, 12))

ax = ax.flatten()

sns.kdeplot(data = df_train, x = "age", ax = ax[0], color = "orange")
sns.kdeplot(data = df_train, x = "balance", ax = ax[1], color = "orange")
sns.kdeplot(data = df_train, x = "duration", ax = ax[2], color = "orange")
sns.kdeplot(data = df_train, x = "campaign", ax = ax[3], color = "orange")
sns.kdeplot(data = df_train[df_train["pdays"] != -1], x = "pdays", ax = ax[4], color = "orange")
sns.kdeplot(data = df_train[df_train["previous"] != 0], x = "previous", ax = ax[5], color = "orange")

ax[0].set_title("KDE Plot for Age Feature")
ax[1].set_title("KDE Plot for Balance Feature")
ax[2].set_title("KDE Plot for Duration Feature")
ax[3].set_title("KDE Plot for Campaign Feature")
ax[4].set_title("KDE Plot for Pdays not Equal to -1")
ax[5].set_title("KDE Plot for Previous not Equal to 0")

plt.show()


def chi_square_test(df: pd.DataFrame, feature: str, target: str ="y", alpha=0.05):
    """
    Performs Chi-Square Test of Independence to determine if there is a 
    statistically significant relationship between a categorical feature 
    and the target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
        
    feature : str
        The categorical feature to analyze (e.g., 'job', 'education', 'marital')
        
    target : str, optional (default='y')
        The target variable (binary: 0/1 or 'no'/'yes')
        
    alpha : float, optional (default=0.05)
        Significance level for hypothesis testing
    
    Returns:
    --------
    dict : Contains:
        - 'feature': Name of feature tested
        - 'chi2_statistic': Chi-Square test statistic
        - 'p_value': Two-tailed p-value
        - 'degrees_of_freedom': Degrees of freedom
        - 'is_significant': Boolean (True if p < alpha)
        - 'alpha': Significance level used
        - 'crosstab': The observed frequency table
        - 'expected': The expected frequency table (if independent)
        - 'interpretation': Human-readable result
    """
    
    # Validate inputs
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in dataframe")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe")
    
    # Convert target to binary if needed
    df_copy = df.copy()
    if df_copy[target].dtype == 'object':
        df_copy[target] = (df_copy[target].str.lower() == 'yes').astype(int)
    
    # Create crosstab (observed frequencies)
    crosstab = pd.crosstab(df_copy[feature], df_copy[target])
    
    # Perform Chi-Square test
    chi2_stat, p_value, dof, expected_freq = chi2_contingency(crosstab)
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Create interpretation
    if is_significant:
        interpretation = (
            f"âœ… SIGNIFICANT RELATIONSHIP DETECTED (p < {alpha})\n"
            f"   â†’ There IS a statistically significant relationship between\n"
            f"      '{feature}' and the target variable.\n"
            f"   â†’ Chi-Square = {chi2_stat:.4f}, p-value = {p_value:.6f}\n"
            f"   â†’ The feature has predictive power."
        )
    else:
        interpretation = (
            f"â�Œ NO SIGNIFICANT RELATIONSHIP (p â‰¥ {alpha})\n"
            f"   â†’ There is NO statistically significant relationship between\n"
            f"      '{feature}' and the target variable.\n"
            f"   â†’ Chi-Square = {chi2_stat:.4f}, p-value = {p_value:.6f}\n"
        )
    
    results = {
        "feature": feature,
        "chi2_statistic": chi2_stat,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "is_significant": is_significant,
        "alpha": alpha,
        "crosstab": crosstab,
        "expected": expected_freq,
        "interpretation": interpretation
    }
    
    return results


# ========================================================================================================================

def cramers_v(df: pd.DataFrame, feature: str, target="y"):
    """
    Calculates CramÃ©r's V statistic to measure the strength of association 
    between a categorical feature and the target variable.
    
    CramÃ©r's V normalizes the Chi-Square statistic to account for sample size,
    providing a measure of effect size between 0 and 1.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
        
    feature : str
        The categorical feature to analyze
        
    target : str, optional (default='y')
        The target variable (binary: 0/1 or 'no'/'yes')
    
    Returns:
    --------
    dict : Contains:
        - 'feature': Name of feature
        - 'cramers_v': CramÃ©r's V coefficient (0 to 1)
        - 'effect_size': Categorical interpretation ('negligible', 'small', 'medium', 'large')
        - 'n_observations': Sample size
        - 'interpretation': Human-readable explanation of strength
    
    Formula:
    --------
    CramÃ©r's V = âˆš[ Ï‡Â² / (n Ã— min(k-1, r-1)) ]
    
    Where:
        Ï‡Â² = Chi-Square statistic
        n = Sample size
        k = Number of columns (target categories)
        r = Number of rows (feature categories)
    
    Interpretation Guide: (Cohen's General Rule for Cramer's V)
    ---------------------
    V = 0:      No association
    V < 0.1:    Negligible association
    0.1 â‰¤ V < 0.3:  Small association
    0.3 â‰¤ V < 0.5:  Medium association
    V â‰¥ 0.5:    Large association
    """
    
    # Validate inputs
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in dataframe")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe")
    
    # Convert target to binary if needed
    df_copy = df.copy()
    if df_copy[target].dtype == 'object':
        df_copy[target] = (df_copy[target].str.lower() == 'yes').astype(int)
    
    # Create crosstab
    crosstab = pd.crosstab(df_copy[feature], df_copy[target])
    
    # Get Chi-Square statistic
    chi2_stat, _, _, _ = chi2_contingency(crosstab)
    
    # Get sample size
    n = df_copy.shape[0]
    
    # Get dimensions
    min_dim = min(crosstab.shape) - 1 # min(k-1, r-1). It finds the lesser number of categories of either variable.
    
    # Calculate CramÃ©r's V
    v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0
    
    # Determine effect size category
    if v < 0.01:
        effect_size = "No associate"
    elif v < 0.1:
        effect_size = "negligible"
    elif v < 0.3:
        effect_size = "small"
    elif v < 0.5:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    # Create interpretation
    interpretation = (
        f"CramÃ©r's V = {v:.4f}\n"
        f"   Effect Size: {effect_size.upper()}\n"
        f"   Strength: {_get_strength_description(v)}"
    )
    
    results = {
        'feature': feature,
        'cramers_v': v,
        'effect_size': effect_size,
        'n_observations': n,
        'interpretation': interpretation
    }
    
    return results


def _get_strength_description(v):
    """Helper function to provide descriptive strength interpretation"""
    if v < 0.01:
        return "Virtually no relationship. Feature has negligible predictive power."
    elif v < 0.1:
        return "Very weak relationship. Feature has limited predictive power."
    elif v < 0.3:
        return "Weak to moderate relationship. Feature has some predictive power."
    elif v < 0.5:
        return "Moderate to strong relationship. Feature has good predictive power."
    else:
        return "Very strong relationship. Feature is highly predictive."


# ========================================================================================================================

def two_proportion_test(df: pd.DataFrame, feature: str, category: str, target: str ="y", alpha=0.05):
    """
    Performs Two Proportion Z-Test to determine if the subscription rate 
    within a specific category differs significantly from the overall rate.
    
    This test answers: "Does this category's subscription rate differ 
    significantly from the general population?"
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
        
    feature : str
        The categorical feature (e.g., 'job', 'education')
        
    category : str
        The specific category within the feature (e.g., 'admin.', 'tertiary')
        
    target : str, optional (default='y')
        The target variable (binary)
        
    alpha : float, optional (default=0.05)
        Significance level
    
    Returns:
    --------
    dict : Contains:
        - 'feature': Name of feature
        - 'category': Specific category tested
        - 'n_overall': Total sample size
        - 'n_category': Sample size within category
        - 'subscriptions_overall': Total subscriptions
        - 'subscriptions_category': Subscriptions in category
        - 'p_overall': Overall subscription rate
        - 'p_category': Category subscription rate
        - 'z_statistic': Z-test statistic
        - 'p_value': Two-tailed p-value
        - 'is_significant': Boolean (True if p < alpha)
        - 'interpretation': Human-readable result
    
    Hypothesis:
    -----------
    Hâ‚€: p_category = p_overall (category rate equals overall rate)
    Hâ‚�: p_category â‰  p_overall (category rate differs from overall rate)
    """
    
    # Validate
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found")
    if category not in df[feature].values:
        raise ValueError(f"Category '{category}' not found in feature '{feature}'")
    
    # Convert target to binary
    df_copy = df.copy()
    if df_copy[target].dtype == 'object':
        df_copy[target] = (df_copy[target].str.lower() == 'yes').astype(int)
    
    # Calculate overall statistics
    n_overall = df_copy.shape[0]
    subscriptions_overall = (df_copy[target] == 1).sum()
    p_overall = subscriptions_overall / n_overall # Rate of people who subscribed for target variable
    
    # Calculate category statistics
    category_data = df_copy[df_copy[feature] == category]
    n_category = category_data.shape[0]
    subscriptions_category = (category_data[target] == 1).sum()
    p_category = subscriptions_category / n_category # Rate of people who subscibed in feature (e.g., job)
    
    # Perform Two Proportion Z-Test
    count = np.array([subscriptions_overall, subscriptions_category])
    nobs = np.array([n_overall, n_category])
    z_statistic, p_value = proportions_ztest(count, nobs)
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Create interpretation
    difference_pp = (p_category - p_overall) * 100
    direction = "HIGHER" if p_category > p_overall else "LOWER"
    
    if is_significant:
        interpretation = (
            f"âœ… SIGNIFICANT DIFFERENCE (p < {alpha})\n"
            f"   â†’ Category '{category}' has a SIGNIFICANTLY {direction} rate\n"
            f"   â†’ Category: {p_category*100:.2f}% | Overall: {p_overall*100:.2f}%\n"
            f"   â†’ Difference: {difference_pp:+.2f} percentage points\n"
            f"   â†’ This category shows distinct behavior"
        )
    else:
        interpretation = (
            f"â�Œ NO SIGNIFICANT DIFFERENCE (p â‰¥ {alpha})\n"
            f"   â†’ Category '{category}' rate is NOT significantly different\n"
            f"   â†’ Category: {p_category*100:.2f}% | Overall: {p_overall*100:.2f}%\n"
            f"   â†’ Difference: {difference_pp:+.2f} percentage points\n"
            f"   â†’ This category behaves like the general population"
        )
    
    results = {
        'feature': feature,
        'category': category,
        'n_overall': n_overall,
        'n_category': n_category,
        'subscriptions_overall': subscriptions_overall,
        'subscriptions_category': subscriptions_category,
        'p_overall': p_overall,
        'p_category': p_category,
        'z_statistic': z_statistic,
        'p_value': p_value,
        'is_significant': is_significant,
        'interpretation': interpretation
    }
    
    return results


# ========================================================================================================================

def analyze_categorical_feature(df: pd.DataFrame, feature: str, target: str ="y", alpha=0.05, show_all_data = False):
    """
    MASTER FUNCTION: Performs complete analysis of a categorical feature's 
    relationship with the target variable.
    
    Workflow:
    ---------
    STEP 1: Chi-Square Test
        Is there ANY relationship? (Yes/No)
        
    STEP 2: CramÃ©r's V (if Chi-Square is significant)
        How STRONG is the relationship? (Effect size)
        
    STEP 3: Two Proportion Tests for Each Category
        WHICH categories drive the relationship?
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
        
    feature : str
        The categorical feature to analyze
        
    target : str, optional (default='y')
        The target variable (binary)
        
    alpha : float, optional (default=0.05)
        Significance level for all tests
    
    Returns:
    --------
    dict : Comprehensive results containing:
        - 'feature': Name of feature
        - 'chi_square': Chi-Square test results
        - 'cramers_v': CramÃ©r's V results (if significant)
        - 'two_proportion_tests': Results for each category
        - 'summary': Overall summary and recommendations
        - 'should_continue': Boolean (True if feature is significant)
    
    Workflow Example:
    -----------------
    >>> results = analyze_categorical_feature(df_train, 'job')
    
    If Chi-Square is NOT significant:
        Analysis stops here
        Feature is likely independent from target
        Skip feature in modeling
    
    If Chi-Square IS significant:
        Calculate CramÃ©r's V to measure strength
        Test each job category against overall rate
        Identify which categories drive subscription
        Recommend for feature engineering
    """

    if show_all_data:
        print("\n" + "=" * 90)
        print(f"CATEGORICAL FEATURE ANALYSIS: {feature.upper()}")
        print("=" * 90)
        
        print(f"\n[STEP 1/3] Chi-Square Test of Independence")
        print("-" * 90)


    # ==================== STEP 1: CHI-SQUARE TEST ====================
    chi_square_results = chi_square_test(df, feature, target, alpha)

    if show_all_data:
        print(f"Feature: {feature}")
        print(f"Chi-Square Statistic: {chi_square_results['chi2_statistic']:.4f}")
        print(f"P-value: {chi_square_results['p_value']:.6f}")
        print(f"Degrees of Freedom: {chi_square_results['degrees_of_freedom']}")
        print(f"\n{chi_square_results['interpretation']}")
    
    # If NOT significant, stop here
    if not chi_square_results["is_significant"]:
        print(f"\n{'!' * 90}")
        print("âš ï¸�  ANALYSIS STOPPED HERE")
        print("Since there is NO significant relationship, further analysis is not warranted.")
        print("This feature appears to be independent from the target variable.")
        print(f"{'!' * 90}\n")
        
        results = {
            'feature': feature,
            'chi_square': chi_square_results,
            'cramers_v': None,
            'two_proportion_tests': None,
            'summary': "Feature shows no significant relationship with target. Skip in modeling.",
            'should_continue': False
        }
        
        return results
    
    # ==================== STEP 2: CRAMÃ‰R'S V ====================
    
    
    cramers_results = cramers_v(df, feature, target)

    if show_all_data:
        print(f"\n[STEP 2/3] Effect Size Analysis (CramÃ©r's V)")
        print("-" * 90)
        print(f"CramÃ©r's V: {cramers_results['cramers_v']:.4f}")
        print(f"Effect Size Category: {cramers_results['effect_size'].upper()}")
        print(f"\n{cramers_results['interpretation']}")


        
    # ==================== STEP 3: TWO PROPORTION TESTS ====================
    
    # Convert target for category checking
    df_copy = df.copy()
    if df_copy[target].dtype == "object":
        df_copy[target] = (df_copy[target].str.lower() == "yes").astype(int)
    
    categories = sorted(df_copy[feature].unique())
    two_proportion_results = {}
    if show_all_data:
        print(f"\n[STEP 3/3] Two Proportion Tests for Each Category")
        print("-" * 90)
        print(f"\nTesting {len(categories)} categories in '{feature}':\n")
    
    for i, category in enumerate(categories, 1):
        tp_result = two_proportion_test(df, feature, category, target, alpha)
        two_proportion_results[category] = tp_result

        if show_all_data:
            significance = "âœ… SIGNIFICANT" if tp_result["is_significant"] else "â�Œ NOT SIGNIFICANT"
            rate_diff = (tp_result["p_category"] - tp_result["p_overall"]) * 100
    
            cat_field = 20 if i < 10 else 19
            
            print(f"  {i}. {category:{cat_field}} | Rate: {tp_result['p_category']*100:6.2f}% | " # {category:20}: Place this variable within a 20 character field
                  f"Overall: {tp_result['p_overall']*100:6.2f}% | "
                  f"Diff: {rate_diff:+6.2f}pp | {significance}")
        
    # ==================== SUMMARY ====================
    if show_all_data:
        print(f"\n{'=' * 90}")
        print("SUMMARY & RECOMMENDATIONS")
        print(f"{'=' * 90}\n")
        
    # Count significant categories
    significant_categories = [cat for cat, res in two_proportion_results.items() 
                             if res["is_significant"]]

    if show_all_data:
        summary = (
            f"Feature: {feature}\n"
            f"Chi-Square P-value: {chi_square_results['p_value']:.6f} â†’ SIGNIFICANT âœ…\n"
            f"CramÃ©r's V: {cramers_results['cramers_v']:.4f} â†’ {cramers_results['effect_size'].upper()}\n"
            f"Categories with significant difference: {len(significant_categories)}/{len(categories)}\n\n"
        )
        
        if cramers_results["cramers_v"] < 0.1:
            summary += "âš ï¸�  NOTE: While statistically significant, the effect size is SMALL.\n"
            summary += "Consider including for completeness but monitor in model.\n\n"
        elif cramers_results["cramers_v"] >= 0.3:
            summary += "ğŸ’ª STRONG EFFECT: This feature has substantial predictive power.\n"
            summary += "Highly recommended for inclusion in model.\n\n"
        else:
            summary += "âœ… MODERATE EFFECT: This feature has decent predictive power.\n"
            summary += "Include in model and consider feature engineering.\n\n"
        
        if significant_categories:
            summary += f"Categories driving the relationship:\n"
            for cat in significant_categories:
                res = two_proportion_results[cat]
                direction = "Higher" if res["p_category"] > res["p_overall"] else "Lower"
                summary += f"  â€¢ {cat}: {res['p_category']*100:.2f}% {direction}\n"
    
           
        print(summary)
    else:
        summary = "There is no summary because you set show-all_data parameter is False"


    
    results = {
        "feature": feature,
        "chi_square": chi_square_results,
        "cramers_v": cramers_results,
        "two_proportion_tests": two_proportion_results,
        "summary": summary,
        "should_continue": True
    }
    
    return results


# ========================================================================================================================

def display_styled_summary(df: pd.DataFrame):
    """Display a nicely formatted table"""
    
    # Styling
    def style_effect_size(val):
        if val >= 0.30:
            return 'background-color: #90EE90'  # Green
        elif val >= 0.10:
            return 'background-color: #FFFFE0'  # Yellow
        else:
            return 'background-color: #FFB6C6'  # Red
    
    styled = (df.style
              .format({'Cramers V': '{:.4f}', 'P_Value': '{:.6f}'})
              .map(style_effect_size,subset=['Cramers V'])
             ) 

    return styled

def analyze_all_categorical_features(df: pd.DataFrame = df_train, target: str ="y", alpha=0.05, features=None, show_all_data= False):
    """
    Analyzes ALL categorical features in the dataset using the master function.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
        
    target : str, optional (default='y')
        The target variable
        
    alpha : float, optional (default=0.05)
        Significance level
        
    features : list, optional
        Specific features to analyze. If None, analyzes all non-numeric columns
    
    Returns:
    --------
    dict : Results for all features with 'significant_features' summary
    """
    
    if features is None:
        # Auto-detect categorical features (exclude target)
        features = df.select_dtypes(include=["object"]).columns.tolist()
        if target in features:
            features.remove(target)
    
    all_results = {}
    significant_features = []
    
    for feature in features:
        result = analyze_categorical_feature(df, feature, target, alpha)
        all_results[feature] = result
        
        if result["should_continue"]:
            significant_features.append({
                'feature': feature,
                'cramers_v': result['cramers_v']['cramers_v'],
                'effect_size': result['cramers_v']['effect_size']
            })
    
    # Sort by CramÃ©r's V
    significant_features.sort(key=lambda x: x['cramers_v'], reverse=True)
    
    # Print final summary
    print("\n" + "=" * 90)
    print("FINAL SUMMARY - ALL CATEGORICAL FEATURES")
    print("=" * 90)
    print(f"\nTotal features analyzed: {len(features)}")
    print(f"Significant features (p < {alpha}): {len(significant_features)}\n")
    
    if significant_features:
        print("SIGNIFICANT FEATURES RANKED BY EFFECT SIZE:")
        print("-" * 90)
        data = []
        for i, feat in enumerate(significant_features, 1):
            data.append({
                "Feature"   : feat["feature"],
                "Cramers V" : feat["cramers_v"],
                "Effect"    : feat["effect_size"]
            })
            
            #print(f"{i}. {feat['feature']:20} | Cramer's V: {feat['cramers_v']:.4f} | "
            #      f"Effect: {feat['effect_size'].upper()}")
            
        df_summary = pd.DataFrame(data = data).sort_values("Cramers V", ascending = False)
        html_table = display_styled_summary(df_summary).to_html()
        display(HTML(html_table))
    else:
        print("âš ï¸� No significant features found.")
    #return {
        #'all_results': all_results,
        #'significant_features': significant_features,
        #'total_features': len(features)
    #}


analyze_all_categorical_features(df_train)


# Plot correlation map for numeric faetures
df_numeric = df_train[numeric_features].copy()
corr_ = df_numeric.corr(method = "pearson")
mask_ = np.triu(np.ones_like(corr_, dtype = bool), k = 1)

plt.figure(figsize = (12, 6))
sns.heatmap(
    corr_,
    mask = mask_,
    cmap = "RdBu_r",
    center = 0,
    square = True,
    linewidths = 2,
    linecolor = "white",
    annot = True,
    fmt = ".3f"
)

plt.title(
    "Correlation Matrix for Numeric Features",
    fontsize = 15,
    fontweight = "bold",
    color = "#1a57fa"

)

plt.show()


def scatter2d_byTarget(dataframe: pd.DataFrame,
                      x: str,
                      y: str,
                      target: str = "y"):
    """
        Parameters:
        -----------
        df : pandas.DataFrame
            The input dataset
            
        x : str
            First numeric feature (X-axis)
            
        y : str
            Second numeric feature (Y-axis)
            
        target : str, optional (default='y')
    
    
    """
    plt.figure(figsize = (12, 6))
    sns.scatterplot(data = dataframe,
                   x = x,
                   y = y,
                   hue = target,
                   palette = "Set2",
                   )

    plt.title(f"{x.upper()} vs {y.upper()} by {target.upper()}", fontsize = 15, fontweight = "bold", color = "#AB4632")
    plt.show()


scatter2d_byTarget(df_train, x = "balance", y = "duration", target = "y")


scatter2d_byTarget(df_train, x = "balance", y = "campaign", target = "y")


scatter2d_byTarget(df_train, x = "balance", y = "age", target = "y")


scatter2d_byTarget(df_train, x = "age", y = "duration", target = "y")


scatter2d_byTarget(df_train, x = "age", y = "campaign", target = "y")


scatter2d_byTarget(df_train, x = "duration", y = "campaign", target = "y")


def hexbin2d_byTarget(dataframe: pd.DataFrame, x: str, y: str, target: str = "y"):

    """
        This fucntion plots 2 dimension hexbin for relationship between numeric features to target (No:0, Yes:1)

         Parameters:
        -----------
        df : pandas.DataFrame
            The input dataset
            
        x : str
            First numeric feature (X-axis)
            
        y : str
            Second numeric feature (Y-axis)
            
        target : str, optional (default='y')
    """

    subset_yes  = dataframe[dataframe[target] == 1].copy()
    subset_no   = dataframe[dataframe[target] == 0].copy()

    # Calculate minimum count parameter in hexbin for each of target class
    total_yes = subset_yes.shape[0]
    total_no  = subset_no.shape[0]

    mincount_yes = 100
    mincount_no = int(mincount_yes * (total_no/total_yes))

    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 5))

    # Plot hexbin for target == 1
    ax[0].hexbin(
        subset_yes[x],
        subset_yes[y],
        gridsize = 25,
        cmap = "YlOrRd",
        mincnt = mincount_yes,
        edgecolors = "black",
        linewidths = 0.2,
        alpha = 0.9
    )
    # Plot hexbin for target == 0
    ax[1].hexbin(
        subset_no[x],
        subset_no[y],
        gridsize = 25,
        cmap = "YlOrRd",
        mincnt = mincount_no,
        edgecolors = "black",
        linewidths = 0.2,
        alpha = 0.9
    )

    # Set c and y labels for each ax
    ax[0].set_xlabel(x)
    ax[0].set_ylabel(y)
    ax[1].set_xlabel(x)
    ax[1].set_ylabel(y)

    # Set title for each ax
    ax[0].set_title(f"{x.upper()} and {y.upper()} by Target = 1")
    ax[1].set_title(f"{x.upper()} and {y.upper()} Target = 0")

    # Show graph
    plt.show()


hexbin_features = {
    1: ["balance", "duration"],
    2: ["balance", "campaign"],
    3: ["balance", "age"],
    4: ["age", "duration"],
    5: ["age", "campaign"]
}

for key, value in hexbin_features.items():
    x = value[0]
    y = value [1]
    hexbin2d_byTarget(dataframe = df_train, x = x, y = y, target = "y")


plt.figure(figsize = (8, 6))
sns.boxplot(data = df_train, x = "poutcome", y = "pdays")

plt.show()


# Add difference between campaing to previous
df_train["sum_contacts"] = df_train["campaign"] + df_train["previous"]
df_test["sum_contacts"]  = df_test["campaign"] + df_test["previous"]

# Remove previous feature.
df_train = df_train.drop(["previous", "campaign"], axis = 1).copy()
df_test  = df_test.drop(["previous", "campaign"], axis = 1).copy()


#replace -1 with 999 for pdays
df_train["pdays"] = df_train["pdays"].replace(-1, 999)
df_test["pdays"] = df_test["pdays"].replace(-1, 999)


# Catagorize duration
def categorize_duration(d):
    if d < 200:
        return "Rejected_Fast" # 0-200 sn
    elif d < 500:
        return "Uncertain"     # 200-500 sn 
    elif d < 1000:
        return "Interested"    # 500-1000 sn 
    elif d < 2500:
        return "Hot_Lead"      # 1000-2500 sn 
    else:
        return "Outlier_Long"  # Too long

df_train["call_segment"] = df_train["duration"].apply(categorize_duration)
df_test["call_segment"] = df_test["duration"].apply(categorize_duration)




# Log transformation of balance and duration features beacause of outliers.
df_train["balance"] = np.sign(df_train["balance"]) * np.log1p(np.abs(df_train["balance"])) #balance feature has negative values so I used this.
df_test["balance"] = np.sign(df_test["balance"]) * np.log1p(np.abs(df_test["balance"]))

df_train["duration"] = np.log1p(df_train["duration"])
df_test["duration"] = np.log1p(df_test["duration"])



# merge job and education categoric fatures 
df_train["job_education"] = df_train["job"] + "_" + df_train["education"]
df_test["job_education"] = df_test["job"] + "_" + df_test["education"]

df_train = df_train.drop(["job", "education"], axis = 1)
df_test = df_test.drop(["job", "education"], axis = 1)


df_train["housing"] = df_train["housing"].map({"no": 0, "yes": 1})
df_test["housing"]  = df_test["housing"].map({"no": 0, "yes": 1})

df_train["loan"] = df_train["loan"].map({"no": 0, "yes": 1})
df_test["loan"]  = df_test["loan"].map({"no": 0, "yes": 1})

df_train["default"] = df_train["default"].map({"no": 0, "yes": 1})
df_test["default"]  = df_test["default"].map({"no": 0, "yes": 1})

df_train["loan_state"] = df_train["loan"] + df_train["housing"] + df_train["default"]
df_test["loan_state"] = df_test["loan"] + df_test["housing"] + df_test["default"]

df_train = df_train.drop(["housing", "loan", "default"], axis = 1)
df_test = df_test.drop(["housing", "loan", "default"], axis = 1)


# One Hot Vector to Categorial Features
cat_features_one_hot_vector = ["job_education" ,"marital", "contact", "day", "month", "poutcome", "loan_state", "call_segment"]

df_train_encoded = pd.get_dummies(df_train, columns = cat_features_one_hot_vector, drop_first = False)
df_test_encoded  = pd.get_dummies(df_test, columns = cat_features_one_hot_vector, drop_first = False)

one_hot_features = df_train_encoded.select_dtypes(include = ["bool"]).columns

df_train_encoded[one_hot_features] = df_train_encoded[one_hot_features].astype(int)
df_test_encoded[one_hot_features]  = df_test_encoded[one_hot_features].astype(int)


df_train = df_train_encoded.copy()
df_test  = df_test_encoded.copy()



from sklearn.preprocessing import MinMaxScaler, RobustScaler


numeric_features = [ "balance", "duration","age", "sum_contacts", "pdays"]
scaler = RobustScaler()

df_train[numeric_features] = scaler.fit_transform(df_train[numeric_features])
df_test[numeric_features] = scaler.transform(df_test[numeric_features])


from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


target = "y"
feature = [col for col in df_train.columns if not col == "y"]
print(f"We have a {len(feature)} in dataset and we ara trying to predict y variable.")


X_train = df_train[feature].copy()
y_train = df_train[target].copy()

X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                    y_train,
                                                    test_size = 0.1,
                                                    random_state = 42,
                                                    stratify = y_train  #Preserve class disstribution
                                                   )

X_sub  = df_test[feature].copy()

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
print(f"We are using KFold with {cv.n_splits} splits")




# Models used

LogisticRegression_model = LogisticRegression(l1_ratio = 0.1,
                                           max_iter = 1000,
                                           solver = "saga", # Faster for large datasets
                                           #class_weight = "balanced", #Give more attetion to minority class
                                           random_state = 42
                                       )



RandomForestClassifier_model = RandomForestClassifier(n_estimators = 100,
                                                      max_depth = 10,
                                                      min_samples_split = 20, # Minimum number of observations before node splitting
                                                      min_samples_leaf = 10, # Minimum number of observations in a leaf node (overfitting prevention)
                                                      n_jobs = -1,
                                                      #class_weight = "balanced",
                                                      random_state = 42)

XGBoost_model = xgb.XGBClassifier(n_estimators = 10000,
                                  objective = "binary:logistic",
                                  min_child_weight = 10, # At least N examples in leaf nodes
                                  reg_lambda = 10,
                                  reg_alpha = 0.5,
                                  subsample = 1, # What percentage of the dataset is used in each iteration?
                                  colsample_bytree= 0.8,
                                  device = "cpu",  # If you have gpu you should write gpu here
                                  max_depth = 5,
                                  learning_rate = 0.1,
                                  scale_pos_weight= 1, # Useful for unbalanced classes. sum(negative instances) / sum(positive instances)
                                  eval_metric = "logloss",
                                  tree_method = "hist",
                                  early_stopping_rounds = 250 # If the validation loss does not improve over 250 iterations, automatically stop training.
                                 )


CatBoostClassifier_model = CatBoostClassifier(loss_function = "Logloss",
                                              iterations = 2000,
                                              depth = 5,
                                              learning_rate = 0.1,
                                              l2_leaf_reg = 10,
                                              eval_metric = "Logloss",
                                              custom_metric = ["Accuracy", "Precision", "Recall"],
                                              grow_policy = "Depthwise",
                                              min_data_in_leaf = 20,
                                              #auto_class_weights = "Balanced", # Useful for unbalanced classes,
                                              #class_weights={0: 1.0, 1: 7.0}, #Class weights with manuel. Dont use with "auto_class_weights" parameter
                                              boosting_type =  "Plain", #For Faster model
                                              boost_from_average = True, # For Faster convergence use it
                                              use_best_model = True,
                                              verbose = 250,
                                              early_stopping_rounds = 250,
                                              random_state = 42
                                             )

models = {
    "Logistic Regression": LogisticRegression_model,
    "Random Forest Classifier": RandomForestClassifier_model,
    "XGBOOST": XGBoost_model,
    "CatBoost": CatBoostClassifier_model

}

print(f"Models to test: {', '.join(models.keys())}")
print()


def create_models(
                    X: pd.DataFrame,
                    y: pd.Series,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    cv: StratifiedKFold,
                    model,
                    model_name: str

):

    test_preds = []
    models = []
    test_metrics = {
        "Fold"         : [],
        "Accuracy"     : [],
        "Precision"    : [],
        "Recall"       : [],
        "F1"           : [],
        "ROC_AUC"      : []
    }


    train_metrics = {
        "Fold"               : [],
        "Accuracy_Train"     : [],
        "Precision_Train"    : [],
        "Recall_Train"       : [],
        "F1_Train"           : [],
        "ROC_AUC_Train"        : [],
        "Accuracy_Val"       : [],
        "Precision_Val"      : [],
        "Recall_Val"         : [],
        "F1_Val"             : [],
        "ROC_AUC_Val"        : []

    }
    elapsed_time = 0

    for idx, (train_idx, val_idx) in enumerate(cv.split(X = X, y = y)):
        start_time = time.time()

        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()

        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx].copy()

        try:# For XGBoost, CatBoost
            model.fit(X_train,
                      y_train,
                      eval_set = [(X_train, y_train), (X_val, y_val)],
                      verbose = 250)
        except: #For LR and RF
            model.fit(X_train, y_train)


        fold_train_pred = model.predict(X_train)
        fold_val_pred   = model.predict(X_val)
        fold_test_pred  = model.predict(X_test)

        try:
            fold_test_pred_proba = model.predict_proba(X_test)[:, 1]
            fold_train_pred_proba = model.predict_proba(X_train)[:, 1]
            fold_val_pred_proba = model.predict_proba(X_val)[:, 1]

            fold_roc_auc_score = roc_auc_score(y_test, fold_test_pred_proba)
            fold_roc_auc_score_train = roc_auc_score(y_train, fold_train_pred_proba)
            fold_roc_auc_score_val = roc_auc_score(y_val, fold_val_pred_proba)

        except:
            fold_roc_auc_score = 0
            fold_roc_auc_score_train = 0
            fold_roc_auc_score_val = 0

        test_preds.append(fold_test_pred)
# ------------------------------------------TEST SET----------------------------------------------------------
        # Determine performance metrics for fold on test set
        fold_acc_score = accuracy_score(y_true = y_test, y_pred = fold_test_pred)
        fold_recall_score = recall_score(y_true = y_test, y_pred = fold_test_pred)
        fold_precision_score = precision_score(y_true = y_test, y_pred = fold_test_pred)
        fold_f1_score = f1_score(y_true = y_test, y_pred = fold_test_pred, zero_division = 0)


        # Add fold performance metrics to model performance metric on test set
        test_metrics["Fold"].append(idx + 1)
        test_metrics["Accuracy"].append(fold_acc_score)
        test_metrics["Precision"].append(fold_precision_score)
        test_metrics["Recall"].append(fold_recall_score)
        test_metrics["F1"].append(fold_f1_score)
        test_metrics["ROC_AUC"].append(fold_roc_auc_score)
# ------------------------------------------TRAIN and VAL SET----------------------------------------------------------
        # Determine performance metrics for fold on train set
        fold_acc_score_train = accuracy_score(y_true = y_train, y_pred = fold_train_pred)
        fold_recall_score_train = recall_score(y_true = y_train, y_pred = fold_train_pred)
        fold_precision_score_train = precision_score(y_true = y_train, y_pred = fold_train_pred)
        fold_f1_score_train = f1_score(y_true = y_train, y_pred = fold_train_pred, zero_division = 0)


        # Add fold performance metrics to model performance metric on train set
        train_metrics["Fold"].append(idx + 1)
        train_metrics["Accuracy_Train"].append(fold_acc_score_train)
        train_metrics["Precision_Train"].append(fold_precision_score_train)
        train_metrics["Recall_Train"].append(fold_recall_score_train)
        train_metrics["F1_Train"].append(fold_f1_score_train)
        train_metrics["ROC_AUC_Train"].append(fold_roc_auc_score_train)



        # Determine performance metrics for fold on val set
        fold_acc_score_val = accuracy_score(y_true = y_val, y_pred = fold_val_pred)
        fold_recall_score_val = recall_score(y_true = y_val, y_pred = fold_val_pred)
        fold_precision_score_val = precision_score(y_true = y_val, y_pred = fold_val_pred)
        fold_f1_score_val = f1_score(y_true = y_val, y_pred = fold_val_pred, zero_division = 0)


        # Add fold performance metrics to model performance metric
        train_metrics["Accuracy_Val"].append(fold_acc_score_val)
        train_metrics["Precision_Val"].append(fold_precision_score_val)
        train_metrics["Recall_Val"].append(fold_recall_score_val)
        train_metrics["F1_Val"].append(fold_f1_score_val)
        train_metrics["ROC_AUC_Val"].append(fold_roc_auc_score_val)
#-----------------------------------------------------------------------------------------


        models.append(model)

        # Calculate fold time
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time to {model_name} Model for Fold {idx + 1}: {elapsed}")

        # Show Metrics (Test, Train, Validation)
        print(f"  Fold {idx + 1} Metrics:")
        print(f"    {'Metric':<15} {'Train':<12} {'Val':<12} {'Test':<12}")
        print(f"    {'-'*51}")
        print(f"    {'Accuracy':<15} {fold_acc_score_train:.4f}     {fold_acc_score_val:.4f}     {fold_acc_score:.4f}")
        print(f"    {'Precision':<15} {fold_precision_score_train:.4f}     {fold_precision_score_val:.4f}     {fold_precision_score:.4f}")
        print(f"    {'Recall':<15} {fold_recall_score_train:.4f}     {fold_recall_score_val:.4f}     {fold_recall_score:.4f}")
        print(f"    {'F1-Score':<15} {fold_f1_score_train:.4f}     {fold_f1_score_val:.4f}     {fold_f1_score:.4f}")
        print(f"    {'ROC-AUC':<15} {fold_roc_auc_score_train:.4f}     {fold_roc_auc_score_val:.4f}     {fold_roc_auc_score:.4f}")

        elapsed_time += elapsed


    print(f"{model_name} model run time is : {elapsed_time} seconds")

    print(f"\n{model_name} - SUMMARY STATISTICS")
    print("=" * 100)
    print(f"{'Metric':<15} {'Train Mean':<15} {'Val Mean':<15} {'Test Mean':<15}")
    print("-" * 100)
    print(f"{'Accuracy':<15} {np.mean(train_metrics['Accuracy_Train']):.4f}         {np.mean(train_metrics['Accuracy_Val']):.4f}         {np.mean(test_metrics['Accuracy']):.4f}")
    print(f"{'Precision':<15} {np.mean(train_metrics['Precision_Train']):.4f}         {np.mean(train_metrics['Precision_Val']):.4f}         {np.mean(test_metrics['Precision']):.4f}")
    print(f"{'Recall':<15} {np.mean(train_metrics['Recall_Train']):.4f}         {np.mean(train_metrics['Recall_Val']):.4f}         {np.mean(test_metrics['Recall']):.4f}")
    print(f"{'F1-Score':<15} {np.mean(train_metrics['F1_Train']):.4f}         {np.mean(train_metrics['F1_Val']):.4f}         {np.mean(test_metrics['F1']):.4f}")
    print(f"{'ROC-AUC':<15} {np.mean(train_metrics['ROC_AUC_Train']):.4}         {np.mean(train_metrics['ROC_AUC_Val']):.4f}         {np.mean(test_metrics['ROC_AUC']):.4f}")
    print("=" * 100)

    return {
        "Test Metrics": test_metrics,
        "Train Metrics": train_metrics,
        "Models": models,
        "Total Time": elapsed_time
    }


length_ = int(len(X_train)/10)
cat_return = create_models(X = X_train[:length_],
                           y = y_train[:length_],
                           X_test = X_test[:length_],
                           y_test = y_test[:length_],
                           cv = cv,
                           model = models["CatBoost"],
                           model_name = "CatBoost")


length_ = int(len(X_train)/10)
xgb_return = create_models(X = X_train[:length_],
                           y = y_train[:length_],
                           X_test = X_test[:length_],
                           y_test = y_test[:length_],
                           cv = cv,
                           model = models["XGBOOST"],
                           model_name = "XGBOOST")


length_ = int(len(X_train)/10)
lgb_return = create_models(X = X_train[:length_],
                           y = y_train[:length_],
                           X_test = X_test[:length_],
                           y_test = y_test[:length_],
                           cv = cv,
                           model = models["Logistic Regression"],
                           model_name = "Logistic Regression")


length_ = int(len(X_train)/10)
RF_return = create_models(X = X_train[:length_],
                           y = y_train[:length_],
                           X_test = X_test[:length_],
                           y_test = y_test[:length_],
                           cv = cv,
                           model = models["Random Forest Classifier"],
                           model_name = "Random Forest Classifier")


# Step 1: Collect all results and model names in a list
all_models_results = [
    ("XGBoost", xgb_return),
    ("CatBoost", cat_return),
    ("Logistic Regression", lgb_return), 
    ("Random Forest", RF_return)
]

# Step 2: Extract data and append to the list
results_data = []

for model_name, result in all_models_results:
    # Retrieve the "Test Metrics" dictionary returned
    metrics = result["Test Metrics"]
    
    # Calculate the average of each metric (across 5 Folds)
    results_data.append({
        "Model": model_name,
        "Accuracy": np.mean(metrics["Accuracy"]),
        "Precision": np.mean(metrics["Precision"]),
        "Recall": np.mean(metrics["Recall"]),
        "F1 Score": np.mean(metrics["F1"]),
        "ROC-AUC": np.mean(metrics["ROC_AUC"])
    })

# Step 3: Create the DataFrame
df_results = pd.DataFrame(results_data)

# Step 4: Sorting by F1 Score in descending order
df_results = df_results.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)


# Melt the dataframe into long format
df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(15, 8))
ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="viridis")


# Iterate over each bar container
for container in ax.containers:
    ax.bar_label(container, 
                 fmt='%.4f',    # Show 4 decimal places (e.g., 0.8541)
                 padding=3,     # Distance of the text from the bar
                 fontsize=9,    # Font size
                 rotation=45)   # Rotate text (45 degrees for better fit)

plt.title("Compare Models Performance (Test Set)", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Submision the results of XGBOOST
xgb_model = models["XGBOOST"]


xgb_model.fit(X_train,
              y_train,
              eval_set = [(X_train, y_train), (X_test, y_test)],
              verbose = 250)

results_sub = 1- xgb_model.predict_proba(X_sub)

df_sub = pd.read_csv(config.dir_sub)
df_sub["y"] = results_sub

df_sub = df_sub.set_index("id")

df_sub.to_csv("/kaggle/working/submission.csv")

