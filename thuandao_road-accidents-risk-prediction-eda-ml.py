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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Machine learning
from sklearn.linear_model import Ridge, Lasso
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

# Optuna
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
df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

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


# Drop column id
df_train.drop(columns="id", axis=1, inplace=True)
list_test_id = df_test["id"].copy().to_list()
df_test.drop("id", axis=1, inplace=True)

num_features = ["curvature"]
cat_features = ["num_lanes", "speed_limit", "num_reported_accidents", "road_type", "lighting", "weather",
                "time_of_day", "road_signs_present", "public_road", "holiday", "school_season"]

print("Train Data describe:")
cm = sns.light_palette("blue", as_cmap=True)
display(df_train[num_features].describe().T.style.background_gradient(cmap=cm))

print("\nTest Data describe:")
display(df_test[num_features].describe().T.style.background_gradient(cmap=cm))


def convert_cat(features, df):
    for feature in features:
        if feature in df.columns:
            df[feature] = df[feature].astype("category")

convert_cat(features=cat_features, df=df_train)
convert_cat(features=cat_features, df=df_test)

# Display information about the DataFrames
print("Train Data Info:")
df_train.info()

print("\nTest Data Info:")
df_test.info()


print("Train Data describe:")
display(df_train[cat_features].describe().T.style.background_gradient(cmap="Blues", subset=["unique", "freq"]))

print("\nTest Data describe:")
display(df_test[cat_features].describe().T.style.background_gradient(cmap="Blues", subset=["unique", "freq"]))


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
        if len(outliers) > 0:
            outlier_info.append({
                "Feature": feature,
                "Outlier Count": len(outliers),
                # "Outlier Detail": outliers.tolist()
            })
    
    if len(outlier_info) == 0:
        print("âœ… No outliers detected in the selected features.")
        return None
    else:
        return pd.DataFrame(outlier_info)

checking_outlier(list_feature=num_features, df=df_train, dataset_name="Training data")


checking_outlier(list_feature=num_features, df=df_test, dataset_name="Test data")


def color(n_colors=2):
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    positions = np.linspace(0, 1, n_colors)
    colors = [cmap(p) for p in positions]
    return colors


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

sns.boxplot(data=df_train, y = "accident_risk", ax=ax[0], color="#00BFC4")
ax[0].set_title(f"Box plot of Accident Risk", fontsize=14, pad=20, weight="bold")
ax[0].grid(axis="x", color="gray", linestyle=":", linewidth=0.7)
ax[0].set_ylabel("Accident Risk")
sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)

sns.histplot(data=df_train, x = "accident_risk", ax=ax[1], color="#00BFC4", kde=True, bins=40)
ax[1].set_title(f"Histogram of Accident Risk", fontsize=14, pad=20, weight="bold")
ax[1].set_xlabel("Accident Risk")
ax[1].set_ylabel("Frequency")
# ax[1].grid(axis="x", color="gray", linestyle=":", linewidth=0.7)
sns.despine(ax=ax[1], top=True, right=True, left=False, bottom=False)

# Extract KDE values to find peaks
kde = sns.kdeplot(df_train["accident_risk"], ax=ax[1], color="#00BFC4").lines[0].get_data()
kde_x, kde_y = kde[0], kde[1]
peaks, _ = find_peaks(kde_y)

# Highlight peaks
for peak_idx in peaks:
    plt.plot(kde_x[peak_idx], kde_y[peak_idx], "ro")  # Red dots on peaks

plt.tight_layout()
plt.show()


def plot_numerical_features(df_train, df_test, num_features):
    colors = color(n_colors=2)  # The color function you defined earlier
    n = len(num_features)

    fig, ax = plt.subplots(n, 1, figsize=(10, n * 4))
    if n == 1:
        ax = [ax]  # Ensure ax is iterable when there is only one feature

    for i, feature in enumerate(num_features):
        # Combine data for violin plot
        df_plot = pd.concat([
            pd.DataFrame({"Dataset": "Train data", feature: df_train[feature]}),
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
        # ax[i].grid(color="gray", linestyle=":", linewidth=0.7)
        sns.despine(left=False, bottom=False, ax=ax[i])

    plt.tight_layout()
    plt.show()

# Call the function
plot_numerical_features(
    df_train=df_train,
    df_test=df_test,
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

num_features_train = ["curvature", "accident_risk"]
num_features_test = ["curvature"]
skew_feature_train, skew_train_df = check_skewness(df_train, "Train Data", numerical_features=num_features_train)
skew_feature_test, skew_test_df = check_skewness(df_test, "Test Data", numerical_features=num_features_test)


corr_matrix = df_train.drop(columns=cat_features, axis=1).corr(numeric_only=True, method="pearson")
# one_like can build a matrix of boolean(True, False) with the same shape as our data
ones_corr = np.ones_like(corr_matrix, dtype=bool)
mask = np.triu(ones_corr)
adjusted_mask = mask[1:, :-1]
adjusted_cereal_corr = corr_matrix.iloc[1:, :-1]

fig, ax = plt.subplots(figsize = (8, 7))
# That method uses HUSL colors, so you need hue, saturation, and lightness. 
# I used hsluv.org to select the colors of this chart.
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

sns.heatmap(data=adjusted_cereal_corr, mask=adjusted_mask,
            annot=True, fmt=".2f", cmap=cmap,
            vmin=-1, vmax=1, linecolor="white", linewidths=0.5)

title = "Correlation Matrix Composition\n"
ax.set_title(title, loc="center", fontsize=14, weight="bold", pad=20)

plt.tight_layout()
plt.show()


# Group num_reported_accidents
def group_num_accidents(row):
    if row == 0:
        return "0"
    elif row == 1:
        return "1"
    elif row == 2:
        return "2"
    else:
        return "> 2"
df_train["num_reported_accidents_group"] = df_train["num_reported_accidents"].apply(group_num_accidents)
df_test["num_reported_accidents_group"] = df_test["num_reported_accidents"].apply(group_num_accidents)

# Display few rows of each dataset
print("Train Data Preview:")
display(df_train.head())

print("\nTest Data Preview:")
display(df_test.head())


cat_features = ["num_lanes", "speed_limit", "num_reported_accidents_group", "road_type", "lighting", "weather",
                "time_of_day", "road_signs_present", "public_road", "holiday", "school_season"]

def plot_categorical_distribution(cat_features, df_train, df_test, order=None):
    for feature in cat_features:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

        # Determine order dynamically if not provided
        if order is None:
            unique_vals = sorted(df_train[feature].dropna().unique())
        else:
            unique_vals = order

        # COUNT PLOT â€“ TRAIN
        sns.countplot(data=df_train, x=feature, ax=ax[0, 0],
                      palette=color(n_colors=len(unique_vals)), order=unique_vals)
        ax[0, 0].set_title(f"[Train] Count plot of {feature}", fontsize=13, pad=12, weight="bold")
        # ax[0, 0].grid(axis="y", linestyle=":", linewidth=0.7)
        sns.despine(ax=ax[0, 0], left=False, bottom=False)
        for container in ax[0, 0].containers:
            ax[0, 0].bar_label(container, fmt='%d', label_type="edge", fontsize=10, weight="bold")

        # COUNT PLOT â€“ TEST
        sns.countplot(data=df_test, x=feature, ax=ax[0, 1],
                      palette=color(n_colors=len(unique_vals)), order=unique_vals)
        ax[0, 1].set_title(f"[Test] Count plot of {feature}", fontsize=13, pad=12, weight="bold")
        # ax[0, 1].grid(axis="y", linestyle=":", linewidth=0.7)
        sns.despine(ax=ax[0, 1], left=False, bottom=False)
        for container in ax[0, 1].containers:
            ax[0, 1].bar_label(container, fmt='%d', label_type="edge", fontsize=10, weight="bold")

        # PIE CHART â€“ TRAIN
        train_percent = df_train[feature].value_counts(normalize=True) * 100
        train_percent = train_percent.reindex(unique_vals).fillna(0)
        wedges, texts, autotexts = ax[1, 0].pie(train_percent.values, labels=train_percent.index,
                                                autopct='%1.1f%%', startangle=90, shadow= True,
                                                colors=color(n_colors=len(unique_vals)))
        ax[1, 0].set_title(f"[Train] Percentage Distribution of {feature}", pad=10, weight="bold")

        # PIE CHART â€“ TEST
        test_percent = df_test[feature].value_counts(normalize=True) * 100
        test_percent = test_percent.reindex(unique_vals).fillna(0)
        wedges, texts, autotexts = ax[1, 1].pie(test_percent.values, labels=test_percent.index,
                                                autopct='%1.1f%%', startangle=90, shadow= True,
                                                colors=color(n_colors=len(unique_vals)))
        ax[1, 1].set_title(f"[Test] Percentage Distribution of {feature}", pad=10, weight="bold")

        plt.tight_layout()
        plt.show()

plot_categorical_distribution(cat_features=cat_features, df_train = df_train, df_test = df_test)


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
target_feature = "accident_risk"
def perform_statical_testing(total_categories, feature, df_train = df_train, target_feature = target_feature):
    cal_normaltest(cat_feature=feature, num_feature=target_feature, df=df_train)
    if total_categories == 2:
        cal_mannwhitneyu(dataframe=df_train, categorical_feature=feature, num_feature=target_feature)
    else:
        perform_kruskal_test(df=df_train, categorical_feature=feature, numeric_feature=target_feature)

def plot_categorical_distribution_by_target_feature(feature, df_train = df_train, target_feature = target_feature, order = None):
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
        Mean_target_feature = (target_feature, "mean"),
        Median_target_feature = (target_feature, "median"),
        Std_target_feature = (target_feature, "std")
    )
    df_summary_feature = df_summary_feature.sort_values(by="Mean_target_feature", ascending=False)    

    summary_data = [
        ("Total Categories", f"{df_summary_feature.shape[0]}"),
        ("Overall Target Mean", f"{df_train[target_feature].mean():.2f}")
    ]
    summary_html = "<ul>" + "".join([f"<li><b>{k}:</b> {v}</li>" for k, v in summary_data]) + "</ul>"
    display(HTML(summary_html))
    display(df_summary_feature.style.background_gradient(cmap=cm).set_table_attributes('style="width:75%; margin:auto;"'))

    perform_statical_testing(total_categories=df_summary_feature.shape[0], 
                             feature=feature, df_train=df_train, target_feature=target_feature)

    # Plot distribution
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.violinplot(x=feature, y=target_feature, data=df_train, hue=feature, 
                palette=color(n_colors=df_train[feature].nunique()), ax=ax)
    ax.set_title(f"Violin plot of {target_feature} distribution by {feature}", pad=15, weight = "bold")
    ax.set_xlabel(feature, labelpad=10)
    ax.set_ylabel(target_feature, labelpad=10)
    # plt.grid(axis="y", color="gray", linestyle=":", alpha=0.7)
    sns.despine(left=False, bottom=False, ax=ax)

    plt.tight_layout()
    plt.show()

for feature in cat_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {target_feature} by {feature}</b></h2>"))
    plot_categorical_distribution_by_target_feature(feature=feature)


# high_risk_condition
def classify_high_risk(row):
    if row["speed_limit"] >= 60 and row["weather"] in ["foggy", "rainy"]:
        return "High risk"
    return "Normal risk"

df_train["high_risk_condition"] = df_train.apply(classify_high_risk, axis=1)
df_test["high_risk_condition"] = df_test.apply(classify_high_risk, axis=1)

# expected_light_condition
def check_expected_light(row):
    if row["time_of_day"] in ["morning", "afternoon"] and row["lighting"] != "daylight":
        return "unexpected"
    elif row["time_of_day"] == "evening" and row["lighting"] not in ["dim", "night"]:
        return "unexpected"
    else:
        return "expected"

df_train["expected_light_condition"] = df_train.apply(check_expected_light, axis=1)
df_test["expected_light_condition"] = df_test.apply(check_expected_light, axis=1)

# holiday_public_risk
def check_holiday_public(row):
    if row["holiday"] == True and row["public_road"] == True:
        return "holiday_public"
    elif row["holiday"] == True and row["public_road"] == False:
        return "holiday_private"
    else:
        return "non_holiday"

df_train["holiday_public_risk"] = df_train.apply(check_holiday_public, axis=1)
df_test["holiday_public_risk"] = df_test.apply(check_holiday_public, axis=1)

# school_rush_hour_risk
def check_school_rush_hour(row):
    if row["school_season"] == True and row["time_of_day"] in ["morning", "afternoon"]:
        return "school_rush_hour"
    else:
        return "normal"

df_train["school_rush_hour_risk"] = df_train.apply(check_school_rush_hour, axis=1)
df_test["school_rush_hour_risk"] = df_test.apply(check_school_rush_hour, axis=1)

# Holiday night: holiday + night time
def check_holiday_night(row):
    if row["holiday"] == True and row["lighting"] == "night":
        return "Yes"
    else:
        return "No"

df_train["is_holiday_night"] = df_train.apply(check_holiday_night, axis=1)
df_test["is_holiday_night"] = df_test.apply(check_holiday_night, axis=1)

# Polynomial features for key numerical variables
df_train["curvature_squared"] = df_train["curvature"] ** 2
df_test["curvature_squared"] = df_test["curvature"] ** 2

df_train["curvature_cubed"] = df_train["curvature"] ** 3
df_test["curvature_cubed"] = df_test["curvature"] ** 3

df_train["speed_squared"] = df_train["speed_limit"].astype("int8") ** 2
df_test["speed_squared"] = df_test["speed_limit"].astype("int8") ** 2

# Core interactions
df_train["curv_speed"] = df_train["curvature"] * df_train["speed_limit"].astype("int8")
df_test["curv_speed"] = df_test["curvature"] * df_test["speed_limit"].astype("int8")

df_train["lane_speed"] = df_train["num_lanes"].astype("int8") * df_train["speed_limit"].astype("int8")
df_test["lane_speed"] = df_test["num_lanes"].astype("int8") * df_test["speed_limit"].astype("int8")

df_train["accidents_speed"] = df_train["num_reported_accidents"].astype("int8") * df_train["speed_limit"].astype("int8")
df_test["accidents_speed"] = df_test["num_reported_accidents"].astype("int8") * df_test["speed_limit"].astype("int8")

df_train["accidents_curv"] = df_train["num_reported_accidents"].astype("int8") * df_train["curvature"]
df_test["accidents_curv"] = df_test["num_reported_accidents"].astype("int8") * df_test["curvature"]

# Risk scores
df_train["risk_intensity"] = (df_train["curvature"] * df_train["speed_limit"].astype("int8")) / 50
df_test["risk_intensity"] = (df_test["curvature"] * df_test["speed_limit"].astype("int8")) / 50

df_train["lane_capacity_risk"] = (5 - df_train["num_lanes"].astype("int8")) * df_train["speed_limit"].astype("int8")
df_test["lane_capacity_risk"] = (5 - df_test["num_lanes"].astype("int8")) * df_test["speed_limit"].astype("int8")

df_train["accidents_per_lane"] = df_train["num_reported_accidents"].astype("int8") / (df_train["num_lanes"].astype("int8") + 1)
df_test["accidents_per_lane"] = df_test["num_reported_accidents"].astype("int8") / (df_test["num_lanes"].astype("int8") + 1)

# Binary indicators
df_train["high_risk_combo"] = ((df_train["curvature"] > 0.5) & 
                                (df_train["speed_limit"].astype("int8") >= 60)).astype(int)
df_test["high_risk_combo"] = ((df_test["curvature"] > 0.5) & 
                                (df_test["speed_limit"].astype("int8") >= 60)).astype(int)

# Drop column "num_reported_accidents"
df_train.drop(columns="num_reported_accidents", axis=1, inplace=True)
df_test.drop(columns="num_reported_accidents", axis=1, inplace=True)


new_cat_features = ["high_risk_condition", "expected_light_condition", "holiday_public_risk", "school_rush_hour_risk", 
                    "is_holiday_night", "high_risk_combo"]

new_num_features = ["curvature_squared", "curvature_cubed", "speed_squared", "curv_speed", "lane_speed", "accidents_speed",
                    "accidents_curv", "risk_intensity", "lane_capacity_risk", "accidents_per_lane"]

cat_features = ["num_lanes", "speed_limit", "num_reported_accidents_group", "road_type", "lighting", "weather",
                "time_of_day", "road_signs_present", "public_road", "holiday", "school_season", "high_risk_condition",
                "expected_light_condition", "holiday_public_risk", "school_rush_hour_risk", "is_holiday_night", "high_risk_combo"]

convert_cat(features=cat_features, df=df_train)
convert_cat(features=cat_features, df=df_test)

# # We need to update the data for the columns, this helps to reduce memory.
# df_train = df_train.astype({
#     "curvature": "float16",
#     "accident_risk": "float16",
#     "curvature_squared": "float16",
#     "curvature_cubed": "float16",
#     "curv_speed": "float16",
#     "accidents_curv": "float16",
#     "risk_intensity": "float16",
#     "accidents_per_lane": "float16"
# })

# df_test = df_test.astype({
#     "curvature": "float16",
#     "curvature_squared": "float16",
#     "curvature_cubed": "float16",
#     "curv_speed": "float16",
#     "accidents_curv": "float16",
#     "risk_intensity": "float16",
#     "accidents_per_lane": "float16"
# })

# Display information about the DataFrames
print("Train Data Info:")
df_train.info()

print("\nTest Data Info:")
df_test.info()


corr_matrix = df_train.drop(columns=cat_features, axis=1).corr(numeric_only=True)
# one_like can build a matrix of boolean(True, False) with the same shape as our data
ones_corr = np.ones_like(corr_matrix, dtype=bool)
mask = np.triu(ones_corr)
adjusted_mask = mask[1:, :-1]
adjusted_cereal_corr = corr_matrix.iloc[1:, :-1]

fig, ax = plt.subplots(figsize = (8, 7))
# That method uses HUSL colors, so you need hue, saturation, and lightness. 
# I used hsluv.org to select the colors of this chart.
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

sns.heatmap(data=adjusted_cereal_corr, mask=adjusted_mask,
            annot=True, fmt=".2f", cmap=cmap,
            vmin=-1, vmax=1, linecolor="white", linewidths=0.5)

title = "Correlation Matrix Composition\n"
ax.set_title(title, loc="center", fontsize=14, weight="bold", pad=20)

plt.tight_layout()
plt.show()


for feature in new_cat_features:
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:blue;'><b>Distribution of {target_feature} by {feature}</b></h2>"))
    plot_categorical_distribution_by_target_feature(feature=feature)


num_features_train = ["curvature_squared", "curvature_cubed", "speed_squared", "curv_speed", "lane_speed", "accidents_speed",
                      "accidents_curv", "risk_intensity", "lane_capacity_risk", "accidents_per_lane", "accident_risk"]
num_features_test = ["curvature_squared", "curvature_cubed", "speed_squared", "curv_speed", "lane_speed", "accidents_speed",
                      "accidents_curv", "risk_intensity", "lane_capacity_risk", "accidents_per_lane"]
skew_feature_train, skew_train_df = check_skewness(df_train, "Train Data", numerical_features=num_features_train)
skew_feature_test, skew_test_df = check_skewness(df_test, "Test Data", numerical_features=num_features_test)


from sklearn.preprocessing import PowerTransformer

def handle_skewed_features(
    df,
    zero_threshold=0.9,
    skew_threshold=0.5,
    num_features=None,
    exclude_cols=None
    
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


processed_train_df, transformed_columns, sparse_columns, skewed_columns = handle_skewed_features(df=df_train, num_features=skew_feature_train)
num_features_train = ["PT_curvature_squared", "PT_curvature_cubed", "speed_squared", "PT_curv_speed", "PT_lane_speed", "PT_accidents_speed",
                      "PT_accidents_curv", "PT_risk_intensity", "PT_lane_capacity_risk", "PT_accidents_per_lane", "accident_risk"]
skew_feature_train, skew_train_df = check_skewness(data=processed_train_df, numerical_features=num_features_train,
                                                   dataset_name= "Train data")


processed_test_df, transformed_columns, sparse_columns, skewed_columns = handle_skewed_features(df=df_test, num_features=skew_feature_test)
num_features_test = ["PT_curvature_squared", "PT_curvature_cubed", "speed_squared", "PT_curv_speed", "PT_lane_speed", "PT_accidents_speed",
                      "PT_accidents_curv", "PT_risk_intensity", "PT_lane_capacity_risk", "PT_accidents_per_lane"]
skew_feature_test, skew_test_df = check_skewness(data=processed_test_df, numerical_features=num_features_test,
                                                   dataset_name= "Test data")


checking_outlier(list_feature=num_features_train, df=processed_train_df, dataset_name="Training data")


checking_outlier(list_feature=num_features_test, df=processed_test_df, dataset_name="Test data")


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(processed_train_df, processed_train_df["high_risk_condition"]):
    start_train_set = processed_train_df.iloc[train_index]
    start_val_set = processed_train_df.iloc[test_index]


df_train_new = start_train_set.drop("accident_risk", axis=1)
df_train_label_new = start_train_set["accident_risk"].copy()


list_feature_num_stand = ["PT_curvature_squared", "PT_curvature_cubed", "speed_squared", "PT_curv_speed", "PT_lane_speed", "PT_accidents_speed",
                          "PT_accidents_curv", "PT_risk_intensity", "PT_lane_capacity_risk"]
list_feature_cat_onehot = ["num_lanes", "speed_limit", "num_reported_accidents_group", "road_type", "lighting", "weather",
                           "time_of_day", "road_signs_present", "public_road", "holiday", "school_season", "high_risk_condition",
                           "expected_light_condition", "holiday_public_risk", "school_rush_hour_risk", "is_holiday_night", "high_risk_combo"]
list_feature_num_robust = ["PT_accidents_per_lane"]


num_robust_transformer = Pipeline(steps=[
    ("scaler", RobustScaler()),
    ("imputer", SimpleImputer(strategy="median"))
])

num_stand_transformer = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("imputer", SimpleImputer(strategy="median"))
])

cat_onehot_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_robust", num_robust_transformer, list_feature_num_robust),
    ("num_standard", num_stand_transformer, list_feature_num_stand),
    ("cat_onehot", cat_onehot_transformer, list_feature_cat_onehot)
])

preprocessor.fit(df_train_new)

df_train_new_prepared = preprocessor.transform(df_train_new)
list_feature_prepared = preprocessor.get_feature_names_out().tolist()
list_feature_prepared


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


# Function to evaluate regression models
def evaluate_model(model, X_train, X_val, y_train, y_val):
    RESET = "\033[0m"
    BLUE = "\033[94m"
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    y_val_real = y_val
    y_pred_real = y_pred
    
    # Metrics: RMSE
    rmse = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
    print(f"Model: {model.__class__.__name__}{RESET}")
    print(f"Root Mean Squared Error (RMSE): {BLUE}{rmse:.4f}{RESET}")
    print("-" * 80)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # ----- Plot 1: Predicted vs. Actual -----
    axs[0].scatter(y_val_real, y_pred_real, alpha=0.4, color="royalblue", edgecolors="none", rasterized=True, s=8)
    axs[0].plot(
        [y_val_real.min(), y_val_real.max()],
        [y_val_real.min(), y_val_real.max()],
        "r--", lw=2, label="Perfect Prediction (y=x)"
    )
    axs[0].set_xlabel("Actual Values (Accident Risk)")
    axs[0].set_ylabel("Predicted Values (Accident Risk)")
    axs[0].set_title("Predicted vs. Actual (Validation Set)", fontsize=14, weight="bold", pad=20)
    axs[0].legend()
    axs[0].grid(True, alpha=0.2)

    # ----- Plot 2: Residual Plot -----
    residuals = y_val_real - y_pred_real
    axs[1].scatter(y_val_real, residuals, alpha=0.5, color="royalblue", edgecolors="none", rasterized=True, s=8)
    axs[1].axhline(0, color="red", linestyle="--", lw=2)
    axs[1].set_xlabel("Actual Values (Accident Risk)")
    axs[1].set_ylabel("Prediction Error (Residuals)")
    axs[1].set_title("Residual Plot", fontsize=14, weight="bold", pad=20)
    axs[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


X_val = start_val_set.drop("accident_risk", axis=1)
y_val = start_val_set["accident_risk"].copy()
X_val_prepared = preprocessor.transform(X_val)


param_cb = {
	"bootstrap_type": "Bernoulli",
	"iterations": 4295,
	"depth": 7,
	"learning_rate": 0.01191499084022728,
	"l2_leaf_reg": 0.22534606586222317,
	"border_count": 229,
	"subsample": 0.9889297683948703,
	"random_seed": 42,
	"loss_function": "RMSE",
	"eval_metric": "RMSE",
	"verbose": 0,
	"allow_writing_files": False,
    "task_type": "GPU"
}

model_cb = CatBoostRegressor(**param_cb)
evaluate_model(model = model_cb, X_train = df_train_new_prepared, 
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val)


param_lgbm = {
    "n_estimators": 847,
    "learning_rate": 0.013874553617662705,
    "max_depth": 12,
    "num_leaves": 247,
    "min_child_samples": 5,
    "subsample": 0.8475456932944789,
    "colsample_bytree": 0.7616639678117201,
    "reg_alpha": 0.6825559182811963,
    "reg_lambda": 1.2521080121201102,
    "boosting_type": "gbdt",
    # fixed params
    "objective": "rmse",
    "metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

model_lgbm_gbdt = LGBMRegressor(**param_lgbm)
evaluate_model(model = model_lgbm_gbdt, X_train = df_train_new_prepared, 
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val)


param_xgb = {
    "n_estimators": 2189, 
    "max_depth": 11, 
    "learning_rate": 0.03639366292085837, 
    "subsample": 0.9666713809959383, 
    "colsample_bytree": 0.8477146935580193, 
    "colsample_bylevel": 0.940078973046911, 
    "colsample_bynode": 0.9267147332508152, 
    "gamma": 0.025540535127965093, 
    "min_child_weight": 19, 
    "reg_alpha": 0.004730358315321747, 
    "reg_lambda": 0.06096607718373648,
	"random_state": 42,
	"tree_method": "gpu_hist",      # GPU training
	"device": "cuda",
	"verbosity": 0,
	"n_jobs": -1,
	"objective": "reg:squarederror",
	"eval_metric": "rmse"
}

model_xgb = XGBRegressor(**param_xgb)
evaluate_model(model = model_xgb, X_train = df_train_new_prepared, 
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val)


ests = [("cb", model_cb), ("xgb", model_xgb), ("lgbm", model_lgbm_gbdt)]
preds = {name: m.predict(X_val_prepared) for name, m in ests}
rmse_each = {name: np.sqrt(mean_squared_error(y_val, preds[name])) for name,_ in ests}

display(rmse_each)

A = np.column_stack([preds[name] for name,_ in ests])  # (n_val, n_models)

def obj_w(trial):
    w = np.array([trial.suggest_float(f"w_{i}", 0.0, 5.0) for i in range(A.shape[1])])
    if w.sum() == 0: return 1e9
    y_hat = A.dot(w / w.sum())
    return np.sqrt(mean_squared_error(y_val, y_hat))

study_w = optuna.create_study(direction="minimize")
study_w.optimize(obj_w, n_trials=1000, show_progress_bar=True)
w = np.array([study_w.best_params[f"w_{i}"] for i in range(A.shape[1])])
weights = (w / w.sum()).tolist()
print("Best weights:", weights)


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
voting_reg = VotingRegressor(estimators=[
    ("cb", model_cb),
    ("xgb", model_xgb),
    ("lgbm", model_lgbm_gbdt)
], n_jobs=1, weights=weights)

cv_scores = cross_val_score(
    voting_reg,
    X=df_train_new_prepared,
    y=df_train_label_new,
    cv=kfold,
    scoring="neg_root_mean_squared_error",
    n_jobs=1
)

mean_score = -cv_scores.mean()
std_score = cv_scores.std()

print(f"Cross-validated RMSE (mean Â± std): {mean_score:.4f} Â± {std_score:.4f}")

evaluate_model(model = voting_reg, X_train = df_train_new_prepared, 
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val)


df_test_prepared = preprocessor.transform(processed_test_df)
y_pred_test = voting_reg.predict(df_test_prepared)

submission_df = pd.DataFrame({
    "id": list_test_id,
    "accident_risk": y_pred_test
})

submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file saved!")
submission_df.head(10)


fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(15, 6))

sns.boxplot(data=submission_df, y = "accident_risk", ax=ax[0], color="#00BFC4")
ax[0].set_title(f"Box plot of Beats Per Minute", fontsize=14, pad=20, weight="bold")
ax[0].grid(axis="x", color="gray", linestyle=":", linewidth=0.7)
ax[0].set_ylabel("Accident risk")
sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)

sns.histplot(data=submission_df, x = "accident_risk", ax=ax[1], color="#00BFC4", kde=True, bins=40)
ax[1].set_title(f"Histogram of Beats Per Minute", fontsize=14, pad=20, weight="bold")
ax[1].set_xlabel("Accident risk")
ax[1].set_ylabel("Frequency")
sns.despine(ax=ax[1], top=True, right=True, left=False, bottom=False)

# Extract KDE values to find peaks
kde = sns.kdeplot(submission_df["accident_risk"], ax=ax[1], color="#00BFC4").lines[0].get_data()
kde_x, kde_y = kde[0], kde[1]
peaks, _ = find_peaks(kde_y)

# Highlight peaks
for peak_idx in peaks:
    plt.plot(kde_x[peak_idx], kde_y[peak_idx], "ro")  # Red dots on peaks

plt.tight_layout()
plt.show()


shap_plot(model=voting_reg.named_estimators_["cb"], X_test=df_test_prepared[:1000],
          list_feature=list_feature_prepared, type="bar")


shap_plot(model=voting_reg.named_estimators_["cb"], X_test=df_test_prepared[:1000],
          list_feature=list_feature_prepared)

