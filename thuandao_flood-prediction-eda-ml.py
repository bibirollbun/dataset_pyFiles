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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500) # To display all the columns of dataframe
pd.set_option("max_colwidth", None) # To set the width of the column to maximum


# Load the datasets
# train.csv: Features and target labels
# test.csv: Features only

df_train = pd.read_csv("/kaggle/input/playground-series-s4e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e5/test.csv")

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


df_train.drop("id", axis=1, inplace=True)
list_test_id = df_test["id"].copy().to_list()
df_test.drop("id", axis=1, inplace=True)


# Check memory before
print("Before conversion:")
print(df_train.info(memory_usage="deep"))

# Identify integer columns (excluding target)
int_cols = df_train.select_dtypes(include=["int64"]).columns.tolist()

# Convert integer columns to int8
df_train[int_cols] = df_train[int_cols].astype("int8")
df_test[int_cols] = df_test[int_cols].astype("int8")

# Check after conversion
print("\nAfter conversion:")
print(df_train.info(memory_usage="deep"))
print(df_test.info(memory_usage="deep"))


print("Train Data describe:")
cmap = sns.light_palette("blue", as_cmap=True)
display(df_train[int_cols].describe().T.style.background_gradient(cmap=cmap))

print("\nTest Data describe:")
display(df_test[int_cols].describe().T.style.background_gradient(cmap=cmap))


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
    print("=" * 50)
    print(f"ğŸ”� {dataset_name} - Checking Outliers")
    print("=" * 50)
    
    outlier_info = []

    for feature in list_feature:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]
        outlier_count = len(outliers)
        total_count = len(df)
        outlier_percent = (outlier_count / total_count) * 100

        if outlier_count > 0:
            outlier_info.append({
                "Feature": feature,
                "Outlier Count": outlier_count,
                "Outlier %": round(outlier_percent, 4)
            })
    
    if len(outlier_info) == 0:
        print("âœ… No outliers detected in the selected features.")
    else:
        outlier_df = pd.DataFrame(outlier_info).sort_values(by="Outlier %", ascending=False).reset_index(drop=True)
        print(f"\nâš ï¸� Outlier Summary ({dataset_name}):")
        display(outlier_df)
        print(f"\nTotal features with outliers: {len(outlier_df)}/{len(list_feature)}")


checking_outlier(list_feature=int_cols, df=df_train, dataset_name="Training Data")


checking_outlier(list_feature=int_cols, df=df_train, dataset_name="Test Data")


def color(n_colors=2):
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    positions = np.linspace(0, 1, n_colors)
    colors = [cmap(p) for p in positions]
    return colors


from scipy.signal import find_peaks
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(15, 6))

sns.boxplot(data=df_train, y = "FloodProbability", ax=ax[0], color="#00BFC4")
ax[0].set_title(f"Box plot of Flood Probability", fontsize=14, pad=20, weight="bold")
ax[0].grid(axis="x", color="gray", linestyle=":", linewidth=0.7)
ax[0].set_ylabel("Flood Probability")
sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)

sns.histplot(data=df_train, x = "FloodProbability", ax=ax[1], color="#00BFC4", kde=True, bins=40)
ax[1].set_title(f"Histogram of Flood Probability", fontsize=14, pad=20, weight="bold")
ax[1].set_xlabel("Flood Probability")
ax[1].set_ylabel("Frequency")
# ax[1].grid(axis="x", color="gray", linestyle=":", linewidth=0.7)
sns.despine(ax=ax[1], top=True, right=True, left=False, bottom=False)

# Extract KDE values to find peaks
kde = sns.kdeplot(df_train["FloodProbability"], ax=ax[1], color="#00BFC4").lines[0].get_data()
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

        sns.violinplot(data=df_plot, x=feature, y="Dataset",  palette=colors, ax=ax[i])
        ax[i].set_title(f"Violin plot of {feature}", pad=20, weight="bold")
        # ax[i].grid(color="gray", linestyle=":", linewidth=0.7)
        sns.despine(left=False, bottom=False, ax=ax[i])

    plt.tight_layout()
    plt.show()

# Call the function
plot_numerical_features(df_train=df_train, df_test=df_test, num_features=int_cols)


def check_skewness(data, dataset_name, numerical_features = int_cols, highlight=True, sort=True):
    skewness_dict = {}
    skew_feature = []
    for feature in numerical_features:
        skew = data[feature].skew(skipna=True)
        skewness_dict[feature] = skew

    skew_df = pd.DataFrame.from_dict(skewness_dict, orient="index", columns=["Skewness"])
    if sort:
        skew_df = skew_df.reindex(skew_df["Skewness"].abs().sort_values(ascending=False).index)
    
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

num_features_train = ["MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation", "Urbanization", 
                      "ClimateChange", "DamsQuality", "Siltation", "AgriculturalPractices", "Encroachments", 
                      "IneffectiveDisasterPreparedness", "DrainageSystems", "CoastalVulnerability", "Landslides", 
                      "Watersheds", "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss", "InadequatePlanning", 
                      "PoliticalFactors", "FloodProbability"]
num_features_test = ["MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation", "Urbanization", 
                      "ClimateChange", "DamsQuality", "Siltation", "AgriculturalPractices", "Encroachments", 
                      "IneffectiveDisasterPreparedness", "DrainageSystems", "CoastalVulnerability", "Landslides", 
                      "Watersheds", "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss", "InadequatePlanning", 
                      "PoliticalFactors"]

skew_feature_train, skew_train_df = check_skewness(df_train, "Train Data", numerical_features=num_features_train)
skew_feature_test, skew_test_df = check_skewness(df_test, "Test Data", numerical_features=num_features_test)


# --- Train correlation ---
corr_train = df_train[num_features_train].corr(numeric_only=True)
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
adjusted_mask_train = mask_train[1:, :-1]
adjusted_corr_train = corr_train.iloc[1:, :-1]

# --- Test correlation ---
corr_test = df_test[num_features_test].corr(numeric_only=True)
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))
adjusted_mask_test = mask_test[1:, :-1]
adjusted_corr_test = corr_test.iloc[1:, :-1]

# --- Set up subplots ---
fig, ax = plt.subplots(1, 2, figsize=(25, 7))

# --- Custom color map ---
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

# --- Train Heatmap ---
sns.heatmap(data=adjusted_corr_train, mask=adjusted_mask_train, annot=True, fmt=".2f", 
            cmap=cmap, vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[0])
ax[0].set_title("Correlation Matrix (Train Data)", fontsize=14, weight="bold", pad=15)
# --- Test Heatmap ---
sns.heatmap(data=adjusted_corr_test, mask=adjusted_mask_test, annot=True, fmt=".2f",
            cmap=cmap, vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[1])
ax[1].set_title("Correlation Matrix (Test Data)", fontsize=14, weight="bold", pad=15)
# --- Layout ---
plt.tight_layout()
plt.show()


def feature_engineering(df):
    df = df.copy()
    
    # --- Interaction Features ---
    df["Rain_Urban"] = df["MonsoonIntensity"] * df["Urbanization"]
    df["Forest_River"] = df["Deforestation"] * df["RiverManagement"]
    df["Dams_Drainage"] = df["DamsQuality"] * df["DrainageSystems"]

    # --- Group Risk Indices ---
    natural_cols = [
        "MonsoonIntensity", "TopographyDrainage", "Landslides",
        "ClimateChange", "Siltation", "Watersheds"
    ]
    df["NaturalRisk"] = df[natural_cols].mean(axis=1)

    human_cols = [
        "Urbanization", "Deforestation", "DrainageSystems",
        "DamsQuality", "InadequatePlanning", "DeterioratingInfrastructure"
    ]
    df["InfrastructureRisk"] = df[human_cols].mean(axis=1)

    governance_cols = [
        "Encroachments", "IneffectiveDisasterPreparedness",
        "PoliticalFactors", "PopulationScore"
    ]
    df["GovernanceRisk"] = df[governance_cols].mean(axis=1)

    # --- Nonlinear Transformations ---
    nonlinear_cols = ["MonsoonIntensity", "Urbanization", "DrainageSystems"]
    for col in nonlinear_cols:
        df[f"{col}_squared"] = df[col].astype(np.int16) ** 2
        df[f"{col}_sqrt"] = np.sqrt(df[col].astype(np.float32))
    
    return df


df_train_fe = feature_engineering(df_train)
df_test_fe  = feature_engineering(df_test)

# Check the shape difference
print(f"Train shape before: {df_train.shape} â†’ after: {df_train_fe.shape}")
print(f"Test shape before:  {df_test.shape}  â†’ after: {df_test_fe.shape}")

# Display few rows of each dataset
print("Train Data Preview:")
display(df_train_fe.head())

print("\nTest Data Preview:")
display(df_test_fe.head())


# Check memory before
print("Before conversion:")
print(df_train_fe.info(memory_usage="deep"))
print(df_test_fe.info(memory_usage="deep"))

# Identify float columns (excluding target)
float_cols_train = df_train_fe.select_dtypes(include=["float64"]).columns.tolist()
float_cols_test = df_test_fe.select_dtypes(include=["float64"]).columns.tolist()

# Convert float64 columns to float32
df_train_fe[float_cols_train] = df_train_fe[float_cols_train].astype("float32")
df_test_fe[float_cols_test] = df_test_fe[float_cols_test].astype("float32")

# Check after conversion
print("\nAfter conversion:")
print(df_train_fe.info(memory_usage="deep"))
print(df_test_fe.info(memory_usage="deep"))


num_features_train = df_train_fe.select_dtypes(include=[np.number]).columns.tolist()
skew_feature_train, skew_train_df = check_skewness(data=df_train_fe, dataset_name="Train Data",
                                                   numerical_features=num_features_train)


num_features_test = df_test_fe.select_dtypes(include=[np.number]).columns.tolist()
skew_feature_test, skew_test_df = check_skewness(data=df_test_fe, dataset_name="Test Data",
                                                 numerical_features=num_features_test)


from sklearn.preprocessing import PowerTransformer

def handle_skewed_features(df, zero_threshold=0.9, skew_threshold=0.5, num_features=None, exclude_cols=None):
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


processed_train_df, transformed_columns, sparse_columns, skewed_columns, pt_dict_train = handle_skewed_features(df=df_train_fe, num_features=skew_feature_train)
num_features_train = ["MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation", "Urbanization", "ClimateChange", "DamsQuality",
                      "Siltation", "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness", "DrainageSystems", "CoastalVulnerability",
                      "Landslides", "Watersheds", "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss", "InadequatePlanning", "PoliticalFactors",
                      "FloodProbability", "PT_Rain_Urban", "PT_Forest_River", "PT_Dams_Drainage", "NaturalRisk", "InfrastructureRisk", "GovernanceRisk", "PT_MonsoonIntensity_squared",
                      "MonsoonIntensity_sqrt", "PT_Urbanization_squared", "Urbanization_sqrt", "PT_DrainageSystems_squared", "DrainageSystems_sqrt"]
skew_feature_train, skew_train_df = check_skewness(data=processed_train_df, numerical_features=num_features_train,
                                                   dataset_name= "Train data")


processed_test_df, transformed_columns_test, sparse_columns_test, skewed_columns_test, pt_dict_test = handle_skewed_features(df=df_test_fe, num_features=skew_feature_test)
num_features_test = ["MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation", "Urbanization", "ClimateChange", "DamsQuality",
                      "Siltation", "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness", "DrainageSystems", "CoastalVulnerability",
                      "Landslides", "Watersheds", "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss", "InadequatePlanning", "PoliticalFactors",
                      "PT_Rain_Urban", "PT_Forest_River", "PT_Dams_Drainage", "NaturalRisk", "InfrastructureRisk", "GovernanceRisk", "PT_MonsoonIntensity_squared",
                      "MonsoonIntensity_sqrt", "PT_Urbanization_squared", "Urbanization_sqrt", "PT_DrainageSystems_squared", "DrainageSystems_sqrt"]
skew_feature_test, skew_test_df = check_skewness(data=processed_test_df, numerical_features=num_features_test,
                                                   dataset_name= "Test data")


# --- Train correlation ---
corr_train = processed_train_df[num_features_train].corr(numeric_only=True)
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
adjusted_mask_train = mask_train[1:, :-1]
adjusted_corr_train = corr_train.iloc[1:, :-1]

# --- Test correlation ---
corr_test = processed_test_df[num_features_test].corr(numeric_only=True)
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))
adjusted_mask_test = mask_test[1:, :-1]
adjusted_corr_test = corr_test.iloc[1:, :-1]

# --- Set up subplots ---
fig, ax = plt.subplots(1, 2, figsize=(35, 10))

# --- Custom color map ---
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

# --- Train Heatmap ---
sns.heatmap(data=adjusted_corr_train, mask=adjusted_mask_train, annot=True, fmt=".2f", 
            cmap=cmap, vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[0])
ax[0].set_title("Correlation Matrix (Train Data)", fontsize=14, weight="bold", pad=15)
# --- Test Heatmap ---
sns.heatmap(data=adjusted_corr_test, mask=adjusted_mask_test, annot=True, fmt=".2f",
            cmap=cmap, vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[1])
ax[1].set_title("Correlation Matrix (Test Data)", fontsize=14, weight="bold", pad=15)
# --- Layout ---
plt.tight_layout()
plt.show()


checking_outlier(list_feature=num_features_train, df=processed_train_df, dataset_name="Training Data")


checking_outlier(list_feature=num_features_test, df=processed_test_df, dataset_name="Test Data")


processed_train_df["NaturalRisk_Cat"] = pd.qcut(processed_train_df["NaturalRisk"],
                                              q=5,
                                              labels=[1, 2, 3, 4, 5])

plt.figure(figsize=(8, 5))
sns.histplot(data=processed_train_df, x="NaturalRisk_Cat", color="lightblue", edgecolor="black")
sns.despine(top=True, right=True, left=False, bottom=False)
plt.title("Distribution of NaturalRisk_Cat", fontsize=14, weight="bold", pad=20)
plt.xlabel("NaturalRisk_Cat", fontsize=12)
plt.ylabel("")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, val_index in split.split(processed_train_df, processed_train_df["NaturalRisk_Cat"]):
    start_train_set = processed_train_df.iloc[train_index]
    start_val_set = processed_train_df.iloc[val_index]


# Now we should remove the NaturalRisk_Cat attribute so the data is back to its original state:
for set_ in (start_train_set, start_val_set): 
    set_.drop("NaturalRisk_Cat", axis=1, inplace=True)


df_train_new = start_train_set.drop("FloodProbability", axis=1)
df_train_label_new = start_train_set["FloodProbability"].copy()


robust_features = [
    "PT_DrainageSystems_squared", "DrainageSystems_sqrt",
    "Encroachments", "DamsQuality", "DrainageSystems",
    "RiverManagement", "WetlandLoss", "PT_Dams_Drainage",
    "Deforestation", "PT_Forest_River", "PT_Rain_Urban"
]

standard_features = [
    "CoastalVulnerability", "PoliticalFactors", "TopographyDrainage",
    "InadequatePlanning", "PopulationScore", "Watersheds",
    "MonsoonIntensity", "Urbanization", "Siltation", "AgriculturalPractices",
    "DeterioratingInfrastructure", "IneffectiveDisasterPreparedness",
    "Landslides", "ClimateChange",
    "NaturalRisk", "InfrastructureRisk", "GovernanceRisk",
    "MonsoonIntensity_sqrt", "Urbanization_sqrt",
    "PT_MonsoonIntensity_squared", "PT_Urbanization_squared"
]

num_robust_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

num_stand_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num_robust", num_robust_transformer, robust_features),
    ("num_stand", num_stand_transformer, standard_features)
])

preprocessor.fit(df_train_new)
df_train_new_prepared = preprocessor.transform(df_train_new)
list_feature_prepared = preprocessor.get_feature_names_out().tolist()
clean_features = [col.replace("num_stand__", "").replace("num_robust__", "") for col in list_feature_prepared]
clean_features_2 = [col.replace("PT_", "") for col in clean_features]
clean_features_2


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
    
    # Metrics: R2
    r2 = r2_score(y_val_real, y_pred_real)
    print(f"Model: {model.__class__.__name__}{RESET}")
    print(f"Coefficient of Determination (R2): {BLUE}{r2:.4f}{RESET}")
    print("-" * 80)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # ----- Plot 1: Predicted vs. Actual -----
    axs[0].scatter(y_val_real, y_pred_real, alpha=0.5, color="royalblue", edgecolors="none", rasterized=True, s=10)
    axs[0].plot(
        [y_val_real.min(), y_val_real.max()],
        [y_val_real.min(), y_val_real.max()],
        "r--", lw=2, label="Perfect Prediction (y=x)"
    )
    axs[0].set_xlabel("Actual Values (FloodProbability)")
    axs[0].set_ylabel("Predicted Values (FloodProbability)")
    axs[0].set_title("Predicted vs. Actual (Validation Set)", fontsize=14, weight="bold", pad=20)
    axs[0].legend()
    axs[0].grid(True, alpha=0.2)

    # ----- Plot 2: Residual Plot -----
    residuals = y_val_real - y_pred_real
    axs[1].scatter(y_val_real, residuals, alpha=0.5, color="royalblue", edgecolors="none", rasterized=True, s=10)
    axs[1].axhline(0, color="red", linestyle="--", lw=2)
    axs[1].set_xlabel("Actual Values (FloodProbability)")
    axs[1].set_ylabel("Prediction Error (Residuals)")
    axs[1].set_title("Residual Plot", fontsize=14, weight="bold", pad=20)
    axs[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


X_val = start_val_set.drop("FloodProbability", axis=1)
y_val = start_val_set["FloodProbability"].copy()
X_val_prepared = preprocessor.transform(X_val)


from catboost import CatBoostRegressor

cat_params = {
    "n_estimators":12000,
    "l2_leaf_reg": 0.0017992898021052064, 
    "max_bin": 200, 
    "learning_rate": 0.016714889518285515, 
    "max_depth": 7, 
    "min_data_in_leaf": 288,
    "random_seed": 42,
    "loss_function": "RMSE",
    "eval_metric": "R2",
    "verbose": 0,
    "allow_writing_files": False,
    "bootstrap_type": "Bernoulli",
    "thread_count": -1
}

model_cb = CatBoostRegressor(**cat_params)
evaluate_model(model = model_cb, X_train = df_train_new_prepared, 
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val)


df_test_prepared = preprocessor.transform(processed_test_df)
y_pred_test = model_cb.predict(df_test_prepared)

submission_df = pd.DataFrame({
    "id": list_test_id,
    "FloodProbability": y_pred_test
})

submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file saved!")
submission_df.head(5)


fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(15, 6))

sns.boxplot(data=submission_df, y = "FloodProbability", ax=ax[0], color="#00BFC4")
ax[0].set_title(f"Box plot of Flood Probability", fontsize=14, pad=20, weight="bold")
ax[0].grid(axis="x", color="gray", linestyle=":", linewidth=0.7)
ax[0].set_ylabel("Flood Probability")
sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)

sns.histplot(data=submission_df, x = "FloodProbability", ax=ax[1], color="#00BFC4", kde=True, bins=40)
ax[1].set_title(f"Histogram of Flood Probability", fontsize=14, pad=20, weight="bold")
ax[1].set_xlabel("Flood Probability")
ax[1].set_ylabel("Frequency")
sns.despine(ax=ax[1], top=True, right=True, left=False, bottom=False)

# Extract KDE values to find peaks
kde = sns.kdeplot(submission_df["FloodProbability"], ax=ax[1], color="#00BFC4").lines[0].get_data()
kde_x, kde_y = kde[0], kde[1]
peaks, _ = find_peaks(kde_y)

# Highlight peaks
for peak_idx in peaks:
    plt.plot(kde_x[peak_idx], kde_y[peak_idx], "ro")  # Red dots on peaks

plt.tight_layout()
plt.show()


shap_plot(model=model_cb, X_test=df_test_prepared[:1000],
          list_feature=clean_features_2, type="bar")

