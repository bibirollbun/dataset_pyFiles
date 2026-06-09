# Core data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Statistical functions
from scipy.stats import skew
from scipy.signal import find_peaks

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
df_train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
df_origin = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# Verify shapes
print("Train Data Shape:", df_train.shape)
print("\nOrigin Data Shape:", df_origin.shape)
print("\nTest Data Shape:", df_test.shape)


# Display few rows of each dataset
print("Train Data Preview:")
display(df_train.tail())

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


# Remove space in name columns
df_train.columns = (
    df_train.columns
    .str.strip()
    .str.replace(" ", "")
)

df_origin.columns = (
    df_origin.columns
    .str.strip()
    .str.replace(" ", "")
)

df_test.columns = (
    df_test.columns
    .str.strip()
    .str.replace(" ", "")
)

# Drop columns id
df_train.set_index(df_train.id, inplace=True)
df_train.drop(columns="id", axis=1, inplace=True)
df_test.set_index(df_test.id, inplace=True)
df_test.drop(columns="id", axis=1, inplace=True)

# We need to update the data for the columns, this helps to reduce memory.
df_train = df_train.astype({
    "RhythmScore": "float32",
    "AudioLoudness": "float32",
    "VocalContent": "float32",
    "AcousticQuality": "float32",
    "InstrumentalScore": "float32",
    "LivePerformanceLikelihood": "float32",
    "MoodScore": "float32",
    "TrackDurationMs": "float64",
    "Energy": "float32",
    "BeatsPerMinute": "float32"
})

df_origin = df_origin.astype({
    "RhythmScore": "float32",
    "AudioLoudness": "float32",
    "VocalContent": "float32",
    "AcousticQuality": "float32",
    "InstrumentalScore": "float32",
    "LivePerformanceLikelihood": "float32",
    "MoodScore": "float32",
    "TrackDurationMs": "float64",
    "Energy": "float32",
    "BeatsPerMinute": "float32"
})

df_test = df_test.astype({
    "RhythmScore": "float32",
    "AudioLoudness": "float32",
    "VocalContent": "float32",
    "AcousticQuality": "float32",
    "InstrumentalScore": "float32",
    "LivePerformanceLikelihood": "float32",
    "MoodScore": "float32",
    "TrackDurationMs": "float64",
    "Energy": "float32"
})


# Display information about the DataFrames
print("Train Data Info:")
df_train.info()

print("\nOrigin Data Info:")
df_origin.info()

print("\nTest Data Info:")
df_test.info()


cm = sns.light_palette("blue", as_cmap=True)
print("Train data describe:")
display(df_train.drop(columns=["BeatsPerMinute"], axis=1).describe().T.style.background_gradient(cmap=cm))
print("\nOrigin data describe:")
display(df_origin.drop(columns=["BeatsPerMinute"], axis=1).describe().T.style.background_gradient(cmap=cm))
print("\nTest data describe:")
display(df_test.describe().T.style.background_gradient(cmap=cm))


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


num_features = ["RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality", "InstrumentalScore", "LivePerformanceLikelihood",
                "MoodScore", "TrackDurationMs", "Energy"]

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


fig, axes = plt.subplots(nrows=2, ncols=2, sharey=False, figsize=(15, 10))
datasets = [("Train Data", df_train), ("Original Data", df_origin)]

for i, (title, df_) in enumerate(datasets):
    ax_box  = axes[i, 0]
    ax_hist = axes[i, 1]

    # If the target column is missing, skip and display a message on the plot
    if 'BeatsPerMinute' not in df_.columns:
        ax_box.axis('off'); ax_hist.axis('off')
        ax_box.text(0.5, 0.5, f"{title}: missing 'BeatsPerMinute'", ha='center', va='center')
        continue

    # Boxplot
    sns.boxplot(y=df_["BeatsPerMinute"], ax=ax_box, color="#00BFC4")
    ax_box.set_title(f"Box plot of Beats Per Minute in {title}", fontsize=14, pad=20, weight="bold")
    ax_box.grid(axis="x", color="gray", linestyle=":", linewidth=0.7)
    sns.despine(ax=ax_box, top=True, right=True, left=False, bottom=False)

    # Histogram + KDE (draw KDE once with histplot)
    sns.histplot(df_["BeatsPerMinute"], ax=ax_hist, bins=40, kde=True, color="#00BFC4")
    ax_hist.set_title(f"Histogram of Beats Per Minute in {title}", fontsize=14, pad=20, weight="bold")
    ax_hist.set_xlabel("BeatsPerMinute"); ax_hist.set_ylabel("Frequency")
    sns.despine(ax=ax_hist, top=True, right=True, left=False, bottom=False)

    # Get the KDE curve drawn by histplot (last line on the axis)
    kde_line = ax_hist.lines[-1]
    kde_x, kde_y = kde_line.get_data()

    # Find peaks on KDE (tune prominence/distance for smoother detection)
    peaks_idx, _ = find_peaks(kde_y, prominence=0.001, distance=10)
    ax_hist.plot(kde_x[peaks_idx], kde_y[peaks_idx], "ro", ms=4)

plt.tight_layout()
plt.show()


def plot_numerical_features(df_train, df_test, df_origin, num_features):
    colors = color(n_colors=3)
    n = len(num_features)

    fig, ax = plt.subplots(n, 2, figsize=(12, n * 4))
    ax = np.array(ax).reshape(n, 2)

    for i, feature in enumerate(num_features):
        sns.histplot(data=df_train[feature], color=colors[0], bins=20, kde=True, ax=ax[i, 0], label="Train data")
        sns.histplot(data=df_origin[feature], color=colors[1], bins=20, kde=True, ax=ax[i, 0], label="Origin data")
        sns.histplot(data=df_test[feature], color=colors[2], bins=20, kde=True, ax=ax[i, 0], label="Test data")
        ax[i, 0].set_title(f"Histogram of {feature}", pad=20, weight="bold")
        ax[i, 0].legend()
        ax[i, 0].set_ylabel("")
        sns.despine(left=False, bottom=False, ax=ax[i, 0])

        df_plot = pd.concat([
            pd.DataFrame({"Dataset": "Train data", feature: df_train[feature]}),
            pd.DataFrame({"Dataset": "Origin data", feature: df_origin[feature]}),
            pd.DataFrame({"Dataset": "Test data", feature: df_test[feature]})
        ]).reset_index(drop=True)

        sns.boxplot(data=df_plot, x=feature, y="Dataset", palette=colors, orient="h", ax=ax[i, 1])
        ax[i, 1].set_title(f"Horizontal Box plot of {feature}", pad=20, weight="bold")
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
    corr_train = df_train.corr(numeric_only=True, method="pearson")
    corr_origin = df_origin.corr(numeric_only=True, method="pearson")
    corr_test = df_test.corr(numeric_only=True, method="pearson")

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

plot_correlation(df_train=df_train,
                 df_origin=df_origin,
                 df_test=df_test)


df_merged = pd.concat([df_train, df_origin], axis=0, ignore_index=True)
df_merged = df_merged.reset_index(drop=True)
print("Shape of merged:", df_merged.shape)


# Combines energy and loudness â†’ represents the â€œintensityâ€� or strength of the track.
df_merged["Energy_AudioLoudness"] = df_merged["Energy"] * df_merged["AudioLoudness"]
df_test["Energy_AudioLoudness"] = df_test["Energy"] * df_test["AudioLoudness"]

# Combines mood score and acoustic quality â†’ captures the â€œacoustic emotionâ€� (e.g., sad acoustic ballad or cheerful acoustic music).
df_merged["Mood_Acoustic"] = df_merged["MoodScore"] * df_merged["AcousticQuality"]
df_test["Mood_Acoustic"] = df_test["MoodScore"] * df_test["AcousticQuality"]

# The track duration expressed in minutes (converted from milliseconds).
df_merged["TrackDurationMin"] = df_merged["TrackDurationMs"] / 60000
df_test["TrackDurationMin"] = df_test["TrackDurationMs"] / 60000

# The ratio of overall energy to acoustic quality.
df_merged["Energy_Acoustic_Ratio"] = df_merged["Energy"] / (df_merged["AcousticQuality"] + 1e-5)
df_test["Energy_Acoustic_Ratio"] = df_test["Energy"] / (df_test["AcousticQuality"] + 1e-5)

# Measures the balance between vocal and instrumental elements.
df_merged["Vocal_Instrument_Balance"] = df_merged["VocalContent"] / (df_merged["InstrumentalScore"] + 1e-5)
df_test["Vocal_Instrument_Balance"] = df_test["VocalContent"] / (df_test["InstrumentalScore"] + 1e-5)

# Captures the alignment between the trackâ€™s mood and rhythm.
df_merged["MoodRhythm"] = df_merged["MoodScore"] * df_merged["RhythmScore"]
df_test["MoodRhythm"] = df_test["MoodScore"] * df_test["RhythmScore"]

# Represents the intensity of performance, combining the likelihood of live performance with loudness.
df_merged["PerformanceIntensity"] = df_merged["LivePerformanceLikelihood"] * df_merged["AudioLoudness"]
df_test["PerformanceIntensity"] = df_test["LivePerformanceLikelihood"] * df_test["AudioLoudness"]

# The energy level amplified by rhythm.
df_merged["RhythmEnergy"] = df_merged["RhythmScore"] * df_merged["Energy"]
df_test["RhythmEnergy"] = df_test["RhythmScore"] * df_test["Energy"]

def plot_correlation(df_merge, df_test, merge_name="Merge Data", test_name="Test Data"):
    corr_merge = df_merge.corr(numeric_only=True, method="pearson")
    corr_test = df_test.corr(numeric_only=True, method="pearson")

    mask_merge = np.triu(np.ones_like(corr_merge, dtype=bool))
    adjusted_mask_merge = mask_merge[1:, :-1]
    adjusted_cereal_corr_merge = corr_merge.iloc[1:, :-1]

    mask_test = np.triu(np.ones_like(corr_test, dtype=bool))
    adjusted_mask_test = mask_test[1:, :-1]
    adjusted_cereal_corr_test = corr_test.iloc[1:, :-1]

    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    fig, ax = plt.subplots(1, 2, figsize=(24, 10))

    sns.heatmap(data=adjusted_cereal_corr_merge, mask=adjusted_mask_merge,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[0])
    ax[0].set_title(f"Correlation Heatmap of {merge_name}", fontsize=16, weight="bold")

    sns.heatmap(data=adjusted_cereal_corr_test, mask=adjusted_mask_test,
                annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linecolor="white", linewidths=0.5, ax=ax[1])
    ax[1].set_title(f"Correlation Heatmap of {test_name}", fontsize=16, weight="bold")

    plt.tight_layout()
    plt.show()

plot_correlation(df_merge=df_merged, df_test=df_test)


new_features = ["Energy_AudioLoudness", "Mood_Acoustic", "TrackDurationMin", "Energy_Acoustic_Ratio", 
                "Vocal_Instrument_Balance", "MoodRhythm", "PerformanceIntensity", "RhythmEnergy"]

def plot_numerical_new_features(df_merged, df_test, num_features):
    colors = color(n_colors=2)
    n = len(num_features)

    fig, ax = plt.subplots(n, 2, figsize=(12, n * 4))
    ax = np.array(ax).reshape(n, 2)

    for i, feature in enumerate(num_features):
        sns.histplot(data=df_merged[feature], color=colors[0], bins=20, kde=True, ax=ax[i, 0], label="Merge data")
        sns.histplot(data=df_test[feature], color=colors[1], bins=20, kde=True, ax=ax[i, 0], label="Test data")
        ax[i, 0].set_title(f"Histogram of {feature}", pad=20, weight="bold")
        ax[i, 0].legend()
        ax[i, 0].set_ylabel("")
        sns.despine(left=False, bottom=False, ax=ax[i, 0])

        df_plot = pd.concat([
            pd.DataFrame({"Dataset": "Merge data", feature: df_merged[feature]}),
            pd.DataFrame({"Dataset": "Test data", feature: df_test[feature]})
        ]).reset_index(drop=True)

        sns.boxplot(data=df_plot, x=feature, y="Dataset", palette=colors, orient="h", ax=ax[i, 1])
        ax[i, 1].set_title(f"Horizontal Box plot of {feature}", pad=20, weight="bold")
        sns.despine(left=False, bottom=False, ax=ax[i, 1])

    plt.tight_layout()
    plt.show()
plot_numerical_new_features(df_merged = df_merged, df_test = df_test, num_features=new_features)


num_features = ["RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality", "InstrumentalScore", "LivePerformanceLikelihood",
                "MoodScore", "TrackDurationMs", "Energy", "Energy_AudioLoudness", "Mood_Acoustic", "BeatsPerMinute", "TrackDurationMin", 
                "Energy_Acoustic_Ratio", "Vocal_Instrument_Balance", "MoodRhythm", "PerformanceIntensity", "RhythmEnergy"]
skew_feature_merge, skew_merge_df = check_skewness(data=df_merged, dataset_name="Merge Data",
                                                   numerical_features=num_features)


num_features_test = ["RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality", "InstrumentalScore", "LivePerformanceLikelihood",
                "MoodScore", "TrackDurationMs", "Energy", "Energy_AudioLoudness", "Mood_Acoustic", "TrackDurationMin", 
                "Energy_Acoustic_Ratio", "Vocal_Instrument_Balance", "MoodRhythm", "PerformanceIntensity", "RhythmEnergy"]
skew_feature_test, skew_test_df = check_skewness(data=df_test, dataset_name="Test Data",
                                                 numerical_features=num_features_test)


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
            print("AAA")
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


processed_merge_df, transformed_columns, sparse_columns, skewed_columns = handle_skewed_features(df=df_merged, num_features=skew_feature_merge)
num_features = ["RhythmScore", "AudioLoudness", "PT_VocalContent", "PT_AcousticQuality", "PT_InstrumentalScore", "LivePerformanceLikelihood",
                "MoodScore", "TrackDurationMs", "Energy", "PT_Energy_AudioLoudness", "PT_Mood_Acoustic", "BeatsPerMinute", "TrackDurationMin", 
                "PT_Energy_Acoustic_Ratio", "PT_Vocal_Instrument_Balance", "MoodRhythm", "PT_PerformanceIntensity", "RhythmEnergy"]
skew_feature_merge, skew_merge_df = check_skewness(data=processed_merge_df, dataset_name="Merge Data", numerical_features=num_features)


processed_test_df, transformed_columns_test, sparse_columns_test, skewed_columns_test = handle_skewed_features(df=df_test, num_features=skew_feature_test)
num_features = ["RhythmScore", "AudioLoudness", "PT_VocalContent", "PT_AcousticQuality", "PT_InstrumentalScore", "LivePerformanceLikelihood",
                "MoodScore", "TrackDurationMs", "Energy", "PT_Energy_AudioLoudness", "PT_Mood_Acoustic", "TrackDurationMin", 
                "PT_Energy_Acoustic_Ratio", "PT_Vocal_Instrument_Balance", "MoodRhythm", "PT_PerformanceIntensity", "RhythmEnergy"]
skew_feature_test, skew_test_df = check_skewness(data=processed_test_df, numerical_features=num_features, dataset_name= "Test data")


checking_outlier(list_feature=num_features, df=processed_merge_df, dataset_name="Merge data")


checking_outlier(list_feature=num_features, df=processed_test_df, dataset_name="Test data")


processed_merge_df["PT_Mood_Acoustic_Cat"] = pd.qcut(processed_merge_df["PT_Mood_Acoustic"],
                                              q=5,
                                              labels=[1, 2, 3, 4, 5])

plt.figure(figsize=(8, 5))
sns.histplot(data=processed_merge_df, x="PT_Mood_Acoustic_Cat", color="lightblue", edgecolor="black")
sns.despine(top=True, right=True, left=False, bottom=False)
plt.title("Distribution of PT_Mood_Acoustic_Cat", fontsize=14, weight="bold", pad=20)
plt.xlabel("PT_Mood_Acoustic_Cat", fontsize=12)
plt.ylabel("")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(processed_merge_df, processed_merge_df["PT_Mood_Acoustic_Cat"]):
    start_train_set = processed_merge_df.iloc[train_index]
    start_test_set = processed_merge_df.iloc[test_index]


# Now we should remove the PT_Mood_Acoustic_Cat attribute so the data is back to its original state:
for set_ in (start_train_set, start_test_set): 
    set_.drop("PT_Mood_Acoustic_Cat", axis=1, inplace=True)


df_train_new = start_train_set.drop("BeatsPerMinute", axis=1)
df_train_label_new = start_train_set["BeatsPerMinute"].copy()


list_feature_num_robust = ["RhythmScore","AudioLoudness", "LivePerformanceLikelihood", "TrackDurationMs", "TrackDurationMin", 
                           "MoodRhythm", "RhythmEnergy"]
list_feature_num_stand = ["PT_VocalContent", "PT_AcousticQuality", "PT_InstrumentalScore",
                "MoodScore", "Energy", "PT_Energy_AudioLoudness", "PT_Mood_Acoustic",
                "PT_Energy_Acoustic_Ratio", "PT_Vocal_Instrument_Balance", "PT_PerformanceIntensity"]

num_robust_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

num_stand_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num_robust", num_robust_transformer, list_feature_num_robust),
    ("num_stand", num_stand_transformer, list_feature_num_stand)
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
def evaluate_model(model, X_train, X_val, y_train, y_val, show_shap_plot = True):
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
    axs[0].scatter(y_val_real, y_pred_real, alpha=0.4, color="royalblue")
    axs[0].plot(
        [y_val_real.min(), y_val_real.max()],
        [y_val_real.min(), y_val_real.max()],
        "r--", lw=2, label="Perfect Prediction (y=x)"
    )
    axs[0].set_xlabel("Actual Values (BeatsPerMinute)")
    axs[0].set_ylabel("Predicted Values (BeatsPerMinute)")
    axs[0].set_title("Predicted vs. Actual (Validation Set)", fontsize=14, weight="bold", pad=20)
    axs[0].legend()
    axs[0].grid(True, alpha=0.2)

    # ----- Plot 2: Residual Plot -----
    residuals = y_val_real - y_pred_real
    axs[1].scatter(y_val_real, residuals, alpha=0.5)
    axs[1].axhline(0, color="red", linestyle="--", lw=2)
    axs[1].set_xlabel("Actual Values (BeatsPerMinute)")
    axs[1].set_ylabel("Prediction Error (Residuals)")
    axs[1].set_title("Residual Plot", fontsize=14, weight="bold", pad=20)
    axs[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show() 

    if show_shap_plot:
        shap_plot(model = model, X_test = X_val, list_feature = list_feature_prepared)


X_val = start_test_set.drop("BeatsPerMinute", axis=1)
y_val = start_test_set["BeatsPerMinute"].copy()
X_val_prepared = preprocessor.transform(X_val)


from catboost import CatBoostRegressor

param_cb = {
	"iterations": 601, 
	"learning_rate": 0.010916330886941803, 
	"depth": 5, 
	"l2_leaf_reg": 90.94596820625567, 
	"random_strength": 1.8922481051459825, 
	"border_count": 218, 
	"leaf_estimation_iterations": 6, 
	"bootstrap_type": "Bernoulli", 
	"subsample": 0.7711311287541387,
	"loss_function": "RMSE",
	"eval_metric": "RMSE",
	"random_seed": 42,
	"verbose": 0,
	"task_type": "GPU",
	"allow_writing_files": False
}

model_cb = CatBoostRegressor(**param_cb)
model_cb


evaluate_model(model = model_cb, X_train = df_train_new_prepared, 
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val, show_shap_plot = False)


from xgboost import XGBRegressor

param_xgb = {
"n_estimators": 522, 
"learning_rate": 0.010771844406737502, 
"max_depth": 4, 
"min_child_weight": 5.571386310846427, 
"subsample": 0.7851504418160667, 
"colsample_bytree": 0.6000243214641255, 
"gamma": 3.867737354082032, 
"reg_alpha": 3.1150492652714474e-07, 
"reg_lambda": 1.7275254522103365, 
"max_bin": 323,
"random_state": 42,
"tree_method": "hist",
"device": "cuda",
"n_jobs": 1,
"objective": "reg:squarederror",
"eval_metric": "rmse"
}

model_xgb = XGBRegressor(**param_xgb)
model_xgb


evaluate_model(model = model_xgb, X_train = df_train_new_prepared, 
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val, show_shap_plot = False)


from lightgbm import LGBMRegressor

param_lgbm = {
"learning_rate": 0.001502328415098844,
"num_leaves": 79, 
"max_depth": 14,
"feature_fraction": 0.8933016300882094,
"bagging_fraction": 0.9754103048412501,
"bagging_freq": 7, 
"min_child_samples": 40,
"lambda_l1": 7.10897934678165e-07,
"lambda_l2": 7.81564014894075e-08,
"random_state" : 42,
"n_jobs" : -1,
"verbosity": -1,
"n_estimators": 643,
"objective": "rmse",
"metric": "rmse",
"boosting_type": "gbdt"
}

model_lgbm = LGBMRegressor(**param_lgbm)
model_lgbm


evaluate_model(model = model_lgbm, X_train = df_train_new_prepared, 
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val, show_shap_plot = False)


from sklearn.linear_model import Ridge

param_ridge = {
"alpha": 9999.679917972731, 
"fit_intercept": True, 
"tol": 5.6573144072058315e-05, 
"max_iter": 11944,
"solver": "auto",
"random_state": 42
}

model_ridge = Ridge(**param_ridge)
model_ridge


evaluate_model(model = model_ridge, X_train = df_train_new_prepared, 
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val, show_shap_plot = False)


from sklearn.linear_model import Lasso

param_lasso = {
"alpha": 0.0033159447142899045, 
"fit_intercept": True, 
"tol": 0.0007706141125664155, 
"max_iter": 18300, 
"selection": "random", 
"warm_start": True, 
"positive": True,
"random_state": 42
}

model_lasso = Lasso(**param_lasso)
model_lasso


evaluate_model(model = model_lasso, X_train = df_train_new_prepared, 
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val, show_shap_plot = False)


from sklearn.linear_model import ElasticNet

param_elasticNet = {
"alpha": 0.031876022671564974, 
"l1_ratio": 0.03158293835811394, 
"tol": 5.2073690492099974e-06, 
"max_iter": 29330, 
"selection": "random",
"random_state": 42
}

model_elasticNet = ElasticNet(**param_elasticNet)
model_elasticNet


evaluate_model(model = model_elasticNet, X_train = df_train_new_prepared, 
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val, show_shap_plot = False)


ests = [("cb", model_cb), ("xgb", model_xgb), ("lgbm", model_lgbm),
        ("ridge", model_ridge), ("lasso", model_lasso), ("elastic", model_elasticNet)]
preds = {name: m.predict(X_val_prepared) for name, m in ests}
corr = pd.DataFrame(preds).corr()
rmse_each = {name: mean_squared_error(y_val, preds[name], squared=False) for name,_ in ests}
display(corr)
display(rmse_each)

A = np.column_stack([preds[name] for name,_ in ests])  # (n_val, n_models)

def obj_w(trial):
    w = np.array([trial.suggest_float(f"w_{i}", 0.0, 5.0) for i in range(A.shape[1])])
    if w.sum() == 0: return 1e9
    y_hat = A.dot(w / w.sum())
    return mean_squared_error(y_val, y_hat, squared=False)

study_w = optuna.create_study(direction="minimize")
study_w.optimize(obj_w, n_trials=3000, show_progress_bar=True)
w = np.array([study_w.best_params[f"w_{i}"] for i in range(A.shape[1])])
weights = (w / w.sum()).tolist()
print("Best weights:", weights)


from sklearn.ensemble import VotingRegressor
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
voting_reg = VotingRegressor(estimators=[
    ("cb", model_cb),
    ("xgb", model_xgb),
    ("lgbm", model_lgbm),
    ("ridge", model_ridge),
    ("lasso", model_lasso),
    ("elasticNet", model_elasticNet),
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
               X_val = X_val_prepared , y_train = df_train_label_new, y_val=y_val, show_shap_plot = False)


df_test_prepared = preprocessor.transform(processed_test_df)
y_pred_test = voting_reg.predict(df_test_prepared)

submission_df = pd.DataFrame({
    "id": df_test.index,
    "BeatsPerMinute": y_pred_test
})

submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file saved!")
submission_df.head()


fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(15, 6))

sns.boxplot(data=submission_df, y = "BeatsPerMinute", ax=ax[0], color="#00BFC4")
ax[0].set_title(f"Box plot of Beats Per Minute", fontsize=14, pad=20, weight="bold")
ax[0].grid(axis="x", color="gray", linestyle=":", linewidth=0.7)
ax[0].set_ylabel("BeatsPerMinute")
sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)

sns.histplot(data=submission_df, x = "BeatsPerMinute", ax=ax[1], color="#00BFC4", kde=True, bins=40)
ax[1].set_title(f"Histogram of Beats Per Minute", fontsize=14, pad=20, weight="bold")
ax[1].set_xlabel("BeatsPerMinute")
ax[1].set_ylabel("Frequency")
sns.despine(ax=ax[1], top=True, right=True, left=False, bottom=False)

# Extract KDE values to find peaks
kde = sns.kdeplot(submission_df["BeatsPerMinute"], ax=ax[1], color="#00BFC4").lines[0].get_data()
kde_x, kde_y = kde[0], kde[1]
peaks, _ = find_peaks(kde_y)

# Highlight peaks
for peak_idx in peaks:
    plt.plot(kde_x[peak_idx], kde_y[peak_idx], "ro")  # Red dots on peaks

plt.tight_layout()
plt.show()


shap_plot(model=voting_reg.named_estimators_["cb"], X_test=df_test_prepared[:1500], 
          list_feature=list_feature_prepared, type="bar")


shap_plot(model=voting_reg.named_estimators_["cb"], X_test=df_test_prepared[:1500], 
          list_feature=list_feature_prepared)

