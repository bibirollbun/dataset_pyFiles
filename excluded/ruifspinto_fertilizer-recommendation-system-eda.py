# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings


# Load training and testing datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col=0)
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col=0)
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


def dataframe_summary(df):
    info_df = pd.DataFrame(
        {
            "Feature": df.columns,
            "Non-Null Count": df.notnull().sum().values,
            "Dtype": df.dtypes.values,
        }
    )

    # Descriptive statistics
    describe_df = (
        df.describe(include="all")
        .transpose()
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    # Missing values per column
    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ["Feature", "Missing Values"]

    # Unique value counts per column
    unique_vals = df.nunique().reset_index()
    unique_vals.columns = ["Feature", "Unique Values"]

    # Mode and frequency for categorical columns
    mode_df = (
        df.mode()
        .transpose()
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Mode"})
    )
    freq_df = (
        df.apply(lambda x: x.value_counts().iloc[0] if x.dtype == "object" else None)
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Most Frequent Frequency"})
    )

    # Merge all pieces
    summary_df = info_df.merge(describe_df, on="Feature", how="left")
    summary_df = summary_df.merge(missing_df, on="Feature", how="left")
    summary_df = summary_df.merge(unique_vals, on="Feature", how="left")
    summary_df = summary_df.merge(mode_df, on="Feature", how="left")
    summary_df = summary_df.merge(freq_df, on="Feature", how="left")

    # Rearranging columns for readability
    preferred_order = [
        "Feature",
        "Dtype",
        "Non-Null Count",
        "Missing Values",
        "Unique Values",
        "Mode",
        "Most Frequent Frequency",
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
    summary_df = summary_df[
        [col for col in preferred_order if col in summary_df.columns]
    ]

    # Count duplicated rows
    duplicate_count = df.duplicated().sum()

    return summary_df, duplicate_count


print("\nSummary Statistics for Numerical Features:")
display(train.describe())


summary, dup_count = dataframe_summary(train)

print(f"\nTotal duplicated rows: {dup_count}")
summary.style.background_gradient(
    cmap="viridis", subset=["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
)


summary, dup_count = dataframe_summary(test)

print(f"\nTotal duplicated rows: {dup_count}")
summary.style.background_gradient(
    cmap="viridis", subset=["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
)


categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()
print("\nCategorical Summary:")
for col in categorical_cols:
    print(f"\n{col} - Unique Values: {train[col].nunique()}")


def analyze_categorical_feature(df, feature_name, target="Fertilizer Name"):
    """
    Analyze a categorical feature against the target variable with improved visualizations
    """
    if feature_name not in df.columns or feature_name == target:
        print(f"Invalid feature name or same as target: {feature_name}")
        return

    # Create a contingency table
    ct = pd.crosstab(df[feature_name], df[target])

    # Normalize to show percentage distribution
    ct_normalized = pd.crosstab(df[feature_name], df[target], normalize="index") * 100

    # Get categories and split into two groups
    categories = sorted(df[feature_name].unique())
    half_point = len(categories) // 2
    first_half = categories[:half_point]
    second_half = categories[half_point:]

    # 1. BAR CHARTS: Split visualization by categories
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # Get top target values for each category
    top_values = {}
    for cat in ct.index:
        top_values[cat] = ct.loc[cat].nlargest(3).index.tolist()

    # First subplot - first half categories
    selected_data = ct_normalized.loc[first_half]
    selected_data.plot(kind="barh", ax=axes[0], colormap="viridis")
    axes[0].set_title(
        f"{feature_name} vs {target} - First Group (%)",
        fontsize=16,
        fontweight="bold",
    )
    axes[0].set_xlabel("Percentage (%)", fontsize=12)
    axes[0].set_ylabel(feature_name, fontsize=12)
    axes[0].grid(axis="x", linestyle="--", alpha=0.7)
    axes[0].legend(title=target, bbox_to_anchor=(1.02, 1), loc="upper left")

    # Second subplot - second half categories
    selected_data = ct_normalized.loc[second_half]
    selected_data.plot(kind="barh", ax=axes[1], colormap="viridis")
    axes[1].set_title(
        f"{feature_name} vs {target} - Second Group (%)",
        fontsize=16,
        fontweight="bold",
    )
    axes[1].set_xlabel("Percentage (%)", fontsize=12)
    axes[1].grid(axis="x", linestyle="--", alpha=0.7)
    axes[1].legend(title=target, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout(pad=3.0)
    plt.show()

    # 2. HEATMAP: Split into two heatmaps for better readability
    fig, axes = plt.subplots(2, 1, figsize=(18, 16))

    # First heatmap - raw counts
    sns.heatmap(
        ct.loc[first_half],
        annot=True,
        fmt="d",
        cmap="viridis",
        linewidths=0.5,
        ax=axes[0],
        cbar_kws={"label": "Count"},
    )
    axes[0].set_title(
        f"{feature_name} vs {target} - First Group (Counts)",
        fontsize=16,
        fontweight="bold",
    )
    axes[0].set_xlabel("") 

    # Second heatmap
    sns.heatmap(
        ct.loc[second_half],
        annot=True,
        fmt="d",
        cmap="viridis",
        linewidths=0.5,
        ax=axes[1],
        cbar_kws={"label": "Count"},
    )
    axes[1].set_title(
        f"{feature_name} vs {target} - Second Group (Counts)",
        fontsize=16,
        fontweight="bold",
    )
    axes[1].set_xlabel(target, fontsize=14)

    plt.tight_layout(pad=3.0)
    plt.show()

    # 3. Percentage heatmap for better understanding of relative distributions
    fig, axes = plt.subplots(2, 1, figsize=(18, 16))

    # First heatmap - percentages
    sns.heatmap(
        ct_normalized.loc[first_half],
        annot=True,
        fmt=".1f",  # Format to 1 decimal place
        cmap="viridis",
        linewidths=0.5,
        ax=axes[0],
        cbar_kws={"label": "Percentage (%)"},
    )
    axes[0].set_title(
        f"{feature_name} vs {target} - First Group (Percentages)",
        fontsize=16,
        fontweight="bold",
    )
    axes[0].set_xlabel("")

    # Second heatmap - percentages
    sns.heatmap(
        ct_normalized.loc[second_half],
        annot=True,
        fmt=".1f",
        cmap="viridis",
        linewidths=0.5,
        ax=axes[1],
        cbar_kws={"label": "Percentage (%)"},
    )
    axes[1].set_title(
        f"{feature_name} vs {target} - Second Group (Percentages)",
        fontsize=16,
        fontweight="bold",
    )
    axes[1].set_xlabel(target, fontsize=14)

    plt.tight_layout(pad=3.0)
    plt.show()


analyze_categorical_feature(train, "Crop Type")


analyze_categorical_feature(train, "Soil Type")


plt.figure(figsize=(16, 14))
train["Fertilizer Name"].value_counts().plot.pie(
    autopct="%1.1f%%",
    shadow=True,
    startangle=90,
    explode=[0.05] * len(train["Fertilizer Name"].unique()),
    textprops={"fontsize": 16},
    wedgeprops={
        "linewidth": 1.5,
        "edgecolor": "white",
    },
)
plt.title("Distribution of Fertilizer Name", fontsize=20)
plt.ylabel("")
plt.show()


def analyze_numerical_feature(df, feature_name, target="Fertilizer Name"):
    """
    Analyze a numerical feature against the target variable with visualizations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    if feature_name not in df.columns or feature_name == target:
        print(f"Invalid feature name or same as target: {feature_name}")
        return

    with pd.option_context('mode.use_inf_as_na', True):
        # 1. Box plot to show distribution by target class
        plt.figure(figsize=(16, 8))
        sns.boxplot(x=target, y=feature_name, data=df)
        plt.title(f"Distribution of {feature_name} by {target}", fontsize=16)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        # 2. Violin plot for more detailed distribution
        plt.figure(figsize=(16, 8))
        sns.violinplot(x=target, y=feature_name, data=df, inner="quartile")
        plt.title(f"Violin Plot of {feature_name} by {target}", fontsize=16)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        # 3. Calculate summary statistics by target class
        stats = df.groupby(target)[feature_name].agg(
            ["mean", "median", "std", "min", "max"]
        )
        print(f"\nSummary Statistics for {feature_name} by {target}:")
        display(stats)

        # 4. Create a heatmap showing the mean values
        plt.figure(figsize=(12, 6))
        sns.heatmap(stats[["mean"]].T, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Mean {feature_name} by {target}", fontsize=16)
        plt.tight_layout()
        plt.show()

        # 5. Distribution histogram with KDE for each class
        plt.figure(figsize=(16, 10))
        for fertilizer in df[target].unique():
            subset = df[df[target] == fertilizer][feature_name]
            sns.kdeplot(subset, label=fertilizer, fill=True, alpha=0.3)

        plt.title(f"Distribution of {feature_name} by {target}", fontsize=16)
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(title=target)
        plt.grid(linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()



with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)

    for col in train.select_dtypes(include=[np.number]).columns:
        if col != "Fertilizer Name":
            analyze_numerical_feature(train, col, target="Fertilizer Name")

