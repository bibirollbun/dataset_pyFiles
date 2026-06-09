import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")


def check_df(dataframe, head=5):
    """Shows basic information about the dataset"""
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Tail #####################')
    print(dataframe.tail(head))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def plot_feature_distributions(dataframe, cols, hue_col='label', n_cols=3, n_rows=None, figsize=None):
    """Visualizes distributions of features"""
    if n_rows is None:
        n_rows = (len(cols) + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (n_cols * 6, n_rows * 4)

    plt.figure(figsize=figsize)

    for i, col in enumerate(cols):
        if i < n_rows * n_cols:
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(data=dataframe, x=col, hue=hue_col, bins=30, kde=True, alpha=0.6)
            plt.title(f"{col} Distribution")

    plt.tight_layout()
    plt.savefig(f'feature_distributions_{"_".join(cols[:3])}.png')
    plt.show()


def plot_correlation_matrix(dataframe, target_col=None, mask_upper=True, figsize=(20, 16)):
    """Visualizes correlation matrix"""
    corr = dataframe.corr()

    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix', fontsize=20)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.show()

    # If target column specified, show relationships
    if target_col:
        correlations = corr[target_col].sort_values(ascending=False)

        # Highest correlated features
        plt.figure(figsize=(12, 8))
        top_corr = correlations.iloc[1:21]  # Excluding self, top 20
        sns.barplot(x=top_corr.values, y=top_corr.index)
        plt.title(f'Top 20 Features Correlated with {target_col}', fontsize=16)
        plt.tight_layout()
        plt.savefig('top_correlations.png')
        plt.show()

        # Lowest correlated features
        plt.figure(figsize=(12, 8))
        bottom_corr = correlations.iloc[-20:]  # Bottom 20
        sns.barplot(x=bottom_corr.values, y=bottom_corr.index)
        plt.title(f'Bottom 20 Features Correlated with {target_col}', fontsize=16)
        plt.tight_layout()
        plt.savefig('bottom_correlations.png')
        plt.show()

        return correlations


def check_zero_ratio(dataframe):
    """Calculates the ratio of zero values for each column"""
    zero_counts = (dataframe == 0).sum() / len(dataframe)
    return zero_counts.sort_values(ascending=False)


def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    """Calculates outlier thresholds"""
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """Checks for presence of outliers"""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def plot_feature_importances_for_rf(dataframe, target_col, n_features=20):
    """Calculates and visualizes feature importances using Random Forest"""
    from sklearn.ensemble import RandomForestClassifier

    X = dataframe.drop(target_col, axis=1)
    y = dataframe[target_col]

    # Simple Random Forest model training
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Convert feature importances to DataFrame
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(n_features))
    plt.title('Feature Importances (Random Forest)', fontsize=16)
    plt.tight_layout()
    plt.savefig('feature_importances_rf.png')
    plt.show()

    return feature_imp


def plot_pairplot(dataframe, features, target_col, sample_size=5000):
    """Creates pairplot showing relationships between selected features"""
    # Sampling for large datasets
    if len(dataframe) > sample_size:
        sampled_df = dataframe.sample(sample_size, random_state=42)
    else:
        sampled_df = dataframe

    plt.figure(figsize=(15, 15))
    sns.pairplot(sampled_df, vars=features, hue=target_col, diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle('Pairplot of Selected Features', y=1.02, fontsize=16)
    plt.savefig('pairplot_features.png')
    plt.show()


def analyze_feature_by_target(dataframe, feature, target_col):
    """Examines distribution of a specific feature by target variable"""
    plt.figure(figsize=(14, 6))

    # Left side: Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(x=target_col, y=feature, data=dataframe)
    plt.title(f'Boxplot of {feature} by {target_col}')

    # Right side: Violin plot
    plt.subplot(1, 2, 2)
    sns.violinplot(x=target_col, y=feature, data=dataframe, inner='quartile')
    plt.title(f'Violin Plot of {feature} by {target_col}')

    plt.tight_layout()
    plt.savefig(f'analysis_{feature}_by_{target_col}.png')
    plt.show()


print("Loading datasets...")
train_data = pd.read_csv("/kaggle/input/higgs-boson-detection-2025/train.csv")
test_data = pd.read_csv("/kaggle/input/higgs-boson-detection-2025/test.csv") 
submission = pd.read_csv("/kaggle/input/higgs-boson-detection-2025/sample_submission.csv")

print("Train:", train_data.shape)
print("Test:", test_data.shape)
print("Submission:", submission.shape)

# Dataset examination
print("\n=== Dataset Examination ===")
check_df(train_data)


print("\n=== Label Distribution ===")
print(train_data['label'].value_counts())
print(train_data['label'].value_counts() / len(train_data))

# Label distribution visualization
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=train_data)
plt.title('Label Distribution')
plt.savefig('label_distribution.png')
plt.show()


# Ratio of zero values
print("\n=== Ratio of Zero Values ===")
zero_ratios = check_zero_ratio(train_data)
print(zero_ratios)

# Features with high zero content
high_zero_features = zero_ratios[zero_ratios > 0.3].index.tolist()
print("\n=== Features with High Zero Content ===")
print(high_zero_features)

# Visualization of features with high zero content
if len(high_zero_features) > 0:
    print("\n=== Distribution of Features with High Zero Content ===")
    plot_feature_distributions(train_data, high_zero_features)


# Correlation analysis
print("\n=== Correlation Analysis ===")
correlations = plot_correlation_matrix(train_data, 'label')

# Detailed examination of top correlated features
top_corr_features = correlations.iloc[1:6].index.tolist()  # Top 5 features (excluding label)
print("\n=== Detailed Examination of Top Correlated Features ===")
for feature in top_corr_features:
    analyze_feature_by_target(train_data, feature, 'label')


# Outlier analysis
print("\n=== Outlier Analysis ===")
for col in train_data.columns[1:]:  # All features except label
    has_outlier = check_outlier(train_data, col)
    print(f"{col}: {'Has outliers' if has_outlier else 'No outliers'}")


# Feature distributions - visualize all features in small groups
print("\n=== Feature Distributions ===")
all_features = train_data.columns[1:].tolist()  # All features except label
for i in range(0, len(all_features), 6):  # Show 6 features at a time
    feature_group = all_features[i:i + 6]
    plot_feature_distributions(train_data, feature_group)


# Random Forest feature importance
print("\n=== Random Forest Feature Importance ===")
feature_imp = plot_feature_importances_for_rf(train_data, 'label')


# Pairplot - visualize most important features
top_features = feature_imp['Feature'].head(5).tolist()
print("\n=== Pairplot Analysis of Top Features ===")
plot_pairplot(train_data, top_features, 'label')


# Summary information
print("\n=== EDA Summary ===")
print(f"Total number of records: {len(train_data)}")
print(f"Number of features: {train_data.shape[1] - 1}")  # Excluding label
print(f"Positive class ratio: {train_data['label'].mean():.4f}")
print(f"Number of features with high zero content: {len(high_zero_features)}")
print(f"Number of features with outliers: {sum([check_outlier(train_data, col) for col in train_data.columns[1:]])}")

