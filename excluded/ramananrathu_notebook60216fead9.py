# main libraries
import pandas as pd
import numpy as np

# Libraries to visualize 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Statistitcal library
from scipy.stats import skew

# Display utilities for Jupyter notebooks
from IPython.display import display

# AI libraries 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb



# Data set load 
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_data = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
# Verify shapes
print("Train Data Shape:", train_data.shape)
print("\nTest Data Shape:", test_data.shape)


# Data frame information
print("Train Data Information:")
train_data.info()

print("\nTest Data Information:")
test_data.info()


The training data consists of 750,000 rows and 10 columns , while the test data contains 250,000 rows and 9 columns â€” notably missing the "Fertilizer Name" column, which suggests that this is the target variable to be predicted.

All columns in both datasets are fully populated with no missing values, indicating a clean dataset. The training set includes a mix of numerical (int64) and categorical (object) data types: 7 integer columns (e.g., Temperature, Humidity, Nitrogen) and 3 categorical columns (Soil Type, Crop Type, Fertilizer Name). The test set mirrors the 7 numerical features but excludes the target variable.


def generate_stats(df, exclude_columns=None):
    """
    Generates descriptive statistics for a DataFrame after excluding specified columns.
    Returns transposed statistics with gradient styling.
    """
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors='ignore')
    return df.describe().T.style.background_gradient(cmap='RdYlGn')

# Train Data Statistics
print("Train Data Descriptive Statistics:")
train_stats = generate_stats(train_data, exclude_columns=['id'])
display(train_stats)

# Test Data Statistics
print("\nTest Data Descriptive Statistics:")
test_stats = generate_stats(test_data, exclude_columns=['id'])
display(test_stats)


def show_categorical_insights(dataframe, cat_columns, set_name="Dataset"):
    """
    Displays insights about categorical columns in a given dataset.
    Shows number of unique values, list of values, and most frequent value.
    """
    print("\n" + "-" * 60)
    print(f"Categorical Feature Insights: {set_name}")
    print("-" * 60)

    for column in cat_columns:
        distinct_values = dataframe[column].unique()
        freq_distribution = dataframe[column].value_counts()
        most_common_value = freq_distribution.idxmax()
        highest_count = freq_distribution.max()

        print(f"\nFeature: '{column}'")
        print(f"Total Unique Values: {dataframe[column].nunique()}")
        print(f"All Unique Values: {sorted(distinct_values)}")
        print(f"Most Frequent Value: '{most_common_value}' (Count: {highest_count})")

# Categorical Columns.
rewritingMethod_train_categories = ['Soil Type', 'Crop Type', 'Fertilizer Name']
rewritingMethod_test_categories = ['Soil Type', 'Crop Type']

# Run Analysis
show_categorical_insights(train_data, rewritingMethod_train_categories, "Train Data")
show_categorical_insights(test_data, rewritingMethod_test_categories, "Test Data")


The categorical analysis shows consistent categories across train and test datasets. Both share the same 5 soil types and 11 crop types, with 'Sandy' soil and 'Paddy' crop being most frequent. The train set includes 7 fertilizer types, commonly '14-35-14', suggesting targeted application patterns. Uniformity in categorical features indicates a well-prepared dataset for modeling. However, slight differences in frequency distributions may affect model generalization if not properly balanced during training.


def check_missing_values(dataframe, set_label):
    """
    Check missing value information for a given dataset.
    """
    total_missing = dataframe.isna().sum().sum()
    total_rows = len(dataframe)

    print("-" * 45)
    print(f"Missing Value Analysis: {set_label}")
    print("-" * 45)

    if total_missing == 0:
        print(f"âœ… No missing values in {total_rows:,} rows.")
    else:
        print(f" Total missing values: {total_missing} in {total_rows:,} rows.")

# Dictionary mapping dataset names to their respective DataFrames
rewritingMethod_dataset_registry = {
    "Train Dataset": train_data,
    "Test Dataset": test_data
}

# Run missing value check across all datasets
for dataset_name, df in rewritingMethod_dataset_registry.items():
    check_missing_values(df, dataset_name)
    print()  # Add spacing between reports


The missing value analysis confirms that both the Train Dataset (750,000 rows) and Test Dataset (250,000 rows) are clean and complete , with no null or missing entries. This suggests the data has been preprocessed effectively before modeling. The absence of missing values simplifies the analysis and modeling process, reducing the need for imputation or data-cleaning steps. It also increases confidence in the reliability of the dataset for training and evaluating machine learning models without introducing bias or inaccuracies due to missing information.


def duplicate_entries(dataframe, set_name):
    """
    Checks for duplicate rows in a DataFrame and prints a detailed report.
    """
    duplicate_total = dataframe.duplicated().sum()
    row_count = len(dataframe)

    print("-" * 50)
    print(f"Duplicate Entry Analysis: {set_name}")
    print("-" * 50)

    if duplicate_total == 0:
        print(f" No duplicate records found in {row_count:,} rows.")
    else:
        duplication_rate = duplicate_total / row_count
        print(f"   Duplicates detected: {duplicate_total:,} entries")
        print(f"   Out of total rows: {row_count:,}")
        print(f"   Duplication Rate: {duplication_rate:.2%}")

# Dictionary of datasets to analyze
rewritingMethod_dataset_pool = {
    "Train Dataset": train_data,
    "Test Dataset": test_data,
}

# Store results in a summary dictionary
duplicate_insights = {}

# Run analysis for each dataset
for dataset_label, df in rewritingMethod_dataset_pool.items():
    duplicate_entries(df, dataset_label)
    duplicate_insights[dataset_label] = {
        'duplicate_count': df.duplicated().sum(),
        'total_records': len(df)
    }
    print()  # Add spacing between reports

# Optional: Print the insights dictionary
print("ðŸ“Š Duplicate Insights Summary:")
print(duplicate_insights)


The duplicate analysis confirms that both the Train Dataset (750,000 rows) and Test Dataset (250,000 rows) are free of duplicate entries , ensuring data uniqueness and integrity. The insights summary further validates this, showing a duplicate_count of 0 for both datasets. This indicates high-quality, well-prepared data, which is essential for reliable model training and evaluation. The absence of duplicates reduces the risk of overfitting and ensures that patterns learned by the model are based on genuine variations in the data rather than repeated instances.


Data analyzing 


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def rewritingMethod_plot_fertilizer_distribution(data_dict, target='Fertilizer Name'):
    """
    Generates count plots and pie charts to visualize fertilizer distribution across datasets.
    Handles any number of datasets dynamically.
    """
    num_datasets = len(data_dict)
    fig, axes = plt.subplots(num_datasets, 2, figsize=(14, 6 * num_datasets))

    # If only one dataset, wrap axes in list to keep indexing consistent
    if num_datasets == 1:
        axes = [axes]

    for idx, (title, df) in enumerate(data_dict.items()):
        ax_bar = axes[idx][0]
        count_data = df[target].value_counts()
        
        # Count Plot
        sns.countplot(y=target, data=df, ax=ax_bar, palette='RdYlGn', order=count_data.index)
        ax_bar.set_title(f'{title}: Fertilizer Category Counts', fontsize=14, pad=20)
        ax_bar.set_xlabel('Number of Samples')
        ax_bar.set_ylabel('Fertilizer Name')
        ax_bar.grid(axis='x', linestyle=':', color='gray', alpha=0.7)
        sns.despine(ax=ax_bar, top=True, right=True, left=False, bottom=False)

        # Annotate bars
        for patch in ax_bar.patches:
            y_value = patch.get_y() + patch.get_height() / 2
            x_value = patch.get_width() + count_data.max() * 0.01
            ax_bar.text(x_value, y_value, f'{int(patch.get_width())}', 
                        ha='left', va='center', fontweight='bold', fontsize=10)

        # Pie Chart
        ax_pie = axes[idx][1]
        counts = df[target].value_counts().sort_index()
        colors = sns.color_palette("RdYlGn", n_colors=len(counts))

        wedges, texts, autotexts = ax_pie.pie(
            counts,
            labels=counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.4, edgecolor='white'),
            textprops={'fontsize': 10, 'fontweight': 'bold'}  # Corrected here
        )

        centre_circle = plt.Circle((0, 0), 0.7, fc='white')
        ax_pie.add_patch(centre_circle)
        ax_pie.set_title(f'{title}: Fertilizer Distribution (%)', fontsize=14, pad=25)
        ax_pie.axis('equal')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()

# Dataset mapping
rewritingMethod_dataset_mapping = {
    "Training Data": train_data,
    
}

# Call the function
rewritingMethod_plot_fertilizer_distribution(rewritingMethod_dataset_mapping, target='Fertilizer Name')


def analyze_numerical_distributions(training_set, testing_set, full_dataset, numeric_columns):
    """
    Visualizes the distribution of numerical features using histograms and boxplots.
    Compares distributions across training, testing, and original datasets.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Define colors for visualization
    dataset_colors = sns.color_palette('RdYlGn', 3)

    # Create subplots: one row per feature, two columns (histogram + boxplot)
    fig, axes = plt.subplots(len(numeric_columns), 2, figsize=(14, len(numeric_columns) * 4))
    axes = np.atleast_2d(axes)  # Ensure 2D structure even if single row

    for idx, column in enumerate(numeric_columns):
        # Histogram with KDE overlay
        sns.histplot(
            data=training_set[column],
            color=dataset_colors[0],
            label='Training Set',
            bins=20,
            kde=True,
            ax=axes[idx, 0]
        )
        sns.histplot(
            data=testing_set[column],
            color=dataset_colors[1],
            label='Testing Set',
            bins=20,
            kde=True,
            ax=axes[idx, 0]
        )

        axes[idx, 0].set_title(f'Distribution of {column}', fontsize=12)
        axes[idx, 0].legend()
        axes[idx, 0].set_facecolor("whitesmoke")
        axes[idx, 0].grid(True, linestyle=':', alpha=0.6)

        # Horizontal Boxplot
        sns.boxplot(
            data=[
                training_set[column],
                testing_set[column],
                full_dataset[column]
            ],
            palette=dataset_colors,
            orient='h',
            ax=axes[idx, 1]
        )
        axes[idx, 1].set_title(f'Variation Across Datasets: {column}', fontsize=12)
        axes[idx, 1].set_yticklabels(['Train', 'Test'])
        axes[idx, 1].set_xlabel(column)
        axes[idx, 1].set_facecolor("whitesmoke")
        axes[idx, 1].grid(True, axis='x', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()


def check_skewness(dataframe, set_label, highlight=True, sort_by_skew=True):
    """
    Analyzes skewness for numerical features 
    """
    import pandas as pd
    import numpy as np

    feature_skew = {}

    # Calculate skewness for numeric columns only
    for feature in dataframe.select_dtypes(include=[np.number]).columns:
        feature_skew[feature] = dataframe[feature].skew(skipna=True)

    # Convert to DataFrame
    skew_frame = pd.DataFrame.from_dict(feature_skew, orient='index', columns=['Skewness'])

    # Sort by absolute skew value if enabled
    if sort_by_skew:
        skew_frame = skew_frame.reindex(skew_frame['Skewness'].abs().sort_values(ascending=False).index)

    # Print formatted output
    print(f"\n Skewness Analysis: {set_label}")
    print("-" * 60)
    print(f"{'Feature':<18} | {'Skewness':<10} | {'Distribution Status'}")
    print("-" * 60)

    for feature, row in skew_frame.iterrows():
        skew_value = row['Skewness']
        abs_skew = abs(skew_value)

        # Determine remark and color based on skew level
        if abs_skew > 1:
            status = "Highly skewed"
            color_code = '\033[91m'  # Red
        elif abs_skew > 0.5:
            status = "Moderately skewed"
            color_code = '\033[92m'  # Green
        else:
            status = "Approx. symmetric"
            color_code = ''

        reset_color = '\033[0m'

        # Conditional highlighting
        if highlight and color_code:
            print(f"{color_code}{feature:<18} | {skew_value:>+9.4f} | {status}{reset_color}")
        else:
            print(f"{feature:<18} | {skew_value:>+9.4f} | {status}")

    print("-" * 60)
    return skew_frame

# Run skewness analysis across datasets
rewritingMethod_skew_train = check_skewness(train_data, "Training Set")
rewritingMethod_skew_test = check_skewness(test_data, "Testing Set")


The skewness analysis of both Training Set and Testing Set reveals that all numerical features exhibit very low skewness , indicating nearly symmetric distributions. Features like Moisture, Nitrogen, and `Humidity show slight negative skewness, but values are close to zero, suggesting balanced data spread. The absence of highly or moderately skewed features implies the data is well-distributed, reducing the need for transformations like log scaling. This consistency across datasets enhances model reliability, as similar distributions in train and test sets help ensure stable performance without bias from skewed inputs.


def correlation_heatmaps(data_groups, feature_list):
    """
    Plots correlation heatmaps for multiple datasets to compare feature relationships.
    
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    num_datasets = len(data_groups)
    fig, axes = plt.subplots(ncols=num_datasets, figsize=(6 * num_datasets, 6))
    axes = np.ravel(axes)

    for idx, (title, dataframe) in enumerate(data_groups.items()):
        # Filter features that exist in current DataFrame
        present_features = [f for f in feature_list if f in dataframe.columns]
        df_relevant = dataframe[present_features]

        # Compute correlation matrix
        correlation_matrix = df_relevant.corr()

        # Plot heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": .7},
            ax=axes[idx],
            annot_kws={"fontsize": 10}
        )
        axes[idx].set_title(f'Feature Correlations: {title}', fontsize=14)

    plt.tight_layout()
    plt.show()

# Define numerical features and datasets
rewritingMethod_feature_set = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']

rewritingMethod_dataset_collections = {
    "Training Set": train_data,
    "Evaluation Set": test_data
}

# Call the rewritten method
correlation_heatmaps(rewritingMethod_dataset_collections, rewritingMethod_feature_set)


import pandas as pd

def analyze_correlations(datasets, features):
    """
    Analyzes feature correlations 
    """
    correlation_results = {}

    for name, df in datasets.items():
        # Filter only available features in the current dataset
        available_features = [f for f in features if f in df.columns]
        df_subset = df[available_features]

        # Compute correlation matrix
        corr_matrix = df_subset.corr()

        # Store in dictionary
        correlation_results[name] = corr_matrix

        # Print upper triangle only for readability
        print(f"\Correlation Matrix - {name}")
        print("-" * 60)
        print(corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack())
    
    return correlation_results


# Example usage:

# Define numerical features
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']

# Define your datasets
data_groups = {
    "Train Data": train_data,
    "Test Data": test_data,
}

# Run the correlation analysis
correlation_tables = analyze_correlations(data_groups, numerical_features)


The correlation matrices for both the Train Data and Test Data show that all feature pairs have very low correlation coefficients, mostly close to zero. This indicates that there is no strong linear relationship between any of the numerical features such as Temparature, Humidity, Moisture, Nitrogen, Phosphorous, and Potassium. In the Train Data, the highest positive correlation is between Humidity and Moisture at 0.0034, while in the Test Data, itâ€™s Moisture and Potassium at 0.0045. These near-zero values suggest that each feature contributes unique information and multicollinearity is not a concern. The absence of high positive or negative correlations also implies that models trained on this data wonâ€™t suffer from unstable coefficient estimates due to redundant features. Additionally, the similarity in correlation patterns across train and test datasets supports the assumption that both are drawn from the same underlying distribution, which is crucial for generalization. Overall, this low-correlation environment is ideal for many machine learning algorithms, especially those sensitive to feature independence, and suggests that all features should be retained without dimensionality reduction based on correlation alone.


from scipy.stats import chi2_contingency

# Contingency table between 'Crop Type' and 'Fertilizer Name'
contingency_table = pd.crosstab(train_data['Crop Type'], train_data['Fertilizer Name'])

chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square Statistic: {chi2:.4f}, p-value: {p:.4f}")
if p < 0.05:
    print("Reject null hypothesis: Variables are dependent.")
else:
    print("Fail to reject null hypothesis: Variables are independent.")


from scipy.stats import f_oneway

# Example: Does Nitrogen level vary by Soil Type?
groups = [group['Nitrogen'].values for name, group in train_data.groupby('Soil Type')]
f_stat, p_val = f_oneway(*groups)
print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")


from scipy.stats import kruskal

# Same as above but non-parametric
h_stat, p_val = kruskal(*groups)
print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}, p-value: {p_val:.4f}")


from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
X = train_data[numeric_features]

# VIF DataFrame
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


from scipy.stats import shapiro

feature = train_data['Nitrogen']
stat, p = shapiro(feature)
print(f'Shapiro-Wilk Test for {feature.name}: Statistics={stat:.4f}, p-value={p:.4f}')
if p > 0.05:
    print("Sample looks Gaussian (fail to reject Hâ‚€)")
else:
    print("Sample not Gaussian (reject Hâ‚€)")


from scipy.stats import ttest_ind

train_vals = train_data['Nitrogen']
test_vals = test_data['Nitrogen']

t_stat, p_val = ttest_ind(train_vals, test_vals)
print(f"T-test: t-statistic={t_stat:.4f}, p-value={p_val:.4f}")


# Group by Crop Type and Fertilizer Name
target_analysis = pd.crosstab(
    index=train_data['Crop Type'],
    columns=train_data['Fertilizer Name'],
    margins=True
)
print(target_analysis)


from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

X = train_data[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium', 'Soil Type', 'Crop Type']]
y = train_data['Fertilizer Name']

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
baseline_acc = dummy_clf.score(X_test, y_test)
print(f"Baseline Accuracy: {baseline_acc:.2%}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

 =model RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))


import pandas as pd

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
top_features = feature_importance.sort_values(ascending=False).head(10)
print("Top Predictive Features:")
print(top_features)

# Optional: Plot
import matplotlib.pyplot as plt
top_features.plot(kind='barh', color='skyblue')
plt.title('Feature Importance')
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.show()


new_sample = pd.DataFrame([{
    'Temparature': 30,
    'Humidity': 60,
    'Moisture': 25,
    'Nitrogen': 120,
    'Phosphorous': 50,
    'Potassium': 40,
    'Soil Type': 'Loamy',
    'Crop Type': 'Paddy'
}])
new_sample = pd.get_dummies(new_sample)

# Align columns with training set
new_sample = new_sample.reindex(columns=X.columns, fill_value=0)

predicted_fertilizer = model.predict(new_sample)
print(f"Predicted Fertilizer: {predicted_fertilizer[0]}")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# Define features and target
features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium', 'Soil Type', 'Crop Type']
target = 'Fertilizer Name'

X = train_data[features]
y = train_data[target]

X_test = test_data[features]

# Encode target labels
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

# Identify categorical and numerical columns
categorical_cols = [col for col in X.select_dtypes(include='object').columns]
numerical_cols = [col for col in X.select_dtypes(exclude='object').columns]

# Preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Build model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data and train
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
val_preds = model.predict(X_val)
print(f"Model Accuracy: {accuracy_score(y_val, val_preds):.2%}")
print("\nClassification Report:")
print(classification_report(y_val, val_preds, target_names=target_le.classes_))

# Predict probabilities on test data
test_probs = model.predict_proba(X_test)

# Get top 3 classes for each test sample
test_top3_indices = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]

# Map indices back to original fertilizer names
fertilizer_decoder = dict(zip(range(len(target_le.classes_)), target_le.classes_))
submission_labels = [
    ' '.join([fertilizer_decoder[idx] for idx in row]) for row in test_top3_indices
]

# Create submission DataFrame
submission_df = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer Name': submission_labels
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
print("\nâœ… Submission file saved as 'submission.csv'")

