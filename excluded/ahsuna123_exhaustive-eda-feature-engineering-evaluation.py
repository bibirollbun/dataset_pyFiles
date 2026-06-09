# Importing Libraries
!pip install imbalanced-learn==0.11.0

import warnings
warnings.filterwarnings("ignore")

import optuna
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, median_absolute_error
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import catboost as cb
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
original_data = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv')


train_data.sample(5)


test_data.sample(5)


original_data.sample(5)


# Checking the number of rows and columns

num_train_rows, num_train_columns = train_data.shape

num_test_rows, num_test_columns = test_data.shape

num_original_rows, num_original_columns = original_data.shape

print("Training Data:")
print(f"Number of Rows: {num_train_rows}")
print(f"Number of Columns: {num_train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {num_test_rows}")
print(f"Number of Columns: {num_test_columns}\n")

print("Original Data:")
print(f"Number of Rows: {num_original_rows}")
print(f"Number of Columns: {num_original_columns}")



# Creating a table for missing values, unique values and data types of the features

missing_values_train = pd.DataFrame({'Feature': train_data.columns,
                              '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_data.isnull().sum().values)/len(train_data)*100)})

missing_values_test = pd.DataFrame({'Feature': test_data.columns,
                             '[TEST] No.of Missing Values': test_data.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_data.isnull().sum().values)/len(test_data)*100)})

missing_values_original = pd.DataFrame({'Feature': original_data.columns,
                             '[ORIGINAL] No.of Missing Values': original_data.isnull().sum().values,
                             '[ORIGINAL] % of Missing Values': ((original_data.isnull().sum().values)/len(original_data)*100)})

unique_values = pd.DataFrame({'Feature': train_data.columns,
                              'No. of Unique Values[FROM TRAIN]': train_data.nunique().values})

feature_types = pd.DataFrame({'Feature': train_data.columns,
                              'DataType': train_data.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, missing_values_original, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

merged_df.style.background_gradient(cmap='viridis')


# Count duplicate rows in train_data
train_duplicates = train_data.duplicated().sum()

# Count duplicate rows in test_data
test_duplicates = test_data.duplicated().sum()

# Count duplicate rows in original_data
original_duplicates = original_data.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")
print(f"Number of duplicate rows in original_data: {original_duplicates}")



# Having a look at the description of all the numerical columns present in the dataset
print('Description of all the numerical columns present in the train dataset')
train_data.describe().T.style.background_gradient(cmap='viridis')


# Having a look at the description of all the numerical columns present in the dataset
print('Description of all the numerical columns present in the test dataset')
test_data.describe().T.style.background_gradient(cmap='viridis')


# Having a look at the description of all the numerical columns present in the dataset
print('Description of all the numerical columns present in the original dataset')
original_data.describe().T.style.background_gradient(cmap='viridis')


numerical_variables = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
target_variable = 'Personality' 
categorical_variables = ['Stage_fear', 'Drained_after_socializing']


# Analysis of all NUMERICAL features

# Define a custom color palette
custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Add 'Dataset' column to distinguish between train and test data
train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'
original_data['Dataset'] = 'Original'

# Function to create and display a row of plots for a single variable
def create_variable_plots(variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train_data, test_data,original_data.dropna()]), x=variable, y="Dataset", palette=custom_palette)
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")

    # Separate Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=test_data, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    sns.histplot(data=original_data.dropna(), x=variable, color=custom_palette[2], kde=True, bins=30, label="Original")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} [TRAIN, TEST & ORIGINAL]")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for each variable
for variable in numerical_variables:
    create_variable_plots(variable)

# Drop the 'Dataset' column after analysis
train_data.drop('Dataset', axis=1, inplace=True)
test_data.drop('Dataset', axis=1, inplace=True)
original_data.drop('Dataset', axis=1, inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define colors for Train, Test, and Original data
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

# Function to create and display a grouped count plot for a single categorical variable
def create_categorical_barplot(variable):
    sns.set_style('whitegrid')

    # Combine the datasets and create a new column indicating the source
    train_data_copy = train_data.copy()
    test_data_copy = test_data.copy()
    original_data_copy = original_data.dropna().copy()

    train_data_copy['Dataset'] = 'Train'
    test_data_copy['Dataset'] = 'Test'
    original_data_copy['Dataset'] = 'Original'

    combined_data = pd.concat([train_data_copy, test_data_copy, original_data_copy])

    # Get sorted order of categories based on Train data count (small to big)
    train_counts = train_data[variable].value_counts().sort_values(ascending=True).index.tolist()

    # Plot grouped countplot (Horizontal bars)
    plt.figure(figsize=(14, 7))
    sns.countplot(
        data=combined_data, 
        x=variable,  # Swapped axes
        hue="Dataset", 
        palette=custom_palette, 
        dodge=True,  # Ensures grouped bars
        width=0.85,  # Further increased bar width
        order=train_counts  # Sorting categories by Train data count (small to big)
    )

    plt.ylabel("Count")
    plt.xlabel(variable)
    plt.title(f"Grouped Count Plot for {variable} [TRAIN, TEST & ORIGINAL]")
    plt.legend(title="Dataset")

    # Rotate x labels for better visibility
    plt.xticks(rotation=45, ha="right")

    # Show the plot
    plt.show()

# Perform univariate analysis for each categorical variable
for variable in categorical_variables:
    create_categorical_barplot(variable)


pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779', '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']

countplot_color = '#5C67A3'

# Function to create and display a row of plots for a single target variable
def create_target_plots(variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie Chart
    plt.subplot(1, 2, 1)
    train_data[variable].value_counts().plot.pie(
        autopct='%1.1f%%', colors=pie_chart_palette, wedgeprops=dict(width=0.3), startangle=140
    )
    plt.title(f"Pie Chart for {variable}")

    # Bar Graph
    plt.subplot(1, 2, 2)
    sns.countplot(
        data=pd.concat([train_data, original_data.dropna()]), 
        x=variable, 
        color=countplot_color,  # Using a single color for the countplot
        alpha=0.8  # Setting 80% opacity
    )
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title(f"Bar Graph for {variable} [TRAIN & ORIGINAL Combined]")

    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Show the plots
    plt.show()

# Perform univariate analysis for target variable
create_target_plots(target_variable)


variables = [col for col in train_data.columns if col in numerical_variables]

# Adding variables to the existing list
test_variables = variables
train_variables = variables

# Calculate correlation matrices for train_data and test_data
corr_train = train_data[train_variables].corr()
corr_test = test_data[test_variables].corr()

# Create masks for the upper triangle
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

# Set the text size and rotation
annot_kws = {"size": 8, "rotation": 45}

# Generate heatmaps for train_data
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
ax_train = sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
                      square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data')

# Generate heatmaps for test_data
plt.subplot(1, 2, 2)
ax_test = sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
                     square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# Selecting numerical features + target variable
variables = [col for col in train_data.columns if col in numerical_variables]
train_variables = variables 

# Compute correlation with 'rainfall' and transpose for horizontal display
corr_train = train_data[train_variables].corr().T  # Transpose for horizontal orientation

# Set the text size and rotation
annot_kws = {"size": 10}  # Increased size for better visibility

# Generate horizontal heatmap without color bar
plt.figure(figsize=(10, 2))  # Adjusted for a horizontal layout
ax_train = sns.heatmap(corr_train, cmap='viridis', annot=True, 
                      square=False, linewidths=0.5, annot_kws=annot_kws, 
                      cbar=False)  # **Removed color bar**

# Formatting
plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
plt.title('Correlation Heatmap - Train Data')
plt.yticks(rotation=0)  # Keep y-labels horizontal

# Show plot
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Define color schemes
train_color = '#3498db'
test_color = '#e74c3c'
personality_colors = {'Introvert': '#8e44ad', 'Extrovert': '#27ae60'}

# Identify numerical columns
numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'id' in numerical_columns:
    numerical_columns.remove('id')

# Plot loop
for column in numerical_columns:
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1])

    # --- Trend Plot: ID vs Feature ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(train_data['id'], train_data[column], linestyle='-', color=train_color, label='Train Data', alpha=0.7)
    ax0.plot(test_data['id'], test_data[column], linestyle='-', color=test_color, label='Test Data', alpha=0.7)
    ax0.set_xlabel('ID', fontsize=14)
    ax0.set_ylabel(column, fontsize=14)
    ax0.set_title(f'Trend Plot: {column} vs ID', fontsize=16, fontweight='bold')
    ax0.legend(fontsize=12)
    ax0.grid(True, linestyle='--', alpha=0.5)

    # --- KDE Plot: Feature Distribution by Personality ---
    ax1 = fig.add_subplot(gs[1, 0])
    sns.kdeplot(
        data=train_data, x=column, hue='Personality',
        palette=personality_colors, ax=ax1, fill=True, common_norm=False, alpha=0.6
    )
    ax1.set_xlabel(column, fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.set_title(f'Distribution (KDE) of {column} by Personality', fontsize=16, fontweight='bold')
    ax1.legend(title='Personality', fontsize=12, title_fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Layout
    plt.tight_layout(pad=3.0)
    plt.show()

    # Separator line
    plt.figure(figsize=(16, 0.3))
    plt.axhline(y=0, color='gray', linewidth=5, linestyle='-') 
    plt.axis('off')
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Define color schemes
train_color = '#3498db'
test_color = '#e74c3c'
personality_colors = {'Introvert': '#8e44ad', 'Extrovert': '#27ae60'}

# Identify numerical columns
numerical_columns = original_data.select_dtypes(include=['int64', 'float64']).columns.tolist()


# Plot loop
for column in numerical_columns:
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1])

    # --- Trend Plot: ID vs Feature ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(original_data[column], linestyle='-', color=train_color, label='Original Data', alpha=0.7)
    ax0.plot(test_data[column], linestyle='-', color=test_color, label='Test Data', alpha=0.7)
    ax0.set_xlabel('ID', fontsize=14)
    ax0.set_ylabel(column, fontsize=14)
    ax0.set_title(f'Trend Plot: {column} vs ID', fontsize=16, fontweight='bold')
    ax0.legend(fontsize=12)
    ax0.grid(True, linestyle='--', alpha=0.5)

    # --- KDE Plot: Feature Distribution by Personality ---
    ax1 = fig.add_subplot(gs[1, 0])
    sns.kdeplot(
        data=original_data, x=column, hue='Personality',
        palette=personality_colors, ax=ax1, fill=True, common_norm=False, alpha=0.6
    )
    ax1.set_xlabel(column, fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.set_title(f'Distribution (KDE) of {column} by Personality', fontsize=16, fontweight='bold')
    ax1.legend(title='Personality', fontsize=12, title_fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Layout
    plt.tight_layout(pad=3.0)
    plt.show()

    # Separator line
    plt.figure(figsize=(16, 0.3))
    plt.axhline(y=0, color='gray', linewidth=5, linestyle='-') 
    plt.axis('off')
    plt.show()



from sklearn.impute import SimpleImputer
# 1. Handle categorical columns (Yes/No) â€” impute with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_variables] = cat_imputer.fit_transform(train_data[categorical_variables])
test_data[categorical_variables] = cat_imputer.transform(test_data[categorical_variables])
original_data[categorical_variables] = cat_imputer.fit_transform(original_data[categorical_variables])

# 2. Handle numerical columns â€” impute with median
num_imputer = SimpleImputer(strategy='median')
train_data[numerical_variables] = num_imputer.fit_transform(train_data[numerical_variables])
test_data[numerical_variables] = num_imputer.transform(test_data[numerical_variables])
original_data[numerical_variables] = num_imputer.transform(original_data[numerical_variables])


import numpy as np
import pandas as pd
import scipy.stats as stats  # For Box-Cox, Yeo-Johnson

def perform_feature_engineering(df):
    """
    Applies feature engineering to the dataframe for personality prediction.
    """

    # 1. Binary Encoding for Categorical Columns
    df['Stage_fear_bin'] = (df['Stage_fear'] == 'Yes').astype(int)
    df['Drained_bin'] = (df['Drained_after_socializing'] == 'Yes').astype(int)

    # 2. Interaction Features
    df['alone_x_social'] = df['Time_spent_Alone'] * df['Social_event_attendance']
    df['alone_x_drained'] = df['Time_spent_Alone'] * df['Drained_bin']
    df['outgoing_score'] = df['Going_outside'] + df['Social_event_attendance']
    df['social_energy_ratio'] = df['Social_event_attendance'] / (df['Time_spent_Alone'] + 1)
    df['friends_post_ratio'] = df['Friends_circle_size'] / (df['Post_frequency'] + 1)

    # 3. Grouped/Binned Features
    df['post_freq_level'] = pd.cut(df['Post_frequency'], bins=[-1, 2, 5, 15], labels=['Low', 'Medium', 'High'])
    df['friends_group'] = pd.cut(df['Friends_circle_size'], bins=[-1, 3, 8, 20], labels=['Small', 'Medium', 'Large'])
    df['alone_level'] = pd.cut(df['Time_spent_Alone'], bins=[-1, 2, 6, 24], labels=['Low', 'Moderate', 'High'])

    # 4. Log & Power Transforms (handle skewness)
    df['log_alone_time'] = np.log1p(df['Time_spent_Alone'])
    df['sqrt_post_freq'] = np.sqrt(df['Post_frequency'])

    # 5. Box-Cox and Yeo-Johnson transforms (for advanced models)
    df['alone_boxcox'], _ = stats.boxcox(df['Time_spent_Alone'] + 1)
    df['postfreq_yeojohnson'], _ = stats.yeojohnson(df['Post_frequency'])

    # 6. Composite Behavior Score
    df['extroversion_score'] = (
        df['Social_event_attendance'] + 
        df['Going_outside'] + 
        df['Friends_circle_size'] + 
        df['Post_frequency'] -
        df['Time_spent_Alone'] -
        df['Drained_bin'] * 2
    )

    # 7. Boolean Flags / Outlier Tagging
    df['highly_active'] = ((df['Post_frequency'] > 7) & (df['Social_event_attendance'] > 7)).astype(int)
    df['low_social_low_friends'] = ((df['Social_event_attendance'] < 2) & (df['Friends_circle_size'] < 3)).astype(int)

    return df

# Save test ids (if needed)
id_test = test_data['id']

# Combine train and test
full_data = pd.concat([train_data, test_data], axis=0).sort_values('id').reset_index(drop=True)

# Apply feature engineering
full_data = perform_feature_engineering(full_data)

# Split back
train_data = full_data[full_data['Personality'].notna()].reset_index(drop=True)
test_data = full_data[full_data['Personality'].isna()].reset_index(drop=True)


newly_created_vars = [
    # 1. Binary encodings
    'Stage_fear_bin', 'Drained_bin',

    # 2. Interactions
    'alone_x_social', 'alone_x_drained', 'outgoing_score',
    'social_energy_ratio', 'friends_post_ratio',

    # 3. Binned groups
    'post_freq_level', 'friends_group', 'alone_level',

    # 4. Transforms
    'log_alone_time', 'sqrt_post_freq', 'alone_boxcox', 'postfreq_yeojohnson',

    # 5. Composite score
    'extroversion_score',

    # 6. Flags
    'highly_active', 'low_social_low_friends'
]

# Categorical new features
categorical_new_feats = ['post_freq_level', 'friends_group', 'alone_level']



import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Encode Target
# -----------------------------
train_data['Personality_encoded'] = train_data['Personality'].map({'Introvert': 0, 'Extrovert': 1})
y = train_data['Personality_encoded']

# -----------------------------
# 1. Correlation (point biserial) for numeric features
# -----------------------------


numerical_variables = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
numeric_engineered = [f for f in newly_created_vars if pd.api.types.is_numeric_dtype(train_data[f])]

numeric_features = numerical_variables + numeric_engineered
num_scores = {}

for feature in numeric_features:
    try:
        corr, _ = pointbiserialr(train_data[feature], y)
        num_scores[feature] = corr
    except Exception as e:
        print(f"Numeric skip: {feature} â†’ {e}")


# 2. Mutual Information for categorical existing + engineered

categorical_variables = ['Stage_fear', 'Drained_after_socializing']
categorical_engineered = ['post_freq_level', 'friends_group', 'alone_level']

categorical_features = categorical_variables + categorical_engineered
X_cat = train_data[categorical_features].apply(LabelEncoder().fit_transform)

mi_scores = mutual_info_classif(X_cat, y, discrete_features=True)
mi_scores_dict = dict(zip(categorical_features, mi_scores))

# -----------------------------
# 3. Combine into one DataFrame
# -----------------------------
combined_data = pd.DataFrame({
    'Feature': list(num_scores.keys()) + list(mi_scores_dict.keys()),
    'Score': list(num_scores.values()) + list(mi_scores_dict.values()),
    'Method': ['PointBiserial'] * len(num_scores) + ['MutualInfo'] * len(mi_scores_dict)
})

combined_data = combined_data.sort_values('Score', ascending=False)

# -----------------------------
# 4. Heatmap Visualization
# -----------------------------
plt.figure(figsize=(14, 3))
sns.heatmap(combined_data.set_index('Feature')[['Score']].T,
            annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
plt.title("ğŸ“Š Feature Relevance to Personality (Combined View)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# Target variable
# ------------------------------
y = train_data['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# ------------------------------
# Encode categorical features
# ------------------------------
categorical_features = ['Stage_fear', 'Drained_after_socializing',
                        'post_freq_level', 'friends_group', 'alone_level']

encoded_data = train_data[categorical_features].apply(LabelEncoder().fit_transform)

# ------------------------------
# Combine all features
# ------------------------------
# Numerical features (existing + engineered)
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                      'Friends_circle_size', 'Post_frequency']

numeric_engineered = [f for f in newly_created_vars if pd.api.types.is_numeric_dtype(train_data[f])]
X_num = train_data[numerical_features + numeric_engineered]

# Final feature matrix
X = pd.concat([X_num, encoded_data], axis=1)

# ------------------------------
# Train Random Forest Classifier
# ------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X, y)

# ------------------------------
# Get Feature Importances
# ------------------------------
feature_importances = rf.feature_importances_
important_features = np.argsort(feature_importances)[::-1][:15]
selected_features = X.columns[important_features]
selected_importance = feature_importances[important_features]

print(f"Top {len(selected_features)} important features for Personality Prediction:")
for f, imp in zip(selected_features, selected_importance):
    print(f"{f:<30} â†’  {imp:.4f}")

# ------------------------------
# Visualization
# ------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x=selected_importance, y=selected_features, palette="magma")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 15 Feature Importances for Personality Classification")
plt.gca().invert_yaxis()
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()




# Compute correlation matrix
corr_matrix = X[selected_features].corr().abs()

# Create a mask to filter highly correlated features 
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_correlation = [column for column in upper.columns if any(upper[column] > 0.80)]

# Remove highly correlated features
final_features = [f for f in selected_features if f not in high_correlation]

# Display final selected features
print(f"Final Selected Features After Correlation Filtering: {final_features}")

# Visualization: Correlation Heatmap Before & After Filtering
plt.figure(figsize=(12, 6))

# Before Filtering
plt.subplot(1, 2, 1)
sns.heatmap(corr_matrix, annot=False, cmap="viridis", linewidths=0.5)
plt.title("Feature Correlation Before Filtering")

# After Filtering (Subset of Final Features)
filtered_corr_matrix = X[final_features].corr().abs()
plt.subplot(1, 2, 2)
sns.heatmap(filtered_corr_matrix, annot=False, cmap="viridis", linewidths=0.5)
plt.title("Feature Correlation After Filtering")

plt.tight_layout()
plt.show()


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression  # Classifier for binary labels
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# ----------------------------
# Step 1: Encode Categorical Target
# ----------------------------
y = train_data['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# ----------------------------
# Step 2: Prepare Features
# ----------------------------
# Categorical features
categorical_features = ['Stage_fear', 'Drained_after_socializing',
                        'post_freq_level', 'friends_group', 'alone_level']
X_cat = train_data[categorical_features].apply(LabelEncoder().fit_transform)

# Numerical features (existing + engineered)
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                      'Friends_circle_size', 'Post_frequency']
numeric_engineered = [f for f in newly_created_vars if pd.api.types.is_numeric_dtype(train_data[f])]
X_num = train_data[numerical_features + numeric_engineered]

# Combine both
X = pd.concat([X_num, X_cat], axis=1)

# ----------------------------
# Step 3: Run RFE
# ----------------------------
n_features_to_select = 10
estimator = LogisticRegression(max_iter=1000)
rfe = RFE(estimator, n_features_to_select=n_features_to_select)
rfe.fit(X, y)

# ----------------------------
# Step 4: Extract Selected Features
# ----------------------------
selected_rfe_features = X.columns[rfe.support_].tolist()

print("ğŸ”� Selected Features using RFE:")
for i, feature in enumerate(selected_rfe_features, 1):
    print(f"{i:>2}. {feature}")



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ------------------------
# Step 1: Identify numerical features
# ------------------------
columns_to_check = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove ID and target columns
columns_to_check = [col for col in columns_to_check if col not in ['id', 'Personality_encoded']]

# ------------------------
# Step 2: Function to remove outliers using IQR
# ------------------------
def remove_outliers_iqr_with_plot(data, column):
    Q1 = data[column].quantile(0.05)
    Q3 = data[column].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    rows_deleted = len(data) - len(filtered_data)
    
    if rows_deleted > 0:
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        sns.boxplot(x=data[column], ax=axes[0], color='lightblue',
                    flierprops={'markerfacecolor': 'red'})
        axes[0].set_title(f'Before Outlier Removal: {column}')
        axes[0].axvline(Q1, color='green', linestyle='--', label='Q1 (5th %)')
        axes[0].axvline(Q3, color='blue', linestyle='--', label='Q3 (95th %)')
        axes[0].axvline(lower_bound, color='red', linestyle='-', label='Lower Bound')
        axes[0].axvline(upper_bound, color='red', linestyle='-', label='Upper Bound')
        axes[0].legend()

        sns.boxplot(x=filtered_data[column], ax=axes[1], color='lightgreen',
                    flierprops={'markerfacecolor': 'red'})
        axes[1].set_title(f'After Outlier Removal: {column}')
        
        plt.suptitle(f'Outlier Detection & Removal: {column}')
        plt.tight_layout()
        plt.show()

        print(f"âœ… {column} â†’ {rows_deleted} rows deleted")

    return filtered_data, rows_deleted

# ------------------------
# Step 3: Apply IQR Outlier Removal
# ------------------------
rows_deleted_total = 0
features_with_outliers = []

for column in columns_to_check:
    train_data_filtered, rows_deleted = remove_outliers_iqr_with_plot(train_data, column)
    
    if rows_deleted > 0:
        train_data = train_data_filtered
        rows_deleted_total += rows_deleted
        features_with_outliers.append(column)

# ------------------------
# Summary
# ------------------------
print("\nğŸ“Š **Outlier Removal Summary:**")
if features_with_outliers:
    print(f"Total rows deleted: {rows_deleted_total}")
    print(f"Features with outliers removed: {features_with_outliers}")
else:
    print("No significant outliers detected.")


y = train_data["Personality_encoded"]


from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Step 1: Define numerical features to scale
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                      'Friends_circle_size', 'Post_frequency']

# Step 2: Add engineered numerical features
numeric_engineered = [f for f in newly_created_vars if pd.api.types.is_numeric_dtype(train_data[f])]
features_to_scale = numerical_features + numeric_engineered

# Step 3: Subset train and test data
X_train_num = train_data[features_to_scale].copy()
X_test_num = test_data[features_to_scale].copy()

# Step 4: Initialize and fit scaler on training data
scaler = MinMaxScaler()
scaler.fit(X_train_num)

# Step 5: Transform train and test data
scaled_train_array = scaler.transform(X_train_num)
scaled_test_array = scaler.transform(X_test_num)

# Step 6: Create scaled DataFrames
scaled_train_df = pd.DataFrame(scaled_train_array, columns=features_to_scale, index=train_data.index)
scaled_test_df = pd.DataFrame(scaled_test_array, columns=features_to_scale, index=test_data.index)



train_data_combined = scaled_train_df

test_data_combined = scaled_test_df


from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define Stratified CV
def stratified_cross_validation(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        yield X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]

# Base models
log_reg = LogisticRegressionCV(
    cv=5,
    penalty='l2',
    solver='liblinear',
    class_weight='balanced',
    random_state=42
)

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

lgbm = LGBMClassifier(
    random_state=42
)

# Voting Classifier
ensemble_model = VotingClassifier(
    estimators=[('lr', log_reg), ('xgb', xgb), ('lgbm', lgbm)],
    voting='soft'  # Soft voting: average probabilities
)

# Train-validation split
X = train_data_combined.copy()
y = y.astype(int)

auc_scores = []
fold = 1

print("Training Ensemble with Stratified K-Fold Cross-Validation...")

for X_train, X_val, y_train, y_val in stratified_cross_validation(X, y, n_splits=5):
    ensemble_model.fit(X_train, y_train)
    y_val_proba = ensemble_model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    fold_auc = roc_auc_score(y_val, y_val_proba)
    print(f"Fold {fold} - ROC-AUC: {fold_auc:.4f}")
    print(classification_report(y_val, y_val_pred))
    print("-" * 40)

    auc_scores.append(fold_auc)
    fold += 1

# Average ROC-AUC
avg_auc = np.mean(auc_scores)
print(f"\nAverage ROC-AUC for Ensemble: {avg_auc:.4f}")

# -------------------------
# Final training on full data
# -------------------------
print("Training on full data...")
ensemble_model.fit(X, y)

# Predict on test data
test_proba = ensemble_model.predict_proba(test_data_combined)[:, 1]
test_preds = (test_proba >= 0.52).astype(int)

# Map predictions
label_map = {0: 'Introvert', 1: 'Extrovert'}
test_labels = [label_map[p] for p in test_preds]

# Create submission
submission_df = pd.DataFrame({
    'id': id_test,
    'Personality': test_labels
})
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())

# -------------------------
# ROC Curve for Best Fold
# -------------------------
best_fold_idx = np.argmax(auc_scores)
best_fold_data = list(stratified_cross_validation(X, y, n_splits=5))[best_fold_idx]
X_train_best, X_val_best, y_train_best, y_val_best = best_fold_data

ensemble_model.fit(X_train_best, y_train_best)
y_val_proba_best = ensemble_model.predict_proba(X_val_best)[:, 1]

fpr, tpr, _ = roc_curve(y_val_best, y_val_proba_best)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="#3498db", label=f"ROC Curve (AUC = {auc_scores[best_fold_idx]:.4f})")
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Ensemble (Best Fold: {best_fold_idx + 1})")
plt.legend()
plt.grid(alpha=0.3)
plt.show()





