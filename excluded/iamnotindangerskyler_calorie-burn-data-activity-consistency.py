import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# --------------------------------------------------------------
# Retro Synthwave Color Palette and Custom Styling
# --------------------------------------------------------------

def set_synthwave_palette(style="whitegrid", context="notebook", font_family="sans-serif"):
    """Set custom Retro Synthwave color palette and styling for visualizations"""
    palette = ['#f72585', '#b5179e', '#7209b7', '#560bad', '#480ca8',
               '#3a0ca3', '#3f37c9', '#4361ee', '#4895ef', '#4cc9f0']
    
    sns.set_palette(palette)
    sns.set_style(style)
    sns.set_context(context)
    
    # Matplotlib global settings
    plt.rcParams.update({
        'axes.titlepad': 20,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'font.family': font_family,
        'figure.autolayout': True,
        'axes.edgecolor': '#3a0ca3',  # Dark purple frame
        'axes.facecolor': '#ffffff',  # White background
        'figure.facecolor': '#ffffff',  # White figure background
        'axes.labelcolor': '#3a0ca3',  # Dark purple labels
        'axes.titlecolor': '#3a0ca3',  # Dark purple title
        'xtick.color': '#3a0ca3',  # Dark purple tick labels
        'ytick.color': '#3a0ca3',
        'grid.color': '#4cc9f0',  # Light blue grid
        'grid.alpha': 0.5
    })
    
    return palette

# Apply global settings
palette = set_synthwave_palette()


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Keep original ID columns
train_id = train['id'].copy()
test_id = test['id'].copy()

# Display basic information
print("Training set shape:", train.shape)
print("Test set shape:", test.shape)

# Check for missing values
print("\nðŸ“Š Missing values in training set:")
print(train.isna().sum())

print("\nðŸ“Š Missing values in test set:")
print(test.isna().sum())

# Display statistical summary
print("\nðŸ“Š Statistical summary of training data:")
display(train.describe().T)

# Remove ID columns for analysis
train_data = train.drop(columns=['id'])
test_data = test.drop(columns=['id'])

# Identify numeric and categorical columns
numeric_columns = [col for col in train_data.columns if col not in ['Sex', 'Calories'] and train_data[col].dtype in [np.int64, np.float64]]
categorical_columns = [col for col in train_data.columns if col not in numeric_columns + ['Calories']]
target_column = 'Calories'

print(f"\nðŸ“Š Number of numeric features: {len(numeric_columns)}")
print(f"ðŸ“Š Number of categorical features: {len(categorical_columns)}")
print(f"ðŸ“Š Target column: {target_column}")


train_data.head()


train_data.info()


print(f'DataFrame contains {train_data.shape[0]} rows (records) and {train_data.shape[1]} columns (attributes).')


# --------------------------------------------------------------
# Target Variable Analysis
# --------------------------------------------------------------

plt.figure(figsize=(10, 4))
sns.histplot(train_data[target_column], kde=True, color=palette[0])
plt.title('Distribution of Target Variable: Calories', fontweight='bold')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Box plot to check for outliers in target
plt.figure(figsize=(10, 4))
sns.boxplot(x=train_data[target_column], color=palette[1])
plt.title('Boxplot of Calories (Target Variable)', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()

# Target variable statistics
print(f"\nðŸ“Š Target variable statistics:")
print(f"Mean Calories: {train_data[target_column].mean():.2f}")
print(f"Median Calories: {train_data[target_column].median():.2f}")
print(f"Min Calories: {train_data[target_column].min():.2f}")
print(f"Max Calories: {train_data[target_column].max():.2f}")
print(f"Standard Deviation: {train_data[target_column].std():.2f}")
print(f"Skewness: {train_data[target_column].skew():.2f}")
print(f"Kurtosis: {train_data[target_column].kurtosis():.2f}")

# --------------------------------------------------------------
# Feature Distribution Analysis
# --------------------------------------------------------------

# Distribution of numeric features
fig = plt.figure(figsize=(15, 12))
for i, feature in enumerate(numeric_columns, 1):
    plt.subplot((len(numeric_columns) // 3) + 1, 3, i)
    sns.histplot(train_data[feature], kde=True, color=palette[i % len(palette)])
    plt.title(f'Distribution of {feature}', fontweight='bold')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Boxplots to check for outliers in features
fig = plt.figure(figsize=(15, 12))
for i, feature in enumerate(numeric_columns, 1):
    plt.subplot((len(numeric_columns) // 3) + 1, 3, i)
    sns.boxplot(y=train_data[feature], color=palette[i % len(palette)])
    plt.title(f'Boxplot of {feature}', fontweight='bold')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Distribution of categorical features
if categorical_columns:
    fig = plt.figure(figsize=(15, 5))
    for i, feature in enumerate(categorical_columns, 1):
        plt.subplot(1, len(categorical_columns), i)
        sns.countplot(x=train_data[feature], palette=palette)
        plt.title(f'Distribution of {feature}', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------
# Correlation Analysis
# --------------------------------------------------------------

# Correlation matrix
plt.figure(figsize=(15, 12))
correlation_matrix = train_data[numeric_columns + [target_column]].corr()
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask,
           linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Numeric Features', fontweight='bold', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Top correlated features with target
correlations = correlation_matrix[target_column].sort_values(ascending=False)
print("\nðŸ“Š Top correlations with Calories:")
print(correlations)

# Visualization of top correlated features
top_correlated = correlations[1:6].index.tolist()
plt.figure(figsize=(15, 12))
for i, feature in enumerate(top_correlated, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=train_data[feature], y=train_data[target_column], alpha=0.6, color=palette[i])
    plt.title(f'{feature} vs {target_column} (corr: {correlation_matrix.loc[feature, target_column]:.2f})', fontweight='bold')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Feature Engineering - Quantile Binning
# --------------------------------------------------------------

# Quantile-based discretization
n_bins = 10
quantile_labels = [f'Q{i+1}' for i in range(n_bins)]

train_quantile = train_data.copy()
test_quantile = test_data.copy()

for col in numeric_columns:
    train_quantile[f'{col}_quantile'], bins = pd.qcut(train_data[col], q=n_bins, labels=quantile_labels, retbins=True)
    test_quantile[f'{col}_quantile'] = pd.cut(test_data[col], bins=bins, labels=quantile_labels, include_lowest=True)

# Visualize quantile distributions with bin ranges
plt.figure(figsize=(15, 12))
for i, col in enumerate(numeric_columns[:6], 1):
    plt.subplot(2, 3, i)
    
    # Get quantile bins and their edges
    train_quantile[f'{col}_quantile'], bins = pd.qcut(train_data[col], q=n_bins, labels=quantile_labels, retbins=True)
    
    # Create a countplot
    sns.countplot(data=train_quantile, x=f'{col}_quantile', palette=palette)
    
    # Customize x-axis labels to show bin ranges
    bin_ranges = [f'[{bins[j]:.2f}, {bins[j+1]:.2f}]' for j in range(len(bins)-1)]
    plt.xticks(ticks=range(len(quantile_labels)), labels=bin_ranges, rotation=45, ha='right')
    
    plt.title(f'{col} Quantile Distribution', fontweight='bold')
    plt.xlabel(f'{col} Bins (Value Ranges)')
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Feature Engineering - Equal-Width Binning
# --------------------------------------------------------------

# Equal-width binning
n_bins_equal = 10
equal_labels = [f'Bin{i+1}' for i in range(n_bins_equal)]

train_equal = train_data.copy()
test_equal = test_data.copy()

for col in numeric_columns:
    train_equal[f'{col}_equal'], bins = pd.cut(train_data[col], bins=n_bins_equal, labels=equal_labels, retbins=True)
    test_equal[f'{col}_equal'] = pd.cut(test_data[col], bins=bins, labels=equal_labels, include_lowest=True)

# Visualize equal-width bin distributions with bin ranges
plt.figure(figsize=(15, 12))
for i, col in enumerate(numeric_columns[:6], 1):
    plt.subplot(2, 3, i)
    
    # Get equal-width bins and their edges
    train_equal[f'{col}_equal'], bins = pd.cut(train_data[col], bins=n_bins_equal, labels=equal_labels, retbins=True)
    
    # Create a countplot
    sns.countplot(data=train_equal, x=f'{col}_equal', palette=palette)
    
    # Customize x-axis labels to show bin ranges
    bin_ranges = [f'[{bins[j]:.2f}, {bins[j+1]:.2f}]' for j in range(len(bins)-1)]
    plt.xticks(ticks=range(len(equal_labels)), labels=bin_ranges, rotation=45, ha='right')
    
    plt.title(f'{col} Equal-Width Bins', fontweight='bold')
    plt.xlabel(f'{col} Bins (Value Ranges)')
    plt.ylabel('Count')
plt.tight_layout()
plt.show()


import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb

# --------------------------------------------------------------
# Polar Plots for Median and Standard Deviation of Numeric Features
# --------------------------------------------------------------

# Calculate median and standard deviation for each numeric feature
medians = train_data[numeric_columns].median()
stds = train_data[numeric_columns].std()

# Prepare data for polar plots
angles = np.linspace(0, 2 * np.pi, len(numeric_columns), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # Close the loop
median_values = medians.values
median_values = np.concatenate((median_values, [median_values[0]]))  # Close the loop
std_values = stds.values
std_values = np.concatenate((std_values, [std_values[0]]))  # Close the loop

# Generate colors for median plot
colors_hue_median = [hsv_to_rgb(v / 100, 1, 1) for v in median_values / max(median_values) * 100]

# Create figure with two subplots side by side
fig, axes = plt.subplots(ncols=2, nrows=1, subplot_kw={'projection': 'polar'}, figsize=(12, 5))

# Median plot
ax = axes[0]
ax.scatter(angles, median_values, c=colors_hue_median, edgecolors="black", s=200, zorder=5)
ax.plot(angles, median_values, color='#979dac')
ax.fill(angles, median_values, alpha=0.3, color='#979dac')
ax.set_title('Median of Numeric Features', fontsize=16, color='black')
ax.set_thetagrids(angles[:-1] * 180 / np.pi, numeric_columns, fontsize=10, zorder=100)
for x, y in zip(angles, median_values):
    label = f"{y:.1f}"
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), 
                ha='center', color=palette[5])
ax.set_yticklabels([])
ax.set_theta_zero_location('N')
ax.set_ylim([0, max(median_values) + max(median_values) / 5])

# Standard deviation plot
ax = axes[1]
ax.plot(angles, std_values, 'o-', color=palette[4], markersize=8)
ax.fill(angles, std_values, alpha=0.3, color=palette[5])
ax.set_title('Standard Deviation of Numeric Features', fontsize=16)
ax.set_thetagrids(angles[:-1] * 180 / np.pi, numeric_columns, fontsize=10)
for x, y in zip(angles, std_values):
    label = f"{y:.1f}"
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), 
                ha='center', color='#979dac')
ax.set_yticklabels([])
ax.set_theta_zero_location('N')
ax.set_ylim([0, max(std_values) + max(std_values) / 5])

plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Polar Plots for Min and Max of Numeric Features
# --------------------------------------------------------------

# Calculate min and max for each numeric feature
stats = {
    'min': train_data[numeric_columns].min(),
    'max': train_data[numeric_columns].max()
}

# Create figure with two subplots side by side
fig, axes = plt.subplots(ncols=2, nrows=1, subplot_kw={'projection': 'polar'}, figsize=(12, 5))

for i, ax in enumerate(axes.flatten()):
    column = stats[list(stats.keys())[i]]
    values = column.values
    values = np.concatenate((values, [values[0]]))  # Close the loop
    colors_hue = [hsv_to_rgb(v / 100, 1, 1) for v in values / max(values) * 100]

    ax.scatter(angles, values, c=colors_hue, edgecolors="black", s=200, zorder=5)
    ax.plot(angles, values, color='#979dac')
    ax.fill(angles, values, alpha=0.3, color='#979dac')
    ax.set_title(f'{list(stats.keys())[i].capitalize()} of Numeric Features', fontsize=16, color='black')
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, numeric_columns, fontsize=10, zorder=100)
    for x, y in zip(angles, values):
        label = f"{y:.1f}"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), 
                    ha='center', color=palette[5])
    ax.set_yticklabels([])
    ax.set_theta_zero_location('N')
    ax.set_ylim([0, max(values) + max(values) / 5])

plt.tight_layout()
plt.show()


# Ensure 'id' is present in both datasets
train_quantile['id'] = train_id
train_equal['id'] = train_id
test_quantile['id'] = test_id
test_equal['id'] = test_id

# Identify binned columns in train_equal/test_equal
equal_binned_columns = [col for col in train_equal.columns if '_equal' in col]

# Select only 'id' and binned columns from train_equal/test_equal
train_equal_subset = train_equal[['id'] + equal_binned_columns]
test_equal_subset = test_equal[['id'] + equal_binned_columns]

# Merge using 'id' as the key
train_combined = pd.merge(train_quantile, train_equal_subset, how='left', on='id')
test_combined = pd.merge(test_quantile, test_equal_subset, how='left', on='id')

# Check row counts to ensure no inflation
print(f"Original train rows: {len(train_data)}")
print(f"Train combined rows: {len(train_combined)}")
print(f"Original test rows: {len(test_data)}")
print(f"Test combined rows: {len(test_combined)}")

# Check the first few rows
print("\nTrain combined head:")
print(train_combined.head())
print("\nTest combined head:")
print(test_combined.head())


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from itertools import combinations

# Clean column names by removing extra spaces
train_combined.columns = train_combined.columns.str.strip()
test_combined.columns = test_combined.columns.str.strip()

# Check for column mismatches between train and test sets
missing_in_test = [col for col in train_combined.columns if col not in test_combined.columns and col not in ['Calories']]
missing_in_train = [col for col in test_combined.columns if col not in train_combined.columns and col != 'id']

if missing_in_test or missing_in_train:
    print("Columns missing in test:", missing_in_test)
    print("Columns missing in train:", missing_in_train)
    # Fill missing columns with zeros
    for col in missing_in_test:
        test_combined[col] = 0
    for col in missing_in_train:
        if col != 'id':
            train_combined[col] = 0

# Identify categorical columns including the Sex column
categorical_columns = ['Sex']  # Add the Sex column explicitly

# Columns to apply Label Encoding
label_encode_columns = [
    'Age_quantile', 'Age_equal', 
    'Height_quantile', 'Height_equal', 
    'Weight_quantile', 'Weight_equal', 
    'Duration_quantile', 'Duration_equal', 
    'Heart_Rate_quantile', 'Heart_Rate_equal', 
    'Body_Temp_quantile', 'Body_Temp_equal'
]

# ------------------- CATEGORICAL ENCODING -------------------
# Handle Sex column separately (binary encoding)
if 'Sex' in train_combined.columns:
    le_sex = LabelEncoder()
    train_combined_encoded = train_combined.copy()
    train_combined_encoded['Sex'] = le_sex.fit_transform(train_combined['Sex'])
    
    test_combined_encoded = test_combined.copy()
    if 'Sex' in test_combined.columns:
        test_combined_encoded['Sex'] = le_sex.transform(test_combined['Sex'])
else:
    train_combined_encoded = train_combined.copy()
    test_combined_encoded = test_combined.copy()

# ------------------- LABEL ENCODING -------------------
label_encoders = {}
# Apply Label Encoding to train set
for col in label_encode_columns:
    if col in train_combined_encoded.columns:
        le = LabelEncoder()
        train_combined_encoded[col] = le.fit_transform(train_combined_encoded[col])
        label_encoders[col] = le

# Apply Label Encoding to test set using the same encoders
for col in label_encode_columns:
    if col in test_combined_encoded.columns and col in label_encoders:
        le = label_encoders[col]
        test_combined_encoded[col] = le.transform(test_combined_encoded[col])

# ------------------- CHECK FOR REMAINING CATEGORICAL COLUMNS -------------------
# Find any remaining object columns that need encoding
object_columns_train = train_combined_encoded.select_dtypes(include=['object']).columns.tolist()
if object_columns_train:
    print(f"Warning: These columns still need encoding: {object_columns_train}")
    # Encode any remaining object columns
    for col in object_columns_train:
        le = LabelEncoder()
        train_combined_encoded[col] = le.fit_transform(train_combined_encoded[col])
        if col in test_combined_encoded.columns:
            test_combined_encoded[col] = le.transform(test_combined_encoded[col])

# Columns to exclude from scaling (ID, Calories, and label-encoded columns)
exclude_from_scaling = ['id', 'Calories'] + label_encode_columns

# ------------------- MINMAX SCALING -------------------
# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Columns to scale in train set - make sure we're only scaling numeric columns
columns_to_scale_train = [col for col in train_combined_encoded.columns 
                          if col not in exclude_from_scaling 
                          and train_combined_encoded[col].dtype in ['int64', 'float64']]

# Columns to scale in test set
columns_to_scale_test = [col for col in test_combined_encoded.columns 
                         if col not in ['id'] + label_encode_columns
                         and test_combined_encoded[col].dtype in ['int64', 'float64']]

# Scale train set
train_combined_scaled = train_combined_encoded.copy()
train_combined_scaled[columns_to_scale_train] = scaler.fit_transform(train_combined_encoded[columns_to_scale_train])

# Scale test set
test_combined_scaled = test_combined_encoded.copy()
test_combined_scaled[columns_to_scale_test] = scaler.transform(test_combined_encoded[columns_to_scale_test])

# ------------------- DATA PREVIEW -------------------
print("\nTrain Combined (Scaled & Encoded) Dataset (First 5 rows):")
print(train_combined_scaled.head())
print("\nTest Combined (Scaled & Encoded) Dataset (First 5 rows):")
print(test_combined_scaled.head())


!pip install xgboost==3.0.0


# --------------------------------------------------------------------------------------------------------
# XGBoost Model Training with Cross-Validation, Evaluation, Prediction, and Feature Importance Analysis
# --------------------------------------------------------------------------------------------------------

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Ignore warnings to keep output clean
warnings.filterwarnings("ignore")

# Define RMSLE metric for evaluation
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


X_train = train_combined_scaled.drop(columns=['Calories'])
y_train = train_combined_scaled['Calories']

X_test = test_combined_scaled[X_train.columns]
test_ids = test_id


# Set up DMatrix with feature names for XGBoost
feature_names = X_train.columns.tolist()
dtrain_full = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, feature_names=feature_names)

# XGBoost hyperparameters
params = {
    'objective': 'reg:squarederror',
    'eta': 0.025635582934494156,
    'max_depth': 9,
    'subsample': 0.9316010302755162,
    'colsample_bytree': 0.6458382344923967,
    'min_child_weight': 3.0537700569962194,
    'gamma': 2.9485894835458426,
    'seed': 42,
    'eval_metric': 'rmse',
    'device': 'cuda',
    'tree_method': 'hist'
}

# Set up 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
rmse_scores = []
rmsle_scores = []
best_iterations = []

# Run cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    # Configure early stopping
    early_stopping = xgb.callback.EarlyStopping(
        rounds=100,
        metric_name='rmse',
        data_name='validation_0',
        save_best=True
    )

    # Train model for this fold
    evals = [(dtrain, 'train'), (dval, 'validation_0')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=evals,
        callbacks=[early_stopping],
        verbose_eval=False
    )

    # Evaluate model
    y_pred = model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmsle_score = rmsle(y_val, y_pred)
    
    rmse_scores.append(rmse)
    rmsle_scores.append(rmsle_score)
    best_iterations.append(model.best_iteration)

    print(f"Fold {fold} - RMSE: {rmse:.4f}, RMSLE: {rmsle_score:.4f}, Best Iteration: {model.best_iteration}")

# Summarize cross-validation results
print("\nCross-Validation Summary:")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} (Â±{np.std(rmse_scores):.4f})")
print(f"Average RMSLE: {np.mean(rmsle_scores):.4f} (Â±{np.std(rmsle_scores):.4f})")
print(f"Average Best Iteration: {np.mean(best_iterations):.1f}")

# Train final model using average best iteration
avg_best_iteration = int(np.mean(best_iterations))
print(f"\nTraining final model with {avg_best_iteration} iterations...")

final_model = xgb.train(
    params,
    dtrain_full,
    num_boost_round=avg_best_iteration,
    verbose_eval=False
)

# Generate test predictions
test_predictions = final_model.predict(dtest)

# Convert 'id' column to integers
test_ids = test_ids.astype(int)

# Create submission file with integer 'id' column
submission = pd.DataFrame({
    'id': test_ids,
    'Calories': test_predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)
print(f"submission.csv")


# Analyze feature importance
print("\nFeature Importance (sorted by gain):")
feature_importance = final_model.get_score(importance_type='gain')
feature_importance_df = pd.DataFrame({
    'Feature': list(feature_importance.keys()),
    'Importance': list(feature_importance.values())
}).sort_values(by='Importance', ascending=False)

# Display feature importance
print(feature_importance_df)

# Visualize top 20 features
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20], color='skyblue')
plt.xlabel('Importance (Gain)')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importance (XGBoost)', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/feature_importance.png')

