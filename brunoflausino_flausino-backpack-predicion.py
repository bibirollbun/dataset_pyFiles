# Libraries
# ===============================
# ğŸ“Š Data Manipulation Libraries
# ===============================
import pandas as pd        # DataFrames and Series handling
import numpy as np         # Numerical computations and arrays

# ===============================
# ğŸ’¾ File Handling
# ===============================
import os                  # File and path operations
import joblib              # Save/load Python objects (e.g., trained models)

# ===============================
# ğŸ“ˆ Data Visualization Libraries
# ===============================
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns           # Statistical plots built on matplotlib

# ===============================
# âš ï¸� Warning Handling
# ===============================
import warnings
warnings.filterwarnings('ignore')  # Suppress warning messages

# ===============================
# âš™ï¸� Data Preprocessing
# ===============================
from sklearn.preprocessing import StandardScaler  # Feature scaling

# ===============================
# ğŸ¤– Machine Learning Model
# ===============================
from lightgbm import LGBMRegressor  # Gradient boosting for regression

# ===============================
# ğŸ“Š Model Evaluation & Training
# ===============================
from sklearn.model_selection import KFold, RandomizedSearchCV  # Cross-validation & hyperparameter tuning
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Regression metrics

# ===============================
# ğŸ§µ Parallel Computing
# ===============================
from joblib import Parallel, delayed  # Parallel task execution



# Loading dataset
base_path = "/kaggle/input/playground-series-s5e2/"
train = pd.read_csv(base_path + "train.csv")
train_ex = pd.read_csv(base_path + "training_extra.csv")
test = pd.read_csv(base_path + "test.csv")

# Output directory (writable in Kaggle)
output_path = "/kaggle/working/"



# DATA SUMMARY
print("\n" + "="*50)
print("DATA SUMMARY".center(50))
print("="*50)
def dataset_summary(df, name):
    print(f"{name} Dataset Summary (First Rows, Shape, Data Types)")
    display(df.head(), df.shape, df.dtypes)

dataset_summary(train, "Train")
dataset_summary(test, "Test")
dataset_summary(train_ex, "Train_ex")

# TARGET DISTRIBUTION - Modified to use broken axis for Price comparison
print("\n" + "="*50)
print("TARGET DISTRIBUTION".center(50))
print("="*50)
print("Price Distributions in Train and Train_ex Datasets")

# Create a broken axis plot for Price distribution
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 3])

# Create histograms on both axes
# In the top axis (for higher values - train_ex)
sns.histplot(train_ex['Price'], bins=100, kde=True, color='green', ax=ax1, label='Train_ex')
sns.histplot(train['Price'], bins=100, kde=True, color='blue', ax=ax1, label='Train')

# In the bottom axis (for lower values - train)
sns.histplot(train_ex['Price'], bins=100, kde=True, color='green', ax=ax2, label='Train_ex')
sns.histplot(train['Price'], bins=100, kde=True, color='blue', ax=ax2, label='Train')

# Set appropriate y-axis limits
# Get max counts
train_counts, _ = np.histogram(train['Price'], bins=100)
train_ex_counts, _ = np.histogram(train_ex['Price'], bins=100)
small_max = train_counts.max() * 1.1  # Add 10% padding
large_max = train_ex_counts.max() * 1.1  # Add 10% padding
break_point = small_max * 1.5

ax1.set_ylim(break_point, large_max)  # Top portion for train_ex (higher values)
ax2.set_ylim(0, small_max)  # Bottom portion for train (lower values)

# Remove x-axis ticks from the top subplot
ax1.set_xticklabels([])

# Add proper labels
fig.suptitle('Price Distribution Comparison', fontsize=16)
ax2.set_xlabel('Price', fontsize=12)
fig.text(0.06, 0.5, 'Count', va='center', rotation='vertical', fontsize=12)

# Hide the bottom spine of the top subplot and the top spine of the bottom subplot
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Add diagonal lines to indicate the break
d = .015  # Size of diagonal lines
kwargs = dict(transform=ax1.transAxes, color='gray', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)# Load the CSV files in the specified order

base_path = "/kaggle/input/playground-series-s5e2/"
train = pd.read_csv(base_path + "train.csv")
train_ex = pd.read_csv(base_path + "training_extra.csv")
test = pd.read_csv(base_path + "test.csv")
output_path = "/kaggle/working/"

# Add a legend to the bottom subplot
ax2.legend(['Train_ex', 'Train'], loc='lower right')

# Add grid
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.05)  # Adjust for title and reduce space between subplots
plt.show()

# Function to create broken axis grouped bar chart
def create_broken_bar_chart(feature, datasets, title, is_categorical=False):
    # Get unique values for the feature
    all_values = set()
    for df in datasets.values():
        all_values.update(df[feature].dropna().unique())
    
    # For categorical features, convert to strings and sort
    if is_categorical:
        all_values = sorted([str(x) for x in all_values])
    else:
        all_values = sorted(all_values)
    
    # Count occurrences for each value in each dataset
    data = {}
    for name, df in datasets.items():
        if is_categorical:
            # Convert to string for categorical to ensure matching
            value_counts = df[feature].astype(str).value_counts()
            counts = pd.Series(0, index=all_values)
            for val, count in value_counts.items():
                if val in all_values:
                    counts[val] = count
        else:
            counts = df[feature].value_counts().reindex(all_values, fill_value=0)
        data[name] = counts
    
    # Create DataFrame from the counts
    df_plot = pd.DataFrame(data)
    
    # Determine break point based on the maximum of train and test vs train_ex
    small_max = max(df_plot['Train'].max(), df_plot['Test'].max())
    large_max = df_plot['Train_ex'].max()
    break_point = small_max * 1.5
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[1, 3])
    
    # Set the width of the bars
    bar_width = 0.25
    
    # Set indices for the bars
    indices = np.arange(len(all_values))
    
    # Create bars for each dataset in both subplots - but don't add labels in ax1
    ax1.bar(indices - bar_width, df_plot['Train'], bar_width, color='blue')
    ax1.bar(indices, df_plot['Test'], bar_width, color='orange')
    ax1.bar(indices + bar_width, df_plot['Train_ex'], bar_width, color='green')
    
    # Add labels only in ax2
    bars1 = ax2.bar(indices - bar_width, df_plot['Train'], bar_width, label='Train', color='blue')
    bars2 = ax2.bar(indices, df_plot['Test'], bar_width, label='Test', color='orange')
    bars3 = ax2.bar(indices + bar_width, df_plot['Train_ex'], bar_width, label='Train_ex', color='green')
    
    # Set the y-axis limits to create the broken axis effect
    ax1.set_ylim(break_point, large_max * 1.1)
    ax2.set_ylim(0, small_max * 1.1)
    
    # Remove the x-axis labels from the top subplot
    ax1.set_xticklabels([])
    
    # Set the x-axis labels on the bottom subplot
    ax2.set_xticks(indices)
    ax2.set_xticklabels(all_values, rotation=45 if is_categorical else 0)
    
    # Hide spines to create break effect
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Add diagonal lines to indicate the break
    d = .015  # Size of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color='gray', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    # Add title and labels
    fig.suptitle(title, fontsize=16)
    fig.text(0.06, 0.5, 'Count', va='center', rotation='vertical', fontsize=12)
    ax2.set_xlabel(feature, fontsize=12)
    
    # Add legend to the bottom subplot (ax2) in the bottom right corner
    ax2.legend([bars1, bars2, bars3], ['Train', 'Test', 'Train_ex'], 
              loc='lower right', framealpha=0.9)
    
    # Add grid for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the title
    
    return fig

# Create datasets dictionary
datasets = {
    'Train': train,
    'Test': test,
    'Train_ex': train_ex
}

# NUMERIC DATA DISTRIBUTION
print("\n" + "="*50)
print("NUMERIC DATA DISTRIBUTION".center(50))
print("="*50)

# Create broken axis grouped bar charts for Compartments
print("\nDistribution of Compartments (Count) with Broken Axis")
fig1 = create_broken_bar_chart('Compartments', datasets, 'Distribution of Compartments (Count)')
plt.show()

# Create broken axis grouped bar charts for Weight Capacity
# Instead of using pd.cut which creates categorical data that's harder to work with,
# let's manually bin the data into a new numerical column
for df in datasets.values():
    # Create bin ranges
    bins = list(range(0, 32, 2))
    bin_labels = [f"{i}" for i in bins[:-1]]
    
    # Categorize the data
    df['Weight Bin'] = pd.cut(df['Weight Capacity (kg)'], 
                              bins=bins,
                              labels=bin_labels,
                              include_lowest=True)

print("\nDistribution of Weight Capacity (kg) (Count) with Broken Axis")
fig2 = create_broken_bar_chart('Weight Bin', datasets, 
                             'Distribution of Weight Capacity (kg) (Count)', 
                             is_categorical=True)
plt.show()

# Create a single histogram for ID counts in each dataset
print("\nID Count Distribution by Dataset")
# Get the count of IDs in each dataset
id_counts = {
    'Train': len(train),
    'Test': len(test),
    'Train_ex': len(train_ex)
}

# Create a bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(['Train', 'Test', 'Train_ex'], 
        [id_counts['Train'], id_counts['Test'], id_counts['Train_ex']], 
        color=['blue', 'green', 'red'])

# Add the count values above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5000,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=10)

plt.title('ID Count by Dataset')
plt.ylabel('Count')
plt.xlabel('Dataset')
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Remove the legend from the ID Count chart
plt.tight_layout()
plt.show()

# Get a list of numeric columns excluding the ones we've already visualized
num_cols = test.select_dtypes(include=['number']).columns
num_cols = [col for col in num_cols if col not in ['Compartments', 'Weight Capacity (kg)', 'id', 'Price']]

# Only create histograms for the remaining numeric columns
if len(num_cols) > 0:
    print("\nOther Numeric Features:")
    plt.figure(figsize=(12, len(num_cols) * 3))
    
    for i, col in enumerate(num_cols):
        plt.subplot(len(num_cols), 3, i*3 + 1)
        sns.histplot(train[col], bins=19, color='blue')
        plt.title(f"Train [{col}] Distribution")
        plt.xlabel(col)
        
        plt.subplot(len(num_cols), 3, i*3 + 2)
        sns.histplot(test[col], bins=19, color='green')
        plt.title(f"Test [{col}] Distribution")
        plt.xlabel(col)
        
        plt.subplot(len(num_cols), 3, i*3 + 3)
        sns.histplot(train_ex[col], bins=19, color='red')
        plt.title(f"Train_ex [{col}] Distribution")
        plt.xlabel(col)
    
    plt.tight_layout()
    plt.show()

# CATEGORICAL DATA DISTRIBUTION
print("\n" + "="*50)
print("CATEGORICAL DATA DISTRIBUTION".center(50))
print("="*50)
cat_cols = train.select_dtypes(include=['object']).columns

for col in cat_cols:
    # Prepare data
    train_counts = train[col].value_counts(normalize=True).sort_index() * 100
    test_counts = test[col].value_counts(normalize=True).sort_index() * 100
    train_ex_counts = train_ex[col].value_counts(normalize=True).sort_index() * 100
    
    # Get all unique categories
    all_categories = sorted(list(set(
        list(train_counts.index) + list(test_counts.index) + list(train_ex_counts.index)
    )))
    
    # Reindex to include all categories
    train_counts = train_counts.reindex(all_categories, fill_value=0)
    test_counts = test_counts.reindex(all_categories, fill_value=0)
    train_ex_counts = train_ex_counts.reindex(all_categories, fill_value=0)
    
    # Create DataFrame for plotting
    df_plot = pd.DataFrame({
        'Train': train_counts,
        'Test': test_counts,
        'Train_ex': train_ex_counts
    })
    
    # Calculate max value for x-axis limit (with padding)
    max_val = df_plot.max().max() * 1.15  # Add 15% padding
    
    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, max(6, len(all_categories)*0.5)))
    df_plot.plot(kind='barh', ax=ax, figsize=(12, max(6, len(all_categories)*0.5)))
    plt.title(f'Distribution of {col} across datasets')
    plt.xlabel('Percentage (%)')
    plt.ylabel(col)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend(title='Dataset', loc='lower right')  # Place legend in bottom right
    plt.xlim(0, max_val)  # Set x-axis limit with padding
    
    # Add percentage labels (with adjustment to prevent overflow)
    for i, dataset in enumerate(['Train', 'Test', 'Train_ex']):
        for j, v in enumerate(df_plot[dataset]):
            if v > 0:
                # Position labels with a maximum limit
                label_pos = min(v + 0.3, max_val - 2)
                plt.text(label_pos, j - 0.1 + i*0.2, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    plt.show()

# MISSING VALUES ANALYSIS
print("\n" + "="*50)
print("MISSING VALUES ANALYSIS".center(50))
print("="*50)
# Prepare missing values data for all three datasets
missing_train = train.isnull().sum() / len(train) * 100
missing_test = test.isnull().sum() / len(test) * 100
missing_train_ex = train_ex.isnull().sum() / len(train_ex) * 100

# Combine missing values data
missing_df = pd.DataFrame({
    'Train': missing_train,
    'Test': missing_test,
    'Train_ex': missing_train_ex
})

# Filter to show only columns with missing values
missing_df = missing_df.loc[(missing_df['Train'] > 0) | (missing_df['Test'] > 0) | (missing_df['Train_ex'] > 0)]
missing_df = missing_df.sort_values(by='Train', ascending=False)

# Calculate max value for x-axis limit (with padding)
max_val = missing_df.max().max() * 1.15  # Add 15% padding

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(12, max(6, len(missing_df)*0.5)))
missing_df.plot(kind='barh', ax=ax, figsize=(12, max(6, len(missing_df)*0.5)))
plt.title('Missing Values Percentage across datasets')
plt.xlabel('Percentage (%)')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='Dataset', loc='lower right')  # Place legend in bottom right
plt.xlim(0, max_val)  # Set x-axis limit with padding

# Add percentage labels (with adjustment to prevent overflow)
for i, dataset in enumerate(['Train', 'Test', 'Train_ex']):
    for j, v in enumerate(missing_df[dataset]):
        if v > 0:
            # Position labels with a maximum limit
            label_pos = min(v + 0.3, max_val - 2)
            plt.text(label_pos, j - 0.1 + i*0.2, f'{v:.1f}%', va='center')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("EDA completed successfully!".center(50))
print("="*50)


# Step 1: Data Augmentation - Combine Train and Train_ex after aligning Price distributions

# Update paths for Kaggle environment
base_path = "/kaggle/input/playground-series-s5e2/"
output_path = "/kaggle/working/"

# Load raw data
train = pd.read_csv(base_path + "train.csv")
train_ex = pd.read_csv(base_path + "training_extra.csv")
test = pd.read_csv(base_path + "test.csv")

# Cap extreme Price values in Train_ex at the 99th percentile
train_ex_price_99th = train_ex['Price'].quantile(0.99)
train_ex['Price'] = train_ex['Price'].clip(upper=train_ex_price_99th)

# Combine Train and Train_ex
assert not any(train['id'].isin(train_ex['id'])), "Overlap in IDs between Train and Train_ex"
combined_train = pd.concat([train, train_ex], ignore_index=True)

# Impute missing values
numerical_cols = combined_train.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = combined_train.select_dtypes(include=['object']).columns

for col in numerical_cols:
    combined_train[col].fillna(combined_train[col].median(), inplace=True)
    if col != 'Price':
        test[col].fillna(test[col].median(), inplace=True)

for col in categorical_cols:
    combined_train[col].fillna(combined_train[col].mode()[0], inplace=True)
    if col in test.columns:
        test[col].fillna(test[col].mode()[0], inplace=True)

# Step 2: Apply log1p to Price
combined_train['Price'] = np.log1p(combined_train['Price'])

# Step 3: Drop ID column
combined_train = combined_train.drop(columns=['id'])
test = test.drop(columns=['id'])

# Step 4: Encode categorical features
categorical_cols = ['Brand', 'Material', 'Size', 'Style', 'Color']
combined_train = pd.get_dummies(combined_train, columns=categorical_cols, drop_first=True)
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

# Align test to match combined_train
combined_train, test = combined_train.align(test, join='left', axis=1, fill_value=0)

# Encode binary features
binary_cols = ['Laptop Compartment', 'Waterproof']
for col in binary_cols:
    combined_train[col] = combined_train[col].map({'Yes': 1, 'No': 0})
    test[col] = test[col].map({'Yes': 1, 'No': 0})

# Step 6: Scale numeric features
numerical_cols = ['Compartments', 'Weight Capacity (kg)']
scaler = StandardScaler()
combined_train[numerical_cols] = scaler.fit_transform(combined_train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])

# Step 8: Feature Engineering - Create Material_Waterproof interaction
material_cols = [col for col in combined_train.columns if col.startswith('Material_')]
for material_col in material_cols:
    interaction_col = f"{material_col}_Waterproof"
    combined_train[interaction_col] = combined_train[material_col] * combined_train['Waterproof']
    test[interaction_col] = test[material_col] * test['Waterproof']

# Drop target from test if present
if 'Price' in test.columns:
    test = test.drop(columns=['Price'])

# Save processed files
combined_train.to_csv(output_path + "preprocessed_train.csv", index=False)
test.to_csv(output_path + "preprocessed_test.csv", index=False)

print("Pre-processing completed successfully!")
print(f"Pre-processed training data shape: {combined_train.shape}")
print(f"Pre-processed test data shape: {test.shape}")



# Load pre-processed datasets
output_path = "/kaggle/working/"
train_processed = pd.read_csv(output_path + "preprocessed_train.csv")
test_processed = pd.read_csv(output_path + "preprocessed_test.csv")

# 1. Verify shapes
print("Shapes Verification:")
print(f"Train Processed Shape: {train_processed.shape}")  # Expecting: (3994318, N)
print(f"Test Processed Shape: {test_processed.shape}")    # Expecting: (200000, N-1)

# 2. Check for missing values
print("\nMissing Values Check:")
print("Train Missing Values:", train_processed.isnull().sum().sum())  # Expected: 0
print("Test Missing Values:", test_processed.isnull().sum().sum())    # Expected: 0

# 3. Visualize Price distribution (log1p transformed)
plt.figure(figsize=(8, 4))
sns.histplot(train_processed['Price'], bins=50, kde=True, color='blue')
plt.title("Log-Transformed Price Distribution (Train)")
plt.xlabel("log(Price)")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 4. Check one-hot encoding and binary mapping
print("\nEncoding Check:")
print("Sample of one-hot encoded Brand columns:")
print(train_processed.filter(like='Brand_').head(2))
print("Unique values for 'Laptop Compartment':", train_processed['Laptop Compartment'].unique())

# 5. Verify scaling (mean â‰ˆ 0, std â‰ˆ 1)
print("\nScaling Check (StandardScaler):")
numerical_cols = ['Compartments', 'Weight Capacity (kg)']
for col in numerical_cols:
    print(f"{col} - Mean: {train_processed[col].mean():.2f}, Std: {train_processed[col].std():.2f}")

# Plot scaled features
plt.figure(figsize=(10, 4))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(1, 2, i)
    sns.histplot(train_processed[col], bins=50, kde=True, color='green')
    plt.title(f"Scaled {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 6. Verify feature engineering (Material_Waterproof interaction terms)
print("\nFeature Engineering Check:")
interaction_cols = [col for col in train_processed.columns if 'Material_' in col and '_Waterproof' in col]
print("Detected Interaction Features:", interaction_cols)

# Visualize one of the interaction features
if interaction_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=train_processed[interaction_cols[0]], color='orange')
    plt.title(f"Distribution of {interaction_cols[0]}")
    plt.xlabel(interaction_cols[0])
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

print("âœ… Preprocessing validation completed successfully.")



# ============================
# ğŸ“¦ Model Training & Tuning
# ============================

import pandas as pd
import numpy as np
import time
import os
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import joblib

# Define paths in Kaggle environment
input_path = "/kaggle/input/playground-series-s5e2/"
output_path = "/kaggle/working/"

# Load preprocessed datasets
train_processed = pd.read_csv(output_path + "preprocessed_train.csv")
test_processed = pd.read_csv(output_path + "preprocessed_test.csv")

# Fix column names (remove spaces)
train_processed.columns = train_processed.columns.str.replace(' ', '_')
test_processed.columns = test_processed.columns.str.replace(' ', '_')

# Separate features and target
X = train_processed.drop(columns=['Price']).to_numpy()
y = train_processed['Price'].to_numpy()
X_test = test_processed.to_numpy()

# ================================
# ğŸ“Š Evaluation Function (CV Loop)
# ================================
def evaluate_fold(train_idx, val_idx, model, X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start

    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_val_orig = np.expm1(y_val)

    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred))
    mae = mean_absolute_error(y_val_orig, y_pred)

    print(f"[Fold] Time: {duration:.2f}s | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse, mae

def evaluate_model(model, X, y, cv=5, n_jobs=-1):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    folds = [(train_idx, val_idx) for train_idx, val_idx in kf.split(X)]

    start = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_fold)(train_idx, val_idx, model, X, y) for train_idx, val_idx in folds
    )
    duration = time.time() - start

    rmse_scores, mae_scores = zip(*results)
    print(f"\n[CV] Total Time: {duration:.2f}s")
    print(f"[CV] RMSE Mean: {np.mean(rmse_scores):.4f} Â± {np.std(rmse_scores):.4f}")
    print(f"[CV] MAE  Mean: {np.mean(mae_scores):.4f} Â± {np.std(mae_scores):.4f}")

    # Save metrics
    with open(os.path.join(output_path, "cv_results.txt"), "w") as f:
        f.write(f"CV Duration: {duration:.2f} seconds\n")
        f.write(f"RMSE Mean: {np.mean(rmse_scores):.4f} Â± {np.std(rmse_scores):.4f}\n")
        f.write(f"MAE  Mean: {np.mean(mae_scores):.4f} Â± {np.std(mae_scores):.4f}\n")

    return np.mean(rmse_scores), np.mean(mae_scores)

# =================================
# ğŸ§ª Step 1: Initial LightGBM Model
# =================================
print("\n[Step 1] Initial LightGBM Training")
lgbm = LGBMRegressor(device='gpu', gpu_platform_id=0, gpu_device_id=0, random_state=42, n_jobs=1)
evaluate_model(lgbm, X, y)

# ==========================================
# ğŸ”� Step 2: Hyperparameter Randomized Search
# ==========================================
print("\n[Step 2] Hyperparameter Tuning")
param_dist = {
    'num_leaves': [31, 50, 70, 100],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [5, 10, 15, -1],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

subset_idx = np.random.choice(len(X), size=int(0.1 * len(X)), replace=False)
X_subset = X[subset_idx]
y_subset = y[subset_idx]

random_search = RandomizedSearchCV(
    estimator=LGBMRegressor(device='gpu', gpu_platform_id=0, gpu_device_id=0, random_state=42, n_jobs=1),
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_subset, y_subset)
best_params = random_search.best_params_
print("\nBest Parameters:")
print(best_params)

# Save best params
pd.Series(best_params).to_csv(os.path.join(output_path, "best_params.csv"))

# ===================================
# ğŸ�� Step 3: Final Model Evaluation
# ===================================
print("\n[Step 3] Final Model Evaluation")
best_model = random_search.best_estimator_
evaluate_model(best_model, X, y)

# Save final model
joblib.dump(best_model, os.path.join(output_path, "final_model.pkl"))



# ======================================
# ğŸ“Š Model Evaluation and Diagnostics
# ======================================

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Define path for output files
output_path = "/kaggle/working/"

# Load model and training data
model = joblib.load(output_path + "final_model.pkl")
train_data = pd.read_csv(output_path + "preprocessed_train.csv")

# Prepare features and target
X = train_data.drop(columns=['Price']).to_numpy()
y_log = train_data['Price'].to_numpy()
y_true = np.expm1(y_log)  # Convert back to original price scale

# Make predictions
y_pred_log = model.predict(X)
y_pred = np.expm1(y_pred_log)

# Calculate residuals
residuals = y_true - y_pred

# ---------------------------
# ğŸ“ˆ Predicted vs. True Values
# ---------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs. True Values")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path + "plot_pred_vs_true.png")
plt.show()

# ---------------------------
# ğŸ“ˆ Residual Plot
# ---------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual (True - Predicted)")
plt.title("Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path + "plot_residuals.png")
plt.show()

# ---------------------------
# ğŸ“ˆ Distribution of Residuals
# ---------------------------
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.tight_layout()
plt.savefig(output_path + "plot_residual_dist.png")
plt.show()

# ---------------------------
# ğŸ“Š Feature Importance
# ---------------------------
importances = model.feature_importances_
features = train_data.drop(columns=['Price']).columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title("Top 20 Feature Importances (LightGBM)")
plt.tight_layout()
plt.savefig(output_path + "plot_feature_importance.png")
plt.show()

# ---------------------------
# ğŸ“‹ Print Key Metrics
# ---------------------------
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RÂ²:   {r2:.4f}")



# ================================
# ğŸ“¤ Submission File Generation
# ================================

# Define Kaggle paths
base_path = "/kaggle/input/playground-series-s5e2/"
output_path = "/kaggle/working/"
model_path = os.path.join(output_path, "final_model.pkl")
test_data_path = os.path.join(output_path, "preprocessed_test.csv")
raw_test_path = os.path.join(base_path, "test.csv")  # Source of original test IDs

# Load trained model and preprocessed test features
model = joblib.load(model_path)
X_test = pd.read_csv(test_data_path).to_numpy()

# Predict and apply inverse log transformation
print("[Submission] Predicting on test set...")
test_predictions_log = model.predict(X_test)
test_predictions = np.expm1(test_predictions_log)

# Build submission DataFrame
submission = pd.DataFrame({
    'id': pd.read_csv(raw_test_path)['id'],
    'Price': test_predictions
})

# Save submission file
submission_path = os.path.join(output_path, "submission.csv")
submission.to_csv(submission_path, index=False)

# Display sample of predictions
print(f"[Submission] Predictions saved to: {submission_path}")
print(submission.head())
print(submission.describe())


