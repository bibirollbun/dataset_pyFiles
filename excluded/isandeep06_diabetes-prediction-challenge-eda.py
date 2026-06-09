import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import warnings
from scipy.stats import skew
warnings.filterwarnings("ignore")


# Load the datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")



df_train.head()


df_test.head()


df_train.info()


df_train.describe()


df_train.isna().sum()


# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# Initialize the figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('Diabetes Prediction: Train vs Test & Key Drivers', fontsize=24, weight='bold', y=0.96)

# --- Plot 1: Target Distribution (Train Only) ---
# Goal: Check class imbalance
sns.countplot(data=df_train, x='diagnosed_diabetes', palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Target Distribution (Train Set)', fontsize=16)
axes[0, 0].set_xlabel('Diagnosed Diabetes (0=No, 1=Yes)', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)
for container in axes[0, 0].containers:
    axes[0, 0].bar_label(container, fmt='%.0f', fontsize=12)

# --- Plot 2: Numerical Feature Distribution (BMI) - Train vs Test ---
# Goal: Ensure BMI distribution is consistent between datasets
sns.kdeplot(df_train['bmi'], fill=True, label='Train', color='#3498db', ax=axes[0, 1])
sns.kdeplot(df_test['bmi'], fill=True, label='Test', color='#e74c3c', ax=axes[0, 1])
axes[0, 1].set_title('Distribution Comparison: BMI', fontsize=16)
axes[0, 1].set_xlabel('BMI', fontsize=12)
axes[0, 1].legend()

# --- Plot 3: Categorical Feature (Gender) - Train vs Test ---
# Goal: Check demographic consistency
# Prepare data for plotting
train_gender = df_train['gender'].value_counts(normalize=True).reset_index()
train_gender['Set'] = 'Train'
test_gender = df_test['gender'].value_counts(normalize=True).reset_index()
test_gender['Set'] = 'Test'
gender_comp = pd.concat([train_gender, test_gender])

sns.barplot(data=gender_comp, x='gender', y='proportion', hue='Set', palette=['#3498db', '#e74c3c'], ax=axes[1, 0])
axes[1, 0].set_title('Gender Proportion: Train vs Test', fontsize=16)
axes[1, 0].set_ylabel('Proportion', fontsize=12)

# --- Plot 4: Correlation Heatmap (Top Correlations to Target) ---
# Goal: See what physically drives the diagnosis
# Select numerical columns only
numeric_df = df_train.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()[['diagnosed_diabetes']].sort_values(by='diagnosed_diabetes', ascending=False)
# Remove the target itself from the top of the list for cleaner view
corr = corr.drop('diagnosed_diabetes')

sns.heatmap(corr.head(10), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1, 1], linewidths=1)
axes[1, 1].set_title('Top 10 Features Correlated with Diabetes', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# Find categorical columns
cat_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Categorical Columns: {cat_cols}")
# Visual the distribution of each category 
plt.figure(figsize=(15,5*len(cat_cols)))

for i, col in enumerate(cat_cols):
    plt.subplot(len(cat_cols),1,i+1)
    sns.countplot(data=df_train,x=col, palette="viridis")
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



# Set the visual style
sns.set_theme(style="whitegrid")

# Create a grid of plots
num_cats = len(cat_cols)
fig, axes = plt.subplots(nrows=(num_cats + 1) // 2, ncols=2, figsize=(16, 5 * ((num_cats + 1) // 2)))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(data=df_train, x=col, hue='diagnosed_diabetes', ax=axes[i], palette='viridis')
    axes[i].set_title(f'Distribution of {col} by Diabetes Status')
    axes[i].tick_params(axis='x', rotation=45)

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


num_cols = df_train.select_dtypes(include=['float64', 'int64']).drop(columns=['id', 'diagnosed_diabetes']).columns

df_train[num_cols].hist(figsize=(15, 12), bins=30, color='skyblue', edgecolor='black')
plt.suptitle('Distribution of Numerical Features', fontsize=16)
plt.tight_layout()
plt.show()


# Select numerical columns (exclude id and target)
num_cols = (
    df_train
    .select_dtypes(include=['int64', 'float64'])
    .drop(columns=['id', 'diagnosed_diabetes','family_history_diabetes', 'hypertension_history',
       'cardiovascular_history'])
    .columns
)

# Grid size
n_cols = 4
n_rows = math.ceil(len(num_cols) / n_cols)

plt.figure(figsize=(20, 4 * n_rows))

for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(y=df_train[col])
    plt.title(col)
    plt.tight_layout()

plt.suptitle('Boxplots of Numerical Features', fontsize=18, y=1.02)
plt.show()



print(df_train.columns)


# Identifying Outliers Mathematically (IQR Method)
df = df_train
clinical_cols = [
    'age',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides',
 ]
def detect_outliers_iqr(df_train, columns):
    outlier_indices = []
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = df_train[(df_train[col] < lower_bound) | (df_train[col] > upper_bound)].index
        print(f"{col}: {len(outliers)} outliers found")
        outlier_indices.extend(outliers)
    
    return list(set(outlier_indices))

outliers_to_investigate = detect_outliers_iqr(df_train, clinical_cols)



# Define clinical and lifestyle columns
clinical_cols = ['age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 
                 'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 
                 'ldl_cholesterol', 'triglycerides']

# Create subplots
plt.figure(figsize=(16, 12))
for i, col in enumerate(clinical_cols):
    plt.subplot(3, 4, i + 1)
    sns.boxplot(data=df_train, y=col, color='salmon')
    plt.title(f'Outliers in {col}')

plt.tight_layout()
plt.show()


# Check separation for Metabolic Indicators
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.kdeplot(data=df_train, x='waist_to_hip_ratio', hue='diagnosed_diabetes', fill=True, ax=axes[0], palette='viridis')
axes[0].set_title('Waist-to-Hip Ratio Distribution')

sns.kdeplot(data=df_train, x='bmi', hue='diagnosed_diabetes', fill=True, ax=axes[1], palette='magma')
axes[1].set_title('BMI Distribution')

plt.show()


from scipy.stats import skew

# Define the lifestyle columns from your dataset
lifestyle_cols = [
    'alcohol_consumption_per_week', 
    'physical_activity_minutes_per_week', 
    'diet_score',
    'sleep_hours_per_day', 
    'screen_time_hours_per_day'
]

plt.figure(figsize=(16, 10))

for i, col in enumerate(lifestyle_cols):
    plt.subplot(2, 3, i + 1)
    
    # Calculate skewness value
    sk = skew(df[col].dropna())
    
    # Plot histogram with KDE
    sns.histplot(df[col], kde=True, color='teal', bins=30)
    
    plt.title(f'{col}\nSkewness: {sk:.2f}', fontsize=12)
    plt.xlabel('')

plt.tight_layout()
plt.show()

# Print advice based on findings
print("ğŸ’¡ Skewness Interpretation:")
print("- If Skewness > 1: Highly right-skewed. Consider Log or Square Root transformation.")
print("- If Skewness < -1: Highly left-skewed.")
print("- If Skewness is between -0.5 and 0.5: Fairly symmetrical.")


# Select only the lifestyle and clinical numerical columns
clinical_cols = ['age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 
                 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']

plt.figure(figsize=(10, 8))
sns.heatmap(df_train[clinical_cols].corr(), annot=True, cmap='RdBu', center=0)
plt.title('Clinical Feature Correlations')
plt.show()


df['target_rolling'] = df['diagnosed_diabetes'].rolling(window=1000).mean()
plt.plot(df['id'],df['target_rolling'])
plt.title('Checking for Target Drift over ID')
plt.show()


# --- 1. Train vs Test Distribution Check ---
cols_to_check = ['age', 'bmi', 'systolic_bp', 'triglycerides']
plt.figure(figsize=(15, 10))
for i, col in enumerate(cols_to_check):
    plt.subplot(2, 2, i+1)
    sns.kdeplot(df[col], label='Train', fill=True, alpha=0.3)
    sns.kdeplot(df_test[col], label='Test', fill=True, alpha=0.3)
    plt.title(f'Distribution Check: {col}')
    plt.legend()
plt.tight_layout()
plt.show()

# --- 2. ID Drift Analysis ---
# We check if the target 'diagnosed_diabetes' is stable across the IDs
plt.figure(figsize=(12, 5))
df['target_rolling'] = df['diagnosed_diabetes'].rolling(window=1000).mean()
plt.plot(df_train['id'], df['target_rolling'], color='red')
plt.title('Target Probability Drift (Rolling Mean over ID)')
plt.xlabel('ID')
plt.ylabel('Diabetes Probability')
plt.axhline(df['diagnosed_diabetes'].mean(), color='black', linestyle='--')
plt.show()

# --- 3. Multicollinearity (Heatmap) ---
# Checking if features like Systolic and Diastolic BP are too similar
plt.figure(figsize=(12, 8))
corr_matrix = df.select_dtypes(include=['float64', 'int64']).drop(columns=['id', 'diagnosed_diabetes', 'target_rolling']).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=np.triu(np.ones_like(corr_matrix)))
plt.title('Clinical Feature Correlation (Red = High Redundancy)')
plt.show()


# Check for rows that are identical except for the ID and Target
features_only = df.drop(columns=['id', 'diagnosed_diabetes'])
duplicates =df[features_only.duplicated(keep=False)]

print(f"Number of rows with identical features: {len(duplicates)}")


from IPython.display import display
summary_rows = len(df_train)
target_counts = (
    df_train['diagnosed_diabetes']
    .value_counts()
    .rename_axis('diagnosed_diabetes')
    .to_frame('count')
)
target_counts['share'] = target_counts['count'] / summary_rows

missing = df_train.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

features_only = df_train.drop(columns=['id', 'diagnosed_diabetes'])
duplicate_feature_rows = int(features_only.duplicated(keep=False).sum())

numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns.drop(['id', 'diagnosed_diabetes', 'target_rolling'])
target_corr = (
    df_train[numeric_cols]
    .corrwith(df_train['diagnosed_diabetes'])
    .dropna()
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

mean_shift = (
    df_train.groupby('diagnosed_diabetes')[numeric_cols]
    .mean()
    .T
)
mean_shift['delta'] = mean_shift[1] - mean_shift[0]
mean_shift_sorted = mean_shift.sort_values('delta', key=lambda s: s.abs(), ascending=False)

train_test_compare = pd.DataFrame({
    'train_mean': df_train[numeric_cols].mean(),
    'test_mean': df_test[numeric_cols].mean(),
})
train_test_compare['pct_gap'] = (
    (train_test_compare['test_mean'] - train_test_compare['train_mean'])
    / train_test_compare['train_mean']
    * 100
)
train_test_compare = train_test_compare.loc[
    ['age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'triglycerides']
]

print(f"Total rows: {summary_rows:,}")
display(target_counts)
print("\nColumns with missing values:" if not missing.empty else "\nNo missing values detected.")
if not missing.empty:
    display(missing.to_frame('missing_count'))
print(f"\nDuplicate feature rows (ignoring id/target): {duplicate_feature_rows}")
print("\nTop correlations with diagnosed_diabetes")
display(target_corr.head(8).to_frame('corr'))
print("\nLargest mean shifts between classes")
display(mean_shift_sorted['delta'].head(8).to_frame('delta'))
print("\nTrain vs Test mean comparison (selected features)")
display(train_test_compare)

