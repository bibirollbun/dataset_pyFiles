import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("libraries sucessfully loaded")

try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
except FileNotFoundError as e:
    print(f"file not found: {e} ")
except Exception as e:
    print(f"Unexpected error: {e}")
print("all data successfully loaded")


print("=== ğŸ“Š DATA SHAPE AND BASIC INFO ===")

print(f"train data shape:{train_df.shape}")
print(f"test data shape:{test_df.shape}")
print(f"submission data shape:{sample_submission.shape}")

print("train data head")
print(train_df.head(10))

print("test data head")
print(test_df.head(10))

print("train data info")
print(train_df.info())

print("all columns in train data")
for i,col in enumerate(train_df.columns,1):
    print(f"{i}.{col} ")


print("=== ğŸ�¯ TARGET VARIABLE ANALYSIS ===")
print("Target Variable Distribution:")
target_counts = train_df['loan_paid_back'].value_counts()
target_percentages = train_df['loan_paid_back'].value_counts(normalize=True) * 100

print(f"total samples {len(train_df)} ")
print(f"Loan Paid Back (1.0): {target_counts[1.0]:,} samples ({target_percentages[1.0]:.2f}%)")
print(f"Loan NOT Paid Back (0.0): {target_counts[0.0]:,} samples ({target_percentages[0.0]:.2f}%)")

print(f"\n=== âš–ï¸� CLASS IMBALANCE CHECK ===")
imbalance_ratio = target_counts[0.0] / target_counts[1.0]
print(f"imbalance ratio (0:1):: {imbalance_ratio:.3f} ")

if imbalance_ratio > 1.5 or imbalance_ratio < 0.70:
    print("imbalance")
else:
    print("its okey")
print(f"\n=== ğŸ“Š TARGET DISTRIBUTION PLOT ===")
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
bars = plt.bar(['Not Paid (0.0)', 'Paid Back (1.0)'], 
               [target_counts[0.0], target_counts[1.0]],
               color=['#ff9999', '#99ff99'])
plt.title('Loan Paid Back - Count')
plt.ylabel('Number of Samples')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,}', ha='center', va='bottom')

plt.subplot(1, 2, 2)
plt.pie([target_percentages[0.0], target_percentages[1.0]],
        labels=['Not Paid (0.0)', 'Paid Back (1.0)'],
        autopct='%1.1f%%', colors=['#ff9999', '#99ff99'])
plt.title('Loan Paid Back - Percentage')

plt.tight_layout()
plt.show()

print("\nğŸ�‰ Target analysis completed!")


# --- IDENTIFY FEATURE TYPES ---
print("=== ğŸ•µï¸� Identifying Feature Types & Checking for Missing Values ===\n")

# VALIDATE MISSING VALUES (Full Check)
print("--- Validating Missing Values (Sum) ---")
missing_train_total = train_df.isnull().sum().sum()
missing_test_total = test_df.isnull().sum().sum()

if missing_train_total == 0:
    print("âœ… Success: Zero missing (NaN) values found in train_df.")
else:
    print(f"â�Œ Warning: Found {missing_train_total} missing values in train_df.")

if missing_test_total == 0:
    print("âœ… Success: Zero missing (NaN) values found in test_df.")
else:
    print(f"â�Œ Warning: Found {missing_test_total} missing values in test_df.")


# SEPARATE FEATURES BY TYPE
print("\n--- Separating Features by Data Type ---")

# Define our key columns
id_col = 'id'
target_col = 'loan_paid_back'

# Find categorical features (data type 'object')
categorical_features = train_df.select_dtypes(include=['object']).columns

print(f"Found {len(categorical_features)} Categorical Features:")
print(list(categorical_features))

# Find numerical features (any type that is NOT 'object')
# We must drop the 'id' and 'target_col' from this list
numerical_features = train_df.select_dtypes(exclude=['object']).columns
numerical_features = numerical_features.drop([id_col, target_col]) # Drop ID and Target

print(f"\nFound {len(numerical_features)} Numerical Features:")
print(list(numerical_features))

print("\nâœ… === Feature Lists Created! === âœ…")


print("=== ğŸ“Š Starting Numerical Feature Analysis ===\n")
print("--- : Plotting Numerical Distributions (Histograms) ---")


plt.figure(figsize=(16, 12))
plt.suptitle("Figure 6.1: Numerical Feature Distributions (Histograms)", fontsize=16, y=1.02)

# Loop through each numerical feature and plot its histogram
for i, col in enumerate(numerical_features):
    ax = plt.subplot(3, 2, i + 1)
    
    # Plot histogram with a Kernel Density Estimate (KDE) line
    sns.histplot(train_df[col], kde=True, bins=50, ax=ax, color='blue')
    
    ax.set_title(f'Distribution of: {col}', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Frequency')

# Adjust layout and display the plots
plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust for suptitle
plt.show()


# --- 6.2: Visualizing Outliers (Box Plots) ---
print("\n--- 6.2: Plotting Numerical Outliers (Box Plots) ---")

# Set up a new plotting area (3 rows, 2 columns)
plt.figure(figsize=(16, 12))
plt.suptitle("Figure 6.2: Numerical Feature Outliers (Box Plots)", fontsize=16, y=1.02)

# Loop through each numerical feature and plot its box plot
for i, col in enumerate(numerical_features):
    ax = plt.subplot(3, 2, i + 1)
    
    # Plotting horizontally (x=) is often clearer
    sns.boxplot(x=train_df[col], ax=ax, color='skyblue')
    
    ax.set_title(f'Box Plot of: {col}', fontsize=12)
    ax.set_xlabel('Value')

# Adjust layout and display the plots
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print("\nâœ… === Numerical Feature Analysis Completed! === âœ…")


# --- CATEGORICAL FEATURE VISUALIZATION ---
print("=== ğŸ“Š Starting Categorical Feature Analysis ===\n")

# --- Visualizing Distributions (Count Plots) ---
print("--- Plotting Categorical Distributions (Count Plots) ---")
print("This shows the frequency (count) of each category.")

# We have 6 categorical features. Let's set up a 3x2 grid.
plt.figure(figsize=(16, 18))
plt.suptitle("Figure 7.1: Categorical Feature Distributions (Count Plots)", fontsize=16, y=1.02)

# Loop through each categorical feature
# We assume 'categorical_features' list is already defined
for i, col in enumerate(categorical_features):
    ax = plt.subplot(3, 2, i + 1)
    
    # Plot a count plot
    # 'order=' sorts the bars from most frequent to least
    sns.countplot(
        x=col, 
        data=train_df, 
        palette='viridis', 
        ax=ax,
        order=train_df[col].value_counts().index 
    )
    
    ax.set_title(f'Distribution of: {col}', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    
    # Rotate x-axis labels if they are long (like in 'grade_subgrade')
    ax.tick_params(axis='x', rotation=45) 

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()


# --- Visualizing Impact on Target (Payback Rate) ---
print("\n--- Plotting Categorical Impact on Target (Payback Rate) ---")
print("This shows the payback rate (percentage of 1.0s) for each category.")

plt.figure(figsize=(16, 18))
plt.suptitle("Figure 7.2: Categorical Features vs. Target (Payback Rate)", fontsize=16, y=1.02)

# Loop through each categorical feature
for i, col in enumerate(categorical_features):
    ax = plt.subplot(3, 2, i + 1)
    
    # Calculate the mean of 'loan_paid_back' for each category.
    # Since 'loan_paid_back' is 0 or 1, the mean IS the payback *rate* (e.g., 0.85 = 85%)
    # We sort the bars by this payback rate to see the trend clearly.
    payback_rates = train_df.groupby(col)[target_col].mean().sort_values()
    
    # Create the bar plot
    sns.barplot(
        x=payback_rates.index, 
        y=payback_rates.values, 
        palette='coolwarm',
        ax=ax,
        order=payback_rates.index
    )
    
    ax.set_title(f'Payback Rate by: {col}', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Payback Rate (Mean)')
    
    # Set the y-axis limit to be between 0 and 1 (0% to 100%)
    ax.set_ylim(0, 1) 
    
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print("\nâœ… === Categorical Feature Analysis Completed! === âœ…")


# --- FEATURE ENGINEERING & PREPROCESSING ---
print("=== âš™ï¸� Starting Feature Engineering & Preprocessing ===\n")

# To avoid warnings, we'll work on copies of the dataframes
X = train_df.drop(columns=[target_col, id_col]).copy()
X_test = test_df.drop(columns=[id_col]).copy()

# --- 1. Dropping Useless Features ---
# Based on our EDA (Payback Rate was the same for all), these are noise.
features_to_drop = ['gender', 'marital_status', 'education_level']

X = X.drop(columns=features_to_drop)
X_test = X_test.drop(columns=features_to_drop)

print(f"Dropped {len(features_to_drop)} useless features: {features_to_drop}")


# --- 2. Manual Ordinal Encoding for 'grade_subgrade' ---
# This is our most important feature, so we encode it carefully.
# We create a mapping: A1 (best) = 34, F5 (worst) = 0.
print("\n--- Applying Manual Ordinal Encoding to 'grade_subgrade' ---")

# Define the explicit order from best (A1) to worst (F5)
grade_order = [
    'A1', 'A2', 'A3', 'A4', 'A5',
    'B1', 'B2', 'B3', 'B4', 'B5',
    'C1', 'C2', 'C3', 'C4', 'C5',
    'D1', 'D2', 'D3', 'D4', 'D5',
    'E1', 'E2', 'E3', 'E4', 'E5',
    'F1', 'F2', 'F3', 'F4', 'F5'
]

# We reverse the list so that F5=0, F4=1, ... A1=34
grade_map = {grade: i for i, grade in enumerate(reversed(grade_order))}

# Apply the mapping to our feature
X['grade_subgrade_encoded'] = X['grade_subgrade'].map(grade_map)
X_test['grade_subgrade_encoded'] = X_test['grade_subgrade'].map(grade_map)

# We can now drop the original 'grade_subgrade' (text) column
X = X.drop(columns=['grade_subgrade'])
X_test = X_test.drop(columns=['grade_subgrade'])


# --- 3. One-Hot Encoding for Remaining Categorical Features ---
# These features don't have an order, so we use get_dummies.
remaining_cats = ['employment_status', 'loan_purpose']

print(f"\n--- Applying One-Hot Encoding to {remaining_cats} ---")

# 'pd.get_dummies' creates new columns for each category (e.g., 'employment_status_Employed')
# 'drop_first=True' is good practice to prevent multicollinearity
X = pd.get_dummies(X, columns=remaining_cats, drop_first=True)
X_test = pd.get_dummies(X_test, columns=remaining_cats, drop_first=True)

# Ensure both train and test sets have the exact same columns after dummies
# Some categories might exist in train but not in test, or vice-versa
X_train_cols, X_test_cols = X.align(X_test, join='inner', axis=1, fill_value=0)

print("\n--- Data after Preprocessing (First 5 Rows) ---")
display(X_train_cols.head())

print(f"\nOriginal feature count: {len(train_df.columns)}")
print(f"New feature count after encoding: {len(X_train_cols.columns)}")
print("\nâœ… === Preprocessing Completed! Data is 100% numerical. === âœ…")


# --- MODEL BUILDING & TRAINING (FINAL WITH ROC VALIDATION) ---
print("=== ğŸš€ Starting Model Building & Training (Final) ===\n")

from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd # Re-importing pandas just in case for display

# 1. Veri AyÄ±rma (Validation iÃ§in %20'yi ayÄ±rÄ±yoruz)
# Bu kodun baÅŸarÄ±lÄ± Ã§alÄ±ÅŸmasÄ± iÃ§in 'X_train_cols', 'y_train', 'test_df' ve 'target_col' tanÄ±mlÄ± olmalÄ±dÄ±r.
y_train = train_df[target_col]
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_cols, 
    y_train,      
    test_size=0.2, # %20'sini validasyon iÃ§in ayÄ±r
    random_state=42, 
    stratify=y_train # Dengeli veri yapÄ±sÄ±nÄ± korur (80/20)
)
print(f"EÄŸitim Seti Boyutu (X_train): {X_train.shape}")
print(f"Validasyon Seti Boyutu (X_val): {X_val.shape}")

# 2. Modeli BaÅŸlat ve EÄŸit
print("\n--- Initializing & Training Optimized LightGBM Model ---")

model = LGBMClassifier(
    random_state=42,
    is_unbalance=True,
    metric='auc',
    n_estimators=300,  # Optimize edilmiÅŸ deÄŸer
    learning_rate=0.1, 
    max_depth=7,       
    num_leaves=31,     
    subsample=0.8,     
    colsample_bytree=0.8,
    n_jobs=-1
)

# Modeli eÄŸitirken validasyon setini (X_val) kullanarak erken durdurma (Early Stopping) ekledik
model.fit(
    X_train, y_train_split,
    eval_set=[(X_val, y_val)], 
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)
print("âœ… Model training completed with validation!")

# 3. ROC EÄŸrisi Ã‡izimi (Validasyon Seti Ãœzerinden)
print("\n--- ğŸ“ˆ Generating ROC Curve Visualization ---")
val_predictions = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, val_predictions)
roc_auc = auc(fpr, tpr)
validation_auc = roc_auc_score(y_val, val_predictions)

# ROC EÄŸrisi GrafiÄŸi
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (YanlÄ±ÅŸ Pozitif OranÄ±)')
plt.ylabel('True Positive Rate (DoÄŸru Pozitif OranÄ±)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
print(f"âœ… Validation ROC AUC Score: {validation_auc:.4f}")

# 4. Final Tahmin ve Submission DosyasÄ± OluÅŸturma
print("\n--- 5. Creating Final Submission File ---")
test_predictions = model.predict_proba(X_test_cols)[:, 1]

# KRÄ°TÄ°K DÃœZELTME: 'id' sÃ¼tununu Ã§ekiyoruz, satÄ±r indexini deÄŸil!
submission_df = pd.DataFrame({
    'id': test_df['id'], # <-- BU HATAYI DÃœZELTEN SATIR!
    'loan_paid_back': test_predictions
})

# Submission DosyasÄ± Format KontrolÃ¼
print("\n--- ğŸ“� Submission Format Check ---")
print(f"Submission Shape: {submission_df.shape}")
print("Expected Columns: id, loan_paid_back")
print("First 5 Rows (Probabilities):")
display(submission_df.head()) # Ä°lk satÄ±rlarda 593994, 593995... gÃ¶rmelisin.

# Save the file
submission_df.to_csv('submission.csv', index=False)

print("\nğŸ�‰ === FULL PIPELINE COMPLETED! SUBMIT FILE! === ğŸ�‰")

