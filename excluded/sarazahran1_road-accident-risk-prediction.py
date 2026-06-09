!pip install xgboost

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42


# Load data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print("Training Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)
print("\nTraining Data Info:")
print(train_data.info())
print("\nMissing Values - Train:", train_data.isnull().sum().sum())
print("Missing Values - Test:", test_data.isnull().sum().sum())



print(" Starting Comprehensive Data Analysis...")

# Set up beautiful styling
plt.style.use('seaborn-v0_8')
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
sns.set_palette(sns.color_palette(colors))

#  DATASET OVERVIEW
print("\n" + "="*50)
print("1. DATASET OVERVIEW")
print("="*50)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Dataset size comparison
datasets = ['Train', 'Test']
sizes = [len(train_data), len(test_data)]
bars = axes[0].bar(datasets, sizes, color=[colors[0], colors[1]], alpha=0.8)
axes[0].set_title(' Dataset Size Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Rows')
for bar, size in zip(bars, sizes):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{size:,}', ha='center', va='bottom', fontweight='bold')

# Features information
numeric_cols = train_data.select_dtypes(include=[np.number]).columns
categorical_cols = train_data.select_dtypes(include=['object']).columns

feature_types = ['Numeric', 'Categorical']
counts = [len(numeric_cols), len(categorical_cols)]
bars = axes[1].bar(feature_types, counts, color=[colors[2], colors[3]], alpha=0.8)
axes[1].set_title(' Feature Types Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Features')
for bar, count in zip(bars, counts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{count}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

#  TARGET VARIABLE ANALYSIS
print("\n" + "="*50)
print("2. TARGET VARIABLE ANALYSIS")
print("="*50)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Distribution of target variable
axes[0,0].hist(train_data['accident_risk'], bins=50, color=colors[0], alpha=0.7, edgecolor='black')
axes[0,0].set_title(' Distribution of Accident Risk', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Accident Risk')
axes[0,0].set_ylabel('Frequency')
axes[0,0].grid(True, alpha=0.3)

# Box plot of target variable
sns.boxplot(y=train_data['accident_risk'], ax=axes[0,1], color=colors[1])
axes[0,1].set_title(' Box Plot of Accident Risk', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Accident Risk')

# Statistical summary
target_stats = train_data['accident_risk'].describe()
axes[1,0].text(0.1, 0.9, f"ğŸ“Š Statistical Summary:", fontsize=12, fontweight='bold', transform=axes[1,0].transAxes)
axes[1,0].text(0.1, 0.8, f"Mean: {target_stats['mean']:.4f}", fontsize=10, transform=axes[1,0].transAxes)
axes[1,0].text(0.1, 0.7, f"Std:  {target_stats['std']:.4f}", fontsize=10, transform=axes[1,0].transAxes)
axes[1,0].text(0.1, 0.6, f"Min:  {target_stats['min']:.4f}", fontsize=10, transform=axes[1,0].transAxes)
axes[1,0].text(0.1, 0.5, f"25%:  {target_stats['25%']:.4f}", fontsize=10, transform=axes[1,0].transAxes)
axes[1,0].text(0.1, 0.4, f"50%:  {target_stats['50%']:.4f}", fontsize=10, transform=axes[1,0].transAxes)
axes[1,0].text(0.1, 0.3, f"75%:  {target_stats['75%']:.4f}", fontsize=10, transform=axes[1,0].transAxes)
axes[1,0].text(0.1, 0.2, f"Max:  {target_stats['max']:.4f}", fontsize=10, transform=axes[1,0].transAxes)
axes[1,0].set_title(' Target Statistics', fontsize=14, fontweight='bold')
axes[1,0].axis('off')

# Q-Q plot for normality check
from scipy import stats
stats.probplot(train_data['accident_risk'], dist="norm", plot=axes[1,1])
axes[1,1].set_title(' Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

#  MISSING VALUES ANALYSIS
print("\n" + "="*50)
print("3. MISSING VALUES ANALYSIS")
print("="*50)

# Calculate missing values
missing_train = train_data.isnull().sum()
missing_test = test_data.isnull().sum()

missing_df = pd.DataFrame({
    'Train_Missing': missing_train,
    'Test_Missing': missing_test,
    'Train_Percentage': (missing_train / len(train_data)) * 100,
    'Test_Percentage': (missing_test / len(test_data)) * 100
}).sort_values('Train_Missing', ascending=False)

# Filter features with missing values
missing_df = missing_df[(missing_df['Train_Missing'] > 0) | (missing_df['Test_Missing'] > 0)]

if len(missing_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Missing values count
    missing_df[['Train_Missing', 'Test_Missing']].plot(kind='bar', ax=axes[0], color=[colors[0], colors[1]])
    axes[0].set_title('ğŸ”� Missing Values Count by Feature', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Missing Values')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Missing values percentage
    missing_df[['Train_Percentage', 'Test_Percentage']].plot(kind='bar', ax=axes[1], color=[colors[2], colors[3]])
    axes[1].set_title('ğŸ“Š Missing Values Percentage', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("Missing values summary:")
    print(missing_df)
else:
    print(" No missing values found in the dataset!")

#  CORRELATION ANALYSIS
print("\n" + "="*50)
print("4. CORRELATION ANALYSIS")
print("="*50)

# Select only numeric columns for correlation
numeric_data = train_data.select_dtypes(include=[np.number])

if len(numeric_data.columns) > 1:
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title(' Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Top correlations with target
    if 'accident_risk' in corr_matrix.columns:
        target_correlations = corr_matrix['accident_risk'].drop('accident_risk').sort_values(ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top positive correlations
        top_positive = target_correlations.head(10)
        sns.barplot(y=top_positive.index, x=top_positive.values, ax=axes[0], palette='viridis')
        axes[0].set_title(' Top Positive Correlations with Target', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Correlation Coefficient')
        
        # Top negative correlations
        top_negative = target_correlations.tail(10)
        sns.barplot(y=top_negative.index, x=top_negative.values, ax=axes[1], palette='viridis_r')
        axes[1].set_title(' Top Negative Correlations with Target', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Correlation Coefficient')
        
        plt.tight_layout()
        plt.show()

#  FEATURE DISTRIBUTIONS
print("\n" + "="*50)
print("5. FEATURE DISTRIBUTIONS")
print("="*50)

# Select first 8 numeric features for visualization (excluding ID and target)
feature_cols = [col for col in numeric_data.columns if col not in ['id', 'accident_risk']][:8]

if len(feature_cols) > 0:
    n_cols = 4
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        if i < len(axes):
            # Plot distribution
            axes[i].hist(train_data[col].dropna(), bins=30, color=colors[i % len(colors)], alpha=0.7, edgecolor='black')
            axes[i].set_title(f'ğŸ“Š {col}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# OUTLIER DETECTION
print("\n" + "="*50)
print("6. OUTLIER DETECTION")
print("="*50)

outlier_cols = feature_cols[:6] if len(feature_cols) >= 6 else feature_cols

if len(outlier_cols) > 0:
    n_cols = 3
    n_rows = (len(outlier_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(outlier_cols):
        if i < len(axes):
            # Create box plot
            sns.boxplot(y=train_data[col], ax=axes[i], color=colors[i % len(colors)])
            axes[i].set_title(f' {col}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Values')
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

#  DATA QUALITY SUMMARY
print("\n" + "="*50)
print("7. DATA QUALITY SUMMARY")
print("="*50)

summary_data = {
    'Metric': [
        'Total Training Samples',
        'Total Test Samples', 
        'Total Features',
        'Numeric Features',
        'Categorical Features',
        'Missing Values in Train',
        'Missing Values in Test',
        'Duplicate Rows in Train',
        'Duplicate Rows in Test'
    ],
    'Value': [
        f"{len(train_data):,}",
        f"{len(test_data):,}",
        f"{train_data.shape[1] - 2:,}",  # Excluding ID and target
        f"{len(numeric_cols) - 2:,}",   # Excluding ID and target
        f"{len(categorical_cols):,}",
        f"{train_data.isnull().sum().sum():,}",
        f"{test_data.isnull().sum().sum():,}",
        f"{train_data.duplicated().sum():,}",
        f"{test_data.duplicated().sum():,}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*50)
print(" COMPREHENSIVE DATA ANALYSIS COMPLETED!")
print("="*50)


X_all = pd.concat([train_data.drop(['id', 'accident_risk'], axis=1), 
                   test_data.drop('id', axis=1)], axis=0)

def create_features(df):
    df_eng = df.copy()
    
 
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols[:3]:  
        df_eng[f'{col}_squared'] = df[col] ** 2
        df_eng[f'{col}_log'] = np.log1p(np.abs(df[col]))
    
    df_eng['feature_mean'] = df[numeric_cols].mean(axis=1)
    df_eng['feature_std'] = df[numeric_cols].std(axis=1)
    
    return df_eng

X_engineered = create_features(X_all)

X_train_eng = X_engineered[:len(train_data)]
X_test_eng = X_engineered[len(train_data):]
y = train_data['accident_risk']

print("Original features:", X_all.shape[1])
print("After feature engineering:", X_train_eng.shape[1])


label_encoders = {}
for col in X_train_eng.columns:
    if X_train_eng[col].dtype == 'object':
        le = LabelEncoder()
        combined = pd.concat([X_train_eng[col], X_test_eng[col]], axis=0)
        le.fit(combined)
        X_train_eng[col] = le.transform(X_train_eng[col])
        X_test_eng[col] = le.transform(X_test_eng[col])
        label_encoders[col] = le
        print(f"Encoded categorical feature: {col}")

imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_eng)
X_test_imputed = imputer.transform(X_test_eng)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print("Preprocessing completed successfully!")
print("Final training shape:", X_train_scaled.shape)


# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, y, test_size=0.2, random_state=RANDOM_STATE
)

models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'Ridge': Ridge(random_state=RANDOM_STATE),
    'Lasso': Lasso(random_state=RANDOM_STATE)
}

validation_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    validation_scores[name] = val_rmse
    print(f"{name} Validation RMSE: {val_rmse:.6f}")

best_model_name = min(validation_scores, key=validation_scores.get)
print(f"\nBest model: {best_model_name} with RMSE: {validation_scores[best_model_name]:.6f}")




print(" Starting training...")

# Load data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

X = train_data.drop(['id', 'accident_risk'], axis=1)
y = train_data['accident_risk']
X_test = test_data.drop('id', axis=1)

# Handle categorical features
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0)
        le.fit(combined)
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

# Handle missing values
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print("Data preprocessing completed!")


print(" Training  ensemble models...")

# Model 1: XGBoost with optimized parameters
xgb_model = xgb.XGBRegressor(
    n_estimators=1500,
    max_depth=7,
    learning_rate=0.025,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)

# Model 2: LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=1200,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Model 3: Gradient Boosting
gbr_model = GradientBoostingRegressor(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

print("Training XGBoost...")
xgb_model.fit(X_scaled, y)

print("Training LightGBM...")
lgb_model.fit(X_scaled, y)

print("Training Gradient Boosting...")
gbr_model.fit(X_scaled, y)

# Generate predictions from all models
xgb_pred = xgb_model.predict(X_test_scaled)
lgb_pred = lgb_model.predict(X_test_scaled)
gbr_pred = gbr_model.predict(X_test_scaled)

ensemble_pred = 0.5 * xgb_pred + 0.3 * lgb_pred + 0.2 * gbr_pred

print(" Ensemble models trained!")




print(" Creating submission files...")

submission_variations = {
    'xgb_only': xgb_pred,
    'lgb_only': lgb_pred, 
    'ensemble': ensemble_pred,
    'ensemble_tuned': 0.6 * xgb_pred + 0.25 * lgb_pred + 0.15 * gbr_pred
}

for name, predictions in submission_variations.items():
    submission = pd.DataFrame({
        'id': test_data['id'],
        'accident_risk': predictions
    })
    
    submission['accident_risk'] = submission['accident_risk'].clip(0, 1)
    
    submission.to_csv(f'{name}_submission.csv', index=False)
    submission.to_csv(f'/kaggle/working/{name}_submission.csv', index=False)
    
    print(f" {name}_submission.csv created")

# Create main submission file (ensemble)
main_submission = pd.DataFrame({
    'id': test_data['id'],
    'accident_risk': ensemble_pred
})

main_submission['accident_risk'] = main_submission['accident_risk'].clip(0, 1)

main_submission.to_csv('submission.csv', index=False)
main_submission.to_csv('/kaggle/working/submission.csv', index=False)
main_submission.to_csv('./submission.csv', index=False)

print("Main submission.csv created in multiple locations!")




print(" Verifying file creation...")

expected_files = [
    'submission.csv',
    '/kaggle/working/submission.csv',
    'xgb_only_submission.csv',
    'lgb_only_submission.csv', 
    'ensemble_submission.csv',
    'ensemble_tuned_submission.csv'
]

missing_files = []
for file_path in expected_files:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f" {file_path} - {file_size} bytes")
    else:
        print(f" {file_path} - NOT FOUND")
        missing_files.append(file_path)

if missing_files:
    print(f"\nâš   {len(missing_files)} files missing. Creating emergency files...")
    
    # Emergency file creation
    for missing_file in missing_files:
        if 'submission.csv' in missing_file:
            main_submission.to_csv(missing_file, index=False)
            print(f" Created emergency: {missing_file}")

# Final verification
print("\n Final directory check:")
current_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'submission' in f]
for file in current_files:
    size = os.path.getsize(file)
    print(f"    {file} - {size} bytes")

if '/kaggle/working' in os.listdir('/kaggle'):
    working_files = [f for f in os.listdir('/kaggle/working') if f.endswith('.csv') and 'submission' in f]
    for file in working_files:
        size = os.path.getsize(f'/kaggle/working/{file}')
        print(f"    /kaggle/working/{file} - {size} bytes")

print("\n SUBMISSION FILES READY!")
print(" Prediction stats for main submission:")
print(f"   Min: {main_submission['accident_risk'].min():.6f}")
print(f"   Max: {main_submission['accident_risk'].max():.6f}") 
print(f"   Mean: {main_submission['accident_risk'].mean():.6f}")

