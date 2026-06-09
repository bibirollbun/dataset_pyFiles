!pip install graphviz


from graphviz import Digraph

dot = Digraph()

dot.attr(rankdir='LR', size='40')

dot.node('A', '1. Data Loading\nand Cleaning')
dot.node('B', '2. Exploratory\nData Analysis (EDA)')
dot.node('C', '3. Feature\nEngineering')
dot.node('D', '4. Feature Selection\nwith SHAP')
dot.node('E', '5. Model\nTraining')
dot.node('F', '6. Prediction &\nEvaluation (MAP@3)')

dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])

dot.render('pipeline', format='png', cleanup=False)
dot


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
from scipy.stats import skew
import shap
from xgboost import XGBClassifier
import xgboost as xgb
import gc
import torch
from itertools import combinations


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
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# Keep original ID columns
train_id = train['id'].copy()
test_id = test['id'].copy()

# Display basic information
print("Training set shape:", train.shape)
print("Test set shape:", test.shape)

# Check for missing values
print("\nğŸ“Š Missing values in training set:")
print(train.isna().sum())

print("\nğŸ“Š Missing values in test set:")
print(test.isna().sum())

# Display statistical summary
print("\nğŸ“Š Statistical summary of training data:")
display(train.describe().T)

# Remove ID columns for analysis
train_data = train.drop(columns=['id'])
test_data = test.drop(columns=['id'])

# Identify numeric and categorical columns
numeric_columns = [col for col in train_data.columns if col not in ['Soil Type', 'Crop Type','Fertilizer Name'] and train_data[col].dtype in [np.int64, np.float64]]
categorical_columns = [col for col in train_data.columns if col not in numeric_columns + ['Fertilizer Name']]
target_column = 'Fertilizer Name'

print(f"\nğŸ“Š Number of numeric features: {len(numeric_columns)}")
print(f"ğŸ“Š Number of categorical features: {len(categorical_columns)}")
print(f"ğŸ“Š Target column: {target_column}")


train_data.head()


train_data.info()


print(f'DataFrame contains {train_data.shape[0]} rows (records) and {train_data.shape[1]} columns (attributes).')


# --------------------------------------------------------------
# Fertilizer Dataset - Exploratory Data Analysis & Feature Engineering
# --------------------------------------------------------------

# Set style and color palette
plt.style.use('default')
palette = sns.color_palette("husl", 10)
sns.set_palette(palette)

# Identify columns based on your dataset structure
numeric_columns = [col for col in train_data.columns if col not in ['Soil Type', 'Crop Type','Fertilizer Name', 'id'] and train_data[col].dtype in [np.int64, np.float64]]
categorical_columns = [col for col in train_data.columns if col not in numeric_columns + ['Fertilizer Name', 'id']]
target_column = 'Fertilizer Name'

print(f"ğŸ“Š Dataset Overview:")

# --------------------------------------------------------------
# Target Variable Analysis (Classification)
# --------------------------------------------------------------

plt.figure(figsize=(12, 6))
target_counts = train_data[target_column].value_counts()
sns.countplot(data=train_data, x=target_column, palette=palette, order=target_counts.index)
plt.title('Distribution of Target Variable: Fertilizer Name', fontweight='bold', fontsize=14)
plt.xlabel('Fertilizer Name')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Class balance visualization (Pie chart instead of bar plot)
plt.figure(figsize=(10, 10))
class_percentages = train_data[target_column].value_counts(normalize=True) * 100
plt.pie(class_percentages, labels=class_percentages.index, autopct='%1.1f%%', startangle=90, colors=palette, 
        textprops={'fontsize': 12}, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
plt.title('Class Distribution (Percentage)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Feature Distribution Analysis
# --------------------------------------------------------------

# Distribution of numeric features
fig = plt.figure(figsize=(15, 12))
n_cols = 3
n_rows = (len(numeric_columns) + n_cols - 1) // n_cols

for i, feature in enumerate(numeric_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(train_data[feature], kde=True, color=palette[i % len(palette)])
    plt.title(f'Distribution of {feature}', fontweight='bold')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Boxplots to check for outliers in features
fig = plt.figure(figsize=(15, 12))
for i, feature in enumerate(numeric_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(y=train_data[feature], color=palette[i % len(palette)])
    plt.title(f'Boxplot of {feature}', fontweight='bold')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Categorical Features Analysis
# --------------------------------------------------------------

if categorical_columns:
    fig = plt.figure(figsize=(15, 6))
    for i, feature in enumerate(categorical_columns, 1):
        plt.subplot(1, len(categorical_columns), i)
        feature_counts = train_data[feature].value_counts()
        sns.countplot(data=train_data, x=feature, palette=palette, order=feature_counts.index)
        plt.title(f'Distribution of {feature}', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print categorical feature details
    print(f"\nğŸ“Š Categorical features details:")
    for col in categorical_columns:
        print(f"\n{col}:")
        print(f"  Unique values: {train_data[col].nunique()}")
        print(f"  Value counts:\n{train_data[col].value_counts().to_string()}")

# --------------------------------------------------------------
# Feature vs Target Analysis
# --------------------------------------------------------------

# Numeric features vs Target
fig = plt.figure(figsize=(15, 12))
for i, feature in enumerate(numeric_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(data=train_data, x=target_column, y=feature, palette=palette)
    plt.title(f'{feature} by Fertilizer Type', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Categorical features vs Target (Stacked bar plots)
if categorical_columns:
    fig = plt.figure(figsize=(15, 8))
    for i, feature in enumerate(categorical_columns, 1):
        plt.subplot(1, len(categorical_columns), i)
        
        # Create cross-tabulation
        crosstab = pd.crosstab(train_data[feature], train_data[target_column], normalize='index') * 100
        
        # Create stacked bar plot
        crosstab.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='Set3')
        plt.title(f'{feature} vs Fertilizer Distribution', fontweight='bold')
        plt.xlabel(feature)
        plt.ylabel('Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Summary statistics
print(f"\nğŸ“Š Dataset Summary:")
print(f"Total samples in training set: {len(train_data):,}")
print(f"Total samples in test set: {len(test_data):,}")
print(f"Number of numeric features: {len(numeric_columns)}")
print(f"Number of categorical features: {len(categorical_columns)}")
print(f"Number of unique fertilizers: {train_data[target_column].nunique()}")
print(f"Class balance ratio (min/max): {train_data[target_column].value_counts().min() / train_data[target_column].value_counts().max():.3f}")


# Cleaning column names (removing extra spaces)
train_data.columns = train_data.columns.str.strip()
test_data.columns = test_data.columns.str.strip()

# Checking for column mismatches between training and test datasets
missing_in_test = [col for col in train_data.columns if col not in test_data.columns and col != 'Fertilizer Name']
missing_in_train = [col for col in test_data.columns if col not in train_data.columns and col != 'id']

if missing_in_test or missing_in_train:
    print("Columns missing in test dataset:", missing_in_test)
    print("Columns missing in training dataset:", missing_in_train)
    # Filling missing columns with zeros
    for col in missing_in_test:
        test_data[col] = 0
    for col in missing_in_train:
        if col != 'id':
            train_data[col] = 0

# Defining numerical and categorical columns
numeric_columns = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_columns = ['Soil Type', 'Crop Type']
target_column = 'Fertilizer Name'

# Defining columns for quantile and equal-width binning
label_encode_columns = [
    'Temparature_quantile', 'Temparature_equal',
    'Humidity_quantile', 'Humidity_equal',
    'Moisture_quantile', 'Moisture_equal',
    'Nitrogen_quantile', 'Nitrogen_equal',
    'Potassium_quantile', 'Potassium_equal',
    'Phosphorous_quantile', 'Phosphorous_equal'
]

# ------------------- CATEGORICAL ENCODING -------------------
# Label Encoding for Soil Type and Crop Type
train_data_encoded = train_data.copy()
test_data_encoded = test_data.copy()

label_encoders = {}

# Encoding Soil Type and Crop Type
for col in categorical_columns:
    if col in train_data.columns:
        le = LabelEncoder()
        train_data_encoded[col] = le.fit_transform(train_data[col])
        label_encoders[col] = le
        if col in test_data.columns:
            test_data_encoded[col] = le.transform(test_data[col])

# ------------------- LABEL ENCODING FOR QUANTILE AND EQUAL-WIDTH BINNING COLUMNS -------------------
# Applying Label Encoding to quantile and equal-width binning columns
for col in label_encode_columns:
    if col in train_data_encoded.columns:
        le = LabelEncoder()
        train_data_encoded[col] = le.fit_transform(train_data_encoded[col].astype(str))  # Treating as categorical
        label_encoders[col] = le
        if col in test_data_encoded.columns:
            test_data_encoded[col] = le.transform(test_data_encoded[col].astype(str))

# ------------------- CHECKING REMAINING CATEGORICAL COLUMNS -------------------
# Finding columns that are still of object (categorical) type
object_columns_train = train_data_encoded.select_dtypes(include=['object']).columns.tolist()
if object_columns_train:
    print(f"Warning: These columns still require encoding: {object_columns_train}")
    # Encoding remaining categorical columns
    for col in object_columns_train:
        if col != 'Fertilizer Name':  # Excluding the target variable from encoding
            le = LabelEncoder()
            train_data_encoded[col] = le.fit_transform(train_data_encoded[col])
            if col in test_data_encoded.columns:
                test_data_encoded[col] = le.transform(test_data_encoded[col])

# ------------------- MINMAX SCALING -------------------
# Initializing MinMaxScaler
scaler = MinMaxScaler()

# Columns to exclude from scaling
exclude_from_scaling = ['id', 'Fertilizer Name'] + label_encode_columns + categorical_columns

# Columns to scale in the training dataset
columns_to_scale_train = [col for col in train_data_encoded.columns 
                         if col not in exclude_from_scaling 
                         and train_data_encoded[col].dtype in ['int64', 'float64']]

# Columns to scale in the test dataset
columns_to_scale_test = [col for col in test_data_encoded.columns 
                        if col not in ['id'] + label_encode_columns + categorical_columns
                        and test_data_encoded[col].dtype in ['int64', 'float64']]

# Scaling the training dataset
train_data_scaled = train_data_encoded.copy()
train_data_scaled[columns_to_scale_train] = scaler.fit_transform(train_data_encoded[columns_to_scale_train])

# Scaling the test dataset
test_data_scaled = test_data_encoded.copy()
test_data_scaled[columns_to_scale_test] = scaler.transform(test_data_encoded[columns_to_scale_test])

# ------------------- DATA PREVIEW -------------------
print("\nTraining Dataset (Scaled and Encoded) (First 5 rows):")
print(train_data_scaled.head())
print("\nTest Dataset (Scaled and Encoded) (First 5 rows):")
print(test_data_scaled.head())

# Checking scaled columns
print("\nScaled columns (training):", columns_to_scale_train)
print("Scaled columns (test):", columns_to_scale_test)


# Numerical and categorical columns
numeric_columns = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_columns = ['Soil Type', 'Crop Type']
target_column = 'Fertilizer Name'

# Create copies for feature engineering
train_data_fe = train_data_scaled.copy()
test_data_fe = test_data_scaled.copy()

# 1. Polynomial Features (2nd Degree)
print("ğŸ“Š Generating 2nd Degree Polynomial Features...")
# Squares
for col in numeric_columns:
    train_data_fe[f'{col}_Squared'] = train_data_fe[col] ** 2
    test_data_fe[f'{col}_Squared'] = test_data_fe[col] ** 2
    print(f"âœ“ {col}_Squared created")

# Cross products
for col1, col2 in combinations(numeric_columns, 2):
    feature_name = f'{col1}_{col2}_Interaction'
    train_data_fe[feature_name] = train_data_fe[col1] * train_data_fe[col2]
    test_data_fe[feature_name] = test_data_fe[col1] * test_data_fe[col2]
    print(f"âœ“ {feature_name} created")

# NPK interaction
train_data_fe['NPK_Interaction'] = train_data_fe['Nitrogen'] * train_data_fe['Potassium'] * train_data_fe['Phosphorous']
test_data_fe['NPK_Interaction'] = test_data_fe['Nitrogen'] * test_data_fe['Potassium'] * test_data_fe['Phosphorous']
print("âœ“ NPK_Interaction created")

# 2. Ratio Features
print("\nğŸ“Š Generating Ratio Features...")
epsilon = 1e-6  # To prevent division by zero
for col1, col2 in combinations(numeric_columns, 2):
    feature_name = f'{col1}_{col2}_Ratio'
    train_data_fe[feature_name] = train_data_fe[col1] / (train_data_fe[col2] + epsilon)
    test_data_fe[feature_name] = test_data_fe[col1] / (test_data_fe[col2] + epsilon)
    print(f"âœ“ {feature_name} created")

# 3. Categorical-Numerical Interactions
print("\nğŸ“Š Generating Categorical-Numerical Interaction Features...")
for cat_col in categorical_columns:
    for num_col in numeric_columns:
        feature_name = f'{num_col}_by_{cat_col}'
        group_means = train_data_fe.groupby(cat_col)[num_col].mean()
        train_data_fe[feature_name] = train_data_fe[cat_col].map(group_means)
        test_data_fe[feature_name] = test_data_fe[cat_col].map(group_means)
        print(f"âœ“ {feature_name} created")

# 4. Categorical Combinations
print("\nğŸ“Š Generating Categorical Combination Features...")
train_data_fe['Soil_Crop_Interaction'] = train_data_fe['Soil Type'].astype(str) + "_" + train_data_fe['Crop Type'].astype(str)
test_data_fe['Soil_Crop_Interaction'] = test_data_fe['Soil Type'].astype(str) + "_" + test_data_fe['Crop Type'].astype(str)

# Label Encoding for Soil_Crop_Interaction
le_interaction = LabelEncoder()
train_data_fe['Soil_Crop_Interaction'] = le_interaction.fit_transform(train_data_fe['Soil_Crop_Interaction'])
le_interaction.classes_ = np.append(le_interaction.classes_, 'unknown')
test_data_fe['Soil_Crop_Interaction'] = test_data_fe['Soil_Crop_Interaction'].map(
    lambda x: x if x in le_interaction.classes_[:-1] else 'unknown'
)
test_data_fe['Soil_Crop_Interaction'] = le_interaction.transform(test_data_fe['Soil_Crop_Interaction'])
print("âœ“ Soil_Crop_Interaction created")

# 5. High Correlation Check and Removal
print("\nğŸ“Š High Correlation Check...")
print("By eliminating one of the features with a correlation of 90% or higher, we select the most meaningful features from the generated ones.")
print("Selecting features with 90% or higher correlation:")

new_numeric_columns = [col for col in train_data_fe.columns 
                       if train_data_fe[col].dtype in ['int64', 'float64'] 
                       and col not in ['id', 'Fertilizer Name', 'Fertilizer Name Encoded']]
correlation_matrix = train_data_fe[new_numeric_columns].corr()
high_corr_pairs = []
to_drop = set()
threshold = 0.9  # Correlation threshold

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) >= threshold:
            feat1, feat2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
            # Select the feature with the longer name to drop
            to_drop.add(feat2 if len(feat2) > len(feat1) else feat1)
            high_corr_pairs.append((feat1, feat2, corr_val))

if high_corr_pairs:
    existing_cols_to_drop = [col for col in to_drop if col in train_data_fe.columns]
    train_data_fe.drop(columns=existing_cols_to_drop, inplace=True)
    test_data_fe.drop(columns=existing_cols_to_drop, inplace=True)
    print(f"Removed highly correlated features: {existing_cols_to_drop}")
else:
    print("No feature pairs found with |correlation| >= 0.9.")

# Remaining features
remaining_features = [col for col in train_data_fe.columns 
                     if col not in ['id', 'Fertilizer Name', 'Fertilizer Name Encoded']]
print(f"Remaining feature count and names: {len(remaining_features)} {remaining_features}")

# 6. Logarithmic Transformations (only for skewed columns)
print("\nğŸ“Š Generating Logarithmic Transformation Features...")
for col in numeric_columns:
    if abs(skew(train_data_fe[col].dropna())) > 0.5:  # Skewness threshold
        train_data_fe[f'Log_{col}'] = np.log1p(train_data_fe[col].clip(lower=0))
        test_data_fe[f'Log_{col}'] = np.log1p(test_data_fe[col].clip(lower=0))
        print(f"âœ“ Log_{col} created")

print("\nNewly created columns:", [col for col in train_data_fe.columns if col not in train_data.columns])


print("ğŸš€ Starting Advanced GPU XGBoost with SHAP Feature Selection...")

# Add 'id' column to feature-engineered datasets
train_data_fe['id'] = train['id']
test_data_fe['id'] = test['id']

# Check GPU availability
try:
    gpu_available = torch.cuda.is_available()
    print(f"âœ… GPU available - XGBoost will run in GPU mode")
except:
    gpu_available = False
    print("âš ï¸� GPU not available - Running in CPU mode")

# Advanced XGBoost parameters (optimized for speed)
xgb_params = {
    'objective': 'multi:softprob',
    'tree_method': 'hist',
    'device': 'cuda' if gpu_available else 'cpu',
    'n_estimators': 500,
    'max_depth': 9,
    'learning_rate': 0.026,
    'subsample': 0.7,
    'colsample_bytree': 0.75, 
    'min_child_weight': 2,
    'gamma': 0.1,
    'reg_alpha': 0.94,
    'reg_lambda': 0.33,
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'n_jobs': 1,
    'enable_categorical': False 
}

# Function to calculate MAP@3 score
def map_at_3(y_true, y_pred_proba, k=3):

    map_score = 0.0
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true  # Convert Series to NumPy array
    for i in range(len(y_true)):
        top_k_preds = np.argsort(y_pred_proba[i])[-k:][::-1]  # Get top k predictions
        if y_true[i] in top_k_preds:
            rank = np.where(top_k_preds == y_true[i])[0][0] + 1
            map_score += 1.0 / rank
    return map_score / len(y_true)

# Encode categorical columns using LabelEncoder
categorical_columns = ['Soil Type', 'Crop Type']
label_encoders = {}
for col in categorical_columns:
    if col in train_data_fe.columns:
        le = LabelEncoder()
        train_data_fe[col] = le.fit_transform(train_data_fe[col])
        label_encoders[col] = le
        if col in test_data_fe.columns:
            test_data_fe[col] = le.transform(test_data_fe[col])

# Separate features and target variable
feature_columns = [col for col in train_data_fe.columns 
                   if col not in ['id', 'Fertilizer Name', 'Fertilizer Name Encoded']]
X_train_fe = train_data_fe[feature_columns]
y_train = train_data_fe['Fertilizer Name']

# Encode target variable
if 'Fertilizer Name Encoded' not in train_data_fe.columns:
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    train_data_fe['Fertilizer Name Encoded'] = y_train_encoded
else:
    y_train_encoded = train_data_fe['Fertilizer Name Encoded']
    le_target = LabelEncoder()
    le_target.fit(y_train)

print(f"ğŸ“Š Total number of features: {len(feature_columns)}")
print(f"ğŸ“Š Number of classes: {len(np.unique(y_train_encoded))}")
print(f"ğŸ“Š Dataset size: {X_train_fe.shape}")

# Check class distribution for imbalance
print("\nğŸ“Š Class distribution:")
print(pd.Series(y_train_encoded).value_counts(normalize=True))

# Split data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_fe, y_train_encoded, test_size=0.3, random_state=42, stratify=y_train_encoded
)

# Train base XGBoost model
print(f"\nğŸš€ Training base XGBoost model ({'GPU' if gpu_available else 'CPU'} mode)...")
base_model = xgb.XGBClassifier(**xgb_params)
base_model.fit(X_train_split, y_train_split, verbose=False)

# Calculate MAP@3 for base model
base_pred_proba = base_model.predict_proba(X_val_split)
base_map_at_3 = map_at_3(y_val_split, base_pred_proba)
print(f"âœ… Base model MAP@3: {base_map_at_3:.4f}")

# Prepare sample for SHAP analysis
sample_size = 10000
X_sample = X_val_split.sample(n=sample_size, random_state=42)
print(f"ğŸ“Š Calculating SHAP values for {sample_size} samples...")

# Create SHAP Explainer
print("\nğŸ”� Creating SHAP Explainer...")
try:
    explainer = shap.TreeExplainer(base_model, check_additivity=False)
    shap_values = explainer.shap_values(X_sample)
    print("âœ… Using native XGBoost SHAP")
except Exception as e:
    print(f"âš ï¸� Native SHAP error: {str(e)}")
    print("ğŸ”„ Falling back to XGBoost feature importance...")
    shap_importance = base_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'shap_importance': shap_importance,
        'xgb_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    shap_values = None
else:
    if isinstance(shap_values, list):
        print("âœ… Multi-class SHAP values (list of arrays)")
        shap_importance = np.mean([np.abs(class_values).mean(axis=0) for class_values in shap_values], axis=0)
    else:
        print("âœ… Binary/Regression SHAP values (single array)")
        shap_importance = np.abs(shap_values).mean(axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'shap_importance': shap_importance,
        'xgb_importance': base_model.feature_importances_
    }).sort_values('shap_importance', ascending=False)

print("\nğŸ“Š Top 20 features (SHAP):")
print(feature_importance[['feature', 'shap_importance']].head(20))

# Test model performance with different feature counts
print("\nğŸ”� Testing model performance with different feature counts (MAP@3)...")
feature_counts = [8 ,9, 10, 12, 15, 20 , 30]
results = []

fast_xgb_params = xgb_params.copy()
fast_xgb_params.update({
    'n_estimators': 500,  # Number of boosting rounds
    'max_depth': 7,  # Maximum tree depth
})

for n_features in feature_counts:
    if n_features > len(feature_importance):
        continue
        
    print(f"   Testing: {n_features} features...")
    selected_features = feature_importance.head(n_features)['feature'].tolist()
    
    X_train_selected = X_train_split[selected_features]
    X_val_selected = X_val_split[selected_features]
    
    model_selected = xgb.XGBClassifier(**fast_xgb_params)
    model_selected.fit(X_train_selected, y_train_split, verbose=False)
    
    # Calculate MAP@3
    y_pred_proba = model_selected.predict_proba(X_val_selected)
    map_score = map_at_3(y_val_split, y_pred_proba)
    
    # Perform cross-validation for MAP@3
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_map_scores = []
    for train_idx, val_idx in cv.split(X_train_selected, y_train_split):
        X_train_cv = X_train_selected.iloc[train_idx]
        y_train_cv = y_train_split[train_idx]
        X_val_cv = X_train_selected.iloc[val_idx]
        y_val_cv = y_train_split[val_idx]
        
        model_cv = xgb.XGBClassifier(**fast_xgb_params)
        model_cv.fit(X_train_cv, y_train_cv, verbose=False)
        y_pred_proba_cv = model_cv.predict_proba(X_val_cv)
        cv_map_scores.append(map_at_3(y_val_cv, y_pred_proba_cv))
    
    map_score_cv = np.mean(cv_map_scores)
    map_std_cv = np.std(cv_map_scores)
    
    results.append({
        'n_features': n_features,
        'map_at_3': map_score_cv,
        'std': map_std_cv,
        'improvement': map_score_cv - base_map_at_3
    })
    
    print(f"   âœ… {n_features:3d} features: CV MAP@3 = {map_score_cv:.4f} (Â±{map_std_cv:.4f})")

# Determine the optimal number of features
results_df = pd.DataFrame(results)
best_result = results_df.loc[results_df['map_at_3'].idxmax()]
optimal_n_features = int(best_result['n_features'])

print(f"\nğŸ�¯ Optimal number of features (MAP@3): {optimal_n_features}")
print(f"ğŸ�¯ Best CV MAP@3: {best_result['map_at_3']:.4f} (Â±{best_result['std']:.4f})")
print(f"ğŸ�¯ Improvement over base model: {best_result['improvement']:+.4f}")

# Select final features
selected_features_final = feature_importance.head(optimal_n_features)['feature'].tolist()

print(f"\nâœ… Selected {len(selected_features_final)} features:")
for i, row in feature_importance.head(optimal_n_features).iterrows():
    print(f"{i+1:2d}. {row['feature']:<40} (SHAP: {row['shap_importance']:.4f}, XGB: {row['xgb_importance']:.4f})")

# Train final model with selected features
print(f"\nğŸš€ Training final model with {optimal_n_features} features...")
X_train_final = X_train_split[selected_features_final]
X_val_final = X_val_split[selected_features_final]

final_model = xgb.XGBClassifier(**xgb_params)
final_model.fit(X_train_final, y_train_split, verbose=False)

# Evaluate final model
y_pred_proba_final = final_model.predict_proba(X_val_final)
final_map_at_3 = map_at_3(y_val_split, y_pred_proba_final)
y_pred_final = final_model.predict(X_val_final)
final_accuracy = classification_report(y_val_split, y_pred_final, target_names=le_target.classes_, digits=4, output_dict=True)

print(f"âœ… Final model MAP@3: {final_map_at_3:.4f}")
print(f"âœ… Improvement: {final_map_at_3 - base_map_at_3:+.4f}")
print(f"\nğŸ“Š Detailed performance report:")
print(classification_report(y_val_split, y_pred_final, target_names=le_target.classes_, digits=4))

# Create final datasets with selected features
print(f"\nğŸš€ Creating datasets with {len(selected_features_final)} selected features...")
train_data_selected = train_data_fe[['id', 'Fertilizer Name', 'Fertilizer Name Encoded'] + selected_features_final].copy()
test_data_selected = test_data_fe[['id'] + selected_features_final].copy()

print(f"\nğŸ“Š Final dataset sizes:")
print(f"   Training set: {train_data_selected.shape}")
print(f"   Test set: {test_data_selected.shape}")


# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_comprehensive_visualizations():
    
    FIGSIZE = (12, 8)  # Larger size for single plots
    
    # ==========================================================================
    # 1. FEATURE IMPORTANCE COMPARISON (Top 20) - Separate Figure
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=FIGSIZE)
    fig1.suptitle('Feature Importance Comparison\n(Top 20 Features)', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    top_20 = feature_importance.head(20)
    y_pos = np.arange(len(top_20))
    
    bars1 = ax1.barh(y_pos - 0.2, top_20['shap_importance'], 
                     height=0.35, alpha=0.8, label='SHAP Importance', 
                     color='#2E86C1', edgecolor='white', linewidth=0.5)
    bars2 = ax1.barh(y_pos + 0.2, top_20['xgb_importance'], 
                     height=0.35, alpha=0.8, label='XGBoost Importance', 
                     color='#E74C3C', edgecolor='white', linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_20['feature'], fontsize=10)
    ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    for i, (shap_val, xgb_val) in enumerate(zip(top_20['shap_importance'], top_20['xgb_importance'])):
        if shap_val > 0.001:
            ax1.text(shap_val + 0.001, i - 0.2, f'{shap_val:.3f}', 
                     va='center', ha='left', fontsize=8, color='#2E86C1')
        if xgb_val > 0.001:
            ax1.text(xgb_val + 0.001, i + 0.2, f'{xgb_val:.3f}', 
                     va='center', ha='left', fontsize=8, color='#E74C3C')
    
    plt.tight_layout()
    plt.show()
    
    # ==========================================================================
    # 2. PERFORMANCE CURVE (MAP@3 vs Number of Features) - Separate Figure
    # ==========================================================================
    fig2, ax2 = plt.subplots(figsize=FIGSIZE)
    fig2.suptitle('Model Performance vs. Number of Features\n(Cross-Validation Results)', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    ax2.plot(results_df['n_features'], results_df['map_at_3'], 
             'o-', linewidth=3, markersize=8, color='#2E86C1', 
             markerfacecolor='white', markeredgewidth=2, 
             markeredgecolor='#2E86C1', label='Cross-Validation MAP@3')
    
    ax2.fill_between(results_df['n_features'], 
                     results_df['map_at_3'] - results_df['std'], 
                     results_df['map_at_3'] + results_df['std'], 
                     alpha=0.2, color='#2E86C1', label='Â±1 Std Deviation')
    
    ax2.axhline(y=base_map_at_3, color='#E74C3C', linestyle='--', linewidth=2,
                label=f'Baseline MAP@3 ({base_map_at_3:.4f})')
    ax2.axvline(x=optimal_n_features, color='#27AE60', linestyle='--', linewidth=2,
                label=f'Optimal Point ({optimal_n_features} features)')
    
    ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cross-Validation MAP@3', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    max_performance = results_df.loc[results_df['map_at_3'].idxmax()]
    ax2.annotate(f'Best: {max_performance["map_at_3"]:.4f}', 
                 xy=(max_performance['n_features'], max_performance['map_at_3']),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.show()
    
    # ==========================================================================
    # 3. CORRELATION HEATMAP (Target Variable and Top Features) - Separate Figure
    # ==========================================================================
    fig3, ax3 = plt.subplots(figsize=FIGSIZE)
    fig3.suptitle('Feature Correlation Matrix\n(Top 12 Features + Target)', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    try:
        top_n_features_for_corr = feature_importance.head(12)['feature'].tolist()
        corr_data = X_train_split[top_n_features_for_corr].copy()
        corr_data['Target'] = y_train_split  
        corr_matrix = corr_data.corr(method='pearson')
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    ax=ax3,
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    annot=True,
                    fmt='.2f',
                    annot_kws={'size': 9, 'weight': 'bold'},
                    cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                    square=True,
                    linewidths=0.5,
                    linecolor='black')
        
        ax3.tick_params(axis='x', labelrotation=45, labelsize=9)
        ax3.tick_params(axis='y', labelrotation=0, labelsize=9)
        
    except Exception as e:
        ax3.text(0.5, 0.5, f'Correlation Analysis Error:\n{str(e)}', 
                 ha='center', va='center', fontsize=12, 
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # ==========================================================================
    # 4. FEATURE CATEGORY DISTRIBUTION - Separate Figure
    # ==========================================================================
    fig4, ax4 = plt.subplots(figsize=FIGSIZE)
    fig4.suptitle('Distribution of Selected Features by Category\n' + 
                  f'Total Features: {len(selected_features_final)}', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    numeric_columns = train_data_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    original_features = [f for f in selected_features_final if f in (set(numeric_columns) | set(categorical_columns))]
    polynomial_features = [f for f in selected_features_final if '_Squared' in f or '_Interaction' in f]
    ratio_features = [f for f in selected_features_final if '_Ratio' in f]
    categorical_interaction_features = [f for f in selected_features_final if '_by_' in f or 'Soil_Crop' in f]
    log_features = [f for f in selected_features_final if f.startswith('Log_')]
    
    categories = ['Original\nFeatures', 'Polynomial\n& Interactions', 'Ratio\nFeatures', 
                  'Categorical\nInteractions', 'Logarithmic\nTransformations']
    counts = [len(original_features), len(polynomial_features), len(ratio_features), 
              len(categorical_interaction_features), len(log_features)]
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
    
    wedges, texts, autotexts = ax4.pie(counts, labels=categories, autopct='%1.1f%%', 
                                      startangle=90, colors=colors, 
                                      textprops={'fontsize': 12, 'fontweight': 'bold'},
                                      wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                                      explode=(0.05, 0.05, 0.05, 0.05, 0.05))
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    legend_labels = [f'{cat.replace(chr(10), " ")}: {count}' for cat, count in zip(categories, counts)]
    ax4.legend(wedges, legend_labels, title="Feature Categories", 
               loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
               fontsize=11, title_fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return True

# Execute visualizations
if __name__ == "__main__":
    success = create_comprehensive_visualizations()


# 1. Validate ID Columns
print("ğŸ“Š Validating ID Columns...")

if 'id' not in train_data_selected.columns:
    train_data_selected['id'] = train_id
if 'id' not in test_data_selected.columns:
    test_data_selected['id'] = test_id
print("âœ“ ID columns added or validated")

# 2. Encode Target Variable
print("\nğŸ“Š Encoding Target Variable...")
le_target = LabelEncoder()
train_data_selected['Fertilizer Name Encoded'] = le_target.fit_transform(train_data_selected['Fertilizer Name'])

# 3. Separate Features and Target
features = [col for col in train_data_selected.columns if col not in ['id', 'Fertilizer Name', 'Fertilizer Name Encoded']]
X = train_data_selected[features]
y = train_data_selected['Fertilizer Name Encoded']
X_test = test_data_selected[features]

# 4. MAP@3 Calculation Function
def map_at_3(y_true, y_pred_proba):
    y_pred_indices = np.argsort(-y_pred_proba, axis=1)[:, :3]
    map_score = 0.0
    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_labels = y_pred_indices[i]
        precision = 0.0
        relevant = 0
        for k in range(len(pred_labels)):
            if pred_labels[k] == true_label:
                relevant += 1
                precision += relevant / (k + 1)
                break
        map_score += precision / min(3, 1)
    return map_score / len(y_true)

# 5. 5-Fold Stratified K-Fold Cross-Validation
print("\nğŸ“Š Starting 5-Fold Stratified K-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
mcc_scores = []
acc_scores = []
map3_scores = []

# Optimized parameters from Optuna
best_params = {
    'learning_rate': 0.030481059889831706,
    'max_depth': 9,
    'min_child_weight': 4,
    'subsample': 0.7314525892865643,
    'colsample_bytree': 0.6728122992503545,
    'reg_alpha': 0.5173990874009716,
    'reg_lambda': 0.5576837741708011,
    'n_estimators': 750,
    'tree_method': 'gpu_hist',
    'device': 'cuda',
    'objective': 'multi:softprob',
    'num_class': len(le_target.classes_),
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nTraining Fold {fold}...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost Model
    model = XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    # Validation set predictions
    val_probs = model.predict_proba(X_val)
    val_preds = model.predict(X_val)

    # Metrics
    f1 = f1_score(y_val, val_preds, average='macro')
    mcc = matthews_corrcoef(y_val, val_preds)
    acc = accuracy_score(y_val, val_preds)
    map3 = map_at_3(y_val.values, val_probs)

    f1_scores.append(f1)
    mcc_scores.append(mcc)
    acc_scores.append(acc)
    map3_scores.append(map3)

    print(f"Fold {fold} Results:")
    print(f"  F1 Score (Macro): {f1:.4f}")
    print(f"  MCC: {mcc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  MAP@3: {map3:.4f}")

# 6. Mean and Standard Deviation
print("\nğŸ“Š Cross-Validation Results Summary:")
print(f"Average F1 Score: {np.mean(f1_scores):.4f} (Â±{np.std(f1_scores):.4f})")
print(f"Average MCC: {np.mean(mcc_scores):.4f} (Â±{np.std(mcc_scores):.4f})")
print(f"Average Accuracy: {np.mean(acc_scores):.4f} (Â±{np.std(acc_scores):.4f})")
print(f"Average MAP@3: {np.mean(map3_scores):.4f} (Â±{np.std(map3_scores):.4f})")

# 7. Train Final Model on Full Data
print("\nğŸ“Š Training Final Model on Full Data...")
final_model = XGBClassifier(**best_params)
final_model.fit(X, y)

# 8. Predict on Test Set
print("ğŸ“Š Making Predictions on Test Set...")
test_probs = final_model.predict_proba(X_test)
test_top3_indices = np.argsort(-test_probs, axis=1)[:, :3]
test_top3_labels = [le_target.inverse_transform(indices) for indices in test_top3_indices]

# 9. Create Submission File
print("ğŸ“Š Creating Submission File...")
submission = pd.DataFrame({
    'id': test_data_selected['id'],
    'Fertilizer Name': [' '.join(labels) for labels in test_top3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ“ Submission file saved as 'submission.csv'.")
print("First 5 rows of submission file:")
print(submission.head())

