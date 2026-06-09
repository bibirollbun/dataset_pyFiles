# Install required packages
!pip install missforest lightgbm  


# Required imports
import pandas as pd
import numpy as np
from missforest import MissForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  
from sklearn.ensemble import StackingClassifier
import os


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


# Load application training data

default_dir_path = "../input/home-credit-default-risk/"
data = pd.read_csv(os.path.join(default_dir_path, 'application_train.csv'))
print(f"Application train shape: {data.shape}")


# Basic data cleaning operations
data = data[data['CODE_GENDER'] != 'XNA']                         # Remove XNA gender records
data = data[data['AMT_INCOME_TOTAL'] < 20000000]                  # Remove unrealistic income values
data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)       # Replace invalid employed days
data['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)   # Replace invalid phone change days


def add_application_features(df):
    """Add basic application features like document count, age features, etc."""

    # Document features
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    df['NEW_DOC_KURT'] = df[docs].kurtosis(axis=1)
    
    # Age feature
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['AGE_RANGE'] = pd.cut(df['AGE_YEARS'], 
                            bins=[0, 27, 40, 50, 65, 99], 
                            labels=[1, 2, 3, 4, 5])
    
    # External source features
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3
    
    for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
        feature_name = f'EXT_SOURCES_{function_name.upper()}'
        df[feature_name] = eval(f'np.{function_name}')(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    
    return df


def add_more_features(df):
    """Add advanced engineered features like polynomial features and ratios"""
    
    # Polynomial features for EXT_SOURCE
    '''
    Captures interaction effects between external scores
    '''
    df['EXT_SOURCE_1_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_1_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_2_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1_2_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    
    # Credit ratios
    '''
    Measures credit burden and repayment capacity
    '''
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Income ratios
    '''
    Indicates income stability and earning potential
    '''
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    
    # Percentage features
    df['CREDIT_TO_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'] * 100
    df['ANNUITY_TO_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'] * 100
    
    # Time ratios
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_PER_AGE'] = df['AMT_INCOME_TOTAL'] / abs(df['DAYS_BIRTH'])
    
    return df

def add_group_statistics(df):
    """Add statistical aggregations by groups"""
    group = ['ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'CODE_GENDER']
    
    # Group statistics for EXT_SOURCES_MEAN
    df = do_mean(df, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_MEAN')
    df = do_std(df, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_STD')
    
    # Group statistics for income
    df = do_mean(df, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_MEAN')
    df = do_std(df, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_STD')
    
    # Group statistics for credit ratios
    df = do_mean(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_MEAN')
    df = do_std(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_STD')
    
    return df

def do_mean(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return df

def do_std(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].std().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return df

# Apply feature engineering
print("Applying feature engineering...")
data = add_application_features(data)
data = add_more_features(data)
data = add_group_statistics(data)
print("Feature engineering completed.")


data['INCOME_TO_EMPLOYED_RATIO'] = data['INCOME_TO_EMPLOYED_RATIO'].replace([np.inf, -np.inf], np.nan)


def handle_missing_values(df):
    """Drop columns with high percentage of missing values"""
    # Drop columns with more than 58% missing values
    threshold = 0.58
    cols_to_drop = df.columns[df.isnull().mean() > threshold]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped columns with more than {threshold*100}% missing values: {list(cols_to_drop)}")
    return df

def encode_categorical_features(df):
    """Encode categorical features before imputation"""
    # Get the categorical columns (type object)
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    print(f"\nFound {len(cat_cols)} categorical columns to encode: {list(cat_cols)}")

    for col in cat_cols:
        # If column has only two unique values, apply Label Encoding
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
            print(f"Label Encoding applied to column: {col}")
        else:
            # Apply One-Hot Encoding for multi-category columns
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            print(f"One-Hot Encoding applied to column: {col}")

    # Convert any Boolean columns to integers (0 and 1)
    for col in df.select_dtypes(include=[bool]).columns:
        df[col] = df[col].astype(int)

    return df

def impute_missing_values(df):
    """Impute missing values using MissForest"""
    # Identify columns with more than 15% missing values
    high_missing_cols = df.columns[df.isnull().mean() > 0.15]
    low_missing_cols = df.columns[(df.isnull().mean() > 0) & (df.isnull().mean() <= 0.15)]

    # Use MissForest for imputation of columns with more than 15% missing values
    if len(high_missing_cols) > 0:
        mf = MissForest()
        df_imputed = mf.fit_transform(df[high_missing_cols])
        df[high_missing_cols] = df_imputed

    # Use simple method for imputation of columns with less than 15% missing values
    if len(low_missing_cols) > 0:
        for col in low_missing_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    return df

# Initial missing values check
print("Initial missing values:")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Step 1: Handle high-missing columns
print("\nStep 1: Handling columns with high missing values...")
data = handle_missing_values(data)

# Step 2: Encode categorical features
print("\nStep 2: Encoding categorical features...")
data = encode_categorical_features(data)

# Step 3: Impute remaining missing values
print("\nStep 3: Imputing remaining missing values...")
data = impute_missing_values(data)

# Verify no missing values remain
print("\nFinal missing values check:")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Assert that all missing values have been handled
assert data.isnull().sum().sum() == 0, "There are still missing values in the dataset"


def detect_and_handle_outliers_isolation_forest(df, contamination=0.01):
    """
    Detects and handles outliers in specified columns using Isolation Forest.
    """
    df_clean = df.copy()
    
    # Define columns for outlier detection based on business logic and importance
    columns_to_check = {
        # Credit-related amounts (use wider bounds as these can vary significantly)
        'AMT_CREDIT': {'quantile_low': 0.01, 'quantile_high': 0.99},
        'AMT_ANNUITY': {'quantile_low': 0.01, 'quantile_high': 0.99},
        'AMT_GOODS_PRICE': {'quantile_low': 0.01, 'quantile_high': 0.99},
        
        # Income and financial ratios (use standard bounds)
        'AMT_INCOME_TOTAL': {'quantile_low': 0.05, 'quantile_high': 0.95},
        'CREDIT_TO_ANNUITY_RATIO': {'quantile_low': 0.05, 'quantile_high': 0.95},
        'CREDIT_TO_GOODS_RATIO': {'quantile_low': 0.05, 'quantile_high': 0.95},
        'ANNUITY_TO_INCOME_RATIO': {'quantile_low': 0.05, 'quantile_high': 0.95},
        'CREDIT_TO_INCOME_RATIO': {'quantile_low': 0.05, 'quantile_high': 0.95},
        
        # External source scores (should be between 0 and 1)
        'EXT_SOURCE_1': {'quantile_low': 0, 'quantile_high': 1},
        'EXT_SOURCE_2': {'quantile_low': 0, 'quantile_high': 1},
        'EXT_SOURCE_3': {'quantile_low': 0, 'quantile_high': 1},
        
        # Time-related features (use conservative bounds)
        'DAYS_EMPLOYED': {'quantile_low': 0.01, 'quantile_high': 0.99},
        'DAYS_REGISTRATION': {'quantile_low': 0.01, 'quantile_high': 0.99},
        'DAYS_ID_PUBLISH': {'quantile_low': 0.01, 'quantile_high': 0.99},
        
        # Payment ratios (use standard bounds)
        'PAYMENT_RATE': {'quantile_low': 0.05, 'quantile_high': 0.95},
        'ANNUITY_TO_CREDIT_RATIO': {'quantile_low': 0.05, 'quantile_high': 0.95},
        
        # Income-related ratios (use standard bounds)
        'INCOME_TO_EMPLOYED_RATIO': {'quantile_low': 0.05, 'quantile_high': 0.95},
        'INCOME_TO_BIRTH_RATIO': {'quantile_low': 0.05, 'quantile_high': 0.95},
        'INCOME_PER_PERSON': {'quantile_low': 0.05, 'quantile_high': 0.95},
        'INCOME_PER_CHILD': {'quantile_low': 0.05, 'quantile_high': 0.95}
    }
    
    print(f"Processing {len(columns_to_check)} selected columns for outliers")
    
    for column, bounds in columns_to_check.items():
        if column in df.columns:  # Check if column exists
            print(f"\nProcessing column: {column}")
            
            # Extract the data for the current column
            data_col = df[column].values.reshape(-1, 1)
            
            # Create and fit the Isolation Forest model
            clf = IsolationForest(contamination=contamination, random_state=42)
            clf.fit(data_col)
            
            # Predict outliers
            outliers_pred = clf.predict(data_col)
            
            # Identify outlier indices
            outlier_indices = np.where(outliers_pred == -1)[0]
            print(f"Number of outliers detected in {column}: {len(outlier_indices)}")
            
            if len(outlier_indices) > 0:
                if isinstance(bounds['quantile_low'], float) and isinstance(bounds['quantile_high'], float):
                    # For fixed bounds (like EXT_SOURCE)
                    lower_bound = bounds['quantile_low']
                    upper_bound = bounds['quantile_high']
                else:
                    # For quantile-based bounds
                    lower_bound = df[column].quantile(bounds['quantile_low'])
                    upper_bound = df[column].quantile(bounds['quantile_high'])
                
                print(f"Capping outliers in {column} between {lower_bound:.2f} and {upper_bound:.2f}")
                
                # Print statistics before capping
                print(f"Before capping - Mean: {df[column].mean():.2f}, Std: {df[column].std():.2f}")
                
                # Cap outliers while preserving original data type
                df_clean.loc[outlier_indices, column] = np.clip(
                    df_clean.loc[outlier_indices, column],
                    lower_bound, 
                    upper_bound
                ).astype(df[column].dtype)
                
                # Print statistics after capping
                print(f"After capping - Mean: {df_clean[column].mean():.2f}, Std: {df_clean[column].std():.2f}")
    
    return df_clean

# Apply outlier detection and handling
print("Starting outlier detection and handling...")
data = detect_and_handle_outliers_isolation_forest(data)
print("\nOutlier detection and handling completed.")


# Visualization function for selected columns
def plot_outlier_comparison(original_df, cleaned_df, columns_to_plot=None):
    """
    Plot before and after distributions for selected columns
    """
    if columns_to_plot is None:
        columns_to_plot = [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
        ]
    
    # Filter existing columns
    columns_to_plot = [col for col in columns_to_plot if col in original_df.columns]
    
    fig, axes = plt.subplots(len(columns_to_plot), 2, figsize=(15, 5*len(columns_to_plot)))
    
    for idx, column in enumerate(columns_to_plot):
        # Boxplot
        sns.boxplot(data=original_df[column], ax=axes[idx, 0], color='skyblue')
        sns.boxplot(data=cleaned_df[column], ax=axes[idx, 1], color='lightgreen')
        
        axes[idx, 0].set_title(f'{column} - Before Outlier Handling')
        axes[idx, 1].set_title(f'{column} - After Outlier Handling')
        
        # Rotate x-labels if needed
        axes[idx, 0].tick_params(axis='x', rotation=45)
        axes[idx, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Create a copy of the original data for comparison
original_data = data.copy()

# Plot comparison
plot_outlier_comparison(original_data, data)


def save_processed_data(df, filename='processed_application_train.csv'):
    """
    Save the processed data as CSV and print basic information.
    
    Args:
        df: Processed DataFrame
        filename: Name of the CSV file to save
    """
    # Save as CSV
    df.to_csv(filename, index=False)
    
    # Print information about the saved data
    print("\nProcessed data saved successfully!")
    print(f"Filename: {filename}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Print file size
    file_size_mb = os.path.getsize(filename) / (1024*1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
# After all processing is done, save the data
save_processed_data(data)


def clean_feature_names(columns):
    """
    Clean feature names to be compatible with LightGBM
    
    Parameters:
    -----------
    columns : array-like
        Original column names
    
    Returns:
    --------
    list
        Cleaned column names
    """
    # Replace special characters and spaces with underscore
    cleaned_names = []
    for col in columns:
        # Replace special characters and spaces with underscore
        clean_name = (str(col).replace('[', '')
                            .replace(']', '')
                            .replace('<', '')
                            .replace('>', '')
                            .replace(':', '')
                            .replace('"', '')
                            .replace("'", '')
                            .replace(',', '')
                            .replace('?', '')
                            .replace('=', '')
                            .replace(' ', '_')
                            .replace('(', '')
                            .replace(')', ''))
        cleaned_names.append(clean_name)
    return cleaned_names

def prepare_data(data):
    """
    Prepare data for modeling by splitting features/target and scaling
    """
    print("Preparing data for modeling...")
    
    # Separate target and features
    print("Separating target and features...")
    y = data['TARGET']
    X = data.drop(columns=['TARGET', 'SK_ID_CURR'])
    
    # Clean feature names
    X.columns = clean_feature_names(X.columns)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split into training and validation sets
    print("\nSplitting into train and validation sets...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_valid.shape}")

    # Scale features while preserving DataFrame structure
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_valid_scaled = pd.DataFrame(
        scaler.transform(X_valid),
        columns=X_valid.columns,
        index=X_valid.index
    )
    
    # Print class distribution
    print("\nClass distribution:")
    print("Training set:")
    print(y_train.value_counts(normalize=True))
    print("\nValidation set:")
    print(y_valid.value_counts(normalize=True))

    return X_train_scaled, X_valid_scaled, y_train, y_valid


def train_model(X_train, y_train, X_valid, model_name, model, feature_names, n_folds=5):
    """
    Train model using stratified k-fold cross-validation
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_valid : array-like
        Validation features
    model_name : str
        Name of the model
    model : classifier object
        The model to train
    feature_names : array-like
        Names of the features
    n_folds : int, default=5
        Number of cross-validation folds
    """
    
    # Initialize cross-validation
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize arrays for predictions
    oof_predictions = np.zeros(X_train.shape[0])
    valid_predictions = np.zeros(X_valid.shape[0])
    
    # Initialize feature importance dataframe
    feature_importance_df = pd.DataFrame()
    
    # Cross-validation loop
    for fold_, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print(f'\nFold {fold_ + 1}')
        
        # Split data for this fold
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Train model based on type
        if model_name == "LightGBM":
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                eval_metric='auc',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=200)
                ]
            )
        elif model_name == "XGBoost":
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                early_stopping_rounds=100,
                verbose=200
            )
        else:
            model.fit(X_fold_train, y_fold_train)
        
        # Generate predictions
        oof_predictions[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        valid_predictions += model.predict_proba(X_valid)[:, 1] / n_folds
        
        # Record feature importance
        if model_name == "Logistic Regression":
            importance = np.abs(model.coef_[0])
        else:
            importance = model.feature_importances_
            
        fold_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
            "fold": fold_ + 1
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)
        
        # Print fold metrics
        print(f'Fold {fold_ + 1} AUC: {roc_auc_score(y_fold_val, oof_predictions[val_idx]):.4f}')
    
    print(f'\nFull OOF AUC: {roc_auc_score(y_train, oof_predictions):.4f}')
    
    return oof_predictions, valid_predictions, feature_importance_df, model


# Prepare the data
X_train_scaled, X_valid_scaled, y_train, y_valid = prepare_data(data)

# Calculate class weight for imbalanced dataset
pos_weight = float(len(y_train[y_train==0])) / len(y_train[y_train==1])
print(f"\nPositive class weight: {pos_weight:.2f}")

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, 
        class_weight='balanced', 
        random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        learning_rate=0.005134,
        n_estimators=10000,
        max_depth=10,
        subsample=1,
        colsample_bytree=0.508716,
        reg_alpha=0.436193,
        reg_lambda=0.479169,
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc'
    ),
    "LightGBM": lgb.LGBMClassifier(
        boosting_type='goss',
        n_estimators=10000,
        learning_rate=0.005134,
        num_leaves=54,
        max_depth=10,
        subsample_for_bin=240000,
        reg_alpha=0.436193,
        reg_lambda=0.479169,
        colsample_bytree=0.508716,
        min_split_gain=0.024766,
        subsample=1,
        random_state=42
    )
}


# Store results
results = {}
feature_importances = {}
final_models = {}

# Train each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    oof_pred, valid_pred, feature_importance, final_model = train_model(
        X_train_scaled, 
        y_train, 
        X_valid_scaled,
        model_name,
        model,
        feature_names=X_train_scaled.columns  # Pass the feature names
    )
    
    # Store results
    results[model_name] = {
        'oof_predictions': oof_pred,
        'valid_predictions': valid_pred,
        'valid_auc': roc_auc_score(y_valid, valid_pred)
    }
    feature_importances[model_name] = feature_importance
    final_models[model_name] = final_model


# Print final results
print("\nFinal Results:")
for model_name in models.keys():
    auc_score = results[model_name]['valid_auc']
    print(f"{model_name} Final AUC: {auc_score:.4f}")


# Plot ROC curves
plt.figure(figsize=(10, 8))
for model_name in models.keys():
    valid_pred = results[model_name]['valid_predictions']
    auc_score = results[model_name]['valid_auc']
    
    fpr, tpr, _ = roc_curve(y_valid, valid_pred)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend()
plt.grid(True)
plt.show()


# Plot feature importance for each model
for model_name in models.keys():
    # Calculate mean importance across folds
    mean_importance = feature_importances[model_name].groupby('feature')['importance'].mean().reset_index()
    mean_importance = mean_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=mean_importance.head(20))
    plt.title(f'Top 20 Most Important Features - {model_name}')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop 10 Most Important Features for {model_name}:")
    print(mean_importance.head(10))


# Save best model (assuming LightGBM is best based on previous results)
best_model = final_models["LightGBM"]
joblib.dump(best_model, 'best_model_lightgbm.pkl')

# Save feature importance for best model
feature_importances["LightGBM"].to_csv('feature_importance_lightgbm.csv', index=False)

print("\nResults saved successfully!")

