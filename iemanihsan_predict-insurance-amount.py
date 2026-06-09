import pandas as pd
import numpy as np
import warnings
from scipy.special import boxcox, inv_boxcox
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import xgboost as xgb

warnings.filterwarnings("ignore")


# --- 1. Data Loading & Initial EDA ---
print("Loading data...")
train_df = pd.read_csv('/kaggle/input/premiumpulse-risk-modeling/train.csv')
test_df = pd.read_csv('/kaggle/input/premiumpulse-risk-modeling/test.csv')
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"Train null values:\n{train_df.isnull().sum()[train_df.isnull().sum() > 0]}")

# Convert date columns
train_df['Policy Start Date'] = pd.to_datetime(train_df['Policy Start Date'])
test_df['Policy Start Date'] = pd.to_datetime(test_df['Policy Start Date'])


# --- 2. Memory Optimization ---
def reduce_mem_usage(df):
    """Reduce memory usage by converting data types."""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage before optimization: {start_mem:.2f} MB')
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            if df[col].max() < 127 and df[col].min() > -128:
                df[col] = df[col].astype('int8')
            elif df[col].max() < 32767 and df[col].min() > -32768:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization: {end_mem:.2f} MB')
    print(f'Reduced by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df


# --- 3. Missing Value Treatment ---
def handle_missing_values(train, test):
    """Handle missing values in train and test dataframes."""
    print("\nHandling missing values...")
    # Missing value indicators
    for col in ['Credit Score', 'Health Score', 'Annual Income']:
        if col in train.columns:
            train[f'{col}_missing'] = train[col].isnull().astype(np.int8)
        if col in test.columns:
            test[f'{col}_missing'] = test[col].isnull().astype(np.int8)

    # Fill categorical NaNs with "Unknown"
    cat_cols_fill_unknown = ['Occupation', 'Marital Status', 'Customer Feedback']
    for col in cat_cols_fill_unknown:
        if col in train.columns:
            train[col] = train[col].fillna('Unknown')
        if col in test.columns:
            test[col] = test[col].fillna('Unknown')

    # Impute predictor columns before income imputation
    predictor_cols_impute = ['Age', 'Occupation', 'Marital Status', 'Number of Dependents']
    for col in predictor_cols_impute:
        if col in train.columns:
            if train[col].dtype.name in ['object', 'category']:
                mode_val = train[col].mode()[0]
                train[col] = train[col].fillna(mode_val)
                if col in test.columns:
                    test[col] = test[col].fillna(mode_val)
            else:
                median_val = train[col].median()
                train[col] = train[col].fillna(median_val)
                if col in test.columns:
                    test[col] = test[col].fillna(median_val)

    # Regression-based imputation for Annual Income
    if 'Annual Income' in train.columns and train['Annual Income'].isnull().sum() > 0:
        income_features = ['Age', 'Occupation', 'Marital Status', 'Number of Dependents']
        income_train_df = train.dropna(subset=['Annual Income'])
        income_missing_train_df = train[train['Annual Income'].isnull()]

        X_income_train = pd.get_dummies(income_train_df[income_features], drop_first=True)
        y_income_train = income_train_df['Annual Income']

        income_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        income_model.fit(X_income_train, y_income_train)

        if not income_missing_train_df.empty:
            X_missing_train = pd.get_dummies(income_missing_train_df[income_features], drop_first=True)
            for col_name in X_income_train.columns:
                if col_name not in X_missing_train.columns:
                    X_missing_train[col_name] = 0
            X_missing_train = X_missing_train[X_income_train.columns]
            predicted_incomes_train = income_model.predict(X_missing_train)
            train.loc[train['Annual Income'].isnull(), 'Annual Income'] = predicted_incomes_train

        if 'Annual Income' in test.columns and test['Annual Income'].isnull().sum() > 0:
            income_missing_test_df = test[test['Annual Income'].isnull()]
            if not income_missing_test_df.empty:
                X_missing_test = pd.get_dummies(income_missing_test_df[income_features], drop_first=True)
                for col_name in X_income_train.columns:
                    if col_name not in X_missing_test.columns:
                        X_missing_test[col_name] = 0
                X_missing_test = X_missing_test[X_income_train.columns]
                predicted_incomes_test = income_model.predict(X_missing_test)
                test.loc[test['Annual Income'].isnull(), 'Annual Income'] = predicted_incomes_test

    # Simple imputers for other numerical features
    median_impute_cols = ['Age', 'Vehicle Age', 'Insurance Duration', 'Number of Dependents']
    mean_impute_cols = ['Health Score', 'Credit Score']

    for strategy, cols_list in {'median': median_impute_cols, 'mean': mean_impute_cols}.items():
        common_cols_impute = [c for c in cols_list if c in train.columns and c in test.columns and train[c].isnull().any()]
        if common_cols_impute:
            imputer = SimpleImputer(strategy=strategy)
            train[common_cols_impute] = imputer.fit_transform(train[common_cols_impute])
            test[common_cols_impute] = imputer.transform(test[common_cols_impute])

    # Fill Previous Claims NaNs with 0
    if 'Previous Claims' in train.columns:
        train['Previous Claims'] = train['Previous Claims'].fillna(0).astype(np.int16)
    if 'Previous Claims' in test.columns:
        test['Previous Claims'] = test['Previous Claims'].fillna(0).astype(np.int16)

    # Final check for remaining NaNs
    if train.isnull().any().any():
        remaining_nans_train = train.columns[train.isnull().any()].tolist()
        print(f"Warning: Train data still has NaN values in columns: {remaining_nans_train}")
        for col_name in remaining_nans_train:
            fill_val = train[col_name].mode()[0] if train[col_name].dtype.name in ['object', 'category'] else train[col_name].median()
            train[col_name] = train[col_name].fillna(fill_val)
            if col_name in test.columns:
                test[col_name] = test[col_name].fillna(fill_val)
    print("Missing value treatment completed.")
    return train, test


# --- 4. Feature Engineering ---
def create_advanced_features(df):
    """Create new features from existing ones."""
    print("\nCreating advanced features...")
    # Temporal features
    df['Policy_Year'] = df['Policy Start Date'].dt.year
    df['Policy_Month'] = df['Policy Start Date'].dt.month
    df['Policy_Quarter'] = df['Policy Start Date'].dt.quarter
    df['Policy_DayOfWeek'] = df['Policy Start Date'].dt.dayofweek
    df['Policy_IsWeekend'] = df['Policy Start Date'].dt.dayofweek.isin([5, 6]).astype(int)

    # Domain-specific features
    if 'Annual Income' in df.columns and 'Number of Dependents' in df.columns:
        df['Income_per_Dependent'] = df['Annual Income'] / (df['Number of Dependents'] + 1)
        df['Log_Annual_Income'] = np.log1p(df['Annual Income'])
        df['Income_Percentile'] = df['Annual Income'].rank(pct=True)

    # Interaction features
    if 'Age' in df.columns and 'Annual Income' in df.columns :
        df['Age_Income_Ratio'] = df['Age'] / (df['Annual Income'] + 1)
        if 'Vehicle Age' in df.columns:
            df['Age_to_VehicleAge_Ratio'] = df['Age'] / (df['Vehicle Age'] + 1)
            df['Vehicle_Age_Squared'] = df['Vehicle Age'] ** 2

    # Combined risk metrics
    if all(col in df.columns for col in ['Health Score', 'Credit Score']):
        df['Risk_Score'] = df['Health Score'] * 0.4 + df['Credit Score'] * 0.6
        df['Risk_Product'] = df['Health Score'] * df['Credit Score']
        df['Health_Band'] = pd.qcut(df['Health Score'].rank(method='first'), 5, labels=False, duplicates='drop')
        df['Credit_Band'] = pd.qcut(df['Credit Score'].rank(method='first'), 5, labels=False, duplicates='drop')
        df['Combined_Risk_Band'] = df['Health_Band'].astype(str) + '_' + df['Credit_Band'].astype(str) # Changed to string concatenation

    # Insurance-specific metrics
    if 'Insurance Duration' in df.columns:
        df['Insurance_Duration_Squared'] = df['Insurance Duration'] ** 2
        df['Log_Insurance_Duration'] = np.log1p(df['Insurance Duration'])
        df['Customer_Tenure'] = pd.cut(df['Insurance Duration'],
                                       bins=[-1, 1, 3, 5, 10, 100],
                                       labels=['New', 'Early', 'Established', 'Loyal', 'VeryLoyal'])
    # Advanced categoricals
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'],
                                 bins=[0, 25, 35, 45, 55, 65, 100],
                                 labels=['Young', 'Young Adult', 'Adult', 'Middle Aged', 'Senior', 'Elderly'])
    if 'Annual Income' in df.columns:
        df['Income_Group'] = pd.qcut(df['Annual Income'].rank(method='first'),
                                     q=5,
                                     labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                                     duplicates='drop')
    if 'Previous Claims' in df.columns:
        df['Claims_History'] = pd.cut(df['Previous Claims'],
                                      bins=[-1, 0, 1, 2, 100],
                                      labels=['None', 'Low', 'Medium', 'High'])
    # Polynomial features
    if 'Age' in df.columns and 'Annual Income' in df.columns and 'Log_Annual_Income' in df.columns:
        df['Age_Income_Interaction'] = df['Age'] * df['Log_Annual_Income']

    # Segmentation features
    if all(col in df.columns for col in ['Marital Status', 'Number of Dependents']):
        conditions = [
            (df['Marital Status'] == 'Single') & (df['Number of Dependents'] == 0),
            (df['Marital Status'] == 'Single') & (df['Number of Dependents'] > 0),
            (df['Marital Status'] == 'Married') & (df['Number of Dependents'] == 0),
            (df['Marital Status'] == 'Married') & (df['Number of Dependents'] > 0)
        ]
        choices = ['Single_No_Deps', 'Single_Parent', 'Couple_No_Deps', 'Family']
        df['Family_Status'] = np.select(conditions, choices, default='Other')

    # Feature Hashing for Occupation (if present and high cardinality)
    if 'Occupation' in df.columns and df['Occupation'].nunique() > 10: # Arbitrary threshold for "high cardinality"
        from sklearn.feature_extraction import FeatureHasher
        hasher = FeatureHasher(n_features=10, input_type='string')
        hashed_occupation = hasher.transform(df['Occupation'].astype(str).values.reshape(-1, 1))
        hashed_cols = pd.DataFrame(hashed_occupation.toarray(),
                                   columns=[f'Occupation_Hash_{i}' for i in range(10)],
                                   index=df.index)
        df = pd.concat([df, hashed_cols], axis=1)

    print(f"Created {len(df.columns)} total features for this dataframe.")
    return df


# --- 5. Feature Selection ---
def select_best_features(X_train_data, y_train_data, feature_thresh=0.95):
    """Select important features using RandomForestRegressor."""
    print("\nSelecting best features...")
    rf_selector = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

    cat_cols_select = X_train_data.select_dtypes(include=['object', 'category']).columns.tolist()
    X_train_encoded_select = X_train_data.copy()

    if cat_cols_select:
        X_train_encoded_select = pd.get_dummies(X_train_encoded_select, columns=cat_cols_select, drop_first=True, dummy_na=False) # Handle NaNs explicitly if any remain

    rf_selector.fit(X_train_encoded_select, y_train_data)
    importances_select = rf_selector.feature_importances_
    indices_select = np.argsort(importances_select)[::-1]
    feature_names_select = X_train_encoded_select.columns
    sorted_importances_select = importances_select[indices_select]
    cumulative_importances_select = np.cumsum(sorted_importances_select)
    num_features_to_keep_select = np.where(cumulative_importances_select >= feature_thresh)[0][0] + 1
    top_feature_indices_select = indices_select[:num_features_to_keep_select]
    top_features_encoded = [feature_names_select[i] for i in top_feature_indices_select]
    print(f"Selected {len(top_features_encoded)} features (after one-hot encoding) based on {feature_thresh*100}% importance.")

    # Map encoded feature names back to original feature names
    selected_original_features = set()
    for feature_name in top_features_encoded:
        original_name = feature_name.split('_')[0] # Simple split, might need refinement for complex names
        if original_name in X_train_data.columns:
             selected_original_features.add(original_name)
        else: # Handle cases where original name is part of a multi-part dummy
            for orig_col in X_train_data.columns:
                if feature_name.startswith(orig_col + "_"):
                    selected_original_features.add(orig_col)
                    break
            else: # If no direct original column found, keep the encoded one (e.g., hashed features)
                 if feature_name in X_train_data.columns: # Keep if it was an original numerical/already encoded feature
                    selected_original_features.add(feature_name)


    # Ensure all original categorical columns that contributed are included
    for cat_col_original in cat_cols_select:
        if any(encoded_feat.startswith(cat_col_original + "_") for encoded_feat in top_features_encoded):
            selected_original_features.add(cat_col_original)

    # Filter out features that might have been created during encoding but don't exist in original X_train_data
    features_to_keep_final = [f for f in list(selected_original_features) if f in X_train_data.columns]

    print(f"Original features to keep: {len(features_to_keep_final)}. Top 10 (potentially encoded): {', '.join(top_features_encoded[:10])}")
    return features_to_keep_final


# --- 6. Target Transformation ---
def apply_boxcox_robust(series):
    """Apply Box-Cox transformation, fallback to log1p."""
    series_positive = series.copy()
    const_added = 0
    if (series_positive <= 0).any():
        const_added = abs(series_positive.min()) + 1 if series_positive.min() <= 0 else 0
        series_positive += const_added
    try:
        transformed_series, fitted_lambda = boxcox(series_positive)
        print(f"Box-Cox applied with lambda={fitted_lambda:.4f}, constant_added={const_added}")
        return transformed_series, fitted_lambda, const_added
    except Exception as e:
        print(f"Box-Cox failed: {e}. Using log1p transformation.")
        if const_added > 0: # If constant was added for Box-Cox attempt, use it for log1p too
             return np.log1p(series_positive), 0, const_added # Lambda 0 for log
        return np.log1p(series), 0, 0 # Lambda 0 for log, no constant if original series was > 0


# --- Main Execution ---
# 1. Preprocessing
train_df, test_df = handle_missing_values(train_df, test_df)

# 2. Feature Engineering
train_df = create_advanced_features(train_df)
test_df = create_advanced_features(test_df)

# 3. Memory Optimization
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)

# 4. Target Transformation
y_original = train_df['Premium Amount'].copy()
print("\nApplying Box-Cox transformation to target...")
y_bc, lambda_target, const_target = apply_boxcox_robust(y_original)
train_df['Premium Amount_bc'] = y_bc

# 5. Prepare Data for Modeling
print("\nPreparing data for modeling...")
X = train_df.drop(['Premium Amount', 'Premium Amount_bc', 'Policy Start Date', 'id'], axis=1, errors='ignore')
y_train_transformed = train_df['Premium Amount_bc'] # Use the transformed target

# Align columns between train and test before splitting - crucial!
common_cols_train_test = list(set(X.columns) & set(test_df.drop(['Policy Start Date', 'id'], axis=1, errors='ignore').columns))
X = X[common_cols_train_test]
test_features_aligned = test_df[common_cols_train_test].copy()


X_train, X_val, y_train_bc_split, y_val_bc_split = train_test_split(
    X, y_train_transformed, test_size=0.2, random_state=42
)
# Store original y_val for final evaluation
y_val_original_split = y_original.loc[y_val_bc_split.index]


# K-means Clustering (Optional, based on feature count)
numerical_features_for_clustering = X_train.select_dtypes(exclude=['object', 'category', 'datetime']).columns
CLUSTER_FEATURE_NAME = 'cluster_label'

if len(numerical_features_for_clustering) > 30: # Arbitrary threshold
    print(f"Applying K-means clustering to {len(numerical_features_for_clustering)} numerical features.")
    scaler_cluster = StandardScaler()
    X_train_scaled_cluster = scaler_cluster.fit_transform(X_train[numerical_features_for_clustering])
    X_val_scaled_cluster = scaler_cluster.transform(X_val[numerical_features_for_clustering])
    test_scaled_cluster = scaler_cluster.transform(test_features_aligned[numerical_features_for_clustering]) # Scale test data too

    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto') # n_init='auto' is default in newer versions
    X_train[CLUSTER_FEATURE_NAME] = kmeans.fit_predict(X_train_scaled_cluster)
    X_val[CLUSTER_FEATURE_NAME] = kmeans.predict(X_val_scaled_cluster)
    test_features_aligned[CLUSTER_FEATURE_NAME] = kmeans.predict(test_scaled_cluster)

    X_train[CLUSTER_FEATURE_NAME] = X_train[CLUSTER_FEATURE_NAME].astype('category')
    X_val[CLUSTER_FEATURE_NAME] = X_val[CLUSTER_FEATURE_NAME].astype('category')
    test_features_aligned[CLUSTER_FEATURE_NAME] = test_features_aligned[CLUSTER_FEATURE_NAME].astype('category')
    print("Added cluster labels to the datasets.")
else:
    print("Skipping K-means clustering as number of numerical features is not high.")
    kmeans = None # Ensure kmeans is defined for later preprocessing function
    scaler_cluster = None

# Feature Selection
features_to_keep_selected = select_best_features(X_train, y_train_bc_split, feature_thresh=0.90)

X_train_selected = X_train[features_to_keep_selected]
X_val_selected = X_val[features_to_keep_selected]
test_final_selected = test_features_aligned[features_to_keep_selected].copy()


# --- Store Final Processed Data ---
print("\nStoring processed data...")

# Create directories if they don't exist
import os
os.makedirs('/kaggle/working/processed_data', exist_ok=True)

# Store the final training data (with selected features)
train_final = pd.concat([X_train_selected, y_train_bc_split], axis=1)
train_final.to_csv('/kaggle/working/processed_data/train_processed.csv', index=False)
print("Saved processed training data to /kaggle/working/processed_data/train_processed.csv")

# Store the validation data
val_final = pd.concat([X_val_selected, y_val_bc_split], axis=1)
val_final.to_csv('/kaggle/working/processed_data/val_processed.csv', index=False)
print("Saved processed validation data to /kaggle/working/processed_data/val_processed.csv")

# Store the test data
test_final_selected.to_csv('/kaggle/working/processed_data/test_processed.csv', index=False)
print("Saved processed test data to /kaggle/working/processed_data/test_processed.csv")

# Store the target transformation parameters
transform_params = pd.DataFrame({
    'lambda': [lambda_target],
    'constant_added': [const_target]
})
transform_params.to_csv('/kaggle/working/processed_data/target_transform_params.csv', index=False)
print("Saved target transformation parameters to /kaggle/working/processed_data/target_transform_params.csv")

# store the original validation targets for evaluation
y_val_original_split.to_csv('/kaggle/working/processed_data/val_original_targets.csv', header=True)
print("Saved original validation targets to /kaggle/working/processed_data/val_original_targets.csv")

# store the full datasets (including non-selected features)
full_train = pd.concat([X_train, y_train_bc_split], axis=1)
full_train.to_csv('/kaggle/working/processed_data/train_full_processed.csv', index=False)

full_test = test_features_aligned.copy()
full_test.to_csv('/kaggle/working/processed_data/test_full_processed.csv', index=False)
print("Saved full processed datasets (including non-selected features)")

