import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from scipy.stats import yeojohnson
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb
import joblib
warnings.filterwarnings('ignore')


class CFG:
    root_dir = '/kaggle/input/playground-series-s4e1'
    train_dir = os.path.join(root_dir, 'train.csv')
    test_dir = os.path.join(root_dir, 'test.csv')


def read_data(path, is_train = True):
    df = pd.read_csv(path)
    df.pop('id')
    df.pop('Surname')
    Id = df.pop('CustomerId')
    return Id, df

train_id, train_df = read_data(CFG.train_dir)
test_id, test_df = read_data(CFG.test_dir)


train_df.head()


# def data_exploration(df):
#     """
#     Comprehensive exploratory data analysis function that includes:
#     - Basic statistics
#     - Distribution analysis
#     - Correlation analysis
#     - Skewness detection and correction
#     - Visualization of distributions before and after skew correction
    
#     Parameters:
#     df (pandas.DataFrame): Input dataframe for analysis
    
#     Returns:
#     dict: Dictionary containing original dataframe, skew-corrected dataframe, 
#           and statistical summaries
#     """
#     results = {}
#     results['original_df'] = df.copy()
    
#     # 1. Basic Statistics
#     print("=" * 50)
#     print("BASIC STATISTICS")
#     print("=" * 50)
    
#     print("\nDataset Shape:", df.shape)
#     print("\nFirst 5 rows:")
#     print(df.head())
    
#     print("\nData Types:")
#     print(df.dtypes)
    
#     print("\nSummary Statistics:")
#     print(df.describe().T)
    
#     # 2. Missing Values
#     print("\n" + "=" * 50)
#     print("MISSING VALUES")
#     print("=" * 50)
#     missing = df.isnull().sum()
#     missing_percent = (missing / len(df)) * 100
#     missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_percent})
#     print(missing_df[missing_df['Count'] > 0])
    
#     # 3. Categorical Variables Analysis
#     print("\n" + "=" * 50)
#     print("CATEGORICAL VARIABLES")
#     print("=" * 50)
    
#     categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#     for col in categorical_cols:
#         print(f"\nCounts for {col}:")
#         print(df[col].value_counts())
#         print(f"Percentage for {col}:")
#         print(df[col].value_counts(normalize=True) * 100)
    
#     # 4. Target Variable Analysis
#     if 'Exited' in df.columns:
#         print("\n" + "=" * 50)
#         print("TARGET VARIABLE ANALYSIS")
#         print("=" * 50)
#         print("\nExited distribution:")
#         print(df['Exited'].value_counts())
#         print("\nExited percentage:")
#         print(df['Exited'].value_counts(normalize=True) * 100)
    
#     # 5. Correlation Analysis
#     print("\n" + "=" * 50)
#     print("CORRELATION ANALYSIS")
#     print("=" * 50)
    
#     # Calculate correlation matrix for numeric columns
#     numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
#     correlation_matrix = df[numeric_cols].corr()
    
#     # Plot correlation heatmap
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
#     plt.title('Correlation Heatmap of Numeric Variables', fontsize=16)
#     plt.tight_layout()
#     plt.show()
    
#     # 6. Skewness Analysis
#     print("\n" + "=" * 50)
#     print("SKEWNESS ANALYSIS")
#     print("=" * 50)
    
#     skewed_cols = []
#     skewness_before = {}
#     skewness_after = {}
    
#     # Create a copy of the DataFrame for skew correction
#     df_corrected = df.copy()
    
#     for col in numeric_cols:
#         # Calculate skewness
#         skew_val = df[col].skew()
#         skewness_before[col] = skew_val
        
#         # Check if column is significantly skewed
#         if abs(skew_val) > 0.3:
#             skewed_cols.append(col)
            
#     print(f"Skewed columns: {skewed_cols}")
#     print("\nSkewness values before correction:")
#     for col, skew in skewness_before.items():
#         print(f"{col}: {skew:.4f}")
    
#     # Create multi-panel plot for skewed columns
#     if skewed_cols:
#         # Plot original distributions
#         fig, axes = plt.subplots(len(skewed_cols), 2, figsize=(16, 4*len(skewed_cols)))
#         fig.suptitle('Distribution Before and After Skew Correction', fontsize=16)
        
#         for i, col in enumerate(skewed_cols):
#             # Skip columns with inappropriate values for log transformation
#             if df[col].min() <= 0:
#                 print(f"Skipping {col} for log transformation as it contains zero or negative values")
#                 continue
                
#             # Original distribution
#             sns.histplot(df[col], kde=True, ax=axes[i, 0])
#             axes[i, 0].set_title(f'Original Distribution of {col}')
#             axes[i, 0].set_xlabel(col)
            
#             # Apply log transformation
#             df_corrected[f'{col}_log'] = np.log1p(df[col])
            
#             # After transformation distribution
#             sns.histplot(df_corrected[f'{col}_log'], kde=True, ax=axes[i, 1])
#             axes[i, 1].set_title(f'Log-Transformed Distribution of {col}')
#             axes[i, 1].set_xlabel(f'{col}_log')
            
#             # Calculate new skewness
#             skewness_after[col] = df_corrected[f'{col}_log'].skew()
        
#         plt.tight_layout()
#         plt.subplots_adjust(top=0.95)
#         plt.show()
        
#         print("\nSkewness values after correction:")
#         for col, skew in skewness_after.items():
#             print(f"{col} (log-transformed): {skew:.4f}")
    
#     # 7. Bivariate Analysis with Target
#     if 'Exited' in df.columns:
#         print("\n" + "=" * 50)
#         print("BIVARIATE ANALYSIS WITH TARGET")
#         print("=" * 50)
        
#         # For numeric variables
#         numeric_cols_bivariate = [col for col in numeric_cols if col != 'Exited']
        
#         # Plot distributions by target
#         for col in numeric_cols_bivariate:
#             plt.figure(figsize=(10, 6))
#             sns.boxplot(x='Exited', y=col, data=df)
#             plt.title(f'Distribution of {col} by Exited Status')
#             plt.show()
            
#             # Calculate mean values by target
#             grouped = df.groupby('Exited')[col].mean()
#             print(f"\nMean {col} by Exited Status:")
#             print(grouped)
            
#         # For categorical variables
#         for col in categorical_cols:
#             plt.figure(figsize=(10, 6))
#             pd.crosstab(df[col], df['Exited'], normalize='index').plot(kind='bar', stacked=True)
#             plt.title(f'Proportion of Exited by {col}')
#             plt.show()
    
# data_exploration(train_df)
# data_exploration(test_df)


train_df.info()


def simple_features_engineering(df):
    df = df.copy()
    
    # Create new columns
    df['HasZeroBalance'] = (df['Balance'] == 0).astype(int)
    df['HasMultipleProducts'] = (df['NumOfProducts'] > 1).astype(int)
    
    # Skewness transformation
    df['Age_log'] = pd.to_numeric(df['Age'], errors='coerce')
    transformed_age, lambda_value = yeojohnson(df['Age_log'].fillna(df['Age_log'].median()))
    df['Age_log'] = transformed_age

    # Interaction features
    df['BalancePerTenure'] = df['Balance'] / df['Tenure'].replace(0, np.nan)
    df['BalancePerTenure'].fillna(0, inplace=True)
    df['BalancePerProduct'] = df['Balance'] / df['NumOfProducts'].replace(0, np.nan)
    df['BalancePerProduct'].fillna(0, inplace=True)
    
    df['ProductsPerTenure'] = df['NumOfProducts'] / df['Tenure'].replace(0, np.nan)
    df['ProductsPerTenure'].fillna(0, inplace=True)
    df['ProductsTimesTenure'] = df['NumOfProducts'] * df['Tenure']

    df['CreditScorePerAge'] = df['CreditScore'] / df['Age'].replace(0, np.nan)
    df['CreditScorePerAge'].fillna(0, inplace=True)
    df['BalancePerAge'] = df['Balance'] / df['Age'].replace(0, np.nan)
    df['BalancePerAge'].fillna(0, inplace=True)
    df['EstimatedSalaryPerAge'] = df['EstimatedSalary'] / df['Age'].replace(0, np.nan)
    df['EstimatedSalaryPerAge'].fillna(0, inplace=True)
    df['EstimatedSalaryTimeAge'] = df['EstimatedSalary'] * df['Age']
    df['EstimatedSalaryPerCreditScore'] = df['EstimatedSalary'] / df['CreditScore'].replace(0, np.nan)
    df['EstimatedSalaryPerCreditScore'].fillna(0, inplace=True)
    df['BalancePerCreditScore'] = df['Balance'] / df['CreditScore'].replace(0, np.nan)
    df['BalancePerCreditScore'].fillna(0, inplace=True)
    
    # Added features: Binning for better categorization
    # Age binning
    df['AgeBin'] = pd.qcut(df['Age'], q=4, labels=False)
    
    # Credit Score binning
    df['CreditScoreBin'] = pd.qcut(df['CreditScore'], q=5, labels=False)
    
    # Balance binning (excluding zeros)
    non_zero_balance = df[df['Balance'] > 0]['Balance']
    if len(non_zero_balance) > 0:
        bins = [0] + list(pd.qcut(non_zero_balance, q=3, retbins=True)[1][1:])
        df['BalanceBin'] = pd.cut(df['Balance'], bins=bins, labels=False, include_lowest=True)
        df['BalanceBin'].fillna(0, inplace=True)
    else:
        df['BalanceBin'] = 0
    
    # New Aggregation features
    # Composite financial health score
    df['FinancialHealthScore'] = (
        df['CreditScore'] / 1000 +
        df['Balance'] / df['EstimatedSalary'].replace(0, np.nan) +
        df['Tenure'] / 10
    ).fillna(0)
    
    # Customer Value Index
    df['CustomerValueIndex'] = (
        df['Balance'] * df['Tenure'] * df['NumOfProducts'] / 
        df['EstimatedSalary'].replace(0, np.nan)
    ).fillna(0)
    
    # Risk Profile Score
    df['RiskProfileScore'] = (
        (1000 - df['CreditScore']) / 1000 * 
        df['NumOfProducts'] /
        df['Tenure'].replace(0, np.nan)
    ).fillna(0)
    
    # Label encoding for Gender column
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    if 'Gender' in df.columns:
        df['Gender_encoded'] = le.fit_transform(df['Gender'])
    
    # One-hot encoding for Geography column
    if 'Geography' in df.columns:
        geography_dummies = pd.get_dummies(df['Geography'], prefix='Geography')
        df = pd.concat([df, geography_dummies], axis=1)

    # Imputation with KNN
    group_of_four = ['Age', 'CreditScore']
    group_of_five = ['EstimatedSalary', 'Balance']

    df_4 = df[group_of_four].copy()
    df_5 = df[group_of_five].copy()

    imputer4 = KNNImputer(n_neighbors=4, weights="uniform", missing_values=np.nan)
    imputer5 = KNNImputer(n_neighbors=5, weights="uniform", missing_values=np.nan)

    df_4_imputed = pd.DataFrame(imputer4.fit_transform(df_4), columns=group_of_four, index=df.index)
    df_5_imputed = pd.DataFrame(imputer5.fit_transform(df_5), columns=group_of_five, index=df.index)

    df['Age_imputed'] = df_4_imputed['Age']
    df['CreditScore_imputed'] = df_4_imputed['CreditScore']
    df['Balance_imputed'] = df_5_imputed['Balance']
    df['EstimatedSalary_imputed'] = df_5_imputed['EstimatedSalary']

    # Base numerical features for scaling and PCA
    base_numerical_cols = [
        'CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Tenure', 'NumOfProducts'
    ]
    
    # Extended numerical features for scaling
    extended_numerical_cols = [
        # Original numerical columns
        'CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Tenure', 'NumOfProducts',
        
        # Imputed columns
        'Age_imputed', 'CreditScore_imputed', 'Balance_imputed', 'EstimatedSalary_imputed',
        
        # Derived features
        'BalancePerTenure', 'BalancePerProduct', 'ProductsPerTenure', 'ProductsTimesTenure',
        'CreditScorePerAge', 'BalancePerAge', 'EstimatedSalaryPerAge', 'Age_log', 'EstimatedSalaryTimeAge',
        'EstimatedSalaryPerCreditScore', 'BalancePerCreditScore', 'FinancialHealthScore',
        'CustomerValueIndex', 'RiskProfileScore'
    ]
    
    # Add Gender_encoded to the columns to scale if it exists
    if 'Gender_encoded' in df.columns:
        extended_numerical_cols.append('Gender_encoded')
        
    cols_to_scale = [col for col in extended_numerical_cols if col in df.columns]
    
    # Scaling
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaled_cols = pd.DataFrame(scaler.fit_transform(df[cols_to_scale]), 
                              columns=[col + '_scaled' for col in cols_to_scale], 
                              index=df.index)
    df = pd.concat([df, scaled_cols], axis=1)
    
    # PCA on base numerical features
    from sklearn.decomposition import PCA
    base_cols_for_pca = [col for col in base_numerical_cols if col in df.columns]
    
    # Ensure we have enough columns for PCA
    if len(base_cols_for_pca) >= 2:
        # Use scaled versions for PCA
        scaled_cols_for_pca = [col + '_scaled' for col in base_cols_for_pca if col + '_scaled' in df.columns]
        
        # Calculate the number of components (min of 2 or number of columns)
        n_components = min(len(scaled_cols_for_pca), 3)
        
        if n_components >= 2:
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(df[scaled_cols_for_pca])
            
            # Add PCA components to dataframe
            pca_cols = pd.DataFrame(
                pca_result,
                columns=[f'PCA_{i+1}' for i in range(n_components)],
                index=df.index
            )
            df = pd.concat([df, pca_cols], axis=1)
            
            # Add explained variance as a feature
            for i in range(n_components):
                df[f'PCA_{i+1}_variance'] = pca.explained_variance_ratio_[i]
    
    # Drop original categorical columns
    if 'Geography' in df.columns:
        df = df.drop(['Geography'], axis=1)
    if 'Gender' in df.columns:
        df = df.drop(['Gender'], axis=1)

    return df


def build_ensemble_and_generate_submission(train_df, test_df, output_file='submission.csv'):
    """
    Build an ensemble model from three different classifiers and generate submission file
    
    Parameters:
    train_df (DataFrame): Training data with features and target
    test_df (DataFrame): Test data for prediction
    output_file (str): Filename for the submission CSV
    
    Returns:
    dict: Contains models, weights, threshold, and submission dataframe
    """
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    import xgboost as xgb
    
    print("Starting the model building process...")
    
    # Ensure target column exists in training data
    if 'Exited' not in train_df.columns:
        raise ValueError("Target column 'Exited' not found in training data")
    
    # Process data
    print("Applying feature engineering to training data...")
    train_processed = simple_features_engineering(train_df)
    
    # Split features and target
    X = train_processed.drop(['Exited'], axis=1)
    y = train_processed['Exited']
    
    # Drop any ID columns if they exist
    if 'CustomerId' in X.columns:
        X = X.drop(['CustomerId'], axis=1)
    if 'RowNumber' in X.columns:
        X = X.drop(['RowNumber'], axis=1)
    
    print(f"Training data shape: {X.shape}")
    
    # Define our models
    model1 = RandomForestClassifier(
        n_estimators=1000, 
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42
    )
    
    model2 = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        min_samples_split=15,
        min_samples_leaf=8,
        subsample=0.8,
        random_state=42
    )
    
    model3 = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=2.0,  # To handle class imbalance
        random_state=42
    )
    
    # Train models on full training data
    print("Training model 1: Random Forest...")
    model1.fit(X, y)
    
    print("Training model 2: Gradient Boosting...")
    model2.fit(X, y)
    
    print("Training model 3: XGBoost...")
    model3.fit(X, y)
    
    # Get predictions on training data for weighting
    train_pred1 = model1.predict_proba(X)[:, 1]
    train_pred2 = model2.predict_proba(X)[:, 1]
    train_pred3 = model3.predict_proba(X)[:, 1]
    
    # Calculate individual model metrics on training data
    roc1 = roc_auc_score(y, train_pred1)
    roc2 = roc_auc_score(y, train_pred2)
    roc3 = roc_auc_score(y, train_pred3)
    
    print(f"Model 1 (Random Forest) ROC-AUC: {roc1:.4f}")
    print(f"Model 2 (Gradient Boosting) ROC-AUC: {roc2:.4f}")
    print(f"Model 3 (XGBoost) ROC-AUC: {roc3:.4f}")
    
    # Calculate weighted ensemble based on training performance
    total_score = roc1 + roc2 + roc3
    weight1 = roc1 / total_score
    weight2 = roc2 / total_score
    weight3 = roc3 / total_score
    
    # Combine predictions with weighted average
    train_ensemble_pred = (weight1 * train_pred1 + weight2 * train_pred2 + weight3 * train_pred3)
    ensemble_roc = roc_auc_score(y, train_ensemble_pred)
    print(f"Ensemble Model ROC-AUC: {ensemble_roc:.4f}")
    
    # Binary predictions using threshold that maximizes performance
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, train_ensemble_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    train_ensemble_binary = (train_ensemble_pred >= optimal_threshold).astype(int)
    ensemble_accuracy = accuracy_score(y, train_ensemble_binary)
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print("\nClassification Report for Ensemble:\n")
    print(classification_report(y, train_ensemble_binary))
    
    # Process test data
    print("Applying feature engineering to test data...")
    test_processed = simple_features_engineering(test_df)
    
    # Save customer ID for submission file
    customer_ids = None
    if 'CustomerId' in test_processed.columns:
        customer_ids = test_processed['CustomerId'].copy()
        test_processed = test_processed.drop(['CustomerId'], axis=1)
    if 'RowNumber' in test_processed.columns:
        if customer_ids is None:
            customer_ids = test_processed['RowNumber'].copy()
        test_processed = test_processed.drop(['RowNumber'], axis=1)
    
    # Ensure test data has same columns as training data
    missing_cols = set(X.columns) - set(test_processed.columns)
    for col in missing_cols:
        test_processed[col] = 0  # Add missing columns with default values
    
    # Ensure column order is the same
    test_processed = test_processed[X.columns]
    
    # Generate predictions from each model
    print("Generating predictions on test data...")
    test_pred1 = model1.predict_proba(test_processed)[:, 1]
    test_pred2 = model2.predict_proba(test_processed)[:, 1]
    test_pred3 = model3.predict_proba(test_processed)[:, 1]
    
    # Create weighted ensemble predictions
    test_ensemble_pred = (weight1 * test_pred1 + weight2 * test_pred2 + weight3 * test_pred3)
    test_ensemble_binary = (test_ensemble_pred >= optimal_threshold).astype(int)
    
    # Create submission dataframe
    print("Creating submission file...")
    if customer_ids is not None:
        submission = pd.DataFrame({
            'CustomerId': customer_ids,
            'Exited': test_ensemble_binary
        })
    else:
        submission = pd.DataFrame({
            'Exited': test_ensemble_binary
        })
    
    # Save submission file
    submission.to_csv(output_file, index=False)
    print(f"Submission file created: {output_file}")
    
    # Save models for future use
    print("Saving models...")
    joblib.dump(model1, 'random_forest_model.pkl')
    joblib.dump(model2, 'gradient_boosting_model.pkl')
    joblib.dump(model3, 'xgboost_model.pkl')
    
    # Return models and submission for further analysis if needed
    return {
        'models': {
            'random_forest': model1,
            'gradient_boosting': model2,
            'xgboost': model3
        },
        'weights': {
            'random_forest': weight1,
            'gradient_boosting': weight2,
            'xgboost': weight3
        },
        'threshold': optimal_threshold,
        'metrics': {
            'roc_auc': ensemble_roc,
            'accuracy': ensemble_accuracy
        },
        'submission': submission
    }


result = build_ensemble_and_generate_submission(
    train_df=train_df,
    test_df=test_df,
    output_file='submission.csv'
)




