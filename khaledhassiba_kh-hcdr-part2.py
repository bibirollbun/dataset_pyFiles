# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score




import time
from tqdm import tqdm

# âœ… Define dataset path
data_path = "/kaggle/input/home-credit-default-risk/"

# âœ… Load datasets (Keep a copy)
datasets = {
    "bureau": pd.read_csv(data_path + 'bureau.csv'),
    "bureau_balance": pd.read_csv(data_path + 'bureau_balance.csv'),
    "pos_cash_balance": pd.read_csv(data_path + 'POS_CASH_balance.csv'),
    "credit_card_balance": pd.read_csv(data_path + 'credit_card_balance.csv'),
    "installment_payments": pd.read_csv(data_path + 'installments_payments.csv'),
    "application_train": pd.read_csv(data_path + 'application_train.csv'),
    "application_test": pd.read_csv(data_path + 'application_test.csv')
}

# âœ… Keep unchanged copies
original_datasets = {name: df.copy() for name, df in datasets.items()}
print("âœ… Datasets loaded successfully!")



def aggregate_dataframe(df, id_column, dataset_name):
    """
    Cleans and aggregates a dataframe:
    - Replaces infinite values with NaN and fills NaNs with 0.
    - Aggregates numeric columns with mean, sum, max, and min.
    - Aggregates categorical columns using the most frequent value.
    - Keeps the ID column for merging but excludes it from aggregation.
    - Applies a prefix to each column based on the dataset name.

    Parameters:
    df (pd.DataFrame): The dataframe to process.
    id_column (str): The column name used for grouping.
    dataset_name (str): Name of the dataset to apply as prefix.

    Returns:
    pd.DataFrame: Aggregated dataframe with flattened column names, keeping the ID column.
    """
    start_time = time.time()
    print(f"ğŸ”„ Aggregating {dataset_name} data...")

    # âœ… Handle missing and infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)

    # âœ… Preserve ID Column for Merging
    id_df = df[[id_column]].drop_duplicates()

    # âœ… Process Numerical Columns
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != id_column]
    if numeric_cols:
        df_numeric_agg = df.groupby(id_column)[numeric_cols].agg(['mean', 'sum', 'max', 'min'])
        df_numeric_agg.columns = [f"{dataset_name}_{'_'.join(col)}" for col in df_numeric_agg.columns]  # Add dataset prefix
        df_numeric_agg.reset_index(inplace=True)
    else:
        df_numeric_agg = pd.DataFrame()

    # âœ… Process Categorical Columns (Fast Mode Calculation)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df_categorical_agg = df.groupby(id_column)[categorical_cols].agg(lambda x: x.value_counts().idxmax() if not x.empty else np.nan)
        df_categorical_agg.columns = [f"{dataset_name}_{col}" for col in df_categorical_agg.columns]  # Add dataset prefix
        df_categorical_agg.reset_index(inplace=True)
    else:
        df_categorical_agg = pd.DataFrame()

    # âœ… Merge Numeric and Categorical Aggregations
    if not df_numeric_agg.empty and not df_categorical_agg.empty:
        df_agg = df_numeric_agg.merge(df_categorical_agg, on=id_column, how='left')
    elif not df_numeric_agg.empty:
        df_agg = df_numeric_agg
    elif not df_categorical_agg.empty:
        df_agg = df_categorical_agg
    else:
        raise ValueError(f"â�Œ No numeric or categorical columns found for {dataset_name}.")

    # âœ… Preserve ID Column for Merging
    df_agg = id_df.merge(df_agg, on=id_column, how='right')

    end_time = time.time()
    print(f"â�³ {dataset_name} aggregation completed in {end_time - start_time:.2f} seconds âœ…")
    
    return df_agg



# âœ… Define the ID columns for each dataset
id_mapping = {
    "bureau_balance": "SK_ID_BUREAU",  # Must be aggregated first, then merged into bureau
    "bureau": "SK_ID_CURR",  # Must be aggregated after bureau_balance is merged
    "pos_cash_balance": "SK_ID_CURR",
    "credit_card_balance": "SK_ID_CURR",
    "installment_payments": "SK_ID_CURR"
}

# âœ… Step 1: Aggregate & Merge `bureau_balance` into `bureau`
print("\nğŸ”„ Aggregating and Merging Bureau Balance into Bureau...\n")
bureau_balance_agg = aggregate_dataframe(datasets["bureau_balance"], id_mapping["bureau_balance"], "bureau_balance")

# âœ… Ensure `SK_ID_BUREAU` exists before merging
if 'SK_ID_BUREAU' not in bureau_balance_agg.columns:
    raise KeyError("â�Œ `SK_ID_BUREAU` is missing in `bureau_balance_agg` before merging!")

# âœ… Merge Bureau Balance Aggregates into Bureau
datasets["bureau"] = datasets["bureau"].merge(bureau_balance_agg, on="SK_ID_BUREAU", how="left")
datasets["bureau"].drop(columns=['SK_ID_BUREAU'], inplace=True)
print("âœ… Bureau balance successfully merged into bureau.")

# âœ… Step 2: Aggregate & Merge `bureau` into `application_train` and `application_test`
print("\nğŸ”„ Aggregating and Merging Bureau into Application Datasets...\n")
bureau_agg = aggregate_dataframe(datasets["bureau"], id_mapping["bureau"], "bureau")

# âœ… Merge into Application Data
datasets["application_train"] = datasets["application_train"].merge(bureau_agg, on="SK_ID_CURR", how="left")
datasets["application_test"] = datasets["application_test"].merge(bureau_agg, on="SK_ID_CURR", how="left")
print("âœ… Bureau data merged into application datasets!")

# âœ… Step 3: Aggregate & Merge Other Datasets
for name in tqdm(["pos_cash_balance", "credit_card_balance", "installment_payments"], desc="ğŸ”„ Aggregating and Merging Other Datasets"):
    agg_df = aggregate_dataframe(datasets[name], id_mapping[name], name)

    # âœ… Merge into Application Data
    datasets["application_train"] = datasets["application_train"].merge(agg_df, on="SK_ID_CURR", how="left")
    datasets["application_test"] = datasets["application_test"].merge(agg_df, on="SK_ID_CURR", how="left")

print("\nâœ… All datasets aggregated and merged successfully! ğŸš€")



# âœ… Define save paths
save_path_train = "/kaggle/working/application_train_processed.csv"
save_path_test = "/kaggle/working/application_test_processed.csv"

# âœ… Save processed datasets
datasets["application_train"].to_csv(save_path_train, index=False)
datasets["application_test"].to_csv(save_path_test, index=False)

print(f"âœ… Processed train data saved to: {save_path_train}")
print(f"âœ… Processed test data saved to: {save_path_test}")
print("ğŸš€ Data saved successfully! You can now reload it in future sessions.")



# âœ… Identify duplicate columns
duplicate_cols_train = application_train.columns[application_train.columns.duplicated()].tolist()
duplicate_cols_test = application_test.columns[application_test.columns.duplicated()].tolist()

if duplicate_cols_train:
    print(f"âš ï¸� Duplicate Columns in Train: {duplicate_cols_train}")
if duplicate_cols_test:
    print(f"âš ï¸� Duplicate Columns in Test: {duplicate_cols_test}")

if not duplicate_cols_train and not duplicate_cols_test:
    print("âœ… No duplicate columns found! Ready for encoding.")
else:
    raise ValueError("â�Œ Duplicate columns detected! Resolve before proceeding.")



import xgboost as xgb
from sklearn.model_selection import train_test_split

# âœ… Define Features and Target
X = application_train.drop(columns=['TARGET', 'SK_ID_CURR'])  # Drop non-feature columns
y = application_train['TARGET']

# âœ… Extract Test Features
X_test = application_test.drop(columns=['SK_ID_CURR'])  # Ensure test set matches training set

# âœ… Split Data into Training and Validation Sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



# âœ… Apply One-Hot Encoding
X_train = pd.get_dummies(X_train, prefix_sep='_', drop_first=True)
X_valid = pd.get_dummies(X_valid, prefix_sep='_', drop_first=True)
X_test = pd.get_dummies(X_test, prefix_sep='_', drop_first=True)

# âœ… Ensure all datasets have the same columns
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

print(f"âœ… One-hot encoding complete! Train shape: {X_train.shape}, Test shape: {X_test.shape}")



# âœ… Initialize XGBoost Model
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,        # Number of trees
    learning_rate=0.05,       # Step size shrinkage
    max_depth=6,              # Maximum tree depth
    subsample=0.8,            # Fraction of samples used per tree
    colsample_bytree=0.8,     # Fraction of features per tree
    eval_metric='auc',        # Evaluation metric
    use_label_encoder=False   # Suppress label encoder warning
)

# âœ… Train the Model
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],  # Validation set
    early_stopping_rounds=100,  # Stop if validation score doesnâ€™t improve after 100 rounds
    verbose=100  # Show progress every 100 iterations
)

print("âœ… Model training complete!")



from sklearn.metrics import roc_auc_score

# âœ… Make Predictions on Validation Set
y_valid_pred = xgb_model.predict_proba(X_valid)[:, 1]  # Probability of class 1

# âœ… Compute AUC Score
auc_score = roc_auc_score(y_valid, y_valid_pred)
print(f"ğŸ�¯ Validation AUC Score: {auc_score:.4f} âœ…")



# âœ… Generate Predictions for Test Set
test_preds = xgb_model.predict_proba(X_test)[:, 1]  # Probability of class 1

# âœ… Prepare Submission File
submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': test_preds})
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file saved as 'submission.csv' ğŸš€âœ…")





