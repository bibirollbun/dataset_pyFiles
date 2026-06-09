# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
print(train.shape)
train.head()


# Load test data
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
print(test.shape)
test.head()


submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

#submission['accident_risk'] = pred_y_test
#submission.to_csv('submission.csv',index=False)
print(submission.shape)


train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


train.info()


train.describe()


class FrequencyBinningEncoder:
    """
    Add frequency encoding and binning features for ML models.
    
    WHAT IT DOES:
    1. Frequency Encoding: Replaces values with their occurrence frequency
    2. Binning: Groups continuous values into discrete categories
    
    WHY IT HELPS:
    - Captures value importance/rarity
    - Reduces noise from outliers
    - Creates non-linear features from linear ones
    - Handles high-cardinality categorical variables
    """
    
    def __init__(self):
        self.frequency_maps = {}
        self.bin_edges = {}
        
    def add_frequency_features(self, df, columns=None, train_df=None):
        """
        Create frequency encoding features.
        
        Parameters:
        - df: DataFrame to transform
        - columns: list of columns to encode (None = all columns)
        - train_df: training data to calculate frequencies from
        
        """
        df = df.copy()
        
        if columns is None:
            columns = df.columns
        
        # Use train_df to calculate frequencies, or df if not provided
        reference_df = train_df if train_df is not None else df
        
        for col in columns:
            # Calculate frequency of each value
            freq_map = reference_df[col].value_counts(normalize=True).to_dict()
            
            # Store for later use (e.g., on test data)
            self.frequency_maps[col] = freq_map
            
            # Create new frequency column
            freq_col_name = f"{col}_freq"
            df[freq_col_name] = df[col].map(freq_map).fillna(0)
            
            print(f"✓ Created {freq_col_name}")
            print(f"  Range: {df[freq_col_name].min():.4f} to {df[freq_col_name].max():.4f}")
        
        return df
    
    def add_binning_features(self, df, numeric_columns, n_bins_list=[5, 10], 
                            train_df=None, strategy='quantile'):
        """
        Create binning features for numeric columns.
        
        Parameters:
        - df: DataFrame to transform
        - numeric_columns: list of numeric columns to bin
        - n_bins_list: list of bin counts to try (e.g., [5, 10])
        - train_df: training data to calculate bin edges from
        - strategy: 'quantile' or 'uniform'
        
        """
        df = df.copy()
        reference_df = train_df if train_df is not None else df
        
        for col in numeric_columns:
            for n_bins in n_bins_list:
                bin_col_name = f"{col}_bin{n_bins}"
                
                # Calculate bin edges from training data
                if strategy == 'quantile':
                    # Equal frequency bins (each bin has ~same number of samples)
                    _, bin_edges = pd.qcut(reference_df[col], q=n_bins, 
                                          retbins=True, duplicates='drop')
                else:
                    # Equal width bins
                    _, bin_edges = pd.cut(reference_df[col], bins=n_bins, 
                                         retbins=True)
                
                # Store bin edges
                self.bin_edges[bin_col_name] = bin_edges
                
                # Apply binning (labels are integers 0, 1, 2, ...)
                df[bin_col_name] = pd.cut(df[col], bins=bin_edges, 
                                         labels=False, include_lowest=True)
                
                # Handle values outside training range
                df[bin_col_name] = df[bin_col_name].fillna(-1).astype(int)
                
                print(f"✓ Created {bin_col_name}")
                print(f"  Bins: {n_bins}, Range: {df[bin_col_name].min()} to {df[bin_col_name].max()}")
        
        return df
    
    def transform_all(self, df, numeric_columns, categorical_columns=None, 
                     train_df=None, n_bins_list=[5, 10]):
        """
        Apply both frequency encoding and binning.
        
        Parameters:
        - df: DataFrame to transform
        - numeric_columns: columns for binning
        - categorical_columns: columns for frequency encoding (None = all non-numeric)
        - train_df: training data for calculating frequencies/bins
        - n_bins_list: bin counts to create
        """
        print("="*50)
        print("FREQUENCY & BINNING FEATURE ENGINEERING")
        print("="*50)
        
        df = df.copy()
        
        # Frequency encoding for all columns
        if categorical_columns is None:
            categorical_columns = [col for col in df.columns if col not in numeric_columns]
        
        print("\n1. Adding Frequency Features...")
        print("-"*50)
        all_cols = list(set(categorical_columns + numeric_columns))
        df = self.add_frequency_features(df, columns=all_cols, train_df=train_df)
        
        # Binning for numeric columns
        print("\n2. Adding Binning Features...")
        print("-"*50)
        df = self.add_binning_features(df, numeric_columns, n_bins_list=n_bins_list, 
                                      train_df=train_df)
        
        print("\n" + "="*50)
        print(f"COMPLETE: {df.shape[1]} total features")
        print("="*50)
        
        return df


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

cat_cols = train.select_dtypes(include=['object']).columns.tolist()
num_cols = train.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols.remove('loan_paid_back')

#print(cat_cols)


# Initialize encoder
encoder = FrequencyBinningEncoder()

# Transform train and test data
print("\n" + "="*50)
print("TRANSFORMING TRAINING DATA")
print("="*50)
train_transformed = encoder.transform_all(
    train,
    numeric_columns=num_cols,
    categorical_columns=cat_cols,
    train_df=train,  # Calculate from train data
    n_bins_list=[3, 5, 10]
)

print("\n" + "="*50)
print("TRANSFORMING TEST DATA (using train statistics)")
print("="*50)
test_transformed = encoder.transform_all(
    test,
    numeric_columns=num_cols,
    categorical_columns=cat_cols,
    train_df=train,  # Use train data for frequencies/bins!
    n_bins_list=[3, 5, 10]
)
    


ordinal_encode = OrdinalEncoder()

for i in cat_cols:
    train_transformed[i]=ordinal_encode.fit_transform(train_transformed[[i]])
    test_transformed[i]=ordinal_encode.transform(test_transformed[[i]])
train_transformed.head()


test_transformed.head()


X = train_transformed.drop('loan_paid_back', axis=1)
y= train['loan_paid_back']


# Import libraries
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


models = {
    "XGBoost": XGBRegressor(
        subsample=0.8,
        reg_lambda=2,
        reg_alpha=1,
        n_estimators=12000,
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=5,
        random_state=42,
        learning_rate=0.01,
        colsample_bytree=0.6
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=400, 
        learning_rate=0.1,  
        max_depth=-1, 
        random_state=42, 
        n_jobs=-1,
        verbose=-1
    ),
    "CatBoost": CatBoostRegressor(
        n_estimators=3000, 
        learning_rate=0.1,   
        custom_metric="AUC",
        depth=6, 
        random_state=42, 
        verbose=0
    )
}


# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
test_preds = {}

for name, model in models.items():
    fold_score = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        try:
            score = roc_auc_score(y_val, pred)
        except ValueError:
            # If y_val has only one class, skip this fold
            print(f"  Fold {fold}: Skipped (only one class in validation set)")
            continue
        
        fold_score.append(score)
        print(f"Fold {fold}: AUC-ROC = {score:.4f}")

    mean_score = np.mean(fold_score)
    results[name] = {
        'fold_scores': fold_score,
        'mean': mean_score
    }
    print(f"\n {name} Mean AUC-ROC: {mean_score:.4f}")

    # Make the predictions
    print(f"  Making the prediction...")
    model.fit(X, y)
    test_pred = model.predict(test_transformed)
    test_preds[name] = test_pred
    print(f"  Test predictions completed!")
    
# Display final results
print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Mean AUC-ROC': [results[m]['mean'] for m in results.keys()]
})

results_df = results_df.sort_values('Mean AUC-ROC', ascending=False)
print(results_df.to_string(index=False))

# Get AUC-ROC scores for the 3 models
xgb_score = results['XGBoost']['mean']
lgb_score = results['LightGBM']['mean']
cat_score = results['CatBoost']['mean']

# Ensemble predictions
ensemble_test_pred = (test_preds['XGBoost'] * xgb_score + 
                      test_preds['LightGBM'] * lgb_score+ 
                      test_preds['CatBoost'] * cat_score) / (xgb_score+lgb_score+cat_score)
print("✓ Ensemble predictions created using simple average")
print(f"  Number of test predictions: {len(ensemble_test_pred)}")

# Add submission
submission['loan_paid_back'] = ensemble_test_pred
submission.to_csv('submission.csv', index=False)

# Print best model
best_model = results_df.iloc[0]['Model']
best_score = results_df.iloc[0]['Mean AUC-ROC']
print(f"\nBest Model: {best_model} with AUC-ROC = {best_score:.4f}")     

