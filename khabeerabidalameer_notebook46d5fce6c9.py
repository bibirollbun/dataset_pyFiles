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


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
import re



warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)




print("Starting the NeurIPS - Open Polymer Prediction 2025 script...")


def is_smiles_string(value):
    if not isinstance(value, str):
        return False
    # Basic SMILES pattern check
    smiles_chars = set('CNOPSFClBrI()[]=#-+@/\\*0123456789cnops')
    return len(value) > 3 and all(c in smiles_chars for c in value)


def extract_smiles_features(smiles_string):
    """Extract basic molecular features from SMILES string."""
    if pd.isna(smiles_string) or not isinstance(smiles_string, str):
        return {
            'length': 0,
            'num_atoms': 0,
            'num_bonds': 0,
            'num_rings': 0,
            'num_aromatic': 0,
            'num_branches': 0,
            'has_nitrogen': 0,
            'has_oxygen': 0,
            'has_sulfur': 0,
            'has_phosphorus': 0,
            'has_fluorine': 0,
            'has_chlorine': 0,
            'has_bromine': 0,
            'num_carbonyls': 0,
            'num_double_bonds': 0,
            'num_triple_bonds': 0,
            'molecular_weight_approx': 0
        }
    
    smiles = str(smiles_string)
    
    # Basic features
    features = {
        'length': len(smiles),
        'num_atoms': len(re.findall(r'[CNOSPF]|Cl|Br', smiles)),
        'num_bonds': smiles.count('-') + smiles.count('=') + smiles.count('#'),
        'num_rings': smiles.count('1') + smiles.count('2') + smiles.count('3') + smiles.count('4'),
        'num_aromatic': smiles.count('c') + smiles.count('n') + smiles.count('o') + smiles.count('s'),
        'num_branches': smiles.count('(') + smiles.count('['),
        'has_nitrogen': 1 if 'N' in smiles or 'n' in smiles else 0,
        'has_oxygen': 1 if 'O' in smiles or 'o' in smiles else 0,
        'has_sulfur': 1 if 'S' in smiles or 's' in smiles else 0,
        'has_phosphorus': 1 if 'P' in smiles else 0,
        'has_fluorine': 1 if 'F' in smiles else 0,
        'has_chlorine': 1 if 'Cl' in smiles else 0,
        'has_bromine': 1 if 'Br' in smiles else 0,
        'num_carbonyls': smiles.count('C=O') + smiles.count('C(=O)'),
        'num_double_bonds': smiles.count('='),
        'num_triple_bonds': smiles.count('#'),
    }
    
    # Approximate molecular weight (very rough estimate)
    carbon_count = smiles.count('C') + smiles.count('c')
    nitrogen_count = smiles.count('N') + smiles.count('n')
    oxygen_count = smiles.count('O') + smiles.count('o')
    sulfur_count = smiles.count('S') + smiles.count('s')
    
    features['molecular_weight_approx'] = (carbon_count * 12 + 
                                         nitrogen_count * 14 + 
                                         oxygen_count * 16 + 
                                         sulfur_count * 32)
    
    return features



def convert_smiles_to_features(df, smiles_columns):
    """Convert SMILES columns to numerical features."""
    print(f"Converting {len(smiles_columns)} SMILES columns to molecular features...")
    
    new_features = []
    
    for col in smiles_columns:
        print(f"  Processing SMILES column: {col}")
        
        # Extract features for each SMILES string
        feature_dicts = []
        for smiles in df[col]:
            features = extract_smiles_features(smiles)
            feature_dicts.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_dicts)
        
        # Rename columns to include original column name
        features_df.columns = [f"{col}_{feat}" for feat in features_df.columns]
        
        # Add to list of new features
        new_features.append(features_df)
        
        print(f"    Generated {len(features_df.columns)} features from {col}")
    
    # Concatenate all new features
    if new_features:
        all_new_features = pd.concat(new_features, axis=1)
        
        # Remove original SMILES columns and add new features
        df_numeric = df.drop(columns=smiles_columns)
        df_numeric = pd.concat([df_numeric, all_new_features], axis=1)
        
        print(f"Total new molecular features created: {len(all_new_features.columns)}")
        return df_numeric
    
    return df


def load_data(train_file='/kaggle/input/neurips-open-polymer-prediction-2025/train.csv', test_file='/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'):
    """Load training and test data from CSV files."""
    try:
        print(f"Loading {train_file} and {test_file}...")
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file '{train_file}' not found.")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file '{test_file}' not found.")
        
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        print("Data loaded successfully.")
        print(f"\nTrain DataFrame shape: {train_df.shape}")
        print(f"Test DataFrame shape: {test_df.shape}")
        print(f"\nTrain DataFrame columns: {train_df.columns.tolist()}")
        
        return train_df, test_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def identify_column_types(df, polymer_properties):
    """Identify different types of columns in the dataset."""
    print("\n=== Column Type Analysis ===")
    
    numeric_columns = []
    smiles_columns = []
    other_columns = []
    
    for col in df.columns:
        if col == 'id' or col in polymer_properties:
            continue
            
        # Sample a few non-null values to check type
        sample_values = df[col].dropna().head(10)
        
        if len(sample_values) == 0:
            other_columns.append(col)
            continue
        
        # Check if any values look like SMILES
        has_smiles = any(is_smiles_string(val) for val in sample_values)
        
        if has_smiles:
            smiles_columns.append(col)
            print(f"  SMILES column: {col}")
            print(f"    Sample: {sample_values.iloc[0]}")
        else:
            # Try to convert to numeric
            try:
                pd.to_numeric(sample_values)
                numeric_columns.append(col)
                print(f"  Numeric column: {col}")
            except:
                other_columns.append(col)
                print(f"  Other column: {col} (type: {type(sample_values.iloc[0])})")
    
    print(f"\nSummary:")
    print(f"  Numeric columns: {len(numeric_columns)}")
    print(f"  SMILES columns: {len(smiles_columns)}")
    print(f"  Other columns: {len(other_columns)}")
    
    return numeric_columns, smiles_columns, other_columns


def prepare_features(train_df, test_df, polymer_properties):
    """Identify and prepare feature columns."""
    print("\n=== Feature Preparation ===")
    
    # Identify column types
    numeric_cols, smiles_cols, other_cols = identify_column_types(train_df, polymer_properties)
    
    # Convert SMILES to features if present
    if smiles_cols:
        print("\nConverting SMILES to molecular features...")
        train_df = convert_smiles_to_features(train_df, smiles_cols)
        test_df = convert_smiles_to_features(test_df, smiles_cols)
        
        # Update numeric columns list (add new molecular features)
        new_molecular_features = [col for col in train_df.columns 
                                if any(smiles_col in col for smiles_col in smiles_cols)]
        numeric_cols.extend(new_molecular_features)
    
    # Handle other non-numeric columns
    if other_cols:
        print(f"\nHandling {len(other_cols)} other columns...")
        for col in other_cols:
            # Try to convert to numeric, otherwise create dummy variables
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
                test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
                numeric_cols.append(col)
                print(f"  Converted {col} to numeric")
            except:
                # Create dummy variables for categorical data
                if train_df[col].nunique() < 50:  # Only if not too many categories
                    train_dummies = pd.get_dummies(train_df[col], prefix=col)
                    test_dummies = pd.get_dummies(test_df[col], prefix=col)
                    
                    # Align dummy columns
                    all_dummy_cols = list(set(train_dummies.columns) | set(test_dummies.columns))
                    for dummy_col in all_dummy_cols:
                        if dummy_col not in train_dummies.columns:
                            train_dummies[dummy_col] = 0
                        if dummy_col not in test_dummies.columns:
                            test_dummies[dummy_col] = 0
                    
                    train_df = pd.concat([train_df.drop(columns=[col]), train_dummies], axis=1)
                    test_df = pd.concat([test_df.drop(columns=[col]), test_dummies], axis=1)
                    numeric_cols.extend(all_dummy_cols)
                    print(f"  Created {len(all_dummy_cols)} dummy variables for {col}")
                else:
                    print(f"  Skipping {col} (too many categories)")
    
    # Final feature columns
    feature_columns = [col for col in numeric_cols if col in train_df.columns and col in test_df.columns]
    
    print(f"\nFinal feature set: {len(feature_columns)} columns")
    if feature_columns:
        print("Sample features:", feature_columns[:10])
        if len(feature_columns) > 10:
            print(f"... and {len(feature_columns) - 10} more")
    
    return train_df, test_df, feature_columns




def analyze_data(train_df, test_df, polymer_properties):
    """Analyze the data and target properties."""
    print("\n=== Data Analysis ===")
    
    print("\nTarget properties analysis:")
    for prop in polymer_properties:
        if prop in train_df.columns:
            non_null_count = train_df[prop].count()
            if non_null_count > 0:
                print(f"  {prop}: {non_null_count} non-null values, "
                      f"mean={train_df[prop].mean():.4f}, "
                      f"std={train_df[prop].std():.4f}, "
                      f"range=[{train_df[prop].min():.4f}, {train_df[prop].max():.4f}]")
            else:
                print(f"  {prop}: No non-null values")
        else:
            print(f"  WARNING: Property '{prop}' not found in training data!")




def calculate_weights(train_df, polymer_properties):
    """Calculate weights for WMAE metric."""
    print("\n=== Calculating WMAE Weights ===")
    
    K = len(polymer_properties)
    n_i_values = {}
    r_i_values = {}
    
    for prop in polymer_properties:
        if prop in train_df.columns:
            n_i_values[prop] = train_df[prop].count()
            if n_i_values[prop] > 0:
                r_i_values[prop] = max(train_df[prop].max() - train_df[prop].min(), 1e-8)
            else:
                r_i_values[prop] = 1.0
        else:
            n_i_values[prop] = 0
            r_i_values[prop] = 1.0
    
    print(f"n_i (non-NaN counts): {n_i_values}")
    print(f"r_i (ranges): {r_i_values}")
    
    sum_sqrt_inv_n = sum(np.sqrt(1 / max(n_i_values[prop], 1)) for prop in polymer_properties)
    if sum_sqrt_inv_n == 0:
        sum_sqrt_inv_n = 1e-9
    
    weights = {}
    for prop in polymer_properties:
        if r_i_values[prop] == 0 or n_i_values[prop] == 0:
            weights[prop] = 0.0
        else:
            term1 = 1 / r_i_values[prop]
            term2 = (K * np.sqrt(1 / n_i_values[prop])) / sum_sqrt_inv_n
            weights[prop] = term1 * term2
        
        print(f"  Weight for {prop}: {weights[prop]:.6f}")
    
    return weights


def impute_and_scale_features(train_df, test_df, feature_columns):
    """Impute missing values and scale features."""
    print(f"\n=== Feature Processing ===")
    
    if not feature_columns:
        print("No features to process!")
        return train_df, test_df, None, None
    
    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    
    # Check for missing values
    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    
    print(f"Missing values in train features: {missing_train}")
    print(f"Missing values in test features: {missing_test}")
    
    # Impute missing values
    imputer = None
    if missing_train > 0 or missing_test > 0:
        imputer = SimpleImputer(strategy='mean')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), 
                              columns=feature_columns, index=X_train.index)
        X_test = pd.DataFrame(imputer.transform(X_test), 
                             columns=feature_columns, index=X_test.index)
        print("Missing values imputed using mean strategy.")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                 columns=feature_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                columns=feature_columns, index=X_test.index)
    
    print("Features scaled using StandardScaler.")
    print(f"Final feature matrix shape - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, imputer, scaler


def train_models_and_predict(train_df, X_train, X_test, test_df, polymer_properties):
    """Train models and make predictions."""
    print("\n=== Model Training and Prediction ===")
    
    models = {}
    predictions_df = pd.DataFrame({'id': test_df['id']})
    
    for prop in polymer_properties:
        print(f"\nTraining model for: {prop}")
        
        if prop not in train_df.columns:
            print(f"  WARNING: Property '{prop}' not found. Predicting 0.0.")
            predictions_df[prop] = 0.0
            continue
        
        y_train = train_df[prop]
        valid_indices = y_train.dropna().index
        
        if len(valid_indices) == 0:
            print(f"  WARNING: No valid training data for '{prop}'. Predicting 0.0.")
            predictions_df[prop] = 0.0
            continue
        
        X_train_prop = X_train.loc[valid_indices]
        y_train_prop = y_train.loc[valid_indices]
        
        print(f"  Training on {len(y_train_prop)} samples with {X_train_prop.shape[1]} features")
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_prop, y_train_prop)
        models[prop] = model
        
        # Cross-validation
        if len(y_train_prop) >= 5:
            cv_scores = cross_val_score(model, X_train_prop, y_train_prop, 
                                       cv=min(5, len(y_train_prop)), 
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=-1)
            print(f"  CV MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Make predictions
        predictions = model.predict(X_test)
        predictions = np.clip(predictions, a_min=0, a_max=None)
        
        predictions_df[prop] = predictions
        
        print(f"  Predictions - Mean: {predictions.mean():.4f}, "
              f"Std: {predictions.std():.4f}, "
              f"Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    return models, predictions_df




def create_submission(predictions_df, polymer_properties, filename='submission.csv'):
    """Create submission file."""
    print(f"\n=== Creating Submission File ===")
    
    submission_columns = ['id'] + polymer_properties
    submission_df = predictions_df[submission_columns]
    
    submission_df.to_csv(filename, index=False)
    
    print(f"Submission file '{filename}' created successfully.")
    print(f"Shape: {submission_df.shape}")
    print("\nFirst 5 rows:")
    print(submission_df.head())
    
    return submission_df


def main():
    """Main execution function."""
    try:
        polymer_properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Load data
        train_df, test_df = load_data()
        
        # Prepare features (handle SMILES and other column types)
        train_df, test_df, feature_columns = prepare_features(train_df, test_df, polymer_properties)
        
        # Analyze data
        analyze_data(train_df, test_df, polymer_properties)
        
        # Calculate weights
        weights = calculate_weights(train_df, polymer_properties)
        
        # Process features
        X_train, X_test, imputer, scaler = impute_and_scale_features(train_df, test_df, feature_columns)
        
        # Train models and predict
        models, predictions_df = train_models_and_predict(train_df, X_train, X_test, test_df, polymer_properties)
        
        # Create submission
        submission_df = create_submission(predictions_df, polymer_properties)
        
        print("\n=== Script completed successfully! ===")
        return models, predictions_df, weights
    
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    models, predictions, weights = main()

