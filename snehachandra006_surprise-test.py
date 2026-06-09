import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SMART DATA LOADER
# ==============================================================================
def read_las_manual(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        data_start = 0
        headers = []
        for i, line in enumerate(lines):
            if line.strip().startswith('~C'):
                for j in range(i+1, len(lines)):
                    if lines[j].startswith('~'): break
                    if lines[j].strip().startswith('#'): continue
                    parts = lines[j].split()
                    if parts: headers.append(parts[0].replace('.', ''))
            if line.strip().startswith('~A'):
                data_start = i + 1
                break
        if data_start > 0:
            df = pd.read_csv(filename, skiprows=data_start, sep='\s+', header=None)
            if len(headers) == df.shape[1]:
                df.columns = headers
            else:
                df.columns = [f"Col_{k}" for k in range(df.shape[1])]
            return df
    except:
        return None
    return None

def find_files():
    print(">>> STEP 1: LOCATING FILES <<<")
    train_paths = []
    test_path = None
    sample_sub_path = None
    
    for root, dirs, files in os.walk('/kaggle/input'):
        for file in files:
            path = os.path.join(root, file)
            if file == 'Test_Data_2.las': # Priority to the new surprise test
                test_path = path
                print(f"  [FOUND] SURPRISE TEST: {file}")
            elif file.endswith('.las') and 'WELL' in file:
                train_paths.append(path)
            elif file == 'sample_submission.csv':
                sample_sub_path = path
                
    # Fallback if specific file not found, try generic test
    if not test_path:
        for root, dirs, files in os.walk('/kaggle/input'):
            for file in files:
                if 'test' in file.lower() and file.endswith('.las'):
                    test_path = os.path.join(root, file)
                    print(f"  [FOUND] TEST WELL: {file}")
                    break
    
    return train_paths, test_path, sample_sub_path

# ==============================================================================
# 2. LOG-SPACE FEATURE ENGINEERING (The "Exact Math" Method)
# ==============================================================================
def prepare_log_linear_data(df, is_train=True):
    # 1. Clean Magic Numbers
    df = df.replace([-999.25, -999, 9999], np.nan)
    
    # 2. Enforce Physics (No negatives allowed in Log space)
    df['Resistivity'] = df['Resistivity'].mask(df['Resistivity'] <= 0, 0.001)
    df['Gamma'] = df['Gamma'].mask(df['Gamma'] <= 0, 0.001)
    
    # 3. Standardize Porosity
    if df['Porosity'].max() > 1.0:
        df['Porosity'] = df['Porosity'] / 100.0
    df['Porosity'] = df['Porosity'].mask(df['Porosity'] <= 0, 0.0001) # Avoid log(0)

    # 4. CREATE LOG FEATURES (Linearizing the Physics)
    # The Equation: Perm = a * Phi^b * Rt^c * GR^d
    # Log Form: log(Perm) = log(a) + b*log(Phi) + c*log(Rt) + d*log(GR)
    
    df['log_Phi'] = np.log10(df['Porosity'])
    df['log_Res'] = np.log10(df['Resistivity'])
    df['log_GR']  = np.log10(df['Gamma'])
    
    if is_train:
        # Clean Target
        df['Perm'] = df['Perm'].mask(df['Perm'] <= 0, np.nan)
        df = df.dropna(subset=['Perm', 'log_Phi', 'log_Res', 'log_GR'])
        # Target Transform
        df['log_Perm'] = np.log10(df['Perm'])
        
    return df

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

train_paths, test_path, sample_sub_path = find_files()

if train_paths and test_path:
    # 1. Load
    print("\n>>> STEP 2: LOADING DATA <<<")
    train_dfs = [read_las_manual(p) for p in train_paths]
    train_df = pd.concat([df for df in train_dfs if df is not None], ignore_index=True)
    test_df = read_las_manual(test_path)
    
    print(f"Training Rows: {len(train_df)}")
    print(f"Test Rows: {len(test_df)}")

    # 2. Prepare for Linear Regression
    print("\n>>> STEP 3: LOG-LINEAR TRANSFORMATION <<<")
    train_df = prepare_log_linear_data(train_df, is_train=True)
    test_df = prepare_log_linear_data(test_df, is_train=False)
    
    # Impute Test Data (Median for logs)
    features = ['log_Phi', 'log_Res', 'log_GR']
    imputer = SimpleImputer(strategy='median')
    
    # Fill missing values in test set so math doesn't break
    test_df[features] = imputer.fit_transform(test_df[features])

    # 3. Train EXACT Equation Finder (Linear Regression)
    print("\n>>> STEP 4: SOLVING THE EQUATION <<<")
    model = LinearRegression()
    
    X = train_df[features]
    y = train_df['log_Perm'] # We predict Log(Perm) first
    
    model.fit(X, y)
    
    # Print the equation found (For your presentation!)
    print("\n--- THE SECRET FORMULA FOUND ---")
    print(f"Log(K) = {model.intercept_:.4f} + {model.coef_[0]:.4f}*Log(Phi) + {model.coef_[1]:.4f}*Log(Rt) + {model.coef_[2]:.4f}*Log(GR)")
    print("--------------------------------\n")
    
    # Check Internal Error
    log_preds_train = model.predict(X)
    preds_train = 10**log_preds_train # Convert back to real number
    mae = mean_absolute_error(train_df['Perm'], preds_train)
    print(f"Internal MAE using Equation: {mae:.6f} mD")

    # 4. Predict on Test
    log_preds_test = model.predict(test_df[features])
    final_preds = 10**log_preds_test # Convert back from Log scale
    
    # 5. Submission
    print("\n>>> STEP 5: SAVING PREDICTIONS <<<")
    
    # Handle IDs exactly as requested (Depth from LAS)
    submission = pd.DataFrame({
        'id': test_df['DEPT'],
        'Permeability': final_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Saved: submission.csv")
    print("\nCHECK THE FILE HEAD:")
    print(submission.head())

