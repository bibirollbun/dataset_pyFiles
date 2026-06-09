import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Silence warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SMART FILE FINDER
# ==============================================================================
def find_and_organize_files():
    print(">>> STEP 1: SEARCHING FOR DATA FILES <<<")
    train_paths = []
    test_path = None
    sample_sub_path = None
    
    # Search recursively in Kaggle input directory
    for root, dirs, files in os.walk('/kaggle/input'):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith('.las'):
                if 'test' in file.lower():
                    test_path = full_path
                    print(f"  FOUND TEST WELL:  {full_path}")
                elif 'well' in file.lower():
                    train_paths.append(full_path)
                    print(f"  FOUND TRAIN WELL: {full_path}")
            elif 'submission.csv' in file.lower() or 'sample' in file.lower():
                 sample_sub_path = full_path
                 print(f"  FOUND SAMPLE SUBMISSION: {full_path}")
    
    if not train_paths or not test_path:
        print("\nCRITICAL ERROR: Missing LAS files. Please check data upload.")
        return None, None, None
        
    return train_paths, test_path, sample_sub_path

# ==============================================================================
# 2. ROBUST DATA LOADER
# ==============================================================================
def read_las_manual(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except:
        return None
    
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
        try:
            df = pd.read_csv(filename, skiprows=data_start, sep='\s+', header=None)
            if len(headers) == df.shape[1]:
                df.columns = headers
            else:
                df.columns = [f"Col_{k}" for k in range(df.shape[1])]
            return df
        except:
            return None
    return None

# ==============================================================================
# 3. PHYSICS-BASED FEATURE ENGINEERING
# ==============================================================================
def process_data(df):
    # 1. Clean Magic Numbers and Infinities
    df = df.replace([-999.25, -999, 9999], np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. Physical Limits
    df['Resistivity'] = df['Resistivity'].mask(df['Resistivity'] <= 0, 0.1)
    df['Gamma'] = df['Gamma'].mask(df['Gamma'] < 0, 0)
    
    # 3. Standardize Porosity
    phi = df['Porosity']
    if phi.max() > 1.0: 
        phi = phi / 100.0
    df['Porosity_Frac'] = phi 
    
    # 4. Calculate Physics Features
    gr_safe = df['Gamma'].fillna(df['Gamma'].median())
    res_safe = df['Resistivity'].fillna(0.1)
    phi_safe = df['Porosity_Frac'].fillna(0)

    # A. Shale Volume
    gr_min = gr_safe.quantile(0.01)
    gr_max = gr_safe.quantile(0.99)
    df['Vsh'] = (df['Gamma'] - gr_min) / (gr_max - gr_min + 1e-5)
    df['Vsh'] = df['Vsh'].clip(0, 1)
    
    # B. Apparent Water Resistivity (Archie)
    df['Rwa'] = res_safe * (phi_safe ** 2)
    
    # C. Timur-Coates Proxy (Permeability Driver)
    df['Timur_Proxy'] = (phi_safe ** 4) * res_safe
    
    # D. Log Transform
    df['Log_Res'] = np.log10(res_safe)
    
    return df

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

# 1. Find and Load Files
train_paths, test_path, sample_sub_path = find_and_organize_files()

if train_paths and test_path:
    print("\n>>> STEP 2: LOADING DATA <<<")
    train_dfs = []
    for path in train_paths:
        df = read_las_manual(path)
        if df is not None:
            train_dfs.append(df)
            
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = read_las_manual(test_path)
    
    print(f"Training Data Size: {train_df.shape}")
    print(f"Test Data Size: {test_df.shape}")

    # 2. Process Data
    print("\n>>> STEP 3: APPLYING PHYSICS & CLEANING <<<")
    train_df = process_data(train_df)
    test_df = process_data(test_df)

    features = ['DEPT', 'Gamma', 'Porosity', 'Resistivity', 'LITH', 
                'Vsh', 'Rwa', 'Timur_Proxy', 'Log_Res']
    target = 'Perm'

    # 3. NUCLEAR CLEANING
    train_df_clean = train_df.dropna(subset=features + [target])
    
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(train_df_clean[features])
    
    X_train_full = train_df_clean[features]
    y_train_full = train_df_clean[target]
    
    X_test_full = pd.DataFrame(imputer.transform(test_df[features]), columns=features)

    # 4. Train & Validate
    print("\n>>> STEP 4: TRAINING MODEL <<<")
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    print(f"Validation MAE (Internal Error): {mae:.4e} mD")

    # 5. Visualize
    print("\n>>> STEP 5: GENERATING GRAPHS <<<")
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.semilogy(y_val.values[:100], label='Actual', color='black', linewidth=1.5)
    plt.semilogy(val_preds[:100], label='Predicted', color='red', linestyle='--', linewidth=1.5)
    plt.title("Permeability Match (Log Scale)")
    plt.ylabel("Permeability (mD)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.subplot(1, 2, 2)
    plt.loglog(y_val, val_preds, 'o', alpha=0.4, color='blue', markersize=4)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Perfect Match')
    plt.title("Predicted vs Actual")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('Team_Prediction_Plot.png', dpi=300)
    plt.show()

    # 6. Save Files (ID MATCHING FIX)
    print("\n>>> STEP 6: SAVING FILES <<<")
    
    model_final = RandomForestRegressor(n_estimators=200, random_state=42)
    model_final.fit(X_train_full, y_train_full)
    test_preds = model_final.predict(X_test_full)

    # --- CRITICAL ID MATCHING SECTION ---
    if sample_sub_path:
        print("Using ID format from sample_submission.csv...")
        sample_sub = pd.read_csv(sample_sub_path)
        
        # Ensure we have the same number of predictions as the sample submission
        if len(test_preds) != len(sample_sub):
            print(f"WARNING: Prediction count ({len(test_preds)}) != Sample ID count ({len(sample_sub)}).")
            print("Adjusting to match sample length...")
            
        # Create submission using Sample IDs and Our Predictions
        submission = pd.DataFrame({
            'id': sample_sub['id'],
            'Permeability': test_preds[:len(sample_sub)] # Slice to be safe
        })
    else:
        print("Sample submission not found. Using index as ID.")
        submission = pd.DataFrame({
            'id': range(len(test_preds)), # Default 0, 1, 2...
            'Permeability': test_preds
        })

    # Save CSVs
    submission.to_csv('submission.csv', index=False)
    submission.to_csv('Team_SnehaChandra.csv', index=False)
    
    print("[1] Saved: submission.csv")
    print("[2] Saved: Team_SnehaChandra.csv")

    # Save Model
    joblib.dump(model_final, 'permeability_model.joblib')
    print("[3] Saved: permeability_model.joblib")

    # Save User Guide
    guide_text = """
    USER GUIDE: PERMEABILITY PREDICTION
    -----------------------------------
    TEAM NAME: Team
    LEADER:    Sneha Chandra
    
    METHODOLOGY:
    1. Feature Engineering: Calculated Vsh (Shale Volume), Rwa (Apparent Water 
       Resistivity), and Timur Proxy (Phi^4 * Rt) to model fluid flow physics.
    2. Model: Random Forest Regressor trained on 4 wells.
    3. Cleaning: Imputed missing values using mean strategy for robust prediction.
    
    USAGE INSTRUCTIONS:
    To reuse this model on new data, you must first apply the feature engineering
    steps defined in the 'process_data' function within the source code.
    
    Step 1: Load raw data (LAS/CSV).
    Step 2: Apply 'process_data(df)' to generate Vsh, Rwa, Timur_Proxy, Log_Res.
    Step 3: Run prediction:
        import joblib
        model = joblib.load('permeability_model.joblib')
        preds = model.predict(processed_data)
    """
    with open('user_guide.txt', 'w') as f:
        f.write(guide_text)
    print("[4] Saved: user_guide.txt")
    
    # Cleanup
    if os.path.exists("GeoMasters_Submission.csv"):
        os.remove("GeoMasters_Submission.csv")

    print("\nSUCCESS! All files are ready in the Output tab.")

