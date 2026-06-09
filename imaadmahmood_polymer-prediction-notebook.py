from IPython.display import Image, display

img_path = "/kaggle/input/polymer-version-1/Open_polymer_1.png"

display(Image(filename=img_path))


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


import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc
import traceback

try:
    # --- File listing for debugging ---
    for dirname, _, filenames in os.walk('/kaggle/input'):
        print(f"Directory: {dirname}")
        for filename in filenames:
            print(f"  {filename}")

    # --- Load datasets ---
    train_path = '/kaggle/input/neurips-polymer-train-descriptors/train_descriptors.csv'
    test_path = '/kaggle/input/neurips-polymer-train-descriptors/test_descriptors.csv'
    sample_sub_path = '/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_sub_path)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print("Sample submission shape:", sample_submission.shape)

    # --- Define target and feature columns ---
    TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    exclude_cols = TARGET_COLS + ['id', 'SMILES'] if 'SMILES' in train_df.columns else TARGET_COLS + ['id']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    # --- Feature cleaning function ---
    max_threshold = 1e10
    def robust_clean(df, features):
        missing_features = set(features) - set(df.columns)
        for col in missing_features:
            print(f"Adding missing feature column: {col}")
            df[col] = np.nan
        X = df[features].replace([np.inf, -np.inf], np.nan)
        X = X.mask(X.abs() > max_threshold, np.nan)
        return X

    # --- Clean train and test features ---
    X_train_cleaned = robust_clean(train_df, feature_cols)
    X_test_cleaned = robust_clean(test_df, feature_cols)

    # --- Impute missing values and scale ---
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_cleaned)
    X_test_imputed = imputer.transform(X_test_cleaned)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # --- Prepare targets ---
    y_train = train_df[TARGET_COLS]

    # --- Train-validation split ---
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    # --- Train LightGBM models ---
    models = {}
    for target in TARGET_COLS:
        print(f"Training model for target: {target}")

        train_mask = ~y_train_final[target].isnull()
        val_mask = ~y_val[target].isnull()

        X_tr = X_train_final[train_mask]
        y_tr = y_train_final[target][train_mask]

        X_vl = X_val[val_mask]
        y_vl = y_val[target][val_mask]

        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        models[target] = model

    gc.collect()

    # --- Predict on test data ---
    preds = pd.DataFrame()
    preds['id'] = test_df['id']
    for target in TARGET_COLS:
        preds[target] = models[target].predict(X_test_scaled)

    # --- Create submission from sample template ---
    submission = sample_submission.copy()

    # Fill predictions using 'id' mapping to ensure exact alignment
    for target in TARGET_COLS:
        submission[target] = submission['id'].map(preds.set_index('id')[target])

    # --- Final Validation ---
    assert submission.shape == sample_submission.shape, "Submission shape mismatch!"
    assert submission['id'].tolist() == sample_submission['id'].tolist(), "ID order mismatch!"
    assert submission[TARGET_COLS].isna().sum().sum() == 0, "NaNs in submission!"
    assert np.isfinite(submission[TARGET_COLS].values).all(), "Invalid values in submission!"

    # --- Save submission ---
    submission.to_csv('submission.csv', index=False)
    print("✅ Final submission.csv created and validated!")

except Exception as e:
    print(f"❌ Error during notebook execution: {e}")
    traceback.print_exc()

    # Save fallback submission if possible
    try:
        sample_submission.to_csv('submission.csv', index=False)
        print("⚠️ Fallback submission saved to avoid failure.")
    except:
        print("❌ Failed to save fallback submission.")


