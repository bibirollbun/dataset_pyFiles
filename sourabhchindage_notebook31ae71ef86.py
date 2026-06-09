import warnings
warnings.filterwarnings('ignore')  # Silences FutureWarnings and others

import os
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'  # Silences debugger noise

# 1. Imports
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# 2. Load data using fixed paths
train = pd.read_csv('/kaggle/input/kaggle-dataset/train_features.csv')
test = pd.read_csv('/kaggle/input/kaggle-dataset/test_features.csv')
submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

# 3. Targets and features
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
feature_cols = [col for col in train.columns if col not in targets + ['SMILES', 'Name']]

# 4. Impute missing values in features
imputer = SimpleImputer(strategy='mean')
X_full = imputer.fit_transform(train[feature_cols])
X_test = pd.DataFrame(imputer.transform(test[feature_cols]), columns=feature_cols)
X_full_df = pd.DataFrame(X_full, columns=feature_cols)

# 5. Train and predict
for target in targets:
    y_full = train[target]
    mask = y_full.notna()
    
    if mask.sum() < 10:
        print(f"⚠️ Skipping {target}: insufficient data after removing NaNs.")
        continue
    
    X_train = X_full_df.loc[mask]
    y_train = y_full[mask]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    submission[target] = predictions

# 6. Save submission
# ✅ Ensure correct column order and no NaNs before saving
id_col = 'Id' if 'Id' in submission.columns else 'id'
submission.rename(columns={id_col: 'Id'}, inplace=True)
submission = submission[['Id'] + targets]
submission = submission.fillna(0)

# ✅ Save submission
# ✅ Final save (already in working directory)
submission.to_csv('submission.csv', index=False)

# ✅ Final check
assert submission.shape[1] == 6, "❌ Submission must have 6 columns."
assert submission.isnull().sum().sum() == 0, "❌ Submission contains NaNs!"
assert submission.columns.tolist() == ['Id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg'], "❌ Column order is wrong!"
print("✅ Submission file generated and validated successfully.")






