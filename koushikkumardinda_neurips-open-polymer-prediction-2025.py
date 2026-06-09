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


!pip install pandas numpy scikit-learn rdkit-pypi lightgbm xgboost catboost matplotlib seaborn optuna torch torchvision torchaudio
# For PyTorch Geometric if using GNNs
!pip install /kaggle/input/neurips-open-polymer-prediction-dataset-2025/torch_geometric-2.6.1-py3-none-any.whl
!pip install /kaggle/input/neurips-open-polymer-prediction-dataset-2025/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# Ensure PyTorch is installed with the correct CUDA version if using GPU
# For specific PyTorch Geometric installation, refer to their official documentation


import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 1. Load the datasets
try:
    train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    # Load supplementary data
    supp_tc_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv')
    supp_smiles_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv')
    supp_density_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv')
    supp_density_2_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure all required CSV files are in the correct directories.")
    exit()

# 2. Data Augmentation
full_train_df = train_df.copy()

if 'Tc' in supp_tc_df.columns and 'SMILES' in supp_tc_df.columns:
    full_train_df = pd.concat([full_train_df, supp_tc_df[['SMILES', 'Tc']]], ignore_index=True)

if 'Density' in supp_density_df.columns and 'SMILES' in supp_density_df.columns:
    full_train_df = pd.concat([full_train_df, supp_density_df[['SMILES', 'Density']]], ignore_index=True)

if 'Density' in supp_density_2_df.columns and 'SMILES' in supp_density_2_df.columns:
    full_train_df = pd.concat([full_train_df, supp_density_2_df[['SMILES', 'Density']]], ignore_index=True)

full_train_df.drop_duplicates(subset='SMILES', keep='first', inplace=True)
full_train_df.reset_index(drop=True, inplace=True)

# 3. Feature Engineering
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.zeros((1,))
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_array)
    
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'TPSA': Descriptors.TPSA(mol)
    }
    
    return fp_array, descriptors

def featurize_dataframe(df):
    fingerprints = []
    descriptors = []
    valid_smiles_indices = []
    
    for i, smiles in enumerate(df['SMILES']):
        fp, desc = smiles_to_features(smiles)
        if fp is not None and desc is not None:
            fingerprints.append(fp)
            descriptors.append(list(desc.values()))
            valid_smiles_indices.append(i)
        else:
            print(f"Warning: Invalid SMILES string at index {i}: {smiles}")

    if not fingerprints:
        return pd.DataFrame() 
    
    fp_df = pd.DataFrame(fingerprints, index=valid_smiles_indices, columns=[f'fp_{i}' for i in range(len(fingerprints[0]))])
    desc_df = pd.DataFrame(descriptors, index=valid_smiles_indices, columns=list(smiles_to_features(df['SMILES'].iloc[valid_smiles_indices[0]])[1].keys()))
    
    df = df.iloc[valid_smiles_indices].reset_index(drop=True)
    features_df = pd.concat([df, fp_df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)
    
    return features_df

print("Featurizing training data...")
# The fix: call the function and assign the result to the variable
processed_train_df = featurize_dataframe(full_train_df) 
print("Featurizing test data...")
processed_test_df = featurize_dataframe(test_df)

# Separate features (X) and targets (y)
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
feature_cols = [col for col in processed_train_df.columns if col not in ['id', 'SMILES'] + target_cols]

X_train = processed_train_df[feature_cols]
y_train = processed_train_df[target_cols]
X_test = processed_test_df[feature_cols]

# Impute missing target values
for col in target_cols:
    y_train[col].fillna(y_train[col].mean(), inplace=True)
    
# 4. Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training and Prediction
print("Training the model...")
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
model.fit(X_train_scaled, y_train)

print("Making predictions on the test set...")
predictions = model.predict(X_test_scaled)
predictions_df = pd.DataFrame(predictions, columns=target_cols)

# 6. Create the submission file
submission_df = pd.DataFrame({
    'id': processed_test_df['id'],
    'Tg': predictions_df['Tg'],
    'FFV': predictions_df['FFV'],
    'Tc': predictions_df['Tc'],
    'Density': predictions_df['Density'],
    'Rg': predictions_df['Rg']
})

submission_df.to_csv('submission1.csv', index=False)
print("Submission file created successfully!")


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# 1. Load preprocessed data
num_train_samples = 1000
num_test_samples = 500
num_features = 2054
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
feature_cols = [f'feature_{i}' for i in range(num_features)]

X_train = pd.DataFrame(np.random.rand(num_train_samples, num_features), columns=feature_cols)
y_train = pd.DataFrame(np.random.rand(num_train_samples, len(target_cols)), columns=target_cols)
X_test = pd.DataFrame(np.random.rand(num_test_samples, num_features), columns=feature_cols)
test_ids = np.arange(num_test_samples)

# 2. Model Initialization
# I'll use a robust XGBoost model with MultiOutputRegressor.
model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42
    )
)

# 3. Model Training
print("Starting model training...")
model.fit(X_train, y_train)
print("Model training complete.")

# 4. Make Predictions
print("Making predictions on the test set...")
predictions = model.predict(X_test)

# Ensure the predictions are in a DataFrame format, even for a single prediction.
# If predictions is a 1D array, reshape it.
if predictions.ndim == 1:
    predictions = predictions.reshape(1, -1)

predictions_df = pd.DataFrame(predictions, columns=target_cols)

# 5. Create Submission File
# Ensure the `id` column is a Series, not a scalar.
# The code below is already robust if `test_ids` is an array.
submission_df = pd.DataFrame({
    'id': test_ids,
    'Tg': predictions_df['Tg'],
    'FFV': predictions_df['FFV'],
    'Tc': predictions_df['Tc'],
    'Density': predictions_df['Density'],
    'Rg': predictions_df['Rg']
})

# Ensure no negative predictions for physical properties
submission_df[target_cols] = submission_df[target_cols].clip(lower=0)

# Save the submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully!")
print(submission_df.head())


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Assume preprocessed data (X_train, y_train, X_test) is already loaded.
num_samples = 1000
num_features = 2054
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
feature_cols = [f'feature_{i}' for i in range(num_features)]

X = pd.DataFrame(np.random.rand(num_samples, num_features), columns=feature_cols)
y = pd.DataFrame(np.random.rand(num_samples, len(target_cols)), columns=target_cols)

# Define the number of splits for cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
mae_scores = []

print("Starting K-Fold Cross-Validation...")

# Iterate over each fold
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"  - Fold {fold+1}/{n_splits}")
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Initialize the model
    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the validation set
    y_pred = model.predict(X_val)
    
    # Calculate MAE for each target and then average them
    current_mae = np.mean([mean_absolute_error(y_val.iloc[:, i], y_pred[:, i]) for i in range(y.shape[1])])
    mae_scores.append(current_mae)

print("\nCross-Validation complete.")
print(f"MAE scores per fold: {mae_scores}")
print(f"Mean MAE across all folds: {np.mean(mae_scores):.4f}")
print(f"Standard deviation of MAE: {np.std(mae_scores):.4f}")

