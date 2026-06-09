# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import subprocess
import os
import shutil
import glob

# 1. Shell script preamble: install from local directory
preamble_configs = """#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

export PIP_DISABLE_PIP_VERSION_CHECK=true
export PIP_FIND_LINKS=$SCRIPT_DIR
export PIP_NO_INDEX=true
"""

github_configs = """
export PIP_NO_BUILD_ISOLATION=false
"""

zip_file_installs = """
for file in "$SCRIPT_DIR"/*.zip; do
    if [ -f "$file" ]; then
        pip install "$file"
    fi
done
"""

# 2. Clean file (remove 'pip install' prefix) → safe for pip install -r
def clean_requirements_file(src="/kaggle/input/packages/input_requirements.txt", dst="/kaggle/working/input_requirements_cleaned.txt"):
    shutil.copy(src, dst)
    with open(dst, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("pip install "):
                line = line.replace("pip install ", "").strip()
            f.write(f"{line}\n")
        f.truncate()
    # print(f"Cleaned requirements saved to {dst}")

# 3. Copy all .whl and .zip files from dataset to working dir
def copy_package_files(src_dir="/kaggle/input/packages", dst_dir="/kaggle/working"):
    for ext in ("*.whl", "*.zip"):
        for file in glob.glob(os.path.join(src_dir, ext)):
            shutil.copy(file, dst_dir)
            # print(f"Copied {os.path.basename(file)} to {dst_dir}")
    # print("All package files copied to working directory")

# 4. Create the install script
def create_install_script(requirements_file="/kaggle/working/input_requirements_cleaned.txt", script_name="/kaggle/working/install_requirements.sh"):
    with open(script_name, "w") as f:
        f.write(preamble_configs)
        f.write(f"pip install -r {requirements_file}\n")
        f.write(github_configs)
        f.write(zip_file_installs)
    print(f"Installation script created at {script_name}")

# 5. Execute the install script
def run_install_script(script_path="/kaggle/working/install_requirements.sh"):
    # print("Running offline install script...\n")
    subprocess.call(["bash", script_path])

# Run all
clean_requirements_file()
copy_package_files()
create_install_script()
run_install_script()



import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import positive, print_summary
from gpflow.utilities.ops import broadcasting_elementwise
from matplotlib import pyplot as plt
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        # We constrain the value of the kernel variance to be positive when it's being optimised
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))

        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * outer_product/denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


# Load datasets
data_train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')   # Contains id, SMILES, Tg, FFV, Tc, Density, Rg
dataset1 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv')  # Contains SMILES, TC_mean (to be renamed to Tc)
dataset3 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv')  # Contains SMILES, Tg
dataset4 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')  # Contains SMILES, FFV

# Rename the column 'TC_mean' to 'Tc' in dataset1
dataset1 = dataset1.rename(columns={'TC_mean': 'Tc'})

# Define full column structure
required_columns = ['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Helper function: standardize dataset to have all required columns (missing ones = NaN)
def standardize_dataset(df):
    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[required_columns]

# Apply standardization
dataset1 = standardize_dataset(dataset1)
dataset3 = standardize_dataset(dataset3)
dataset4 = standardize_dataset(dataset4)

# Combine supplementary datasets
data = pd.concat([data_train, dataset1, dataset3, dataset4], axis=0)


# List of target columns
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Print the number of non-NaN values for each target
for target in targets:
    count = data[target].notna().sum()
    print(f"Number of available values for {target}: {count}")


# Molecule Feature Extraction from SMILES Using RDKit
# Processes a list of SMILES strings to generate molecular fingerprints used as input features for machine learning models.
smiles_list = data['SMILES'].to_list()

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
X = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048) for mol in rdkit_mols]
X = np.asarray(X)
X.shape


# Plotting function
def plot_results(y_true, y_pred, target_name):
    plt.figure(figsize=(6, 6))
    plt.title(f'Prediction for {target_name}')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.scatter(y_true, y_pred, c='blue', alpha=0.7, label='Test')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2, label='Ideal')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Function to train and evaluate a model for a specific target column
def train_predict_gp(target_column):
    print(f"\n--- Training model for: {target_column} ---")

    # Remove NaNs in the target column
    df = data[['SMILES', target_column]].dropna()

    # Convert SMILES to Morgan fingerprint vectors
    smiles_list = df['SMILES'].tolist()
    mols = [MolFromSmiles(smi) for smi in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048) for mol in mols]
    X = np.array(fps).astype(np.float64)
    y = df[target_column].values.reshape(-1, 1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Build GP model
    kernel = Tanimoto()
    mean_func = Constant(np.mean(y_train))
    model = gpflow.models.GPR(data=(X_train, y_train), kernel=kernel, mean_function=mean_func, noise_variance=1.0)

    # Define training function
    def objective_closure():
        return -model.log_marginal_likelihood()

    opt = gpflow.optimizers.Scipy()
    opt.minimize(objective_closure, model.trainable_variables, options=dict(maxiter=100))

    print_summary(model)

    # Predict
    y_pred, y_var = model.predict_f(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{target_column} MAE: {mae:.3f}")

    # Plot
    plot_results(y_test, y_pred, target_column)

    return model, y_test, y_pred
    

# Train and evaluate models 
models = {}
for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    models[target] = train_predict_gp(target)


# Predict on test data
test_data = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
smiles_test = test_data["SMILES"].tolist()
mols_test = [MolFromSmiles(smi) for smi in smiles_test]
fps_test = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048) for mol in mols_test]
X_test_all = np.array(fps_test).astype(np.float64)  
output_df = test_data[["id"]].copy()

for target, model_tuple in models.items():
    model = model_tuple[0]
    y_pred, _ = model.predict_f(X_test_all)
    output_df[target] = y_pred

print(output_df)
    
# Save to CSV
output_df.to_csv('submission.csv', index=False)
print("Predictions saved to 'submission.csv'")

