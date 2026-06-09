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


import warnings
warnings.filterwarnings('ignore')



data=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
data.head()


data.info()





train=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
train.info()


# Supplement datasets
dataset1 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv")
dataset2 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv")  # unused in example but keeping
dataset3 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv")
dataset4 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")

print("___ dataset1 Shape:", dataset1.shape)
print("___ dataset2 Shape:", dataset2.shape)
print("___ dataset3 Shape:", dataset3.shape)
print("___ dataset4 Shape:", dataset4.shape)


id_col = "id"
smiles_col = "SMILES"
targets = ["Tg","FFV","Tc","Density","Rg"]


# Base: keep SMILES and targets
train_base = train[[smiles_col] + targets].copy()

# Supplement alignments
dset1 = dataset1.rename(columns={"TC_mean": "Tc"})  # Tc supplement
dset3 = dataset3.rename(columns={"Tg": "Tg"})       # Tg supplement (already same name)
dset4 = dataset4.rename(columns={"FFV": "FFV"})     # FFV supplement (already same name)

# Merge all into one unified dataset
supp_all = pd.concat([train_base, dset1, dset3, dset4], ignore_index=True)

# Drop duplicates based on SMILES, keeping first occurrence
train = supp_all.drop_duplicates(subset=[smiles_col], keep="first").reset_index(drop=True)

print("Final unified train shape:", train.shape)
print("Columns:", list(train.columns))


train.info()


data_features=train.copy()


!pip install --no-index --find-links=/kaggle/input/rdkit-zip-dataset/rdkit_files/ rdkit


import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

# You can verify it's working by printing the version
print(f"RDKit version {rdkit.__version__} is successfully installed and ready to use.")


!unzip -l /kaggle/input/rdkit-zip-dataset/results.zip


import warnings
from rdkit import RDLogger

# Suppress all RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Suppress Python warnings
warnings.filterwarnings('ignore')



# Install RDKit if not already
# pip install rdkit-pypi

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit import DataStructs
import numpy as np
import pandas as pd

# Load dataset
data = train.copy()

# --- Step 1: Molecular descriptors ---
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return {
        "MolWt": Descriptors.MolWt(mol),
        "NumAtoms": mol.GetNumAtoms(),
        "NumBonds": mol.GetNumBonds(),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumRings": Descriptors.RingCount(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol)
    }

desc_features = [smiles_to_descriptors(s) for s in data['SMILES']]
desc_df = pd.DataFrame(desc_features)
# --- Step 3: Combine everything ---
data_features = pd.concat([data.reset_index(drop=True), desc_df], axis=1)

print("Final dataset shape:", data_features.shape)
data_features.head()



data_features.columns


data_features.info()


import matplotlib.pyplot as plt
import seaborn as sns

# Keep only rows with FFV values
df_ffv = data_features.dropna(subset=["FFV"]).copy()
print("FFV data shape:", df_ffv.shape)

df_ffv["FFV"].describe()



plt.figure(figsize=(8,5))
sns.histplot(df_ffv["FFV"], kde=True, bins=40)
plt.title("Distribution of FFV")
plt.xlabel("FFV")
plt.ylabel("Count")
plt.show()



desc_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds","NumRings",
             "HBD","HBA","LogP","TPSA"]

plt.figure(figsize=(10,6))
sns.heatmap(df_ffv[desc_cols + ["FFV"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation of RDKit Descriptors with FFV")
plt.show()



top_feats = df_ffv[desc_cols].corrwith(df_ffv["FFV"]).abs().sort_values(ascending=False).head(3).index

sns.pairplot(df_ffv, vars=top_feats.tolist() + ["FFV"], diag_kind="kde")
plt.show()



df_ffv.info()


# ===================================================================
# 1. Prepare Data and Handle Outliers
# ===================================================================
target = "FFV"
df_ffv = data_features.dropna(subset=[target]).copy()

print("--- Step 1: Visualizing and Handling Outliers in FFV ---")

# --- Reimagined Visualization: Violin Plot + Strip Plot ---
print("Displaying an enhanced plot to visualize data distribution before outlier handling...")

# Set a modern plot style
plt.style.use('seaborn-v0_8-talk')

# Create the figure
plt.figure(figsize=(18, 9))

# Create a violin plot to show the density distribution
sns.violinplot(x=df_ffv[target], inner='quartile', palette='pastel', linewidth=2)

# Overlay a strip plot to show individual data points, especially outliers
sns.stripplot(x=df_ffv[target], color='crimson', jitter=0.04, alpha=0.6, size=4)

# Add annotations and improve aesthetics
plt.title(f'Advanced Distribution Plot of {target} (Before Outlier Handling)', fontsize=22, weight='bold', pad=20)
plt.xlabel(f'Fractional Free Volume ({target})', fontsize=16, labelpad=15)
plt.ylabel(None)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show the plot
plt.show()


# --- Methodology for Outlier Handling ---
print("\nMethodology: The Interquartile Range (IQR) method is used for outlier detection.")
print("To target only the most extreme outliers, we will use a larger multiplier of 3.0.")
print("Outliers are now defined as data points below Q1 - 3.0 * IQR or above Q3 + 3.0 * IQR.")
print("Instead of removing these rows, we will replace the outlier values with the median.\n")

# --- Calculate IQR and define outlier bounds for EXTREME outliers ---
Q1 = df_ffv[target].quantile(0.25)
Q3 = df_ffv[target].quantile(0.75)
IQR = Q3 - Q1
# Using 3.0 * IQR to identify only extreme outliers
lower_bound = Q1 - 3.0 * IQR
upper_bound = Q3 + 3.0 * IQR

# --- Identify outliers ---
outliers_mask = (df_ffv[target] < lower_bound) | (df_ffv[target] > upper_bound)
num_outliers = outliers_mask.sum()

# --- Impute EXTREME outliers instead of removing them ---
if num_outliers > 0:
    # Using the median is generally more robust than the mean for imputation
    median_ffv = df_ffv[target].median()
    
    print(f"Identified {num_outliers} extreme outliers based on the 3.0 * IQR method.")
    print(f"Valid data range is between {lower_bound:.4f} and {upper_bound:.4f}.")
    print(f"Replacing {num_outliers} outlier values with the median value ({median_ffv:.4f}).")
    
    # Create a copy to work with
    df_ffv_clean = df_ffv.copy()
    
    # Use .loc to safely replace the values in the original dataframe copy
    df_ffv_clean.loc[outliers_mask, target] = median_ffv
    
    print(f"Original dataset size: {len(df_ffv)}, Size after imputation: {len(df_ffv_clean)} (unchanged)\n")
else:
    print("No significant extreme outliers were found in the FFV column.\n")
    df_ffv_clean = df_ffv.copy()



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow / Keras for the BiLSTM part
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Assume 'data_features' is a pre-loaded DataFrame ---
# In a real scenario, you would load your data here, e.g.:
# data_features = pd.read_csv('your_training_data.csv')

# Dummy data for demonstration if 'data_features' doesn't exist
if 'data_features' not in locals():
    print("Info: 'data_features' not found. Creating a dummy DataFrame.")
    dummy_smiles = ['CCO', 'C1=CC=CS1', 'CC(C)C', 'C1CCCCC1', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'] * 20
    data_features = pd.DataFrame({
        'SMILES': dummy_smiles,
        'FFV': np.random.rand(100) * 0.5 + 0.1,
        'MolWt': np.random.rand(100) * 100 + 50,
        'NumAtoms': np.random.randint(5, 20, 100),
        'NumBonds': np.random.randint(5, 25, 100),
        'NumRotatableBonds': np.random.randint(0, 5, 100),
        'NumRings': np.random.randint(0, 3, 100),
        'HBD': np.random.randint(0, 4, 100),
        'HBA': np.random.randint(0, 5, 100),
        'LogP': np.random.rand(100) * 5 - 1,
        'TPSA': np.random.rand(100) * 80
    })
    # Add a few outliers for demonstration
    data_features.loc[5, 'FFV'] = 1.5
    data_features.loc[15, 'FFV'] = -0.8


# ===================================================================
# 1. Prepare Data and Handle Outliers
# ===================================================================
target = "FFV"
df_ffv = data_features.dropna(subset=[target]).copy()

print("--- Step 1: Visualizing and Handling Outliers in FFV ---")

# --- Visualize potential outliers with a box plot ---
print("Displaying box plot to visualize data distribution before outlier handling...")
plt.figure(figsize=(12, 6))
sns.boxplot(x=df_ffv[target])
plt.title(f'Box Plot of {target} Before Outlier Handling', fontsize=16)
plt.xlabel(target, fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- Methodology for Outlier Handling ---
print("\nMethodology: The Interquartile Range (IQR) method is used for outlier detection.")
print("To target only the most extreme outliers, we will use a larger multiplier of 3.0.")
print("Outliers are now defined as data points below Q1 - 3.0 * IQR or above Q3 + 3.0 * IQR.")
print("Instead of removing these rows, we will replace the outlier values with the median.\n")

# --- Calculate IQR and define outlier bounds for EXTREME outliers ---
Q1 = df_ffv[target].quantile(0.25)
Q3 = df_ffv[target].quantile(0.75)
IQR = Q3 - Q1
# Using 3.0 * IQR to identify only extreme outliers
lower_bound = Q1 - 3.0 * IQR
upper_bound = Q3 + 3.0 * IQR

# --- Identify outliers ---
outliers_mask = (df_ffv[target] < lower_bound) | (df_ffv[target] > upper_bound)
num_outliers = outliers_mask.sum()

# --- Impute EXTREME outliers instead of removing them ---
if num_outliers > 0:
    # Using the median is generally more robust than the mean for imputation
    median_ffv = df_ffv[target].median()
    
    print(f"Identified {num_outliers} extreme outliers based on the 3.0 * IQR method.")
    print(f"Valid data range is between {lower_bound:.4f} and {upper_bound:.4f}.")
    print(f"Replacing {num_outliers} outlier values with the median value ({median_ffv:.4f}).")
    
    # Create a copy to work with
    df_ffv_clean = df_ffv.copy()
    
    # Use .loc to safely replace the values in the original dataframe copy
    df_ffv_clean.loc[outliers_mask, target] = median_ffv
    
    print(f"Original dataset size: {len(df_ffv)}, Size after imputation: {len(df_ffv_clean)} (unchanged)\n")
else:
    print("No significant extreme outliers were found in the FFV column.\n")
    df_ffv_clean = df_ffv.copy()


  df_ffv_clean.info()


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 1. Prepare data ---
target = "FFV"
df_ffv = data_features.dropna(subset=[target]).copy()

# Features = descriptors (you can add Morgan fingerprints too if you want)
desc_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
             "NumRings","HBD","HBA","LogP","TPSA"]

X = df_ffv[desc_cols]
y = df_ffv[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (XGBoost works without scaling, but scaling helps other models too)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. Train baseline model ---
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# --- 3. Evaluate ---
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Baseline FFV Model Results:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RÂ²:   {r2:.4f}")

# --- 4. Feature importance ---
import matplotlib.pyplot as plt

xgb.plot_importance(model, importance_type="weight")
plt.title("Feature Importance for FFV Prediction")
plt.show()



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# TensorFlow / Keras for the BiLSTM part
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Use the dataframe with FFV values ---
# Assuming 'data_features' is your dataframe with all precomputed descriptors
df_FFV = data_features.dropna(subset=['FFV']).copy()

# ===================================================================
# 1. SMILES Preprocessing for LSTM
# ===================================================================

# Create a vocabulary of all characters in the SMILES strings
smiles_tokenizer = Tokenizer(char_level=True)
smiles_tokenizer.fit_on_texts(df_FFV['SMILES']) # CORRECTED: Use df_FFV

# Define vocabulary size (add 1 for padding)
vocab_size = len(smiles_tokenizer.word_index) + 1
max_length = 150 # Max length of a SMILES string for padding

# ===================================================================
# 2. BiLSTM Feature Extractor Model
# ===================================================================

# Define model parameters
embedding_dim = 64
lstm_units = 32 # Output will be 2 * 32 = 64 features due to Bidirectional

# Define the model architecture
input_layer = Input(shape=(max_length,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(input_layer)
bilstm_layer = Bidirectional(LSTM(units=lstm_units))(embedding_layer) # This is our feature vector

# Create the feature extractor model
bilstm_feature_extractor = Model(inputs=input_layer, outputs=bilstm_layer)
bilstm_feature_extractor.summary()
print("\\n")


# ===================================================================
# 3. Combined Training Pipeline
# ===================================================================

# --- Define features and target ---
tabular_feature_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
                        "NumRings","HBD","HBA","LogP","TPSA"]

# --- Train/validation split ---
train_df, val_df = train_test_split(
    df_FFV, test_size=0.2, random_state=42
)

# Separate features (X) and target (y) for both sets
X_train_tabular = train_df[tabular_feature_cols]
y_train = train_df["FFV"]

X_val_tabular = val_df[tabular_feature_cols]
y_val = val_df["FFV"] # CORRECTED: Use FFV for validation target

smiles_train = train_df['SMILES']
smiles_val = val_df['SMILES']


# --- Scale the TABULAR features ---
scaler = StandardScaler()
X_train_tabular_scaled = scaler.fit_transform(X_train_tabular)
X_val_tabular_scaled = scaler.transform(X_val_tabular)

# --- Generate LSTM features ---
train_sequences = pad_sequences(smiles_tokenizer.texts_to_sequences(smiles_train), maxlen=max_length)
val_sequences = pad_sequences(smiles_tokenizer.texts_to_sequences(smiles_val), maxlen=max_length)

X_train_lstm_features = bilstm_feature_extractor.predict(train_sequences)
X_val_lstm_features = bilstm_feature_extractor.predict(val_sequences)


# --- Combine Tabular and LSTM Features ---
X_train_combined = np.hstack([X_train_tabular_scaled, X_train_lstm_features])
X_val_combined = np.hstack([X_val_tabular_scaled, X_val_lstm_features])

print(f"Shape of combined training features: {X_train_combined.shape}")
print(f"Shape of combined validation features: {X_val_combined.shape}\\n")


# --- Train XGBoost model on COMBINED features ---
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_combined, y_train)


# --- Evaluate ---
y_val_pred = xgb_model.predict(X_val_combined)
mae = mean_absolute_error(y_val, y_val_pred)
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
r2 = r2_score(y_val, y_val_pred)

print("--- Combined BiLSTM + XGBoost Model Results for FFV ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RÂ²:   {r2:.4f}")
model_FFV=xgb_model


import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===================================================================
# Important Note:
# In a real workflow, you would save your trained components after
# the training script runs and load them here.
# For this example, we'll assume the following objects from your
# training script are available in memory:
#
# - xgb_model: The trained XGBoost regressor.
# - bilstm_feature_extractor: The trained Keras BiLSTM model for feature extraction.
# - smiles_tokenizer: The fitted Keras Tokenizer for SMILES strings.
# - scaler: The fitted StandardScaler for tabular data.
# - max_length: The max length used for padding SMILES sequences (e.g., 150).
# - tabular_feature_cols: The list of column names for tabular features.
#
# Example of how you would save/load them:
#
# --- SAVING ---
# xgb_model.save_model("xgb_model.json")
# bilstm_feature_extractor.save("bilstm_model.h5")
# joblib.dump(smiles_tokenizer, "smiles_tokenizer.pkl")
# joblib.dump(scaler, "scaler.pkl")
#
# --- LOADING ---
# xgb_model = xgb.XGBRegressor()
# xgb_model.load_model("xgb_model.json")
# bilstm_feature_extractor = tf.keras.models.load_model("bilstm_model.h5")
# smiles_tokenizer = joblib.load("smiles_tokenizer.pkl")
# scaler = joblib.load("scaler.pkl")
# ===================================================================


# ===================================================================
# 1. Load Test Data
# ===================================================================
try:
    # This path is common in Kaggle environments
    test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
except FileNotFoundError:
    # Create a dummy test.csv for local execution if the file is not found
    print("Info: '/kaggle/input/' directory not found. Using a dummy 'test.csv'.")
    dummy_data = {'id': range(5), 'SMILES': ['CCO', 'C1=CC=CS1', 'CC(C)C', 'C1=CC=C(C=C1)C(C)(C)C', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']}
    test_df = pd.DataFrame(dummy_data)
    test_df.to_csv('test.csv', index=False)
    test_df = pd.read_csv('test.csv')


# ===================================================================
# 2. Preprocessing Functions for Test Data
# ===================================================================

def smiles_to_descriptors(smiles):
    """Calculates molecular descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return a dictionary of NaNs if SMILES is invalid
        return {col: np.nan for col in tabular_feature_cols}
    return {
        "MolWt": Descriptors.MolWt(mol),
        "NumAtoms": mol.GetNumAtoms(),
        "NumBonds": mol.GetNumBonds(),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumRings": Descriptors.RingCount(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol)
    }

# ===================================================================
# 3. Feature Engineering for the Test Set
# ===================================================================

# --- Generate Tabular Features (Molecular Descriptors) ---
print("Generating molecular descriptors for the test set...")
test_descriptors = [smiles_to_descriptors(s) for s in test_df['SMILES']]
test_desc_df = pd.DataFrame(test_descriptors)

# Ensure the columns are in the same order as during training
X_test_tabular = test_desc_df[tabular_feature_cols]

# Handle any potential missing values from invalid SMILES
if X_test_tabular.isnull().sum().sum() > 0:
    print("Warning: Missing values found in descriptors. Filling with median.")
    X_test_tabular = X_test_tabular.fillna(X_test_tabular.median())

# --- Scale the TABULAR features using the PRE-FITTED scaler ---
print("Scaling tabular features...")
X_test_tabular_scaled = scaler.transform(X_test_tabular)


# --- Generate BiLSTM Features from SMILES ---
print("Generating BiLSTM features from SMILES strings...")
test_smiles = test_df['SMILES']

# Tokenize and pad the SMILES sequences
test_sequences = pad_sequences(
    smiles_tokenizer.texts_to_sequences(test_smiles),
    maxlen=max_length
)

# Extract features using the BiLSTM model
X_test_lstm_features = bilstm_feature_extractor.predict(test_sequences)


# --- Combine Tabular and LSTM Features ---
print("Combining feature sets...")
X_test_combined = np.hstack([X_test_tabular_scaled, X_test_lstm_features])

print(f"Shape of combined test features: {X_test_combined.shape}")

# ===================================================================
# 4. Make Predictions and Create Submission File
# ===================================================================

print("Making predictions on the test set...")
# --- Predict FFV using the trained hybrid model ---
y_test_pred = xgb_model.predict(X_test_combined)

# --- Create the submission DataFrame ---
submission = pd.DataFrame({
    "id": test_df["id"],
    "FFV": y_test_pred
})
s_FFV=submission.copy()

# --- Save the submission file ---
submission.to_csv("submission.csv", index=False)
s_FFV=submission.copy()

print("\nâœ… Submission file saved successfully as submission.csv")
print("--- Submission Head ---")
print(submission.head())



data_features.info()


data_t=data_features.copy()
data_t.info()


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Drop NaN values of Tc
tc_data = data_t["Tc"].dropna()

# --- Histogram & KDE ---
plt.figure(figsize=(8,5))
sns.histplot(tc_data, bins=30, kde=True)
plt.title("Distribution of Tc")
plt.xlabel("Tc")
plt.ylabel("Frequency")
plt.show()

# --- Q-Q Plot ---
plt.figure(figsize=(6,6))
stats.probplot(tc_data, dist="norm", plot=plt)
plt.title("Q-Q Plot of Tc")
plt.show()

# --- Shapiro-Wilk Test ---
shapiro_test = stats.shapiro(tc_data)
print("Shapiro-Wilk Test:")
print(f"Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")

# --- Skewness & Kurtosis ---
print(f"Skewness: {stats.skew(tc_data):.4f}")
print(f"Kurtosis: {stats.kurtosis(tc_data):.4f}")



import matplotlib.pyplot as plt
import seaborn as sns

# Assume 'tc_data' is the pandas Series from your previous code
# tc_data = df['Tc'].dropna()

# --- Create the Box Plot ---
plt.figure(figsize=(10, 4)) # Adjust figure size as needed
sns.boxplot(x=tc_data, color='skyblue')

# --- Add title and labels for clarity ---
plt.title("Box Plot of Thermal Conductivity (Tc)", fontsize=14)
plt.xlabel("Tc Value", fontsize=12)

plt.show()


import pandas as pd
import numpy as np

# Assume 'data_features' is your full DataFrame
df = data_features.copy()

# --- 1. Calculate the outlier boundaries (same as before) ---
tc_data = df['Tc'].dropna()
Q1 = tc_data.quantile(0.25)
Q3 = tc_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# --- 2. Remove the outliers ---
# Keep rows that are either within the bounds OR have a missing Tc value
df_no_outliers = df[(df['Tc'] >= lower_bound) & (df['Tc'] <= upper_bound) | (df['Tc'].isnull())]

# --- 3. Verify the result ---
print(f"Original DataFrame shape: {df.shape}")
print(f"Shape after removing Tc outliers: {df_no_outliers.shape}")
print(f"Number of outliers removed: {df.shape[0] - df_no_outliers.shape[0]}")


data_features=df_no_outliers.copy()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# TensorFlow / Keras for the BiLSTM part
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===================================================================
# 1. Data Preparation
# ===================================================================

# Assume 'data_features' is your DataFrame with all precomputed descriptors
# This example is for the 'Tc' target.
TARGET_VARIABLE = 'Tc'
df = data_features.dropna(subset=[TARGET_VARIABLE]).copy()


# ===================================================================
# 2. SMILES Preprocessing for LSTM
# ===================================================================
smiles_tokenizer = Tokenizer(char_level=True)
smiles_tokenizer.fit_on_texts(df['SMILES'])
vocab_size = len(smiles_tokenizer.word_index) + 1
max_length = 150 # Adjust if your SMILES strings are longer

# ===================================================================
# 3. BiLSTM Training Model
# ===================================================================

embedding_dim = 64
lstm_units = 32

# Define the model architecture for TRAINING
input_layer = Input(shape=(max_length,), name='smiles_input')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(input_layer)
bilstm_layer = Bidirectional(LSTM(units=lstm_units), name='bilstm_feature_layer')(embedding_layer)
output_layer = Dense(1, name='prediction_output')(bilstm_layer) # Output for training on the target

# Create the full training model
bilstm_training_model = Model(inputs=input_layer, outputs=output_layer)
bilstm_training_model.compile(optimizer='adam', loss='mean_squared_error')
print("--- BiLSTM Training Model Summary ---")
bilstm_training_model.summary()
print("\\n")

# ===================================================================
# 4. Train the BiLSTM Network
# ===================================================================

# Prepare data for BiLSTM training
smiles_sequences = pad_sequences(smiles_tokenizer.texts_to_sequences(df['SMILES']), maxlen=max_length)
y = df[TARGET_VARIABLE].values

# Split data for training the BiLSTM
X_train_seq, X_val_seq, y_train_lstm, y_val_lstm = train_test_split(
    smiles_sequences, y, test_size=0.2, random_state=42
)

# --- Define Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10, # Stop after 10 epochs with no improvement
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,  # Reduce LR after 5 epochs with no improvement
    min_lr=1e-6,
    verbose=1
)

# --- Train the BiLSTM model ---
print("\\n--- Training the BiLSTM Model ---")
history = bilstm_training_model.fit(
    X_train_seq, y_train_lstm,
    epochs=100, # Train for more epochs; EarlyStopping will find the best one
    batch_size=32,
    validation_data=(X_val_seq, y_val_lstm),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ===================================================================
# 5. Create the BiLSTM Feature Extractor
# ===================================================================
# Create a new model that outputs the features from the trained BiLSTM layer
bilstm_feature_extractor = Model(
    inputs=bilstm_training_model.input,
    outputs=bilstm_training_model.get_layer('bilstm_feature_layer').output
)
print("\\n--- BiLSTM Feature Extractor is now ready ---")


# ===================================================================
# 6. Combined Training Pipeline for Final XGBoost Model
# ===================================================================

# --- Define tabular features and full target set ---
tabular_feature_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
                        "NumRings","HBD","HBA","LogP","TPSA"]

X_tabular = df[tabular_feature_cols]
X_smiles = df['SMILES']
y_final = df[TARGET_VARIABLE]

# --- Train/validation split for the final XGBoost model ---
X_train_tab, X_val_tab, smiles_train, smiles_val, y_train_xgb, y_val_xgb = train_test_split(
    X_tabular, X_smiles, y_final, test_size=0.2, random_state=42
)

# --- Scale the TABULAR features ---
scaler = StandardScaler()
X_train_tabular_scaled = scaler.fit_transform(X_train_tab)
X_val_tabular_scaled = scaler.transform(X_val_tab)

# --- Generate LSTM features using the TRAINED extractor ---
train_sequences = pad_sequences(smiles_tokenizer.texts_to_sequences(smiles_train), maxlen=max_length)
val_sequences = pad_sequences(smiles_tokenizer.texts_to_sequences(smiles_val), maxlen=max_length)

X_train_lstm_features = bilstm_feature_extractor.predict(train_sequences)
X_val_lstm_features = bilstm_feature_extractor.predict(val_sequences)

# --- Combine Tabular and LSTM Features ---
X_train_combined = np.hstack([X_train_tabular_scaled, X_train_lstm_features])
X_val_combined = np.hstack([X_val_tabular_scaled, X_val_lstm_features])

print(f"Shape of combined training features: {X_train_combined.shape}")
print(f"Shape of combined validation features: {X_val_combined.shape}\\n")

# --- Train FINAL XGBoost model on COMBINED features ---
xgb_model_final = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model_final.fit(X_train_combined, y_train_xgb)

# --- Final Evaluation ---
y_val_pred = xgb_model_final.predict(X_val_combined)
mae = mean_absolute_error(y_val_xgb, y_val_pred)
rmse = mean_squared_error(y_val_xgb, y_val_pred, squared=False)
r2 = r2_score(y_val_xgb, y_val_pred)

print(f"--- Combined BiLSTM + XGBoost Model Results for {TARGET_VARIABLE} ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RÂ²:   {r2:.4f}")


# ===================================================================
# Predict Tc on Test Set with Combined BiLSTM + XGBoost Model
# ===================================================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# NOTE: This script assumes that 'xgb_model_final', 'bilstm_feature_extractor',
# 'scaler', and 'smiles_tokenizer' from the previous training cell are in memory.

# --- 1. Load and Preprocess Test Data ---
print("--- Loading and preprocessing test data... ---")
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

# --- Molecular descriptor function (must be the same as used for training data) ---
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return a dictionary of NaNs with the correct keys
        return {
            "MolWt": np.nan, "NumAtoms": np.nan, "NumBonds": np.nan,
            "NumRotatableBonds": np.nan, "NumRings": np.nan,
            "HBD": np.nan, "HBA": np.nan, "LogP": np.nan, "TPSA": np.nan
        }
    return {
        "MolWt": Descriptors.MolWt(mol),
        "NumAtoms": mol.GetNumAtoms(),
        "NumBonds": mol.GetNumBonds(),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumRings": Descriptors.RingCount(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol)
    }

# --- Extract tabular descriptors for the test set ---
test_desc = [smiles_to_descriptors(s) for s in test_df['SMILES']]
test_desc_df = pd.DataFrame(test_desc)

# --- 2. Feature Engineering for Test Set ---

# A. Process TABULAR features
# These must be the same columns used for training the XGBoost model
tabular_feature_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
                        "NumRings","HBD","HBA","LogP","TPSA"]
X_test_tabular = test_desc_df[tabular_feature_cols]

# Scale tabular features using the PRE-FITTED scaler from training
X_test_tabular_scaled = scaler.transform(X_test_tabular)
print(f"Shape of scaled tabular test features: {X_test_tabular_scaled.shape}")

# B. Process SMILES features for LSTM
# Tokenize and pad SMILES strings using the PRE-FITTED tokenizer
test_sequences = pad_sequences(
    smiles_tokenizer.texts_to_sequences(test_df['SMILES']),
    maxlen=max_length
)

# Generate features using the TRAINED BiLSTM feature extractor
X_test_lstm_features = bilstm_feature_extractor.predict(test_sequences)
print(f"Shape of LSTM test features: {X_test_lstm_features.shape}")


# C. Combine Tabular and LSTM Features
X_test_combined = np.hstack([X_test_tabular_scaled, X_test_lstm_features])
print(f"Shape of combined test features: {X_test_combined.shape}\n")


# --- 3. Make Predictions ---
print("--- Generating predictions on the test set... ---")
# Use the final trained XGBoost model to predict
y_test_pred = xgb_model_final.predict(X_test_combined)

# Note: No back-transformation (like np.expm1) is needed because the target
# variable 'Tc' was not log-transformed during the training of this hybrid model.


# --- 4. Save Submission File ---
submission = pd.DataFrame({
    "id": test_df["id"],
    TARGET_VARIABLE: y_test_pred  # Use the target variable name from training
})
s_TC=submission.copy()

submission_filename = f"submission_{TARGET_VARIABLE.lower()}_hybrid.csv"
submission.to_csv(submission_filename, index=False)

print(f"âœ… Submission file saved as {submission_filename}")
print(submission.head())
model_Tc=xgb_model_final


# ================================
# Density Prediction with XGBoost (Raw Target)
# ================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# -------------------
# Prepare data
# -------------------
data_t = data_features.copy()
df_density = data_t.dropna(subset=['Density'])   # keep only rows with Density

feature_cols = [
    "MolWt","NumAtoms","NumBonds","NumRotatableBonds",
    "NumRings","HBD","HBA","LogP","TPSA"
]

X = df_density[feature_cols]
y = df_density["Density"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------
# Train model
# -------------------
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------
# Evaluate
# -------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("==== Density Model Results (Raw Target) ====")
print("MAE: ", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("RÂ²:  ", round(r2, 4))



import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Extract Density values (drop NA)
density_vals = df_density["Density"].dropna()

# Histogram + KDE
plt.figure(figsize=(10,5))
sns.histplot(density_vals, kde=True, bins=30, color="skyblue")
plt.title("Density Distribution (Raw)")
plt



import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot for Density
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Density"], color="skyblue")
plt.title("Boxplot of Density (Outlier Detection)")
plt.xlabel("Density")
plt.show()



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# TensorFlow / Keras for the BiLSTM part
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===================================================================
# 1. Data Preparation
# ===================================================================

# Assume 'data_features' is your DataFrame with all precomputed descriptors
# This example is for the 'Tc' target.
TARGET_VARIABLE = 'Density'
df = data_features.dropna(subset=[TARGET_VARIABLE]).copy()


# ===================================================================
# 2. SMILES Preprocessing for LSTM
# ===================================================================
smiles_tokenizer = Tokenizer(char_level=True)
smiles_tokenizer.fit_on_texts(df['SMILES'])
vocab_size = len(smiles_tokenizer.word_index) + 1
max_length = 150 # Adjust if your SMILES strings are longer

# ===================================================================
# 3. BiLSTM Training Model
# ===================================================================

embedding_dim = 64
lstm_units = 32

# Define the model architecture for TRAINING
input_layer = Input(shape=(max_length,), name='smiles_input')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(input_layer)
bilstm_layer = Bidirectional(LSTM(units=lstm_units), name='bilstm_feature_layer')(embedding_layer)
output_layer = Dense(1, name='prediction_output')(bilstm_layer) # Output for training on the target

# Create the full training model
bilstm_training_model = Model(inputs=input_layer, outputs=output_layer)
bilstm_training_model.compile(optimizer='adam', loss='mean_squared_error')
print("--- BiLSTM Training Model Summary ---")
bilstm_training_model.summary()
print("\\n")

# ===================================================================
# 4. Train the BiLSTM Network
# ===================================================================

# Prepare data for BiLSTM training
smiles_sequences = pad_sequences(smiles_tokenizer.texts_to_sequences(df['SMILES']), maxlen=max_length)
y = df[TARGET_VARIABLE].values

# Split data for training the BiLSTM
X_train_seq, X_val_seq, y_train_lstm, y_val_lstm = train_test_split(
    smiles_sequences, y, test_size=0.2, random_state=42
)

# --- Define Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10, # Stop after 10 epochs with no improvement
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,  # Reduce LR after 5 epochs with no improvement
    min_lr=1e-6,
    verbose=1
)

# --- Train the BiLSTM model ---
print("\\n--- Training the BiLSTM Model ---")
history = bilstm_training_model.fit(
    X_train_seq, y_train_lstm,
    epochs=100, # Train for more epochs; EarlyStopping will find the best one
    batch_size=32,
    validation_data=(X_val_seq, y_val_lstm),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ===================================================================
# 5. Create the BiLSTM Feature Extractor
# ===================================================================
# Create a new model that outputs the features from the trained BiLSTM layer
bilstm_feature_extractor = Model(
    inputs=bilstm_training_model.input,
    outputs=bilstm_training_model.get_layer('bilstm_feature_layer').output
)
print("\\n--- BiLSTM Feature Extractor is now ready ---")


# ===================================================================
# 6. Combined Training Pipeline for Final XGBoost Model
# ===================================================================

# --- Define tabular features and full target set ---
tabular_feature_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
                        "NumRings","HBD","HBA","LogP","TPSA"]

X_tabular = df[tabular_feature_cols]
X_smiles = df['SMILES']
y_final = df[TARGET_VARIABLE]

# --- Train/validation split for the final XGBoost model ---
X_train_tab, X_val_tab, smiles_train, smiles_val, y_train_xgb, y_val_xgb = train_test_split(
    X_tabular, X_smiles, y_final, test_size=0.2, random_state=42
)

# --- Scale the TABULAR features ---
scaler = StandardScaler()
X_train_tabular_scaled = scaler.fit_transform(X_train_tab)
X_val_tabular_scaled = scaler.transform(X_val_tab)

# --- Generate LSTM features using the TRAINED extractor ---
train_sequences = pad_sequences(smiles_tokenizer.texts_to_sequences(smiles_train), maxlen=max_length)
val_sequences = pad_sequences(smiles_tokenizer.texts_to_sequences(smiles_val), maxlen=max_length)

X_train_lstm_features = bilstm_feature_extractor.predict(train_sequences)
X_val_lstm_features = bilstm_feature_extractor.predict(val_sequences)

# --- Combine Tabular and LSTM Features ---
X_train_combined = np.hstack([X_train_tabular_scaled, X_train_lstm_features])
X_val_combined = np.hstack([X_val_tabular_scaled, X_val_lstm_features])

print(f"Shape of combined training features: {X_train_combined.shape}")
print(f"Shape of combined validation features: {X_val_combined.shape}\\n")

# --- Train FINAL XGBoost model on COMBINED features ---
xgb_model_final = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model_final.fit(X_train_combined, y_train_xgb)

# --- Final Evaluation ---
y_val_pred = xgb_model_final.predict(X_val_combined)
mae = mean_absolute_error(y_val_xgb, y_val_pred)
rmse = mean_squared_error(y_val_xgb, y_val_pred, squared=False)
r2 = r2_score(y_val_xgb, y_val_pred)

print(f"--- Combined BiLSTM + XGBoost Model Results for {TARGET_VARIABLE} ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RÂ²:   {r2:.4f}")


# ===================================================================
# Predict Tc on Test Set with Combined BiLSTM + XGBoost Model
# ===================================================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# NOTE: This script assumes that 'xgb_model_final', 'bilstm_feature_extractor',
# 'scaler', and 'smiles_tokenizer' from the previous training cell are in memory.

# --- 1. Load and Preprocess Test Data ---
print("--- Loading and preprocessing test data... ---")
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

# --- Molecular descriptor function (must be the same as used for training data) ---
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return a dictionary of NaNs with the correct keys
        return {
            "MolWt": np.nan, "NumAtoms": np.nan, "NumBonds": np.nan,
            "NumRotatableBonds": np.nan, "NumRings": np.nan,
            "HBD": np.nan, "HBA": np.nan, "LogP": np.nan, "TPSA": np.nan
        }
    return {
        "MolWt": Descriptors.MolWt(mol),
        "NumAtoms": mol.GetNumAtoms(),
        "NumBonds": mol.GetNumBonds(),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumRings": Descriptors.RingCount(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol)
    }

# --- Extract tabular descriptors for the test set ---
test_desc = [smiles_to_descriptors(s) for s in test_df['SMILES']]
test_desc_df = pd.DataFrame(test_desc)

# --- 2. Feature Engineering for Test Set ---

# A. Process TABULAR features
# These must be the same columns used for training the XGBoost model
tabular_feature_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
                        "NumRings","HBD","HBA","LogP","TPSA"]
X_test_tabular = test_desc_df[tabular_feature_cols]

# Scale tabular features using the PRE-FITTED scaler from training
X_test_tabular_scaled = scaler.transform(X_test_tabular)
print(f"Shape of scaled tabular test features: {X_test_tabular_scaled.shape}")

# B. Process SMILES features for LSTM
# Tokenize and pad SMILES strings using the PRE-FITTED tokenizer
test_sequences = pad_sequences(
    smiles_tokenizer.texts_to_sequences(test_df['SMILES']),
    maxlen=max_length
)

# Generate features using the TRAINED BiLSTM feature extractor
X_test_lstm_features = bilstm_feature_extractor.predict(test_sequences)
print(f"Shape of LSTM test features: {X_test_lstm_features.shape}")


# C. Combine Tabular and LSTM Features
X_test_combined = np.hstack([X_test_tabular_scaled, X_test_lstm_features])
print(f"Shape of combined test features: {X_test_combined.shape}\n")


# --- 3. Make Predictions ---
print("--- Generating predictions on the test set... ---")
# Use the final trained XGBoost model to predict
y_test_pred = xgb_model_final.predict(X_test_combined)

# Note: No back-transformation (like np.expm1) is needed because the target
# variable 'Tc' was not log-transformed during the training of this hybrid model.


# --- 4. Save Submission File ---
submission = pd.DataFrame({
    "id": test_df["id"],
    TARGET_VARIABLE: y_test_pred  # Use the target variable name from training
})


submission_filename = f"submission_{TARGET_VARIABLE.lower()}_hybrid.csv"
submission.to_csv(submission_filename, index=False)

print(f"âœ… Submission file saved as {submission_filename}")
print(submission.head())
s_density=submission.copy()

model_density=model


# Keep only rows with Rg available
df_rg = data_features.dropna(subset=["Rg"])

print("Shape of data with Rg:", df_rg.shape)



feature_cols = [
    "MolWt","NumAtoms","NumBonds","NumRotatableBonds",
    "NumRings","HBD","HBA","LogP","TPSA"
]

X = df_rg[feature_cols]
y = df_rg["Rg"]



import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Histogram + KDE
sns.histplot(y, kde=True, bins=30, color="skyblue")
plt.title("Distribution of Rg")
plt.show()

# Q-Q plot
stats.probplot(y, dist="norm", plot=plt)
plt.title("Q-Q Plot of Rg")
plt.show()



from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Rg Model Results:")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("RÂ²:", round(r2, 4))



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# -------------------
# Split Data
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------
# Normalize Inputs
# -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------
# Yeoâ€“Johnson on target
# -------------------
pt = PowerTransformer(method='yeo-johnson')
y_train_trans = pt.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# -------------------
# Train model
# -------------------
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train_scaled, y_train_trans)

# -------------------
# Evaluate
# -------------------
y_pred_trans = model.predict(X_test_scaled)
y_pred = pt.inverse_transform(y_pred_trans.reshape(-1, 1)).ravel()

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Rg Model Results with Yeoâ€“Johnson:")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("RÂ²:", round(r2, 4))



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# TensorFlow / Keras for the BiLSTM part
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===================================================================
# 1. Data Preparation (for Rg)
# ===================================================================

# Set the target to Rg and create a specific DataFrame
TARGET_VARIABLE_RG = 'Rg'
df_rg = data_features.dropna(subset=[TARGET_VARIABLE_RG]).copy()


# ===================================================================
# 2. SMILES Preprocessing for LSTM (for Rg)
# ===================================================================
# Use a new tokenizer specific to the Rg model
smiles_tokenizer_rg = Tokenizer(char_level=True)
smiles_tokenizer_rg.fit_on_texts(df_rg['SMILES'])
vocab_size_rg = len(smiles_tokenizer_rg.word_index) + 1
max_length_rg = 150 # Keep consistent


# ===================================================================
# 3. BiLSTM Training Model (for Rg)
# ===================================================================
embedding_dim = 64
lstm_units = 32

# Use unique names for layers and the model
input_layer_rg = Input(shape=(max_length_rg,), name='smiles_input_rg')
embedding_layer_rg = Embedding(input_dim=vocab_size_rg, output_dim=embedding_dim, input_length=max_length_rg)(input_layer_rg)
bilstm_layer_rg = Bidirectional(LSTM(units=lstm_units), name='bilstm_feature_layer_rg')(embedding_layer_rg)
output_layer_rg = Dense(1, name='prediction_output_rg')(bilstm_layer_rg)

# Create a specific model for Rg
bilstm_training_model_rg = Model(inputs=input_layer_rg, outputs=output_layer_rg)
bilstm_training_model_rg.compile(optimizer='adam', loss='mean_squared_error')
print(f"--- BiLSTM Training Model Summary for {TARGET_VARIABLE_RG} ---")
bilstm_training_model_rg.summary()
print("\n")

# ===================================================================
# 4. Train the BiLSTM Network (on Rg)
# ===================================================================
smiles_sequences_rg = pad_sequences(smiles_tokenizer_rg.texts_to_sequences(df_rg['SMILES']), maxlen=max_length_rg)
y_rg = df_rg[TARGET_VARIABLE_RG].values

X_train_seq_rg, X_val_seq_rg, y_train_lstm_rg, y_val_lstm_rg = train_test_split(
    smiles_sequences_rg, y_rg, test_size=0.2, random_state=42
)

print(f"\n--- Training the BiLSTM Model on {TARGET_VARIABLE_RG} ---")
history_rg = bilstm_training_model_rg.fit(
    X_train_seq_rg, y_train_lstm_rg,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_seq_rg, y_val_lstm_rg),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ],
    verbose=1
)

# ===================================================================
# 5. Create the BiLSTM Feature Extractor (for Rg)
# ===================================================================
bilstm_feature_extractor_rg = Model(
    inputs=bilstm_training_model_rg.input,
    outputs=bilstm_training_model_rg.get_layer('bilstm_feature_layer_rg').output
)
print(f"\n--- BiLSTM Feature Extractor for {TARGET_VARIABLE_RG} is ready ---")


# ===================================================================
# 6. Combined Training Pipeline for Final XGBoost Model (for Rg)
# ===================================================================
tabular_feature_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
                        "NumRings","HBD","HBA","LogP","TPSA"]

X_tabular_rg = df_rg[tabular_feature_cols]
X_smiles_rg = df_rg['SMILES']
y_final_rg = df_rg[TARGET_VARIABLE_RG]

X_train_tab_rg, X_val_tab_rg, smiles_train_rg, smiles_val_rg, y_train_xgb_rg, y_val_xgb_rg = train_test_split(
    X_tabular_rg, X_smiles_rg, y_final_rg, test_size=0.2, random_state=42
)

# --- Use a new scaler specific to the Rg model ---
scaler_rg = StandardScaler()
X_train_tabular_scaled_rg = scaler_rg.fit_transform(X_train_tab_rg)
X_val_tabular_scaled_rg = scaler_rg.transform(X_val_tab_rg)

# --- Generate LSTM features ---
train_sequences_rg = pad_sequences(smiles_tokenizer_rg.texts_to_sequences(smiles_train_rg), maxlen=max_length_rg)
val_sequences_rg = pad_sequences(smiles_tokenizer_rg.texts_to_sequences(smiles_val_rg), maxlen=max_length_rg)

X_train_lstm_features_rg = bilstm_feature_extractor_rg.predict(train_sequences_rg)
X_val_lstm_features_rg = bilstm_feature_extractor_rg.predict(val_sequences_rg)

# --- Combine features ---
X_train_combined_rg = np.hstack([X_train_tabular_scaled_rg, X_train_lstm_features_rg])
X_val_combined_rg = np.hstack([X_val_tabular_scaled_rg, X_val_lstm_features_rg])

# --- Train FINAL XGBoost model for Rg ---
xgb_model_final_rg = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
xgb_model_final_rg.fit(X_train_combined_rg, y_train_xgb_rg)

# --- Final Evaluation ---
y_val_pred_rg = xgb_model_final_rg.predict(X_val_combined_rg)
mae_rg = mean_absolute_error(y_val_xgb_rg, y_val_pred_rg)
rmse_rg = mean_squared_error(y_val_xgb_rg, y_val_pred_rg, squared=False)
r2_rg = r2_score(y_val_xgb_rg, y_val_pred_rg)

print(f"\n--- Combined BiLSTM + XGBoost Model Results for {TARGET_VARIABLE_RG} ---")
print(f"MAE:  {mae_rg:.4f}")
print(f"RMSE: {rmse_rg:.4f}")
print(f"RÂ²:   {r2_rg:.4f}")



# ===================================================================
# Predict Rg on Test Set with Combined BiLSTM + XGBoost Model
# ===================================================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# NOTE: This script assumes that 'xgb_model_final_rg', 'bilstm_feature_extractor_rg',
# 'scaler_rg', and 'smiles_tokenizer_rg' from the previous training cell are in memory.

# --- 1. Load and Preprocess Test Data ---
print("--- Loading and preprocessing test data for Rg prediction... ---")
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

# --- Molecular descriptor function (must be the same as used for training data) ---
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return a dictionary of NaNs with the correct keys
        return {
            "MolWt": np.nan, "NumAtoms": np.nan, "NumBonds": np.nan,
            "NumRotatableBonds": np.nan, "NumRings": np.nan,
            "HBD": np.nan, "HBA": np.nan, "LogP": np.nan, "TPSA": np.nan
        }
    return {
        "MolWt": Descriptors.MolWt(mol),
        "NumAtoms": mol.GetNumAtoms(),
        "NumBonds": mol.GetNumBonds(),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumRings": Descriptors.RingCount(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol)
    }

# --- Extract tabular descriptors for the test set ---
test_desc = [smiles_to_descriptors(s) for s in test_df['SMILES']]
test_desc_df = pd.DataFrame(test_desc)

# --- 2. Feature Engineering for Test Set ---

# A. Process TABULAR features
# These must be the same columns used for training the XGBoost model
tabular_feature_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
                        "NumRings","HBD","HBA","LogP","TPSA"]
X_test_tabular = test_desc_df[tabular_feature_cols]

# Scale tabular features using the PRE-FITTED scaler from the Rg training
X_test_tabular_scaled = scaler_rg.transform(X_test_tabular)
print(f"Shape of scaled tabular test features: {X_test_tabular_scaled.shape}")

# B. Process SMILES features for LSTM
# Tokenize and pad SMILES strings using the PRE-FITTED tokenizer from Rg training
test_sequences = pad_sequences(
    smiles_tokenizer_rg.texts_to_sequences(test_df['SMILES']),
    maxlen=max_length_rg
)

# Generate features using the TRAINED BiLSTM feature extractor from Rg training
X_test_lstm_features = bilstm_feature_extractor_rg.predict(test_sequences)
print(f"Shape of LSTM test features: {X_test_lstm_features.shape}")


# C. Combine Tabular and LSTM Features
X_test_combined = np.hstack([X_test_tabular_scaled, X_test_lstm_features])
print(f"Shape of combined test features: {X_test_combined.shape}\n")


# --- 3. Make Predictions ---
print("--- Generating Rg predictions on the test set... ---")
# Use the final trained XGBoost model for Rg to predict
y_test_pred = xgb_model_final_rg.predict(X_test_combined)

# Note: No back-transformation (like np.expm1) is needed.

# --- 4. Save Submission File ---
submission = pd.DataFrame({
    "id": test_df["id"],
    TARGET_VARIABLE_RG: y_test_pred  # Use the specific target variable name from Rg training
})

# Create a copy with the specific name you requested
s_Rg = submission.copy()

# Save the submission file with a specific name for Rg
submission_filename = f"submission_{TARGET_VARIABLE_RG.lower()}_hybrid.csv"
submission.to_csv(submission_filename, index=False)

print(f"âœ… Submission file saved as {submission_filename}")
print(submission.head())

# Save the final model in a variable with the name you requested
model_Rg = xgb_model_final_rg



df=pd.read_csv("/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv")


df.info()


# Make a copy to avoid changing the original data
df_clean = df.copy()

# Rename the target column
df_clean = df_clean.rename(columns={'Tg (C)': 'Tg'})

# Drop the unnecessary 'Unnamed: 0' column
df_clean = df_clean.drop(columns=['Unnamed: 0'])

# Check the result
print(df_clean.head())


# Get summary statistics (mean, std, min, max, etc.)
print(df_clean['Tg'].describe())

# Plot a histogram to see the distribution visually
df_clean['Tg'].hist(bins=30, edgecolor='black')


data_features.info()


# Assuming your first DataFrame (662 polymers) is named 'df1'
# and you have already cleaned it
df1_subset = data_features[['SMILES', 'Tg']]

# Assuming your new, larger DataFrame is named 'df2'
df2_subset = df_clean[['SMILES', 'Tg']]


# Drop rows where 'Tg' is NaN (Not a Number)
df2_subset_clean = df2_subset.dropna(subset=['Tg'])


# df1_subset has 662 rows
# df2_subset_clean has 557 rows
final_df = pd.concat([df1_subset, df2_subset_clean], ignore_index=True)

print("Concatenation complete!")
print(f"Total size of final training set: {len(final_df)}")
# Expected output: Total size of final training set: 1219

# Verify the result
print(final_df.info())


import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Let's assume 'final_df' is your DataFrame with a 'SMILES' column.
# If you don't have a 'final_df' to test with, you can create a sample one:
# data = {'SMILES': ['CCO', 'c1ccccc1'], 'Tg': [150.0, 373.0]}
# final_df = pd.DataFrame(data)

def calculate_descriptors(smiles_string):
    """
    Calculates a list of RDKit descriptors for a given SMILES string.
    Returns a dictionary of descriptor names and their values.
    Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        # Return a dictionary of NaNs if the SMILES is invalid
        return {
            "MolWt": None, "NumAtoms": None, "NumBonds": None,
            "NumRotatableBonds": None, "NumRings": None, "HBD": None,
            "HBA": None, "LogP": None, "TPSA": None
        }

    # Calculate descriptors
    descriptors = {
        "MolWt": Descriptors.MolWt(mol),
        "NumAtoms": mol.GetNumAtoms(),
        "NumBonds": mol.GetNumBonds(),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumRings": Descriptors.RingCount(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol)
    }
    return descriptors

# --- Main Execution ---

# 1. Apply the function to each SMILES string in the DataFrame
# This creates a list of dictionaries.
print("Calculating descriptors for all SMILES...")
descriptor_list = final_df['SMILES'].apply(calculate_descriptors)

# 2. Convert the list of dictionaries into a new DataFrame
descriptor_df = pd.DataFrame(descriptor_list.tolist())

# 3. Integrate the new descriptors into the original DataFrame
# This joins the two DataFrames side-by-side.
print("Integrating new features into the DataFrame...")
final_df_with_features = pd.concat([final_df, descriptor_df], axis=1)

# 4. Verify the result
print("\nIntegration complete!")
print(final_df_with_features.head())
print("\nDataFrame Info:")
final_df_with_features.info()


final_df_clean = final_df_with_features.dropna()
final_df_clean.info()


data_features=final_df_clean.copy()


# ================================
# Tg Training using precomputed features
# ================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

# --- Use your dataset ---
data_t = data_features.copy()

# --- Target: Tg ---
df_tg = data_t.dropna(subset=["Tg"]).copy()

feature_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
                "NumRings","HBD","HBA","LogP","TPSA"]

X = df_tg[feature_cols]
y = df_tg["Tg"]

# --- Train/validation split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# --- Train model ---
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# --- Evaluate ---
y_val_pred = model.predict(X_val_scaled)
mae = mean_absolute_error(y_val, y_val_pred)
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
r2 = r2_score(y_val, y_val_pred)

print(f"Optimized Tg Model Results:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RÂ²:   {r2:.4f}")

# --- Save model & scaler for later use ---
joblib.dump(model, "xgb_tg_model.pkl")
joblib.dump(scaler, "scaler_tg.pkl")
print("âœ… Tg model and scaler saved.")



import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- Subset Tg values ---
tg_values = data_features["Tg"].dropna()

# --- Basic statistics ---
print("Tg Summary Statistics:")
print(tg_values.describe())

# --- Histogram & KDE ---
plt.figure(figsize=(12,4))
sns.histplot(tg_values, bins=30, kde=True, color="skyblue")
plt.title("Distribution of Tg")
plt.xlabel("Tg")
plt.ylabel("Frequency")
plt.show()

# --- Q-Q Plot ---
plt.figure(figsize=(5,5))
stats.probplot(tg_values, dist="norm", plot=plt)
plt.title("Q-Q Plot of Tg")
plt.show()

# --- Shapiro-Wilk Normality Test ---
stat, p = stats.shapiro(tg_values)
print("Shapiro-Wilk Test:")
print(f"Statistic={stat:.4f}, p-value={p:.4e}")
if p > 0.05:
    print("âœ… Data looks approximately normal.")
else:
    print("â�Œ Data is not normally distributed.")



import matplotlib.pyplot as plt
import seaborn as sns

# --- Boxplot for Tg ---
plt.figure(figsize=(8,5))
sns.boxplot(x=df_tg["Tg"], color="skyblue")
plt.title("Boxplot of Tg (Outlier Detection)", fontsize=14)
plt.xlabel("Tg")
plt.show()

# --- IQR-based Outlier Detection ---
Q1 = df_tg["Tg"].quantile(0.25)
Q3 = df_tg["Tg"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_tg[(df_tg["Tg"] < lower_bound) | (df_tg["Tg"] > upper_bound)]

print(f"Number of outliers: {len(outliers)}")
print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
print(outliers["Tg"].describe())



 data_features.info()
df_tg= data_features.copy()
print(df_tg.info())


Q1 = df_tg["Tg"].quantile(0.25)
Q3 = df_tg["Tg"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_no_outliers = df_tg[(df_tg["Tg"] >= lower_bound) & (df_tg["Tg"] <= upper_bound)]



print("Before:", df_tg["Tg"].notnull().sum())
print("After:", df_no_outliers["Tg"].notnull().sum())



df_no_outliers.info()


# ================================
# Tg Training using precomputed features (No Outliers)
# ================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# --- Use the cleaned dataset ---
df_tg = df_no_outliers.copy()

# --- Define features and target ---
feature_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
                "NumRings","HBD","HBA","LogP","TPSA"]

X = df_tg[feature_cols]
y = df_tg["Tg"]

# --- Train/validation split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# --- Train model ---
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# --- Evaluate ---
y_val_pred = model.predict(X_val_scaled)
mae = mean_absolute_error(y_val, y_val_pred)
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
r2 = r2_score(y_val, y_val_pred)

print(f"Tg Model Results (No Outliers):")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RÂ²:   {r2:.4f}")



# ===================================================================
# Predict Tg on Test Set with Precomputed Features Model
# ===================================================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# NOTE: This script assumes that 'model' (the trained XGBoost regressor for Tg)
# and 'scaler' (the fitted StandardScaler for Tg features) from the training
# cell are already in memory.

# --- 1. Load and Preprocess Test Data ---
print("--- Loading and preprocessing test data for Tg prediction... ---")
try:
    test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
except FileNotFoundError:
    print("Test file not found. Please ensure the Kaggle dataset is correctly mounted.")
    # As a fallback for local testing, create a dummy test_df
    test_df = pd.DataFrame({
        'id': range(5),
        'SMILES': [
            '*C(C)C(=O)OC', # PMMA monomer
            '*CC(*)(C)C(=O)OC',
            '*C(C)C#N', # PAN monomer
            '*C(C1=CC=CC=C1)C*', # Polystyrene dimer
            '*OCCCCO*' # Poly(butylene oxide)
        ]
    })


# --- Molecular descriptor function (must be the same as used for training data) ---
def smiles_to_descriptors(smiles):
    """Calculates molecular descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return a dictionary of NaNs with the correct keys if SMILES is invalid
        return {
            "MolWt": np.nan, "NumAtoms": np.nan, "NumBonds": np.nan,
            "NumRotatableBonds": np.nan, "NumRings": np.nan,
            "HBD": np.nan, "HBA": np.nan, "LogP": np.nan, "TPSA": np.nan
        }
    return {
        "MolWt": Descriptors.MolWt(mol),
        "NumAtoms": mol.GetNumAtoms(),
        "NumBonds": mol.GetNumBonds(),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumRings": Descriptors.RingCount(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol)
    }

# --- Extract tabular descriptors for the test set ---
test_descriptors = [smiles_to_descriptors(s) for s in test_df['SMILES']]
test_desc_df = pd.DataFrame(test_descriptors)


# --- 2. Feature Engineering for Test Set ---
print("\n--- Preparing features for the test set... ---")
# These must be the exact same feature columns used for training the Tg model
feature_cols = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
                "NumRings","HBD","HBA","LogP","TPSA"]
X_test = test_desc_df[feature_cols]

# Handle potential missing values from invalid SMILES if any
if X_test.isnull().sum().sum() > 0:
    print("Warning: Missing values found in descriptors, filling with column mean.")
    X_test = X_test.fillna(X_test.mean())


# Scale features using the PRE-FITTED scaler from the Tg training
X_test_scaled = scaler.transform(X_test)
print(f"Shape of scaled test features: {X_test_scaled.shape}")


# --- 3. Make Predictions ---
print("\n--- Generating Tg predictions on the test set... ---")
# Use the final trained XGBoost model for Tg to predict
y_test_pred_tg = model.predict(X_test_scaled)


# --- 4. Save Submission File ---
submission_tg = pd.DataFrame({
    "id": test_df["id"],
    "Tg": y_test_pred_tg
})

submission_filename = "submission_tg.csv"
submission_tg.to_csv(submission_filename, index=False)

print(f"\nâœ… Submission file for Tg saved as {submission_filename}")
print(submission_tg.head())

# Optionally, save the final model in a variable with a specific name
model_Tg = model


s_Tg=submission_tg.copy()


model


print(f"Tg Model Results (No Outliers):")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RÂ²:   {r2:.4f}")



df_no_outliers


# ===================================================================
# Full Script: Decoupled Ensemble Strategy
# ===================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------
# 1. DATA PREPARATION
# -----------------------------------
print("--- 1. Preparing Decoupled Feature Sets ---")
df = df_no_outliers.copy()
y = df["Tg"]

# a) Feature Set 1: 9 RDKit Descriptors
feature_cols_orig = ["MolWt","NumAtoms","NumBonds","NumRotatableBonds",
                     "NumRings","HBD","HBA","LogP","TPSA"]
X_original = df[feature_cols_orig]

# b) Feature Set 2: 2048 Morgan Fingerprints
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros(n_bits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)
fp_features = np.array([smiles_to_fingerprint(s) for s in df['SMILES']])
X_fingerprints = pd.DataFrame(fp_features, columns=[f'fp_{i}' for i in range(fp_features.shape[1])])

# c) Create consistent Train/Test splits for both feature sets
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X_original, y, test_size=0.2, random_state=42
)
X_train_fp, X_test_fp, _, _ = train_test_split(
    X_fingerprints, y, test_size=0.2, random_state=42
)

# -----------------------------------
# 2. TRAIN MODEL A: XGBoost on Descriptors
# -----------------------------------
print("\n--- 2. Training Model A: XGBoost on 9 Descriptors (Optimized) ---")
scaler_orig = StandardScaler()
X_train_orig_scaled = scaler_orig.fit_transform(X_train_orig)
X_test_orig_scaled = scaler_orig.transform(X_test_orig)

best_baseline_params = {
    'n_estimators': 361, 'max_depth': 12, 'learning_rate': 0.06539, 'subsample': 0.7297,
    'colsample_bytree': 0.8837, 'reg_alpha': 0.0001825, 'reg_lambda': 0.00177, 'min_child_weight': 2
}
xgb_model = xgb.XGBRegressor(**best_baseline_params, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_orig_scaled, y_train)
preds_xgb = xgb_model.predict(X_test_orig_scaled)
mae_xgb = mean_absolute_error(y_test, preds_xgb)
print(f"âœ… XGBoost Descriptors MAE: {mae_xgb:.4f}")

# -----------------------------------
# 3. TRAIN MODEL B: LightGBM on Fingerprints
# -----------------------------------
print("\n--- 3. Training Model B: LightGBM on 2048 Fingerprints (UNSCALED) ---")
lgbm_model = lgb.LGBMRegressor(objective='mae', n_estimators=1000, random_state=42, n_jobs=-1)
# Fit on the unscaled fingerprint data
lgbm_model.fit(X_train_fp, y_train, eval_set=[(X_test_fp, y_test)],
               callbacks=[lgb.early_stopping(100, verbose=False)])
preds_lgbm = lgbm_model.predict(X_test_fp)
mae_lgbm = mean_absolute_error(y_test, preds_lgbm)
print(f"âœ… LightGBM Fingerprints MAE: {mae_lgbm:.4f}")

# -----------------------------------
# 4. TRAIN MODEL C: MLP on Fingerprints
# -----------------------------------
print("\n--- 4. Training Model C: MLP on 2048 Fingerprints (SCALED) ---")
scaler_fp = StandardScaler()
X_train_fp_scaled = scaler_fp.fit_transform(X_train_fp)
X_test_fp_scaled = scaler_fp.transform(X_test_fp)

mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500,
                         early_stopping=True, random_state=42)
mlp_model.fit(X_train_fp_scaled, y_train)
preds_mlp = mlp_model.predict(X_test_fp_scaled)
mae_mlp = mean_absolute_error(y_test, preds_mlp)
print(f"âœ… MLP Fingerprints MAE: {mae_mlp:.4f}")

# -----------------------------------
# 5. ENSEMBLE AND FINAL EVALUATION
# -----------------------------------
print("\n--- 5. Blending Predictions for Final Ensemble ---")
# Adjust weights based on expected performance. If fingerprint models are weak, lower their weight.
preds_ensemble = (preds_xgb * 0.6) + (preds_lgbm * 0.2) + (preds_mlp * 0.2)
mae_ensemble = mean_absolute_error(y_test, preds_ensemble)
r2_ensemble = r2_score(y_test, preds_ensemble)

print("\nğŸ�† FINAL ENSEMBLE RESULTS ğŸ�†")
print(f"MAE:  {mae_ensemble:.4f}")
print(f"RÂ²:   {r2_ensemble:.4f}")


import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

# =======================================================
# 1. Prepare the FULL Training Data
# =======================================================
print("--- Preparing the full training dataset ---")
# Use all your clean data for training and evaluation
train_df = df_no_outliers.copy() 

# Define features and target from your best model
feature_cols = ["MolWt", "NumAtoms", "NumBonds", "NumRotatableBonds",
                "NumRings", "HBD", "HBA", "LogP", "TPSA"]
X = train_df[feature_cols]
y = train_df["Tg"]

# Scale all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Prepared and scaled {X_scaled.shape[0]} training samples.")

# =======================================================
# 2. Get a Reliable Performance Estimate with Cross-Validation
# =======================================================
print("\n--- Evaluating model with 5-fold cross-validation ---")
# Use your optimized parameters
best_params = {
    'n_estimators': 361, 'max_depth': 12, 'learning_rate': 0.06539, 
    'subsample': 0.7297, 'colsample_bytree': 0.8837, 
    'reg_alpha': 0.0001825, 'reg_lambda': 0.00177, 'min_child_weight': 2
}
cv_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)

# Define the cross-validation strategy
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation to get the average MAE
scores = cross_val_score(cv_model, X_scaled, y, cv=kfold, 
                         scoring='neg_mean_absolute_error', n_jobs=-1)

print(f"âœ… Your model's estimated MAE is: {-scores.mean():.4f} (+/- {scores.std():.4f})")


# =======================================================
# 3. Train the Final Model on 100% of the Data
# =======================================================
print("\n--- Training final XGBoost model on ALL data for submission ---")
final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)

# Fit the model on the entire dataset to maximize learning
final_model.fit(X_scaled, y)
print("âœ… Final model trained successfully.")

# =======================================================
# 4. Prepare the Test Data and Predict
# =======================================================
print("\n--- Loading test data and generating final predictions ---")
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

# --- Molecular descriptor function ---
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: np.nan for k in feature_cols}
    return {
        "MolWt": Descriptors.MolWt(mol), "NumAtoms": mol.GetNumAtoms(),
        "NumBonds": mol.GetNumBonds(), "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumRings": Descriptors.RingCount(mol), "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol), "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol)
    }

# Create features for the test set
test_descriptors = [smiles_to_descriptors(s) for s in test_df['SMILES']]
X_test = pd.DataFrame(test_descriptors)

# Use the scaler that was FITTED on the full training data
X_test_scaled = scaler.transform(X_test)

# Make final predictions
predictions = final_model.predict(X_test_scaled)
print(f"Generated {len(predictions)} predictions.")

# =======================================================
# 5. Create Submission File
# =======================================================
print("\n--- Creating submission file ---")
submission_df = pd.DataFrame({'id': test_df['id'], 'Tg': predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nğŸ�† All done! Your submission.csv file is ready. ğŸ�†")
print(submission_df.head())


s_Tg=submission_df.copy()


import pandas as pd
import numpy as np

# ===================================================================
# 1. (Optional) Create Dummy Prediction Files for Demonstration


# ===================================================================
# 2. Load and Concatenate Files
# ===================================================================
print("--- Loading and merging individual prediction files ---")

# --- Load each prediction file ---
df_tg = s_Tg
df_ffv = s_FFV
df_tc = s_TC
df_density = s_density
df_rg = s_Rg

# --- Merge all DataFrames into one ---
# Start with the first DataFrame and sequentially merge the others based on the 'id' column
submission_df = pd.merge(df_tg, df_ffv, on='id')
submission_df = pd.merge(submission_df, df_tc, on='id')
submission_df = pd.merge(submission_df, df_density, on='id')
submission_df = pd.merge(submission_df, df_rg, on='id')

print("All files merged successfully.")


# ===================================================================
# 3. Prepare and Save Final Output
# ===================================================================

# --- Select and reorder columns to match the required format ---
final_columns = ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
submission_df = submission_df[final_columns]

# --- Save to 'submission.csv' ---
submission_df.to_csv('submission.csv', index=False)

print("\nâœ… Submission file saved as 'submission.csv'")
print("\n--- Final Submission Preview ---")
print(submission_df.head())


print(submission_df)

