# âœ… Install RDKit in Colab
!pip install rdkit-pypi

# Imports
import warnings
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Suppress all warnings
warnings.filterwarnings("ignore")
# Load the dataset
df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
# Loading Test Data
new_data = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
# View first few rows
df.head()


# Info about data types and nulls
df.info()

# Summary of missing values
missing = df.isnull().sum()
print("Missing values in each column:\n", missing[missing > 0])


#Heatmap showing missing data
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='YlOrBr')
plt.title("Missing Data Heatmap")
plt.show()


# Show summary stats for numeric columns
df.describe()


# Select numeric columns only
numeric_cols = df.select_dtypes(include='number').columns.tolist()

# Plot histograms
df[numeric_cols].hist(figsize=(14, 10), bins=30)
plt.suptitle("Distributions of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()


#correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix of Numeric Features")
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Rg', y='Tc', alpha=0.6, s=50, edgecolor='k')

plt.title('Scatter Plot of Thermal Conductivity (Tc) vs Radius of Gyration (Rg)', fontsize=14)
plt.xlabel('Radius of Gyration (Rg)', fontsize=12)
plt.ylabel('Thermal Conductivity (Tc)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * 5
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]

# Compute descriptors
descriptor_names = ['MolWt', 'LogP', 'RotBonds', 'HDonors', 'HAcceptors']
new_data[descriptor_names] = new_data['SMILES'].apply(compute_descriptors).apply(pd.Series)

# Apply to a copy of your SMILES column (safe practice)
descriptor_names = ['MolWt', 'LogP', 'RotBonds', 'HDonors', 'HAcceptors']
descriptor_df = df['SMILES'].apply(compute_descriptors).apply(pd.Series)
descriptor_df.columns = descriptor_names

# Combine original df with new descriptors
df_extended = pd.concat([df, descriptor_df], axis=1)

df_extended


targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
descriptor_names = ['MolWt', 'LogP', 'RotBonds', 'HDonors', 'HAcceptors']

# Loop to train and save each model
for target in targets:
    # Drop rows with missing target values
    data = df_extended.dropna(subset=[target])
    X = data[descriptor_names]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Save model
    joblib.dump(model, f"model_{target}.pkl")

    print(f"âœ… Trained and saved model_{target}.pkl")
    print(f"\nðŸ“Š Target: {target}")
    print(f"âœ… RÂ² Score: {r2:.4f}")
    print(f"ðŸ“‰ MSE: {mse:.4f}")
    print("")


model_tg = joblib.load("model_Tg.pkl")
model_ffv = joblib.load("model_FFV.pkl")
model_tc = joblib.load("model_Tc.pkl")
model_density = joblib.load("model_Density.pkl")
model_rg = joblib.load("model_Rg.pkl")

X_new = new_data[descriptor_names]

# Predict
new_data["Tg_pred"] = model_tg.predict(X_new)
new_data["FFV_pred"] = model_ffv.predict(X_new)
new_data["Tc_pred"] = model_tc.predict(X_new)
new_data["Density_pred"] = model_density.predict(X_new)
new_data["Rg_pred"] = model_rg.predict(X_new)

# View predictions
print(new_data[["id", "Tg_pred", "FFV_pred", "Tc_pred", "Density_pred", "Rg_pred"]])


# save to CSV
new_data.to_csv('submission.csv', index=False)

