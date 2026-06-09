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


# Load datasets
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

# Check shapes
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample submission shape:", sample_submission.shape)

# View first few rows
train_df.head()


import warnings
warnings.filterwarnings("ignore")


train_df.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].isnull(), cbar=False)
plt.title("Missing Targets in Train Set")
plt.show()


for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    train_df[col].dropna().hist(bins=30)
    plt.title(col)
    plt.show()


# from rdkit import Chem
# from rdkit.Chem import Descriptors

# def featurize(smiles):
#     mol = Chem.MolFromSmiles(smiles.replace("*", "H"))  # Replace wildcards
#     if mol is None:
#         return [np.nan]*5
#     return [
#         Descriptors.MolWt(mol),
#         Descriptors.NumValenceElectrons(mol),
#         Descriptors.TPSA(mol),
#         Descriptors.MolLogP(mol),
#         Descriptors.NumRotatableBonds(mol),
#     ]

# train_df[['MolWt', 'ValenceElectrons', 'TPSA', 'LogP', 'RotBonds']] = train_df['SMILES'].apply(lambda x: pd.Series(featurize(x)))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Filter out rows where FFV is not missing
ffv_data = train_df[train_df['FFV'].notnull()].copy()

# Text featurization from SMILES
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)
X = tfidf.fit_transform(ffv_data['SMILES'])
y = ffv_data['FFV'].values

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"FFV RMSE: {rmse:.4f}")


test_tfidf = tfidf.transform(test_df['SMILES'])
ffv_preds = model.predict(test_tfidf)


# Create a new DataFrame for submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Tg': 0,           # placeholder
    'FFV': ffv_preds,  # your predicted values
    'Tc': 0,
    'Density': 0,
    'Rg': 0
})

# Save to CSV in the correct location
submission.to_csv('/kaggle/working/submission.csv', index=False)




