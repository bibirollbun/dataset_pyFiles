import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


train_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")


train_df.head(10)


train_df.info


train_df.describe()


train_df.shape


columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

stats = train_df[columns].agg(['min', 'max', 'mean']).T
stats.columns = ['Min', 'Max', 'Average']

print(stats)



import matplotlib.pyplot as plt
import seaborn as sns

target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

n_cols = 3  
n_rows = (len(target_cols) + n_cols - 1) // n_cols  

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

axes = axes.flatten()  

for i, col in enumerate(target_cols):
    sns.histplot(train_df[col], bins=50, kde=True, color='skyblue', edgecolor='black', ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")
    axes[i].grid(True)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



train_df['SMILES'].head()


import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def clean_and_validate_smiles(smiles):
    if not isinstance(smiles, str):
        return None

    smiles = re.sub(r'[!@#%^&*=+{}\[\]<>?|]', '', smiles)

    while '()' in smiles:
        smiles = smiles.replace('()', '')

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  
    return Chem.MolToSmiles(mol, canonical=True)  

train_df['SMILES'] = train_df['SMILES'].apply(clean_and_validate_smiles)

train_df = train_df.dropna(subset=['SMILES'])

print(f"SMILES Cleaned: {len(train_df)}")



train_df['SMILES'].head()





from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm
import numpy as np
import pandas as pd

def extract_rdkit_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * 7
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol) / mol.GetNumAtoms()
    ]

features = [extract_rdkit_features(smi) for smi in tqdm(train_df['SMILES'], disable=True)]  # هنا تعطيل شريط التقدم

features_df = pd.DataFrame(features, columns=[
    "MolWt", "LogP", "HDonors", "HAcceptors", "TPSA", "RotBonds", "AromaticProp"
])

train_df = pd.concat([train_df.reset_index(drop=True), features_df], axis=1)



print(train_df.columns.tolist())



train_df.shape


target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

for col in target_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_mask = (train_df[col] < lower_bound) | (train_df[col] > upper_bound)
    n_outliers = outliers_mask.sum()

    print(f"Column : {col} |Number of outliers: {n_outliers}")






print("Missing values:")
print(train_df.isnull().sum())





import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

feature_cols = ['MolWt', 'LogP', 'HDonors', 'HAcceptors', 'TPSA', 'RotBonds', 'AromaticProp']

def impute_with_nn(df, target_col):
    print(f"\n>> Column processing: {target_col}")

    if target_col == 'Tg':
        valid_mask = (df[target_col].notna()) & (df[target_col] >= 0)
    else:
        valid_mask = df[target_col].notna()

    df_valid = df[valid_mask]
    X_valid = df_valid[feature_cols]
    y_valid = df_valid[target_col]

    if len(df_valid) < 30:
        mean_val = y_valid.mean()
        print(f" The number of training samples is small. ({len(df_valid)}).Compensation at the average value : {mean_val:.3f}")
        df.loc[~valid_mask, target_col] = mean_val
        return df

    scaler = StandardScaler()
    X_valid_scaled = scaler.fit_transform(X_valid)

    X_train, X_val, y_train, y_val = train_test_split(X_valid_scaled, y_valid, test_size=0.2, random_state=42)

    if target_col in ['Tg', 'Rg']:
        activation_fn = 'relu'
    elif target_col == 'FFV':
        activation_fn = 'sigmoid'
    else:
        activation_fn = 'tanh'

    model = Sequential([
        Dense(64, activation=activation_fn, input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation=activation_fn),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0)

    if target_col == 'Tg':
        missing_mask = (df[target_col].isna()) | (df[target_col] < 0)
    else:
        missing_mask = df[target_col].isna()

    df_missing = df[missing_mask]
    if df_missing.empty:
        print(f" There are no missing or incorrect values to compensate for in {target_col}")
        return df

    X_missing = df_missing[feature_cols]
    X_missing_scaled = scaler.transform(X_missing)

    preds = model.predict(X_missing_scaled).flatten()
    df.loc[missing_mask, target_col] = preds

    print(f" Compensated {len(preds)} Value in the column {target_col}")
    return df

target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
for col in target_cols:
    train_df = impute_with_nn(train_df, col)


train_df.to_csv("Train_df_new.csv", index=False)
print("\n The file has been saved after compensation under the name : Train_df_new.csv")



target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

iteration = 0
while True:
    iteration += 1
    print(f"\n  Repetition number: {iteration} ---")
    non_outlier_mask = pd.Series([True] * len(train_df), index=train_df.index)
    total_outliers = 0

    for col in target_cols:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        is_not_outlier = (train_df[col] >= lower_bound) & (train_df[col] <= upper_bound)

        outliers_count = (~is_not_outlier).sum()
        total_outliers += outliers_count

        print(f"Column : {col} | Number of outlier values removed: {outliers_count}")

        non_outlier_mask &= is_not_outlier

    if total_outliers == 0:
        print("\n There are no remaining outliers in any of the columns.")
        break  

    train_df = train_df[non_outlier_mask].reset_index(drop=True)
    print(f"Deleted {total_outliers} An outlier value in this round. Remains.{len(train_df)} Rows .")



import matplotlib.pyplot as plt
import seaborn as sns

target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

train_df = train_df[train_df['Tg'] >= 0]
train_df = train_df.dropna(subset=['Tg'])

n_cols = len(target_cols)

plt.figure(figsize=(4 * n_cols, 4))

for i, col in enumerate(target_cols):
    plt.subplot(1, n_cols, i + 1)
    sns.histplot(train_df[col], bins=40, kde=True, color='skyblue')
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("")

plt.tight_layout()
plt.suptitle("Distributions of Target Columns After Cleaning", y=1.05, fontsize=14)
plt.show()






import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


numeric_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg', 
                'MolWt', 'LogP', 'HDonors', 'HAcceptors', 'TPSA', 'RotBonds', 'AromaticProp']

corr_matrix = train_df[numeric_cols].corr()

print(corr_matrix)

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix between Features and Targets")
plt.show()



print("Missing values:")
print(train_df.isnull().sum())


train_df.head(10)


train_df.info


train_df.describe()


train_df.shape





import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

feature_cols = ['MolWt', 'LogP', 'HDonors', 'HAcceptors', 'TPSA', 'RotBonds', 'AromaticProp']
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

train_df = train_df[train_df['Tg'] >= 0].dropna(subset=target_cols)

X = train_df[feature_cols]
y = train_df[target_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

xgb_base = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.9, random_state=42)
model = MultiOutputRegressor(xgb_base)

model.fit(X_train, y_train)

preds = model.predict(X_test)

def compute_weights(y_train):
    weights = []
    K = y_train.shape[1]
    n_samples = y_train.shape[0]
    r = y_train.max() - y_train.min()
    
    for i, col in enumerate(y_train.columns):
        ni = n_samples
        numerator = (1 / r[col]) * (K * np.sqrt(1 / ni))
        denominator = np.sum([np.sqrt(1 / n_samples) for _ in range(K)])
        wi = numerator / denominator
        weights.append(wi)
    return np.array(weights)

weights = compute_weights(y_train)

print(" Evaluation Metrics (XGBoost) :")
wMAE = 0
for i, col in enumerate(target_cols):
    r2 = r2_score(y_test.iloc[:, i], preds[:, i])
    rmse = mean_squared_error(y_test.iloc[:, i], preds[:, i], squared=False)
    mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
    weighted_mae = weights[i] * mae
    wMAE += weighted_mae
    
    print(f"\n--- {col} ---")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"w × MAE = {weights[i]:.6f} × {mae:.4f} = {weighted_mae:.6f}")

print(f"\n Final Weighted MAE (wMAE) : {wMAE:.6f}")



from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * 7  
    return [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumRotatableBonds(mol),
        Chem.rdMolDescriptors.CalcNumAromaticRings(mol) / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0
    ]

feature_cols = ['MolWt', 'LogP', 'HDonors', 'HAcceptors', 'TPSA', 'RotBonds', 'AromaticProp']
test_df[feature_cols] = test_df['SMILES'].apply(lambda x: pd.Series(extract_features(x)))

test_df = test_df.dropna(subset=feature_cols)

X_test_scaled = scaler.transform(test_df[feature_cols])

preds = model.predict(X_test_scaled)

target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
pred_df = pd.DataFrame(preds, columns=target_cols)

final_df = pd.concat([test_df[['SMILES']].reset_index(drop=True), pred_df], axis=1)

final_df.to_csv('submission.csv', index=False)
print("Save predicitions as : submission.csv")



show_result_test = pd.read_csv("submission.csv") 
show_result_test

