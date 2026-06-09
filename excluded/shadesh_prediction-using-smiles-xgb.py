!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


# ==============================
# Libraries
# ==============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


# ==============================
# Load Data
# ==============================
df_train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
df_test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

dataset1 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv")
dataset2 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv")
dataset3 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv")
dataset4 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")

# Rename column for consistency
dataset1.rename(columns={'TC_mean': 'Tc'}, inplace=True)

# Combine all training data
df_traindata = pd.concat([df_train, dataset1, dataset2, dataset3, dataset4], ignore_index=True)

print("Original Training Data Summary:\n", df_train.describe())
print("Augmented Training Data Summary:\n", df_traindata.describe())


# ==============================
# Standard Scaling
# ==============================
scaler = StandardScaler()
cols = ["Tg", "FFV", "Tc", "Density", "Rg"]

df_train[cols] = scaler.fit_transform(df_train[cols])
df_traindata[cols] = scaler.fit_transform(df_traindata[cols])


# ==============================
# Error Calculation
# ==============================
def calculate_error(df_test, df_testPred):
    colonne = ["Tg", "FFV", "Tc", "Density", "Rg"]
    n5 = [(1 / np.sqrt(df_test.shape[0])) for _ in colonne]
    total = np.sum(n5)

    weight = []
    for i, col in enumerate(colonne):
        ri = df_test[col].max() - df_test[col].min()
        weight.append((5 * n5[i]) / (total * ri))
    weight = np.array(weight)

    error = 0
    for j in range(df_test.shape[0]):
        true_row = df_test.iloc[j].to_numpy()
        pred_row = df_testPred[j] if isinstance(df_testPred, np.ndarray) else df_testPred.iloc[j].to_numpy()
        difference = true_row - pred_row
        error += difference @ weight.T

    return error / df_test.shape[0]


# ==============================
# Convert SMILES → Fingerprints
# ==============================
def smiles_to_ecfp4(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ==============================
# Predict Missing Values
# ==============================
def predict_missing_value(y, df):
    for col in y.columns:
        df_known = df[["SMILES", col]].dropna()
        df_missing = df[df[col].isnull()][["SMILES", col]]

        if df_missing.empty:
            continue

        X_known = np.stack(df_known['SMILES'].apply(smiles_to_ecfp4).values)
        y_known = df_known[col]

        X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.33, random_state=42)

        X_to_predict = np.stack(df_missing['SMILES'].apply(smiles_to_ecfp4).values)

        model = XGBRegressor(
            booster='gbtree',
            objective='reg:squarederror',
            eval_metric='rmse',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_predicted = model.predict(X_to_predict)
        df.loc[df_missing.index, col] = y_predicted

    return df


y = df_train[cols].copy()
df_train = predict_missing_value(y, df_train)
df_traindata = predict_missing_value(y, df_traindata)


# ==============================
# Training and Evaluation
# ==============================
def print_error(df):
    y = df[cols].copy()
    X = np.stack(df['SMILES'].apply(smiles_to_ecfp4).values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    base_model = XGBRegressor(
        booster='gbtree',
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    error = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    print(f"MAE per property: {error.tolist()}")
    print(f"Weighted MAE: {calculate_error(y_test, y_pred)}")

    return model


model1 = print_error(df_train)
model2 = print_error(df_traindata)


# ==============================
# Predictions on Test Data
# ==============================
X_test = np.stack(df_test['SMILES'].apply(smiles_to_ecfp4).values)

# Model 1 Predictions
y_test1 = np.array(model1.predict(X_test))
df_test_copy = df_test.copy()
for i, col in enumerate(cols):
    df_test_copy[col] = y_test1[:, i]
df_test_copy[cols] = scaler.inverse_transform(df_test_copy[cols])
df_test_copy.to_csv('/kaggle/working/submission_model1.csv', index=False)

# Model 2 Predictions
y_test2 = np.array(model2.predict(X_test))
df_test_copy2 = df_test.copy()
for i, col in enumerate(cols):
    df_test_copy2[col] = y_test2[:, i]
df_test_copy2[cols] = scaler.inverse_transform(df_test_copy2[cols])
df_test_copy2.to_csv('/kaggle/working/submission.csv', index=False)

print("✅ Submissions saved successfully!")


