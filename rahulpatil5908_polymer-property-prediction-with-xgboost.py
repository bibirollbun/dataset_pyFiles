import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import randint, uniform
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.decomposition import PCA

from xgboost import XGBRegressor, plot_importance
from catboost import CatBoostRegressor


train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


train_df.head()


test_df.head()


train_df.info()


test_df.info()


num_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
corr = train_df[num_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


print("ğŸ”� Missing Values:")
msno.heatmap(train_df)
plt.title("Missing Values Heatmap (Train)")
plt.show()


# Target distributions
for col in num_cols:
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f"Distribution of Target: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


num_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
train_df[num_cols].hist(bins=20, figsize=(12, 8))
plt.suptitle("Histograms of Numerical Features")
plt.show()


def extract_smiles_features(smiles):
    features = {}
    if pd.isna(smiles):
        return dict.fromkeys(['SMILES_len', 'C_count', 'N_count', 'O_count', 'F_count',
                              'S_count', 'Cl_count', 'Br_count', 'I_count', 'P_count',
                              'equal_count', 'hash_count', 'ring_count'], 0)
    features['SMILES_len'] = len(smiles)
    features['C_count'] = smiles.count('C')
    features['N_count'] = smiles.count('N')
    features['O_count'] = smiles.count('O')
    features['F_count'] = smiles.count('F')
    features['S_count'] = smiles.count('S')
    features['Cl_count'] = smiles.count('Cl')
    features['Br_count'] = smiles.count('Br')
    features['I_count'] = smiles.count('I')
    features['P_count'] = smiles.count('P')
    features['equal_count'] = smiles.count('=')
    features['hash_count'] = smiles.count('#')
    features['ring_count'] = smiles.count('c') + smiles.count('1') + smiles.count('2')
    return features

train_features = train_df['SMILES'].apply(extract_smiles_features)
test_features = test_df['SMILES'].apply(extract_smiles_features)

train_features_df = pd.DataFrame(train_features.tolist())
test_features_df = pd.DataFrame(test_features.tolist())


descriptor_names = [desc[0] for desc in Descriptors.descList]
def extract_rdkit_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: 0 for name in descriptor_names}
    return {name: func(mol) for name, func in Descriptors.descList}

train_rdkit = pd.DataFrame([extract_rdkit_features(s) for s in tqdm(train_df['SMILES'])])
test_rdkit = pd.DataFrame([extract_rdkit_features(s) for s in tqdm(test_df['SMILES'])])



def morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

train_morgan = np.array([morgan_fp(s) for s in tqdm(train_df['SMILES'])])
test_morgan = np.array([morgan_fp(s) for s in tqdm(test_df['SMILES'])])

train_morgan_df = pd.DataFrame(train_morgan)
test_morgan_df = pd.DataFrame(test_morgan)


X_train_full = pd.concat([train_rdkit, train_morgan_df], axis=1)
X_test_full = pd.concat([test_rdkit, test_morgan_df], axis=1)


X_train_full = train_rdkit.copy()
X_test_full = test_rdkit.copy()

def clean_features(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    df = df.select_dtypes(include=[np.number])
    df = df.clip(lower=-1e5, upper=1e5)  # Clamp large values
    return df

target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
submission = pd.DataFrame({'id': test_df['id']})

X_train_full_clean = clean_features(X_train_full)
X_test_full_clean = clean_features(X_test_full)


target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
submission = pd.DataFrame({'id': test_df['id']})
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ğŸ¤– Train and Predict
for target in target_columns:
    print(f"\nğŸ”§ Training model for target: {target}")
    valid_rows = train_df.dropna(subset=[target])
    X = X_train_full_clean.loc[valid_rows.index]
    y = valid_rows[target]

    if X.shape[0] < 10:
        print(f"â�Œ Not enough data for {target}")
        submission[target] = 0
        continue

    selector = SelectKBest(score_func=f_regression, k=min(300, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    X_test_selected = selector.transform(X_test_full_clean)
    selected_features = X.columns[selector.get_support()]

    # Base models
    model_xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, n_jobs=-1, random_state=42)
    model_cat = CatBoostRegressor(iterations=200, depth=5, learning_rate=0.1, random_seed=42, verbose=False)

    # Stacking
    stacking_model = StackingRegressor(
        estimators=[('xgb', model_xgb), ('cat', model_cat)],
        final_estimator=Ridge(), passthrough=True, cv=5, n_jobs=-1
    )

    # CV Score
    scores = cross_val_score(stacking_model, X_selected, y, cv=kf, scoring='neg_root_mean_squared_error')
    rmse = -np.mean(scores)
    print(f" {target} CV RMSE: {rmse:.4f}")

    # Optional: plot CV score
    sns.boxplot(scores * -1)
    plt.title(f"{target} CV RMSE Distribution")
    plt.ylabel("RMSE")
    plt.show()

    # Train/test split for validation plot
    X_tr, X_val, y_tr, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    stacking_model.fit(X_tr, y_tr)
    preds_val = stacking_model.predict(X_val)

    plt.scatter(y_val, preds_val, alpha=0.6)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{target} - Predicted vs Actual")
    plt.show()

    # Train on full data & predict
    stacking_model.fit(X_selected, y)
    submission[target] = stacking_model.predict(X_test_selected)

    # Optional: plot XGB feature importance
    model_xgb.fit(X_selected, y)
    plot_importance(model_xgb, max_num_features=15)
    plt.title(f"XGB Feature Importance for {target}")
    plt.show()

    # Top feature vs target scatter
    for feat in selected_features[:3]:  # top 3 features
        sns.scatterplot(x=X[feat], y=y)
        plt.xlabel(feat)
        plt.ylabel(target)
        plt.title(f"{feat} vs {target}")
        plt.show()



submission_file_name = 'submission.csv'
submission.to_csv(submission_file_name, index=False)

print(f"\nSubmission file '{submission_file_name}' created successfully.")
print("The first few rows of the generated submission file:")
print(submission.head())


