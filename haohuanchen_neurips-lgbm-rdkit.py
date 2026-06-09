import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd

train_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
dataset1_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv")
dataset2_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv")
dataset3_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv")
dataset4_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")


print(train_df.isnull().sum())
print(dataset1_df.isnull().sum())
print(dataset2_df.isnull().sum())
print(dataset3_df.isnull().sum())
print(dataset4_df.isnull().sum())


print(train_df[train_df['SMILES'].duplicated(keep=False)])
print(dataset1_df[dataset1_df['SMILES'].duplicated(keep=False)])
print(dataset2_df[dataset2_df['SMILES'].duplicated(keep=False)])
print(dataset3_df[dataset3_df['SMILES'].duplicated(keep=False)])
print(dataset4_df[dataset4_df['SMILES'].duplicated(keep=False)])


dataset1_df = dataset1_df.groupby('SMILES', as_index=False).mean()
dataset2_df = dataset2_df.groupby('SMILES', as_index=False).mean()


def merge_dataset(origin_df, df_list):
    merge_df = origin_df.copy()
    for df in df_list:
        merge_df = pd.merge(merge_df, df, on="SMILES", how="outer")
    return merge_df

train_df = merge_dataset(train_df, [dataset1_df, dataset3_df, dataset4_df])


print(train_df.isnull().sum())


def merge_columns(origin_df, col1, col2, new_name):
    merge_df = origin_df.copy()
    merge_df[new_name] = merge_df[[col1, col2]].mean(axis=1, skipna=True)
    
    if col1 == new_name:
        merge_df.drop(columns=[col2], inplace=True)
    elif col2 == new_name:
        merge_df.drop(columns=[col1], inplace=True)
    else:
        merge_df.drop(columns=[col1, col2], inplace=True)
    
    return merge_df

train_df = merge_columns(train_df, "Tc", "TC_mean", "Tc")
train_df = merge_columns(train_df, "Tg_x", "Tg_y", "Tg")
train_df = merge_columns(train_df, "FFV_x", "FFV_y", "FFV")


import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen

def featurize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return pd.Series(np.nan, index=[
            "MolWt","NumAromaticRings","NumRings","FractionCSP3",
            "NumRotatableBonds","TPSA","HBA","HBD","BalabanJ","MolLogP",
            "HeavyAtomCount","NumHetero","NumAliphaticRings","NumSaturatedRings"
        ])
        
    feat = {
        "MolWt": Descriptors.MolWt(mol),
        "NumAromaticRings": Lipinski.NumAromaticRings(mol),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "BalabanJ": Descriptors.BalabanJ(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "NumHetero": Lipinski.NumHeteroatoms(mol),
        "NumAliphaticRings": Lipinski.NumAliphaticRings(mol),
        "NumSaturatedRings": Lipinski.NumSaturatedRings(mol),
        "Chi0n": rdMolDescriptors.CalcChi0n(mol),
        "Chi1n": rdMolDescriptors.CalcChi1n(mol),
        "Kappa1": rdMolDescriptors.CalcKappa1(mol),
        "Kappa2": rdMolDescriptors.CalcKappa2(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "MolMR": Crippen.MolMR(mol),
    }
    
    return pd.Series(feat)


from tqdm import tqdm
tqdm.pandas()

def add_smiles_feat(df: pd.DataFrame):
    feats = df["SMILES"].progress_apply(featurize_smiles)
    return pd.concat([df, feats], axis=1)

train_df = add_smiles_feat(train_df)
test_df = add_smiles_feat(test_df)


import re

def add_smiles_char_counts(df):
    smiles = df["SMILES"].fillna("")
    
    chars = list("CcNnOoSsPpFIBbH[]()=/#\\+-1234567890*@.")

    def colname_for_char(ch):
        return f"cnt_char_{ch}" if ch.isalnum() else f"cnt_char_0x{ord(ch):02x}"

    feat = pd.DataFrame(index=df.index)
    for ch in tqdm(chars):
        feat[colname_for_char(ch)] = smiles.str.count(re.escape(ch))

    return pd.concat([df, feat], axis=1)

train_df = add_smiles_char_counts(train_df)
test_df = add_smiles_char_counts(test_df)


def add_smiles_token_counts(df):
    smiles = df["SMILES"].fillna("")
    
    tokens = ["Cl", "Br", "Si", "Se"]

    feat = pd.DataFrame(index=df.index)
    for tok in tqdm(tokens):
        feat[f"cnt_tok_{tok}"] = smiles.str.count(re.escape(tok))

    return pd.concat([df, feat], axis=1)
    
train_df = add_smiles_token_counts(train_df)
test_df = add_smiles_token_counts(test_df)


print(train_df.isnull().sum())
print(test_df.isnull().sum())


from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

def lgbm_train(
    target,
    n_splits=5,
    n_estimators=30000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    min_child_weight=1e-3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=0,
    objective="mae",
    eval_metric="l1",
    stopping_rounds=500,
    period=1000
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    drop_list = ["id", "SMILES", "Tc", "Density", "Rg", "Tg", "FFV"]
    models = []
    test_preds = 0
    maes = []
    
    train_data = train_df[train_df[target].notna()]
    X = train_data.drop(drop_list, axis=1)
    y = train_data[target]
    X_test = test_df.drop(["id", "SMILES"], axis=1)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold+1}/{n_splits} >>>")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective=objective,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_metric,
            callbacks=[
                early_stopping(stopping_rounds=stopping_rounds),
                log_evaluation(period=period)
            ]
        )

        models.append(model)
    
        # test
        test_preds += model.predict(X_test) / n_splits

        # val
        val_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_pred)
        print(f"MAE: {mae:.6f}")
        maes.append(mae)

    print(f"Mean MAE: {sum(maes) / len(maes):.6f}")
    
    return models, test_preds


models_Tc, preds_Tc = lgbm_train(
    target="Tc",
    learning_rate=0.04,
    num_leaves=31,
    min_child_samples=30,
    min_child_weight=0.001,
    subsample=0.8,
    colsample_bytree=0.4,
    reg_alpha=0.2,
    reg_lambda=0.2,
    eval_metric="l1",
    stopping_rounds=500,
    period=1000
)


models_Density, preds_Density = lgbm_train(
    target="Density", 
    learning_rate=0.12,
    num_leaves=19,
    min_child_samples=22,
    min_child_weight=0.001,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=0,
    eval_metric="l1",
    stopping_rounds=500,
    period=1000
)


models_Rg, preds_Rg = lgbm_train(
    target="Rg",
    learning_rate=0.022,
    num_leaves=31,
    min_child_samples=20,
    min_child_weight=0.001,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0,
    reg_lambda=0,
    eval_metric="l1",
    stopping_rounds=500,
    period=1000
)


models_Tg, preds_Tg = lgbm_train(
    target="Tg",
    learning_rate=0.06,
    num_leaves=31,
    min_child_samples=27,
    min_child_weight=0.001,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_alpha=0,
    reg_lambda=0,
    eval_metric="l1",
    stopping_rounds=500,
    period=1000
)


models_FFV, preds_FFV = lgbm_train(
    target="FFV",
    learning_rate=0.1,
    num_leaves=80,
    min_child_samples=20,
    min_child_weight=0.001,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="l1",
    reg_alpha=0,
    reg_lambda=0,
    stopping_rounds=500,
    period=1000
)


import matplotlib.pyplot as plt
import seaborn as sns

task_names = ["Tc", "Density", "Rg", "Tg", "FFV"]
index = test_df.drop(["id", "SMILES"], axis=1).columns.tolist()

for models, task in zip([models_Tc, models_Density, models_Rg, models_Tg, models_FFV], task_names):
    importances = []
    
    for model in models:
        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)
    
    if importances:
        avg_importances = pd.Series(np.mean(importances, axis=0), index=index)
        avg_importances = avg_importances.sort_values(ascending=False).head(20)
        plt.figure(figsize=(8, 6))
        sns.barplot(
            x=avg_importances.values,
            y=avg_importances.index,
            palette="viridis"
        )
        plt.title(f"Feature Importances - {task}")
        plt.xlabel("Average Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()


submission = pd.DataFrame({
    "id": test_df["id"],
    "Tg": preds_Tg,
    "FFV": preds_FFV,
    "Tc": preds_Tc,
    "Density": preds_Density,
    "Rg": preds_Rg
})
submission.to_csv("submission.csv", index=False)
submission.head()

