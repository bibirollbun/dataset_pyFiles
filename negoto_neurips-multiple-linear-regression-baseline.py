# ! pip install rdkit
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/


import pandas as pd
import numpy as np
from tqdm import tqdm


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


from rdkit import Chem
from rdkit.Chem import Descriptors

# Convert SMILES strings into "Mol" objects
train_smiles = train['SMILES'].to_list()
train_mols = [Chem.MolFromSmiles(smiles) for smiles in train_smiles]

test_smiles = test['SMILES'].to_list()
test_mols = [Chem.MolFromSmiles(smiles) for smiles in test_smiles]

# Generate RDKit descriptor features
train_RDKit = pd.DataFrame([Descriptors.CalcMolDescriptors(mol) for mol in tqdm(train_mols)])
display(train_RDKit.head())

test_RDKit = pd.DataFrame([Descriptors.CalcMolDescriptors(mol) for mol in tqdm(test_mols)])


%%time
from mordred import Calculator, descriptors

# Generate mordred descriptor features (AtomCount)
descList = [
    descriptors.AcidBase,
    descriptors.Aromatic,
    descriptors.AtomCount,
    descriptors.BondCount,
    descriptors.EccentricConnectivityIndex,
    descriptors.FragmentComplexity,
    descriptors.Framework,
    descriptors.InformationContent,
    descriptors.Lipinski,
    descriptors.McGowanVolume,
    descriptors.MolecularId,
#    descriptors.PathCount, # Computationally heavy
    descriptors.Polarizability,
    descriptors.RingCount,
    descriptors.TopologicalIndex,
    descriptors.VertexAdjacencyInformation,
    descriptors.WalkCount,
    descriptors.Weight,
    descriptors.WienerIndex,
    descriptors.ZagrebIndex,
]
train_mordred = Calculator(descList).pandas(train_mols)
display(train_mordred.head())

test_mordred = Calculator(descList).pandas(test_mols)


# Bind molecular-descriptors to the dataframes
train = pd.concat([train, train_RDKit, train_mordred], axis = 1)
display(train.head())

test = pd.concat([test, test_RDKit, test_mordred], axis = 1)


from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def get_features(target, nfeatures = 2, init_features = []):
    best =  ["", np.inf, np.inf]
    train_set = train.loc[train[target].notna(), ]
    features = init_features
    for i in range(nfeatures):
        best[0] =  [""]
        for newfeat in tqdm(bucket):
            if (newfeat in features) or (train[newfeat].isna().sum() > 0) or (train[newfeat].dtype in ['bool', 'O']):
                continue
            X, y = train_set[[newfeat, *features]], train_set[target]
            aics = []
            maes = []
            for seed in seeds:
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X, y, test_size = .25, random_state = seed
                )
                reg = OLS(y_train, add_constant(X_train)).fit()
                aics.append(reg.aic)
                maes.append(mean_absolute_error(y_valid, reg.predict(add_constant(X_valid))))
            aic = np.mean(aics)
            mae = np.mean(maes)
            if mae < best[1]:
                best[0] = newfeat
                best[1] = mae
                best[2] = aic
        if best[0] == "":
            break
        print(f"{best[0]} : MAE(Best) {round(best[1], 3)}, AIC {round(best[2], 3)}")
        features.append(best[0])
    return features, train_set[features], train_set[target]


models = [] # a container for regression models
bucket = train.columns.values[7:] # a list of molecular-descriptors
seeds = [40, 41, 42, 43, 44] # random seeds to perform multiple holdout validations
featdict = {
    "Tg": ['fMF', 'AMID_N', 'fr_imidazole', 'PEOE_VSA3', 'SlogP_VSA4', 'n6aRing', 'Diameter', 'fr_NH1', 'RingCount', 'nG12FHRing', 'MinAbsEStateIndex', 'SIC3', 'n3ARing', 'nFaHRing', 'nP', 'NumAtomStereoCenters', 'n5HRing', 'NumRotatableBonds', 'Kappa3', 'Chi2v'],
    "FFV": ['AMID_O', 'NumHDonors', 'Chi4v', 'VSA_EState8', 'fr_Ar_N', 'fr_ester', 'VSA_EState5', 'AMID_N', 'SMR_VSA10', 'nG12FaRing', 'BIC4', 'ECIndex', 'qed', 'fr_NH1', 'n10FaRing', 'fragCpx', 'Kappa3', 'NumRotatableBonds', 'AMID_X', 'MIC0'],
    "Tc": ['VSA_EState7', 'NumAtomStereoCenters', 'MIC1', 'SlogP_VSA1', 'SPS', 'VSA_EState8', 'fMF', 'BalabanJ', 'n6AHRing', 'BIC0', 'fr_aryl_methyl', 'fr_unbrch_alkane', 'SMR_VSA5', 'MIC2', 'SlogP_VSA12', 'fr_thiophene', 'SRW10', 'FpDensityMorgan3', 'fr_Al_OH', 'fr_Al_OH_noTert'],
    "Density": ['AMW', 'AMID_h', 'AMID', 'NumAliphaticRings', 'fr_Ar_N', 'VSA_EState8', 'BalabanJ', 'nBondsT', 'SMR_VSA9', 'fr_azo', 'n5ARing', 'fr_Al_OH_noTert', 'PEOE_VSA1', 'IC1', 'NumAliphaticCarbocycles', 'fr_ketone', 'n6ARing', 'fr_pyridine', 'nCl', 'fr_nitrile'],
    "Rg": ['NumAtomStereoCenters', 'FpDensityMorgan3', 'VSA_EState8', 'SMR_VSA4', 'SlogP_VSA10', 'BIC0', 'PEOE_VSA14', 'VAdjMat', 'EState_VSA4', 'SlogP_VSA7', 'PEOE_VSA12', 'nAcid', 'VSA_EState10', 'EState_VSA10', 'n5ARing', 'SIC1', 'Diameter', 'CIC1', 'FpDensityMorgan2', 'nG12FRing']
} # initial sets of predictor names

# Update the predictors and build models with the updated predictor lists
for target in targets:
    print(f"[{target}] --------------------------------------------")
    features, X, y = get_features(target, 0, featdict[target])
    print(features)
    featdict[target] = features
    models.append(OLS(y, add_constant(X)).fit())


# Create a submission file
for i in range(5):
    target = targets[i]
    features = featdict[target]
    test_set = test[features]
    test[target] = models[i].predict(add_constant(test_set))
sub = test[['id','Tg', 'FFV', 'Tc', 'Density', 'Rg']]
sub.to_csv('submission.csv',index=False)


sub.head()

