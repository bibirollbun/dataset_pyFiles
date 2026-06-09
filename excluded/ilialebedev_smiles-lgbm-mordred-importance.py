!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install mordred --no-index --find-links='file:///kaggle/input/mordred-1-2-0-py3-none-any/'


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
from sklearn.model_selection import StratifiedKFold,KFold,StratifiedGroupKFold,GroupKFold,train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
import random
import polars as pl
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from mordred import descriptors, Calculator
RDLogger.DisableLog('rdApp.*')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm

warnings.filterwarnings('ignore')


train = pl.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pl.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


def prep(df):
    df_smiles = df['SMILES'].to_list()
    df_mols = [Chem.MolFromSmiles(smiles) for smiles in df_smiles]

    df_RDKit = pl.DataFrame([Descriptors.CalcMolDescriptors(mol) for mol in df_mols])

    df_RDKit = df_RDKit.to_pandas()

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
        descriptors.PathCount,
        descriptors.Polarizability,
        descriptors.RingCount,
        descriptors.TopologicalIndex,
        descriptors.VertexAdjacencyInformation,
        descriptors.WalkCount,
        descriptors.Weight,
        descriptors.WienerIndex,
        descriptors.ZagrebIndex,
    ]

    df_mordred = pl.from_pandas(Calculator(descList).pandas(df_mols))

    df_mordred = df_mordred.to_pandas()

    features = {
        "n_" : ['nH','nB','nC','nN','nO','nS','nP','nF','nCl','nBr','nI','nX'],
        "nBonds" : ['nBonds','nBondsO','nBondsS','nBondsD','nBondsT','nBondsA','nBondsM','nBondsKS','nBondsKD'],
        "IC" : ['IC0','IC1','IC2','IC3','IC4','IC5'],
        "TIC" : ['TIC0','TIC1','TIC2','TIC3','TIC4','TIC5'],
        "SIC" : ['SIC0','SIC1','SIC2','SIC3','SIC4','SIC5'],
        "BIC" : ['BIC0','BIC1','BIC2','BIC3','BIC4','BIC5'],
        "CIC" : ['CIC0','CIC1','CIC2','CIC3','CIC4','CIC5'],
        "MIC" : ['MIC0','MIC1','MIC2','MIC3','MIC4','MIC5'],
        "ZMIC" : ['ZMIC0','ZMIC1','ZMIC2','ZMIC3','ZMIC4','ZMIC5'],
        "MPC" : ['MPC2','MPC3','MPC4','MPC5','MPC6','MPC7','MPC8','MPC9','MPC10'],
        "piPC" : ['piPC1','piPC2','piPC3','piPC4','piPC5','piPC6','piPC7','piPC8','piPC9','piPC10'],
        "MID" : ['MID','MID_h','MID_C','MID_N','MID_O','MID_X'],
        "AMID" : ["AMID","AMID_h","AMID_C","AMID_N","AMID_O","AMID_X"],
        "MPC": ['MPC2','MPC3','MPC4','MPC5','MPC6','MPC7','MPC8','MPC9','MPC10'],
        "piPC": ['piPC1','piPC2','piPC3','piPC4','piPC5','piPC6','piPC7','piPC8','piPC9','piPC10'],
        'nRing': ['nRing','n3Ring','n4Ring','n5Ring','n6Ring','n7Ring','n8Ring','n9Ring','n10Ring','n11Ring','n12Ring'],
        "nHRing": ['nHRing','n3HRing','n4HRing','n5HRing','n6HRing','n7HRing','n8HRing','n9HRing','n10HRing','n11HRing','n12HRing'],
        'naRing': ['naRing','n3aRing','n4aRing','n5aRing','n6aRing','n7aRing','n8aRing','n9aRing','n10aRing','n11aRing','n12aRing'],
        "naHRing": ['naHRing','n3aHRing','n4aHRing','n5aHRing','n6aHRing','n7aHRing','n8aHRing','n9aHRing','n10aHRing','n11aHRing','n12aHRing'],
        "nARing": ['nARing','n3ARing','n4ARing','n5ARing','n6ARing','n7ARing','n8ARing','n9ARing','n10ARing','n11ARing','n12ARing'],
        "nAHRing": ['nAHRing','n3AHRing','n4AHRing','n5AHRing','n6AHRing','n7AHRing','n8AHRing','n9AHRing','n10AHRing','n11AHRing','n12AHRing'],
        'nFRing': ['nFRing','n4FRing','n5FRing','n6FRing','n7FRing','n8FRing','n9FRing','n10FRing','n11FRing','n12FRing'],
        'nFHRing': ['nFHRing','n4FHRing','n5FHRing','n6FHRing','n7FHRing','n8FHRing','n9FHRing','n10FHRing','n11FHRing','n12FHRing'],
        'nFaRing': ['nFaRing','n4FaRing','n5FaRing','n6FaRing','n7FaRing','n8FaRing','n9FaRing','n10FaRing','n11FaRing','n12FaRing'],
        'nFaHRing': ['nFaHRing','n4FaHRing','n5FaHRing','n6FaHRing','n7FaHRing','n8FaHRing','n9FaHRing','n10FaHRing','n11FaHRing','n12FaHRing'],
        'nFARing': ['nFARing','n4FARing','n5FARing','n6FARing','n7FARing','n8FARing','n9FARing','n10FARing','n11FARing','n12FARing'],
        'nFAHRing': ['nFAHRing','n4FAHRing','n5FAHRing','n6FAHRing','n7FAHRing','n8FAHRing','n9FAHRing','n10FAHRing','n11FAHRing','n12FAHRing'],
        'MWC': ['MWC01','MWC02','MWC03','MWC04','MWC05','MWC06','MWC07','MWC08','MWC09','MWC10'],
        'SRW':['SRW02','SRW03','SRW04','SRW05','SRW06','SRW07','SRW08','SRW09','SRW10'],
    }

    for f in features.keys():
        f_feat = features[f]

        df_mordred[f"{f}_mean"] = df_mordred[f_feat].mean(axis=1)
        df_mordred[f"{f}_std"] = df_mordred[f_feat].std(axis=1)
        df_mordred[f"{f}_var"] = df_mordred[f_feat].var(axis=1)
        df_mordred[f"{f}_min"] = df_mordred[f_feat].min(axis=1)
        df_mordred[f"{f}_max"] = df_mordred[f_feat].max(axis=1)
        df_mordred[f"{f}_per25"] = df_mordred[f_feat].quantile(0.25, axis=1)
        df_mordred[f"{f}_median"] = df_mordred[f_feat].quantile(0.5, axis=1)
        df_mordred[f"{f}_per75"] = df_mordred[f_feat].quantile(0.75, axis=1)

    df_mordred_res = df_mordred.join(df_RDKit)
    df_mordred_res = df_mordred_res.drop(
        [
            'VMcGowan', 'bpol', 'apol', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
            'MinAbsPartialCharge', 'BCUT2D_MWHI','BCUT2D_MWLOW','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI',
            'BCUT2D_LOGPLOW','BCUT2D_MRHI','BCUT2D_MRLOW'
        ], axis=1)

    return df_mordred_res


random.seed = 42
random.state = 42
np.random_seed = 42
np.random_state = 42


train_mordred_res = prep(train)
test_mordred_res = prep(test)


def train_target_property(X_target=None, y_target=None, splits=None, params=None):

    if params:
        p = params
    else:
        p = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 120,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.5,
            'lambda_l1': 0.2,
            'lambda_l2': 0.05,
            'min_data_in_leaf': 10,
            'bagging_freq': 1,
            'verbose': -1,
            'random_state': 42,
            'random_seed': 42,
            'device_type' : 'cpu'
        }

    if splits:
        mean_cv = []
        models = {}
        all_val_true = {}
        all_val_pred = {}

        for target, fold_list in splits.items():
            cv_scores = []
            models_l = []
            for fold, (X_train, X_val, y_train, y_val) in enumerate(fold_list):
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

                model = lgb.train(
                    p,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1800,
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(600)]
                )

                val_pred = model.predict(X_val)
                cv_score = mean_absolute_error(y_val, val_pred)
                cv_scores.append(cv_score)
                models_l.append(model)

                all_val_true[target] = y_val
                all_val_pred[target] = val_pred

                print(f"----Fold {fold+1} Complete / MAE = {cv_score:.4f}", flush=True)
                print()

            models[target] = models_l
            cv_mean = np.mean(cv_scores)
            print(f"===CV: {cv_mean:.4f} Â± {np.std(cv_scores):.3f}===")
            print()

        return models, all_val_true, all_val_pred

    else:

        X_train, X_val, y_train, y_val = train_test_split(X_target, y_target, test_size=0.3, shuffle=True, random_state=42)

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            p,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2400,
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(600)]
        )

        val_pred = model.predict(X_val)
        cv_score = mean_absolute_error(y_val, val_pred)

        return cv_score, model


targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


y = train[targets].to_pandas()
X = train_mordred_res

X_test = test_mordred_res

all_features = X.columns
all_features_test = X_test.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


feature_importances_for_target = {}

for target in targets:
    y_target = y[target][y[target].notnull()]
    ind = y_target.index

    score, model = train_target_property(X_scaled[ind], y_target, splits=False, params=None)

    feature_importances_for_target[target] = [all_features[i] for i in model.feature_importance('gain').argsort()[::-1]]


kf = KFold(n_splits=5, shuffle=True, random_state=42)


data_splits = {}
not_remove_feature_ind_list = {}

for target in targets:
    y_target = y[target][y[target].notnull()]
    ind = y_target.index

    remove_feature = feature_importances_for_target[target][::-1][100]

    not_remove_feature_ind = []

    for i, f in enumerate(all_features):
        if f not in remove_feature:
            not_remove_feature_ind.append(i)

    X_target = X_scaled[:, not_remove_feature_ind][ind, :]

    not_remove_feature_ind_list[target] = not_remove_feature_ind

    fold_target = []

    for train_idx, val_idx in kf.split(X_target):

        X_train, X_val = X_target[train_idx], X_target[val_idx]
        y_train, y_val = y_target.iloc[train_idx], y_target.iloc[val_idx]

        fold_target.append((X_train, X_val, y_train, y_val))

    data_splits[target] = fold_target


trained_models, all_cv_true, all_cv_predictions = train_target_property(splits=data_splits)


import open_polymer_2025_metric as metric

cv_true_df = pd.DataFrame()
cv_pred_df = pd.DataFrame()

for target in targets:
    max_len = max(len(all_cv_true[t]) for t in targets)

    true_padded = list(all_cv_true[target]) + [metric.NULL_FOR_SUBMISSION] * (max_len - len(all_cv_true[target]))
    pred_padded = list(all_cv_predictions[target]) + [metric.NULL_FOR_SUBMISSION] * (max_len - len(all_cv_predictions[target]))

    cv_true_df[target] = true_padded
    cv_pred_df[target] = pred_padded

cv_true_df['id'] = range(len(cv_true_df))
cv_pred_df['id'] = range(len(cv_pred_df))

competition_scores = []
for target in targets:
    comp_score = metric.scaling_error(np.array(all_cv_true[target]), np.array(all_cv_predictions[target]), target)
    competition_scores.append(comp_score)

estimated_lb_score = metric.score(cv_true_df, cv_pred_df, 'id')

print("=" * 50)
print(f"Trained: {len(targets)} targets Ã— 5 CV folds = {len(targets) * 5} models")
print(f"Individual competition scores: {[f'{s:.4f}' for s in competition_scores]}")
print(f"ğŸ�¯ ESTIMATED LB SCORE: {estimated_lb_score:.4f}")
print("=" * 50)


def predict_target_property(X_test_scaled, models, scaler, not_rem_f):

    if models is None or scaler is None:
        return np.zeros(len(X_test_scaled))

    X_test_scaled = X_test_scaled[:, not_rem_f]

    fold_predictions = []
    for model in models:
        pred = model.predict(X_test_scaled)
        fold_predictions.append(pred)

    predictions = np.mean(fold_predictions, axis=0)

    return predictions


print(f"\nMAKING PREDICTIONS...")
all_predictions = {}
for target in targets:
    predictions = predict_target_property(X_test_scaled, trained_models[target], scaler=scaler, not_rem_f=not_remove_feature_ind_list[target])
    all_predictions[target] = predictions

submission = pd.DataFrame({'id': test['id']})
for target in targets:
    submission[target] = all_predictions[target]

submission.to_csv('submission.csv', index=False)

print(f"Predicted: {len(test)} test samples")
print(f"Saved: submission.csv")

print(f"\nğŸ‘€ SUBMISSION PREVIEW:")
print(submission.head().to_string(index=False, float_format='%.4f'))

