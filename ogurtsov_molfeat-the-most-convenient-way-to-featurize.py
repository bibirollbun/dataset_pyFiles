import numpy as np
import pandas as pd

import datamol as dm
from molfeat.trans.fp import FPVecTransformer, MoleculeTransformer

dm.disable_rdkit_log()

import warnings
warnings.filterwarnings('ignore')


def get_features(df, kind = "ecfp",  length = 1024, n_jobs = 4):

    transformer = FPVecTransformer(kind = kind, length = length, dtype = np.uint8, n_jobs = n_jobs)

    X = transformer(df.SMILES)

    df = df.reset_index(drop = True)

    cols = [f"col{i:04d}" for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns = cols)

    df = pd.concat([df, X], axis = 1)

    return(df)


df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")


%%time
df = get_features(df, kind = "ecfp", length = 1024, n_jobs = 1)
df


df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")


%%time
df = get_features(df, kind = "ecfp", length = 1024, n_jobs = 4)
df


def get_features(df, featurizer = "mordred", n_jobs = 4):

    transformer = MoleculeTransformer(featurizer = featurizer, n_jobs = n_jobs)

    X = transformer(df.SMILES)

    df = df.reset_index(drop = True)

    cols = [f"col{i:04d}" for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns = cols)

    df = pd.concat([df, X], axis = 1)

    return(df)


df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")


%%time
df = get_features(df, featurizer = "mordred", n_jobs = 4)
df

