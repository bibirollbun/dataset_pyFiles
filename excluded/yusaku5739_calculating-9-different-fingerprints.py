!pip install -q scikit-fingerprints


import gc
import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from skfp.fingerprints import PharmacophoreFingerprint, E3FPFingerprint
from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
import multiprocessing
import gc


def make_fp(smiles):
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)

    if fp_name == "ecfp":
        ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        return np.packbits(ecfp)
    elif fp_name == "apfp":
        from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator
        fp_gen = GetAtomPairGenerator(fpSize=2048)
        fp = np.array(fp_gen.GetFingerprint(mol))
        return np.packbits(fp)
    elif fp_name == "rdfp":
        from rdkit.Chem.rdFingerprintGenerator import GetRDKitFPGenerator
        fp_gen = GetRDKitFPGenerator(fpSize=2048)
        fp = np.array(fp_gen.GetFingerprint(mol))
        return np.packbits(fp)
    elif fp_name == "ttfp":
        from rdkit.Chem.rdFingerprintGenerator import  GetTopologicalTorsionGenerator
        fp_gen = GetTopologicalTorsionGenerator(fpSize=2048)
        fp = np.array(fp_gen.GetFingerprint(mol))
        return np.packbits(fp)
    elif fp_name == "ergfp":
        from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
        fp = np.array(GetErGFingerprint(mol))
        return fp
    elif fp_name == "estate":
        from rdkit.Chem.EState.Fingerprinter import FingerprintMol
        fp = np.array(FingerprintMol(mol))
        fp = fp.flatten().astype(int)
        return fp
    elif fp_name == "pharm2d":
        fp = PharmacophoreFingerprint(fp_size=2048).transform([smiles])
        return np.packbits(fp)
    elif fp_name == "maccs":
        fp = np.array(AllChem.GetMACCSKeysFingerprint(mol))
        return np.packbits(fp)
    #elif fp_name == "e3fp":
    #    mol_from_smiles = MolFromSmilesTransformer()
    #    mols = mol_from_smiles.transform([smiles])
    #    print(smiles)
    #    conf_gen = ConformerGenerator()
    #    mols = conf_gen.transform(mols)
    #    #fp = E3FPFingerprint(fp_size=2048).transform(mols)
    #    fp = fprints_from_smiles(smiles, fprint_params={"bits":2048})
    #    return np.packbits(fp)
    elif fp_name == "mhfp":
        from rdkit.Chem import rdMHFPFingerprint
        #fp = MHFPFingerprint(fp_size=2024).transform([smiles])
        encoder = rdMHFPFingerprint.MHFPEncoder()
        fp = encoder.EncodeMol(mol)
        return np.packbits(fp)
    else:
        raise NotImplementedError

def translate(fp_name, fp_feat_dim, fp_dtype):
    print(f"compute {fp_name}")
    train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    
    fps = np.zeros((train.shape[0], fp_feat_dim), dtype=fp_dtype)
    per_chank = train.shape[0] // n_chank
    
    for i in range(n_chank):
        if i != n_chank-1:
            chank = train.iloc[i*per_chank:(i+1)*per_chank]
        else:
            chank = train.iloc[i*per_chank:]
        smiles = chank["SMILES"].to_list()

        print(f"compute {i} chank fingerprints  ...")
        fp_chank = joblib.Parallel(n_jobs=30)(joblib.delayed(make_fp)(smile) for smile in smiles)
        fp_chank = np.stack(fp_chank, dtype=fp_dtype)
        if i != n_chank-1:
            fps[i*per_chank:(i+1)*per_chank,] = fp_chank
        else:
            fps[i*per_chank:,] = fp_chank
    
        del fp_chank
        gc.collect()
    
    np.save(f"fp_{fp_name}", fps)
    del fps
    gc.collect()


debug = 1
n_chank = 20
fp_name_list = ["ecfp", "apfp", "rdfp", "ttfp", "ergfp", "estate", "pharm2d", "maccs", "mhfp"]
fp_feat_dim_list = [256, 256, 256, 256, 315, 79*2, 4997, 21, 256]
fp_dtype_list = ["uint8", "uint8", "uint8", "uint8", "float32", "int", "uint8", "uint8", "uint8"]


for fp_name, fp_feat_dim, fp_dtype in zip(fp_name_list, fp_feat_dim_list, fp_dtype_list):
    translate(fp_name, fp_feat_dim, fp_dtype)

