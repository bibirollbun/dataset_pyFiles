!pip install biopandas



import pandas as pd
from biopandas.pdb import PandasPdb
from typing import Optional
from glob import glob
from tqdm.notebook import tqdm
import os
import numpy as np
import re


# https://medium.com/@jgbrasier/working-with-pdb-files-in-python-7b538ee1b5e4

def read_pdb_to_dataframe(
    pdb_path: Optional[str] = None,
    model_index: int = 1,
    ) -> pd.DataFrame:
    """
    Read a PDB file, and return a Pandas DataFrame containing the atomic coordinates and metadata.

    Args:
        pdb_path (str, optional): Path to a local PDB file to read. Defaults to None.
        model_index (int, optional): Index of the model to extract from the PDB file, in case
            it contains multiple models. Defaults to 1.
        parse_header (bool, optional): Whether to parse the PDB header and extract metadata.
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the atomic coordinates and metadata, with one row
            per atom
    """
    atomic_df = PandasPdb().read_pdb(pdb_path)
    header = None
    atomic_df = atomic_df.get_model(model_index)
    if len(atomic_df.df["ATOM"]) == 0:
        raise ValueError(f"No model found for index: {model_index}")

    return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]])

def preprocess_pdb(pdb_path):
    df = read_pdb_to_dataframe(pdb_path)
    df = df[df.atom_name=="C1'"].sort_values('residue_number').reset_index(drop = True)
    df = df[['residue_name', 'residue_number', 'x_coord', 'y_coord', 'z_coord']].copy()
    df.columns = ['resname', 'resid', 'x_1', 'y_1', 'z_1']
    df[['x_1', 'y_1', 'z_1']] =  df[['x_1', 'y_1', 'z_1']].astype(np.float32) 
    df['resid'] =  df['resid'].astype(np.int32)
    assert len(df['resid'].unique())  == (df['resid'].max()-df['resid'].min())+1
    return df


def convert_pdb(pdb_paths):
    ext_label_df = []
    for pdb_path in tqdm(pdb_paths):
        try:
            # pdb_id = os.path.basename(pdb_path).split('.', 1)[0]
            pdb_id = pdb_path.split('/')[-2]
            ext_label_df_ = preprocess_pdb(pdb_path)
            ext_label_df_['ID'] =  [f'{pdb_id}_{resid}' for resid in ext_label_df_['resid']]
            ext_label_df_['pdb_id'] =  pdb_id
            ext_label_df.append(ext_label_df_[['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1', 'pdb_id']])
        except Exception as e:
            print(f'Failed to read{pdb_path}')
            print('Error:' + str(e))
    ext_label_df = pd.concat(ext_label_df).reset_index(drop = True)
    ext_sequence_df = ext_label_df[["pdb_id", 'resname']].groupby("pdb_id", as_index = False).apply(lambda x: ''.join(x['resname']), include_groups=False).reset_index(drop = True)
    ext_sequence_df.columns = ["target_id", 'sequence']
    ext_label_df = ext_label_df.drop("pdb_id", axis = 1)
    return ext_label_df, ext_sequence_df


ext_dir = '/kaggle/input/stanford-ribonanza-rna-folding/rhofold_pdbs/rhofold_pdbs'
pdb_paths = glob(f'{ext_dir}/*/*/*/*/*.pdb')
print(len(pdb_paths))


train_label_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
display(train_label_df.head())
test_sequence_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
display(test_sequence_df.head())


ext_label_df, ext_sequence_df = convert_pdb(pdb_paths[:5])
display(ext_label_df.head())
display(ext_sequence_df.head())



ext_label_df, ext_sequence_df = convert_pdb(pdb_paths)
ext_label_df.to_parquet(f'ext_ribonanza_labels.parquet')
ext_sequence_df.to_parquet(f'ext_ribonanza_sequences.parquet')

