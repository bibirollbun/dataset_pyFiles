import numpy as np
import pandas as pd
import torch
import logging
import gc
import warnings
warnings.filterwarnings('ignore')
try:
    import Bio.PDB
except ImportError:
    !pip install --no-index --find-links=/kaggle/input/packages biopython ml-collections python-box dm-tree openmm[cuda12]
    import Bio.PDB
import sys
import os
sys.path.append('/kaggle/input/rhofold-custom')
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils import get_device, save_ss2ct, timing
#from rhofold.relax.relax import AmberRelaxation
from rhofold.utils.alphabet import get_features

print('Finished loading packages')


# prepare fasta files
test_seqs = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
all_RNA_IDs = test_seqs['target_id'].to_list()
print(all_RNA_IDs)

main_output_dir = '/kaggle/working'
fasta_dir = f'{main_output_dir}/fasta-files'
os.makedirs(fasta_dir, exist_ok=True)

for i, row in test_seqs.iterrows():
    each_RNA_ID = row['target_id']
    fasta = f'{fasta_dir}/{each_RNA_ID}.fasta'
    with open(fasta, 'w') as f:
        f.write(f'>{each_RNA_ID}\n')
        f.write(row['sequence'] + '\n')


# do predictions
pretrained = '/kaggle/input/rhofold-custom/pretrained/RhoFold_pretrained.pt'

@torch.no_grad()
def predict(fasta, output_dir, a3m=None, ckpt=pretrained, device='cuda:0', 
            single_seq_mode=False, verbose=False):
    """
    Do inference with RhoFold.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger('RhoFold Inference')
    logger.setLevel(level=logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(f'{output_dir}/log.txt', mode='w')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    model = RhoFold(rhofold_config)
    
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
    model.eval()
    
    if single_seq_mode:
        if verbose:
            logger.info('Use single sequence mode')
        a3m = fasta
    elif a3m is None:
        # MSA search is too slow so we have to provide MSA file.
        if verbose:
            logger.info('Cannot do MSA search currently, please provide MSA file')
            logger.info('No MSA file provided, use single sequence mode')
        a3m = fasta
    else:
        if verbose:
            logger.info(f'Use {a3m} as MSA file')
    
    with timing('RhoFold Inference', logger=logger):
        device = get_device(device)
        if verbose:
            logger.info(f'    Use device {device}')
        model = model.to(device)
        
        data_dict = get_features(fasta, a3m)
        
        outputs = model(tokens=data_dict['tokens'].to(device),
                       rna_fm_tokens=data_dict['rna_fm_tokens'].to(device),
                       seq=data_dict['seq'])
        output = outputs[-1]
        
        ss_prob_map = torch.sigmoid(output['ss'][0, 0]).data.cpu().numpy()
        ss_file = f'{output_dir}/ss.ct'
        save_ss2ct(ss_prob_map, data_dict['seq'], ss_file, threshold=0.5)
        
        npz_file = f'{output_dir}/results.npz'
        np.savez_compressed(npz_file,
                            dist_n = torch.softmax(output['n'].squeeze(0), dim=0).data.cpu().numpy(),
                            dist_p = torch.softmax(output['p'].squeeze(0), dim=0).data.cpu().numpy(),
                            dist_c = torch.softmax(output['c4_'].squeeze(0), dim=0).data.cpu().numpy(),
                            ss_prob_map = ss_prob_map,
                            plddt = output['plddt'][0].data.cpu().numpy(),
                            )
        unrelaxed_model = f'{output_dir}/unrelaxed_model.pdb'
        
        node_cords_pred = output['cord_tns_pred'][-1].squeeze(0)
        model.structure_module.converter.export_pdb_file(data_dict['seq'],
                                                         node_cords_pred.data.cpu().numpy(),
                                                         path=unrelaxed_model, chain_id=None,
                                                         confidence=output['plddt'][0].data.cpu().numpy(),
                                                         logger=logger)
    
    # no relaxation
    return None


# skip some chains as they are too long and will cause memory issue
skip = ['R1126', 'R1136', 'R1138']

for each_RNA_ID in all_RNA_IDs:
    if each_RNA_ID in skip:
        print(f'Skip {each_RNA_ID}')
        continue
    fasta = f'{fasta_dir}/{each_RNA_ID}.fasta'
    output_dir = f'{main_output_dir}/RhoFold-single-seq-predictions/{each_RNA_ID}'
    print(f'Predict {each_RNA_ID}')
    predict(fasta, output_dir, single_seq_mode=True)
    
    # clear memory
    gc.collect()
    torch.cuda.empty_cache()


# define function to parse pdb file
# from github openabc repo
def parse_pdb(pdb_file):
    """
    Load pdb file as pandas dataframe.
    Note there should be only a single frame in the pdb file.
    
    Parameters
    ----------
    pdb_file : str
        Path for the pdb file. 
    
    Returns
    -------
    pdb_atoms : pd.DataFrame
        A pandas dataframe includes atom information. 
    
    """
    def pdb_line(line):
        return dict(recname=str(line[0:6]).strip(),
                    serial=int(line[6:11]),
                    name=str(line[12:16]).strip(),
                    altLoc=str(line[16:17]),
                    resname=str(line[17:20]).strip(),
                    chainID=str(line[21:22]),
                    resSeq=int(line[22:26]),
                    iCode=str(line[26:27]),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    occupancy=0.0 if line[54:60].strip() == '' else float(line[54:60]),
                    tempFactor=0.0 if line[60:66].strip() == '' else float(line[60:66]),
                    element=str(line[76:78].strip()),
                    charge=str(line[78:80].strip()))
    with open(pdb_file, 'r') as pdb:
        lines = []
        for line in pdb:
            if (len(line) > 6) and (line[:6] in ['ATOM  ', 'HETATM']):
                lines += [pdb_line(line)]
    pdb_atoms = pd.DataFrame(lines)
    pdb_atoms = pdb_atoms[['recname', 'serial', 'name', 'altLoc',
                           'resname', 'chainID', 'resSeq', 'iCode',
                           'x', 'y', 'z', 'occupancy', 'tempFactor',
                           'element', 'charge']]
    return pdb_atoms


# prepare submission
df_sample_submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')
df_submission_list = []
for each_RNA_ID in all_RNA_IDs:
    if each_RNA_ID in skip:
        continue
    print(f'Collect coordinates for {each_RNA_ID}')
    output_dir = f'{main_output_dir}/RhoFold-single-seq-predictions/{each_RNA_ID}'
    pdb = f'{output_dir}/unrelaxed_model.pdb'
    assert os.path.exists(pdb)
    atoms = parse_pdb(pdb)
    flag = atoms['name'] == "C1'"
    C1p_atoms = atoms[flag].copy()
    n_C1p_atoms = len(C1p_atoms.index)
    C1p_atoms['serial'] = np.arange(1, len(C1p_atoms.index) + 1)
    coords = C1p_atoms[['x', 'y', 'z']].to_numpy() # in unit angstrom
    each_df_submission = pd.DataFrame()
    each_df_submission['ID'] = [f'{each_RNA_ID}_{i + 1}' for i in range(n_C1p_atoms)]
    each_df_submission['resname'] = C1p_atoms['resname'].to_list()
    each_df_submission['resid'] = 1 + np.arange(n_C1p_atoms)
    
    # just submit one prediction, others are zeros
    each_df_submission[['x_1', 'y_1', 'z_1']] = coords
    for i in range(2, 6):
        each_df_submission[[f'x_{i}', f'y_{i}', f'z_{i}']] = np.zeros((n_C1p_atoms, 3))
    df_submission_list.append(each_df_submission)

# for those skipped, just use zeros
val_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')
val_labels['RNA_ID'] = val_labels['ID'].apply(lambda x: x.split('_')[0])
unique_val_RNA_IDs = val_labels['RNA_ID'].unique()

for each_RNA_ID in skip:
    print(f'Set zero coordinates for {each_RNA_ID}')
    flag = val_labels['RNA_ID'] == each_RNA_ID
    cols = ['ID', 'resname', 'resid']
    each_df_submission = val_labels.loc[flag, cols]
    coord_cols = []
    for i in range(1, 6): 
        coord_cols += [f'x_{i}', f'y_{i}', f'z_{i}']
    n_C1p_atoms = len(each_df_submission.index)
    each_df_submission.loc[:, coord_cols] = np.zeros((n_C1p_atoms, len(coord_cols)))
    df_submission_list.append(each_df_submission)

df_submission = pd.concat(df_submission_list, axis=0, ignore_index=True)
#df_submission['RNA_ID'] = df_submission['ID'].apply(lambda x: x.split('_')[0])
#df_submission = df_submission.sort_values(by=['RNA_ID', 'resid'])
#print(df_submission.head())
#df_submission = df_submission.drop(columns=['RNA_ID'])

# rearrange rows based on sample submission
IDs = df_sample_submission['ID'].to_list()
df_submission.index = df_submission['ID']
df_submission = df_submission.loc[IDs, df_sample_submission.columns]
df_submission.index = np.arange(len(df_submission.index))
df_submission.to_csv(f'{main_output_dir}/submission.csv', index=False)

