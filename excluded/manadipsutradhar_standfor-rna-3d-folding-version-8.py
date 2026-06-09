# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Basic configuration
MODEL_TYPE = 'protenix'
VALIDATION = False


# Create necessary directories and links
! mkdir /af3-dev 
! ln -s /kaggle/input/protenix-checkpoints /af3-dev/release_data
! ls /af3-dev/release_data/

# Copy USalign tool
! cp /kaggle/input/usalign/USalign /kaggle/working/
! sudo chmod u+x /kaggle/working/USalign


!pip install --no-deps protenix
!pip install biopython
!pip install ml-collections
!pip install biotite==1.0.1
!pip install rdkit


# Import required libraries
import os
import sys
import re
import time
import numpy as np
import torch
import pandas as pd
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from tqdm import tqdm

PYTHON = sys.executable
USALIGN = '/kaggle/working/USalign'
DATA_KAGGLE_DIR = '/kaggle/input/stanford-rna-3d-folding'
time0 = time.time()

print('Imports successful!')


# Helper class
class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

# Visualization helper
def set_aspect_equal(ax):
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    max_range = max(x_limits[1] - x_limits[0],
                   y_limits[1] - y_limits[0],
                   z_limits[1] - z_limits[0]) / 2.0

    ax.set_xlim(x_middle - max_range, x_middle + max_range)
    ax.set_ylim(y_middle - max_range, y_middle + max_range)
    ax.set_zlim(z_middle - max_range, z_middle + max_range)


# Data processing functions
def get_truth_df(target_id):
    truth_df = LABEL_DF[LABEL_DF['target_id'] == target_id]
    return truth_df.reset_index(drop=True)

def parse_output_to_df(output, seq, target_id):
    df = []
    chain_data = []
    for i, res in enumerate(seq):
        d = dict(ID=target_id, resname=res, resid=i+1)
        for n in range(len(output)):
            d = {**d, f'x_{n+1}': round(output[n,i,0].item(),3),
                     f'y_{n+1}': round(output[n,i,1].item(),3),
                     f'z_{n+1}': round(output[n,i,2].item(),3)}
        chain_data.append(d)

    if chain_data:
        chain_df = pd.DataFrame(chain_data)
        df.append(chain_df)
    return df

def parse_pdb_to_df(pdb_file, target_id):
    parser = PDBParser()
    structure = parser.get_structure('', pdb_file)

    df = []
    for model in structure:
        for chain in model:
            print(chain)
            chain_data = []
            for residue in chain:
                if residue.get_resname() in ['A', 'U', 'G', 'C']:
                    if 'C1\'' in residue:
                        atom = residue['C1\'']
                        xyz = atom.get_coord()
                        resname = residue.get_resname()
                        resid = residue.get_id()[1]
                        
                        chain_data.append(dict(
                            ID=target_id+'_'+str(resid),
                            resname=resname,
                            resid=resid,
                            x_1=xyz[0],
                            y_1=xyz[1],
                            z_1=xyz[2],
                        ))

            if chain_data:
                chain_df = pd.DataFrame(chain_data)
                df.append(chain_df)
    return df


# USalign functions
def write_target_line(
    atom_name, atom_serial, residue_name, chain_id, residue_num, 
    x_coord, y_coord, z_coord, occupancy=0.9, b_factor=0.0, atom_type='P'
):
    return f'ATOM  {atom_serial:>5d}  {atom_name:<5s} {residue_name:<3s} {residue_num:>3d}    {x_coord:>8.3f}{y_coord:>8.3f}{z_coord:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}           {atom_type}\n'

def write_xyz_to_pdb(df, pdb_file, xyz_id=1):
    resolved_cnt = 0
    with open(pdb_file, 'w') as target_file:
        for _, row in df.iterrows():
            x_coord = row[f'x_{xyz_id}']
            y_coord = row[f'y_{xyz_id}']
            z_coord = row[f'z_{xyz_id}']

            if x_coord > -1e17 and y_coord > -1e17 and z_coord > -1e17:
                resolved_cnt += 1
                target_line = write_target_line(
                    atom_name="C1'",
                    atom_serial=int(row['resid']),
                    residue_name=row['resname'],
                    chain_id='0',
                    residue_num=int(row['resid']),
                    x_coord=x_coord,
                    y_coord=y_coord,
                    z_coord=z_coord,
                    atom_type='C',
                )
                target_file.write(target_line)
    return resolved_cnt

def parse_usalign_for_tm_score(output):
    tm_score_match = re.findall(r'TM-score=\s+([\d.]+)', output)[1]
    if not tm_score_match:
        raise ValueError('No TM score found')
    return float(tm_score_match)

def parse_usalign_for_transform(output):
    matrix_lines = []
    found_matrix = False

    for line in output.splitlines():
        if "The rotation matrix to rotate Structure_1 to Structure_2" in line:
            found_matrix = True
        elif found_matrix and re.match(r'^\d+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+$', line):
            matrix_lines.append(line)
        elif found_matrix and not line.strip():
            break

    rotation_matrix = []
    for line in matrix_lines:
        parts = line.split()
        row_values = list(map(float, parts[1:]))
        rotation_matrix.append(row_values)

    return np.array(rotation_matrix)

def call_usalign(predict_df, truth_df, verbose=1):
    truth_pdb = '~truth.pdb'
    predict_pdb = '~predict.pdb'
    write_xyz_to_pdb(predict_df, predict_pdb, xyz_id=1)
    write_xyz_to_pdb(truth_df, truth_pdb, xyz_id=1)

    command = f'{USALIGN} {predict_pdb} {truth_pdb} -atom " C1\'" -m -'
    output = os.popen(command).read()
    if verbose==1:
        print(output)
    tm_score = parse_usalign_for_tm_score(output)
    transform = parse_usalign_for_transform(output)
    return tm_score, transform

print('Helper functions loaded successfully!')


if MODEL_TYPE == 'protenix':
    from runner.batch_inference import get_default_runner
    from runner.inference import update_inference_configs, InferenceRunner
    from protenix.data.infer_data_pipeline import InferenceDataset

    np.random.seed(244)
    torch.random.manual_seed(244)
    torch.cuda.manual_seed_all(244)

    class DictDataset(InferenceDataset):
        def __init__(
            self,
            seq_list: list,
            dump_dir: str,
            id_list: list = None,
            use_msa: bool = False,
        ) -> None:
            self.dump_dir = dump_dir
            self.use_msa = use_msa
            if isinstance(id_list, type(None)):
                self.inputs = [{"sequences": 
                                [{"rnaSequence": 
                                  {"sequence": seq, 
                                   "count": 1}}],
                                "name": "query"} for seq in seq_list]
            else:
                self.inputs = [{"sequences": 
                                [{"rnaSequence": 
                                  {"sequence": seq, 
                                   "count": 1}}],
                                "name": i} for i, seq in zip(id_list, seq_list)]


if MODEL_TYPE == 'protenix':
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config.config import parse_configs

    configs_base["use_deepspeed_evo_attention"] = (
        os.environ.get("USE_DEEPSPEED_EVO_ATTENTION", False) == "true")
    configs_base["model"]["N_cycle"] = 10
    configs_base["sample_diffusion"]["N_sample"] = (1 if VALIDATION else 5)
    configs_base["sample_diffusion"]["N_step"] = 10
    inference_configs['load_checkpoint_path'] = '/kaggle/input/protenix-checkpoints/model_v0.2.0.pt'
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}

    configs = parse_configs(
        configs=configs,
        fill_required_with_null=True,
    )
    
    # Initialize the runner
    runner = InferenceRunner(configs)


# Load validation data if in validation mode
if VALIDATION:
    LABEL_DF = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
    LABEL_DF['target_id'] = LABEL_DF['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    train_df = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')


# Run validation if enabled
if MODEL_TYPE == 'protenix' and VALIDATION:
    import warnings
    warnings.filterwarnings("ignore")  
    
    train_df['protenix_tm_score'] = None
    dataset = DictDataset(train_df.sequence, dump_dir='output', id_list=train_df.target_id, use_msa=False)
    num_data = len(dataset)
    
    for i, seq in tqdm(enumerate(train_df.sequence), total=num_data):
        if train_df.loc[i, 'protenix_tm_score'] is not None:
            continue
        if len(seq) > 400:
            continue
            
        target_id = train_df.loc[i, 'target_id']
        truth_df = get_truth_df(target_id)
        
        if sum(~np.isnan(truth_df.x_1)) < 3:
            continue
            
        data, atom_array, data_error_message = dataset[i]
        if data_error_message != '':
            continue
            
        new_configs = update_inference_configs(configs, data["N_token"].item())
        runner.update_model_configs(new_configs)
        prediction = runner.predict(data)
        prediction = prediction['coordinate'][:, data['input_feature_dict']['atom_to_tokatom_idx'] == 12]
        
        result = parse_output_to_df(prediction[:1], seq, target_id)[0]
        try:
            tm_score, transform = call_usalign(result, truth_df, verbose=0)
            train_df.loc[i, 'protenix_tm_score'] = tm_score
        except:
            pass
            
        if (time.time() - time0) > (12 * 3600 - 360):
            break
            
    train_df.to_csv('tm_scores.csv', index=False)
    print(f"Average TM score: {train_df.protenix_tm_score.mean()}")


# Generate predictions for test data
if MODEL_TYPE == 'protenix' and not VALIDATION:
    test_df = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
    import warnings
    warnings.filterwarnings("ignore")  
    
    dataset = DictDataset(test_df.sequence, dump_dir='output', id_list=test_df.target_id, use_msa=False)
    num_data = len(dataset)
    
    for i, seq in tqdm(enumerate(test_df.sequence), total=num_data):
        try:
            data, atom_array, data_error_message = dataset[i]
            target_id = data["sample_name"]
            assert target_id == test_df.target_id[i]
            assert data_error_message == ''
            
            new_configs = update_inference_configs(configs, data["N_token"].item())
            runner.update_model_configs(new_configs)
            prediction = runner.predict(data)
            prediction = prediction['coordinate'][:, data['input_feature_dict']['atom_to_tokatom_idx'] == 12]

            result = parse_output_to_df(prediction, seq, target_id)[0]
        except:
            target_id = test_df.target_id[i]
            print(f'Failed to predict {target_id}')
            result = pd.DataFrame(columns=['ID', 'resname', 'resid', 
                                           'x_1', 'y_1', 'z_1', 
                                           'x_2', 'y_2', 'z_2',
                                           'x_3', 'y_3', 'z_3', 
                                           'x_4', 'y_4', 'z_4', 
                                           'x_5', 'y_5', 'z_5'], 
                                  data=[[target_id, x, j+1] + [0.0]*15 for j, x in enumerate(seq)])
            
        result['ID'] = result.apply(lambda x: x.ID + '_' + str(x.resid), axis=1)
        result.to_csv('submission.csv', index=False, mode='a', header=(i==0))
        torch.cuda.empty_cache()




