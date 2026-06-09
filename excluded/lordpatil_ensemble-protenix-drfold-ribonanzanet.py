# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import sys
import pandas as pd
import numpy as np
import torch
import re
import time
import warnings
import yaml
import gc
from tqdm.notebook import tqdm as tqdm_nb  # Use notebook tqdm


# Suppress warnings
warnings.filterwarnings("ignore")

# Timing and Memory Utilities
start_time_global = time.time()



def time_to_str(t, mode='min'):
    if mode == 'min':
        t_int = int(t) / 60
        hr = int(t_int // 60)
        min_val = int(t_int % 60)
        return '%2d hr %02d min' % (hr, min_val)
    elif mode == 'sec':
        t_int = int(t)
        min_val = int(t_int // 60)
        sec = int(t_int % 60)
        return '%2d min %02d sec' % (min_val, sec)
    else:
        raise NotImplementedError

def gpu_memory_use():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3 # Cache
        return int(round(allocated)), int(round(reserved))
    else:
        return 0, 0

# --- Configuration ---
ENSEMBLE_MODELS = ['protenix', 'drfold', 'ribonanzanet']
NUM_CONFORMATIONS = 5
KAGGLE_INPUT_DIR = '/kaggle/input'
KAGGLE_WORKING_DIR = '/kaggle/working'
OUTPUT_FILE = 'submission.csv'

# Paths to model-specific inputs (adjust if necessary based on your Kaggle dataset names)
PROTENIX_CHECKPOINTS = f'{KAGGLE_INPUT_DIR}/protenix-checkpoints'
DRFOLD_DUMMY_DIR = f'{KAGGLE_INPUT_DIR}/hengck23-drfold2-dummy-00'
USALIGN_INPUT = f'{KAGGLE_INPUT_DIR}/usalign'
RIBONANZANET_2D_DIR = f'{KAGGLE_INPUT_DIR}/ribonanzanet2d-final'
RIBONANZANET_3D_WEIGHTS = f'{KAGGLE_INPUT_DIR}/ribonanzanet-3d-finetune'
BIOPYTHON_WHL = f'{KAGGLE_INPUT_DIR}/biopython/biopython-1.85-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'




!python --version


# !pip install /kaggle/input/biotite311/biotite-1.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


# !pip install /kaggle/input/d/lordpatil/protenix/protenix-0.4.6-py3-none-any.whl


import warnings
warnings.filterwarnings("ignore")


# --- Installations and Setup ---
print("--- Installing Dependencies ---")
# Protenix dependencies (assuming it needs to be installed)
# Note: Protenix installation might be complex and require specific versions.
# Ensure the protenix library and its dependencies are available in the Kaggle environment
# or install them here if needed. The original notebook commented these out for submission.
!pip install /kaggle/input/biopython311/biopython-1.85-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install --no-deps /kaggle/input/d/lordpatil/protenix/protenix-0.4.6-py3-none-any.whl 

!pip install /kaggle/input/ml-collections/ml_collections-1.0.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/biotite311/biotite-1.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install /kaggle/input/rdkit311x86/rdkit-2024.9.4-cp311-cp311-manylinux_2_28_x86_64.whl



# DRFold dependencies
try:
    import Bio
    print("Biopython already installed.")
except ImportError:
    print("Installing Biopython...")
    !pip install "{BIOPYTHON_WHL}"

print("Dependency check/installation complete.")

# --- Setup Environment for Models ---
print("\n--- Setting up Model Environments ---")

# Protenix setup
if 'protenix' in ENSEMBLE_MODELS:
    print("Setting up Protenix environment...")
    os.environ['PROTENIX_DATA_ROOT_DIR'] = PROTENIX_CHECKPOINTS
    os.makedirs('/af3-dev', exist_ok=True)
    if not os.path.exists('/af3-dev/release_data'):
        os.symlink(PROTENIX_CHECKPOINTS, '/af3-dev/release_data', target_is_directory=True)
    print("Protenix files linked:")
    !ls /af3-dev/release_data/


# DRFold setup
if 'drfold' in ENSEMBLE_MODELS:
    print("\nSetting up DRFold environment...")
    USALIGN_EXEC = f'{KAGGLE_WORKING_DIR}/USalign'
    if not os.path.exists(USALIGN_EXEC):
        print("Copying USalign...")
        os.system(f'cp {USALIGN_INPUT}/USalign {KAGGLE_WORKING_DIR}/')
        os.system(f'chmod +x {USALIGN_EXEC}')
        print("USalign copied and made executable.")
    else:
        print("USalign already exists.")
    # Add DRFold code path
    sys.path.append(f'{DRFOLD_DUMMY_DIR}/drfold2/cfg_97')

# RibonanzaNet setup
if 'ribonanzanet' in ENSEMBLE_MODELS:
     print("\nSetting up RibonanzaNet environment...")
     sys.path.append(RIBONANZANET_2D_DIR)

print("Model environment setup complete.")

# --- Helper Functions ---

# Generic dotdict
class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


# DataFrame formatting function (common requirement)
def format_predictions_to_df(target_id, sequence, coords_list):
    """
    Formats predictions into the submission DataFrame structure.

    Args:
        target_id (str): The ID of the RNA target.
        sequence (str): The RNA sequence.
        coords_list (list): A list of numpy arrays, where each array is (L, 3)
                             representing one conformation's C1' coordinates.
                             Should contain NUM_CONFORMATIONS arrays.

    Returns:
        pd.DataFrame: DataFrame rows for this sequence.
    """
    L = len(sequence)
    if not coords_list or len(coords_list) != NUM_CONFORMATIONS:
        print(f"Warning: Incorrect number of conformations for {target_id}. Expected {NUM_CONFORMATIONS}, got {len(coords_list)}. Filling with zeros.")
        # Create zero coordinates if prediction failed or returned wrong number
        coords_list = [np.zeros((L, 3), dtype=np.float32) for _ in range(NUM_CONFORMATIONS)]
        
    # Ensure all coordinate arrays have the correct length
    processed_coords = []
    for i, coords in enumerate(coords_list):
        if coords.shape[0] != L:
             print(f"Warning: Coordinate length mismatch for {target_id}, conformation {i+1}. Expected {L}, got {coords.shape[0]}. Padding/truncating.")
             # Simple padding/truncating - adjust if needed
             new_coords = np.zeros((L, 3), dtype=np.float32)
             len_to_copy = min(L, coords.shape[0])
             new_coords[:len_to_copy, :] = coords[:len_to_copy, :]
             processed_coords.append(new_coords)
        else:
             processed_coords.append(coords)
             
    df_rows = []
    for i in range(L):
        row_data = {
            'ID': f'{target_id}_{i + 1}',
            'resname': sequence[i],
            'resid': i + 1
        }
        for k in range(NUM_CONFORMATIONS):
            coords = processed_coords[k]
            row_data[f'x_{k+1}'] = coords[i, 0]
            row_data[f'y_{k+1}'] = coords[i, 1]
            row_data[f'z_{k+1}'] = coords[i, 2]
        df_rows.append(row_data)
    return pd.DataFrame(df_rows)

# Function to create zero predictions for a sequence
def create_zero_predictions_df(target_id, sequence):
    print(f"Creating zero predictions for {target_id} due to failure.")
    zero_coords = [np.zeros((len(sequence), 3), dtype=np.float32) for _ in range(NUM_CONFORMATIONS)]
    return format_predictions_to_df(target_id, sequence, zero_coords)

# --- Load Test Data ---
print("\n--- Loading Test Data ---")
test_df = pd.read_csv(f"{KAGGLE_INPUT_DIR}/stanford-rna-3d-folding/test_sequences.csv")
print(f"Loaded {len(test_df)} test sequences.")
print(test_df.head())

# Dictionary to store predictions from each model
model_predictions = {}


# === Protenix Prediction ===
def predict_protenix(test_sequences_df):
    print("\n--- Running Protenix Predictions ---")
    try:
        from runner.inference import update_inference_configs, InferenceRunner
        from protenix.data.infer_data_pipeline import InferenceDataset
        from configs.configs_base import configs as configs_base
        from configs.configs_data import data_configs
        from configs.configs_inference import inference_configs
        from protenix.config.config import parse_configs
    except ImportError as e:
        print(f"Protenix import failed: {e}. Skipping Protenix.")
        return None

    # Protenix DictDataset specific to its data loading
    class ProtenixDictDataset(InferenceDataset):
        def __init__(self, seq_list: list, id_list: list, dump_dir: str = 'output', use_msa: bool = False) -> None:
            self.dump_dir = dump_dir
            self.use_msa = use_msa
            self.inputs = [{
                "sequences": [{"rnaSequence": {"sequence": seq, "count": 1}}],
                "name": i
            } for i, seq in zip(id_list, seq_list)]

    all_preds_df = pd.DataFrame()
    try:
        # Configure Protenix
        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        configs_base["use_deepspeed_evo_attention"] = (os.environ.get("USE_DEEPSPEED_EVO_ATTENTION", "false").lower() == "true")
        configs_base["model"]["N_cycle"] = 10 # As in notebook
        configs_base["sample_diffusion"]["N_sample"] = NUM_CONFORMATIONS # Ensure 5 samples
        configs_base["sample_diffusion"]["N_step"] = 200 # As in notebook
        inference_configs['load_checkpoint_path'] = f'{PROTENIX_CHECKPOINTS}/model_v0.2.0.pt'
        configs = {**configs_base, **{"data": data_configs}, **inference_configs}
        configs = parse_configs(configs=configs, fill_required_with_null=True)

        runner = InferenceRunner(configs)
        print("Protenix runner initialized.")

        dataset = ProtenixDictDataset(
            test_sequences_df.sequence.tolist(),
            test_sequences_df.target_id.tolist()
        )
        print(f"Protenix dataset created with {len(dataset)} samples.")
        
        for i in tqdm_nb(range(len(dataset)), desc="Protenix"):
            target_id = test_sequences_df.target_id[i]
            seq = test_sequences_df.sequence[i]
            try:
                data, _, data_error_message = dataset[i]
                if data_error_message != '':
                    print(f"Protenix data error for {target_id}: {data_error_message}")
                    raise ValueError(data_error_message)

                # Protenix uses 'N_token' for config update
                N_token = data.get("N_token")
                if N_token is None:
                     print(f"Warning: 'N_token' not found in Protenix data for {target_id}. Cannot update configs dynamically. Using defaults.")
                     # Handle cases where N_token might be missing, maybe use sequence length?
                     # Or skip updating configs, which might be okay for inference.
                     new_configs = configs # Use original configs
                else:
                     new_configs = update_inference_configs(configs, N_token.item())
                
                runner.update_model_configs(new_configs)
                
                with torch.no_grad():
                    prediction_dict = runner.predict(data)

                # Extract C1' atoms (atom index 12 in protenix output)
                # Output shape: [N_sample, N_residue, 3]
                atom_to_tokatom_idx = data.get('input_feature_dict', {}).get('atom_to_tokatom_idx', None)
                if atom_to_tokatom_idx is None:
                    raise ValueError(f"Could not find 'atom_to_tokatom_idx' in Protenix data for {target_id}")
                    
                c1_prime_coords = prediction_dict['coordinate'][:, atom_to_tokatom_idx == 12]
                
                if c1_prime_coords.shape[0] != NUM_CONFORMATIONS or c1_prime_coords.shape[1] != len(seq):
                     print(f"Warning: Protenix output shape mismatch for {target_id}. Expected ({NUM_CONFORMATIONS}, {len(seq)}, 3), got {c1_prime_coords.shape}")
                     # Attempt to reshape or handle error, for now, raise to fall back to zeros
                     raise ValueError("Shape mismatch in Protenix output")

                coords_list = [c1_prime_coords[k].cpu().numpy() for k in range(NUM_CONFORMATIONS)]
                seq_df = format_predictions_to_df(target_id, seq, coords_list)

            except Exception as e:
                print(f"ERROR predicting with Protenix for {target_id}: {e}")
                seq_df = create_zero_predictions_df(target_id, seq)

            all_preds_df = pd.concat([all_preds_df, seq_df], ignore_index=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        print("Protenix predictions finished.")
        return all_preds_df

    except Exception as e:
        print(f"FATAL ERROR during Protenix prediction setup or loop: {e}")
        # Return a DataFrame of zeros for all test sequences if setup fails
        all_preds_df = pd.DataFrame()
        for i in range(len(test_sequences_df)):
            target_id = test_sequences_df.target_id[i]
            seq = test_sequences_df.sequence[i]
            seq_df = create_zero_predictions_df(target_id, seq)
            all_preds_df = pd.concat([all_preds_df, seq_df], ignore_index=True)
        return all_preds_df




# === DRFold Prediction ===
def predict_drfold(test_sequences_df):
    print("\n--- Running DRFold Predictions ---")
    try:
        from EvoMSA2XYZ.Model import MSA2XYZ
        from RNALM2.Model import RNA2nd
        from data import parse_seq, Get_base, BASE_COOR, write_frame_coor_to_pdb, parse_pdb_to_xyz
    except ImportError as e:
        print(f"DRFold import failed: {e}. Skipping DRFold.")
        return None

    MAX_LENGTH_DRFOLD = 480 # From notebook
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # DRFold specific helpers
    def make_data_drfold(seq):
        aa_type = parse_seq(seq)
        base = Get_base(seq, BASE_COOR)
        seq_idx = np.arange(len(seq)) + 1
        msa = aa_type[None, :]
        msa = torch.from_numpy(msa)
        # The notebook duplicates the MSA - replicate that behavior
        msa = torch.cat([msa, msa], 0)
        msa = torch.nn.functional.one_hot(msa.long(), 6).float()
        base_x = torch.from_numpy(base).float()
        seq_idx = torch.from_numpy(seq_idx).long()
        return msa, base_x, seq_idx

    def coord_to_df_drfold(sequence, coord_list, target_id):
        # Adapts solution_to_submit_df logic for a single sequence
        L = len(sequence)
        df_rows = []
        for i in range(L):
            row_data = {
                'ID': f'{target_id}_{i + 1}',
                'resname': sequence[i],
                'resid': i + 1
            }
            for k in range(NUM_CONFORMATIONS):
                 coords = coord_list[k]
                 if coords.shape[0] != L: # Handle length mismatch from padding/truncating
                     print(f"Warning: DRFold internal coord mismatch for {target_id}, conf {k+1}. Len {coords.shape[0]} vs seq {L}.")
                     # Provide zeros if mismatch is severe, otherwise pad/truncate? Using zeros for safety.
                     row_data[f'x_{k+1}'] = 0.0
                     row_data[f'y_{k+1}'] = 0.0
                     row_data[f'z_{k+1}'] = 0.0
                 else:
                    row_data[f'x_{k+1}'] = coords[i, 0]
                    row_data[f'y_{k+1}'] = coords[i, 1]
                    row_data[f'z_{k+1}'] = coords[i, 2]
            df_rows.append(row_data)
        return pd.DataFrame(df_rows)

    all_preds_df = pd.DataFrame()
    try:
        # Load RNALM model (needed by MSA2XYZ)
        rnalm = RNA2nd(dict(s_in_dim=5, z_in_dim=2, s_dim=512, z_dim=128, N_elayers=18))
        rnalm_file = f'{DRFOLD_DUMMY_DIR}/RCLM/epoch_67000'
        print(f"Loading RNALM from {rnalm_file}")
        rnalm.load_state_dict(torch.load(rnalm_file, map_location='cpu', weights_only=True), strict=False)
        rnalm = rnalm.to(DEVICE).eval()
        print("RNALM loaded.")

        msa2xyz_models = []
        model_indices = [0, 1, 2, 8, 9] # Checkpoints used in the notebook
        for k in model_indices:
            msa2xyz = MSA2XYZ(dict(seq_dim=6, msa_dim=7, N_ensemble=1, N_cycle=8, m_dim=64, s_dim=64, z_dim=64))
            msa2xyz_file = f'{DRFOLD_DUMMY_DIR}/cfg_97/model_{k}'
            print(f"Loading MSA2XYZ from {msa2xyz_file}")
            msa2xyz.load_state_dict(torch.load(msa2xyz_file, map_location='cpu', weights_only=True), strict=True)
            msa2xyz.msaxyzone.premsa.rnalm = rnalm # Inject RNALM
            msa2xyz = msa2xyz.to(DEVICE).eval()
            msa2xyz_models.append(msa2xyz)
            print(f"MSA2XYZ model {k} loaded.")

        for i, row in tqdm_nb(test_sequences_df.iterrows(), total=len(test_sequences_df), desc="DRFold"):
            target_id = row.target_id
            sequence = row.sequence
            L = len(sequence)
            coords_list = []

            try:
                # Handle potential length truncation/padding like in the notebook
                if L > MAX_LENGTH_DRFOLD:
                    # The original notebook picked a random start index. For reproducibility/submission,
                    # let's just take the first MAX_LENGTH_DRFOLD residues.
                    # Or, should we process the whole sequence if possible? Let's try processing all first.
                    # If memory error, we might need to truncate.
                    # The original notebook logic:
                    # i0 = np.random.choice(L - MAX_LENGTH_DRFOLD + 1)
                    # i1 = i0 + MAX_LENGTH_DRFOLD
                    # seq_processed = sequence[i0:i1]
                    # L_processed = len(seq_processed)
                    # print(f"Warning: Sequence {target_id} too long ({L}), truncating to {MAX_LENGTH_DRFOLD}")
                    # Let's process the full sequence first. If it fails, we'll know.
                    seq_processed = sequence
                    L_processed = L
                    i0, i1 = 0, L # Keep track for potential later padding (though not needed if processing full seq)
                else:
                    seq_processed = sequence
                    L_processed = L
                    i0, i1 = 0, L

                msa, base_x, seq_idx = make_data_drfold(seq_processed)
                msa, base_x, seq_idx = msa.to(DEVICE), base_x.to(DEVICE), seq_idx.to(DEVICE)

                for model_idx, msa2xyz_model in enumerate(msa2xyz_models):
                    with torch.no_grad():
                        out = msa2xyz_model.pred(msa, seq_idx, None, base_x, np.array(list(seq_processed)))
                    
                    # C1' coordinate is the second atom in the frame (index 1)
                    coord_c1_prime = out['coor'][:, 1, :].cpu().numpy() 

                    if L != L_processed: # Handle padding if truncation occurred (not currently happening)
                         padded_coord = np.zeros((L, 3), dtype=np.float32)
                         padded_coord[i0:i1, :] = coord_c1_prime
                         coords_list.append(padded_coord)
                    else:
                        coords_list.append(coord_c1_prime)
                        
                # Convert list of coordinates to the required format DataFrame
                seq_df = format_predictions_to_df(target_id, sequence, coords_list)
            
            except Exception as e:
                 print(f"ERROR predicting with DRFold for {target_id}: {e}")
                 seq_df = create_zero_predictions_df(target_id, sequence)
            
            all_preds_df = pd.concat([all_preds_df, seq_df], ignore_index=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        print("DRFold predictions finished.")
        return all_preds_df

    except Exception as e:
        print(f"FATAL ERROR during DRFold prediction setup or loop: {e}")
        # Return a DataFrame of zeros for all test sequences if setup fails
        all_preds_df = pd.DataFrame()
        for i in range(len(test_sequences_df)):
            target_id = test_sequences_df.target_id[i]
            seq = test_sequences_df.sequence[i]
            seq_df = create_zero_predictions_df(target_id, seq)
            all_preds_df = pd.concat([all_preds_df, seq_df], ignore_index=True)
        return all_preds_df


# === RibonanzaNet Prediction ===
def predict_ribonanzanet(test_sequences_df):
    print("\n--- Running RibonanzaNet Predictions ---")
    try:
        # Need RNADataset, Config, finetuned_RibonanzaNet from the notebook setup
        from Network import RibonanzaNet # Assuming Network.py is in the path
        from torch.utils.data import Dataset
        
        # RibonanzaNet specific dataset
        class RNADatasetRibonanza(Dataset):
            def __init__(self, data):
                self.data = data
                self.tokens = {nt: i for i, nt in enumerate('ACGU')}
            def __len__(self): return len(self.data)
            def __getitem__(self, idx):
                sequence = [self.tokens[nt] for nt in (self.data.loc[idx, 'sequence'])]
                sequence = np.array(sequence)
                sequence = torch.tensor(sequence, dtype=torch.long) # Ensure long type
                return {'sequence': sequence, 'target_id': self.data.loc[idx, 'target_id']}
        
        # Load config and model
        config_rz = load_config_from_yaml(f"{RIBONANZANET_2D_DIR}/configs/pairwise.yaml")
        
        # Re-define the finetuned model class locally
        class finetuned_RibonanzaNet(RibonanzaNet):
             def __init__(self, config, pretrained=False):
                 config.dropout=0.0 # Set dropout to 0 for eval consistency, but notebook uses 0.2? Let's follow notebook for train() runs.
                 # Notebook uses 0.2 for train runs, 0.0 for eval runs
                 super(finetuned_RibonanzaNet, self).__init__(config)
                 if pretrained: # Not used here, loading finetuned weights below
                     self.load_state_dict(torch.load(f"{RIBONANZANET_2D_DIR}/RibonanzaNet.pt", map_location='cpu'))
                 self.dropout = nn.Dropout(0.0) # Default dropout for eval
                 self.xyz_predictor = nn.Linear(256, 3) # Assuming input dim is 256

             def forward(self, src):
                 # Use the base class method to get embeddings
                 sequence_features, pairwise_features = self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
                 # Apply dropout ONLY if in training mode
                 sequence_features = self.dropout(sequence_features) 
                 xyz = self.xyz_predictor(sequence_features)
                 return xyz

        model_rz = finetuned_RibonanzaNet(config_rz, pretrained=False)
        model_rz.load_state_dict(torch.load(f"{RIBONANZANET_3D_WEIGHTS}/RibonanzaNet-3D.pt", map_location='cpu'))
        model_rz = model_rz.to('cuda' if torch.cuda.is_available() else 'cpu')
        print("RibonanzaNet model loaded.")

        dataset_rz = RNADatasetRibonanza(test_sequences_df)
        print(f"RibonanzaNet dataset created with {len(dataset_rz)} samples.")
        
        all_preds_df = pd.DataFrame()
        
        for i in tqdm_nb(range(len(dataset_rz)), desc="RibonanzaNet"):
            item = dataset_rz[i]
            target_id = item['target_id']
            seq_str = test_sequences_df.sequence[i] # Get string sequence for formatting
            src = item['sequence'].unsqueeze(0).to(model_rz.device) # Add batch dim and move to device
            
            coords_list_np = []
            try:
                # Run 4 times in train mode (with dropout 0.2)
                model_rz.dropout = torch.nn.Dropout(0.2) # Set dropout for train mode runs
                model_rz.train()
                for _ in range(NUM_CONFORMATIONS - 1):
                     with torch.no_grad(): # Still no gradient needed for inference
                          xyz = model_rz(src).squeeze(0) # Remove batch dim
                     coords_list_np.append(xyz.cpu().numpy())
                
                # Run 1 time in eval mode (with dropout 0.0)
                model_rz.dropout = torch.nn.Dropout(0.0) # Set dropout to 0 for eval
                model_rz.eval()
                with torch.no_grad():
                    xyz = model_rz(src).squeeze(0) # Remove batch dim
                coords_list_np.append(xyz.cpu().numpy())
                
                seq_df = format_predictions_to_df(target_id, seq_str, coords_list_np)

            except Exception as e:
                 print(f"ERROR predicting with RibonanzaNet for {target_id}: {e}")
                 seq_df = create_zero_predictions_df(target_id, seq_str)

            all_preds_df = pd.concat([all_preds_df, seq_df], ignore_index=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        print("RibonanzaNet predictions finished.")
        return all_preds_df

    except Exception as e:
        print(f"FATAL ERROR during RibonanzaNet prediction setup or loop: {e}")
        # Return a DataFrame of zeros for all test sequences if setup fails
        all_preds_df = pd.DataFrame()
        for i in range(len(test_sequences_df)):
            target_id = test_sequences_df.target_id[i]
            seq = test_sequences_df.sequence[i]
            seq_df = create_zero_predictions_df(target_id, seq)
            all_preds_df = pd.concat([all_preds_df, seq_df], ignore_index=True)
        return all_preds_df



# Predict using each model
if 'protenix' in ENSEMBLE_MODELS:
    df_protenix = predict_protenix(test_df)
    if df_protenix is not None:
        model_predictions['protenix'] = df_protenix
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if 'drfold' in ENSEMBLE_MODELS:
    df_drfold = predict_drfold(test_df)
    if df_drfold is not None:
        model_predictions['drfold'] = df_drfold
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if 'ribonanzanet' in ENSEMBLE_MODELS:
    df_ribonanzanet = predict_ribonanzanet(test_df)
    if df_ribonanzanet is not None:
        model_predictions['ribonanzanet'] = df_ribonanzanet
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# --- Ensembling Predictions ---
print("\n--- Ensembling Predictions ---")

active_models = list(model_predictions.keys())
num_active_models = len(active_models)

if num_active_models == 0:
    print("ERROR: No models produced predictions. Cannot ensemble.")
    # Create a dummy submission with zeros if needed
    final_submission_df = pd.DataFrame()
    for i in range(len(test_df)):
        target_id = test_df.target_id[i]
        seq = test_df.sequence[i]
        seq_df = create_zero_predictions_df(target_id, seq)
        final_submission_df = pd.concat([final_submission_df, seq_df], ignore_index=True)

elif num_active_models == 1:
    print(f"Warning: Only one model ({active_models[0]}) produced predictions. Using its output directly.")
    final_submission_df = model_predictions[active_models[0]]

else:
    print(f"Ensembling predictions from: {active_models}")
    # Use the first available model's prediction as the base structure (ID, resname, resid)
    base_df = model_predictions[active_models[0]][['ID', 'resname', 'resid']].copy()
    
    # Initialize columns for averaged coordinates
    for k in range(1, NUM_CONFORMATIONS + 1):
        base_df[f'x_{k}'] = 0.0
        base_df[f'y_{k}'] = 0.0
        base_df[f'z_{k}'] = 0.0

    # Sum coordinates from all active models
    for model_name in active_models:
        df = model_predictions[model_name]
        # Ensure alignment (should be correct if generated from test_df in order)
        if not base_df['ID'].equals(df['ID']):
             print(f"Warning: ID mismatch for model {model_name}. Attempting to align...")
             df = df.set_index('ID')
             df = df.reindex(base_df['ID']).reset_index()
             # Fill NaNs that might result from reindexing (e.g., if one model failed on a sequence the base didn't)
             # This shouldn't happen with the current error handling, but good practice.
             df = df.fillna(0.0) 

        for k in range(1, NUM_CONFORMATIONS + 1):
            base_df[f'x_{k}'] += df[f'x_{k}'].astype(float)
            base_df[f'y_{k}'] += df[f'y_{k}'].astype(float)
            base_df[f'z_{k}'] += df[f'z_{k}'].astype(float)

    # Average the coordinates
    for k in range(1, NUM_CONFORMATIONS + 1):
        base_df[f'x_{k}'] /= num_active_models
        base_df[f'y_{k}'] /= num_active_models
        base_df[f'z_{k}'] /= num_active_models
        
    final_submission_df = base_df
    print("Ensembling complete.")




# --- Save Submission ---
print(f"\n--- Saving Final Submission to {OUTPUT_FILE} ---")
print(final_submission_df.head())
print(f"Shape: {final_submission_df.shape}")
# Check for NaNs before saving
if final_submission_df.isnull().values.any():
    print("Warning: NaNs found in the final submission DataFrame. Filling with 0.")
    final_submission_df = final_submission_df.fillna(0.0)

final_submission_df.to_csv(OUTPUT_FILE, index=False)

print("\n--- Script Finished ---")
total_runtime = time.time() - start_time_global
print(f"Total Runtime: {time_to_str(total_runtime, mode='sec')}")
mem_alloc, mem_res = gpu_memory_use()
print(f"Final GPU Memory (GB): Allocated={mem_alloc}, Reserved={mem_res}")
















