MODEL_TYPE='protenix'
VALIDATION=False
local = False





if local:
 !pip install -q --no-deps protenix
 !pip install -q biopython
 !pip install -q ml-collections
 !pip install -q biotite==1.0.1
 !pip install -q rdkit


import Bio
import random
from copy import deepcopy

import pandas as pd
from Bio.PDB import Atom, Model, Chain, Residue, Structure, PDBParser
from Bio import SeqIO
import os, sys
import re
import numpy as np
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import pickle
print('IMPORT OK !!!!')


!export PROTENIX_DATA_ROOT_DIR=/kaggle/input/protenix-checkpoints



! mkdir /af3-dev 
! ln -s /kaggle/input/protenix-checkpoints /af3-dev/release_data
! ls /af3-dev/release_data/


if MODEL_TYPE=='protenix':
    
    
    from runner.batch_inference import get_default_runner
    from runner.inference import update_inference_configs, InferenceRunner

    from protenix.data.infer_data_pipeline import InferenceDataset

    np.random.seed(21)
    torch.random.manual_seed(21)
    torch.cuda.manual_seed_all(21)

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
            if isinstance(id_list,type(None)):
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
                                "name": i} for i, seq in zip(id_list,seq_list)]


import os
import torch
from torch.cuda.amp import autocast

if MODEL_TYPE == 'protenix':
    # Import configs
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config.config import parse_configs

    # Enable DeepSpeed Evo Attention based on environment variable
    configs_base["use_deepspeed_evo_attention"] = (
        os.environ.get("USE_DEEPSPEED_EVO_ATTTENTION", False) == "true"
    )

    # Set model-specific configs
    configs_base["model"]["N_cycle"] = 10
    configs_base["sample_diffusion"]["N_sample"] = 10
    configs_base["sample_diffusion"]["N_step"] = 200


    # Set checkpoint path
    #inference_configs['load_checkpoint_path'] = '/kaggle/input/17april-proteinx/pytorch/default/17/1999_ema_0.995_casp16-2000steps.pt'
    inference_configs['load_checkpoint_path'] = '/kaggle/input/13okt-may/15999_ema_0.995.pt'
    # Merge all configs
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(configs=configs, fill_required_with_null=True)

    # Optional precision field for the runner
    configs["precision"] = "bfloat16"

    # Check GPU and bfloat16 support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Your GPU does not support bfloat16.")

   

    # Inference with autocast for bfloat16
    with autocast(dtype=torch.bfloat16):
         runner = InferenceRunner(configs)  # Replace with the actual method in your runner, e.g., 'predict()' or 'inference()'


def parse_c1_coords_to_df(c1_coords_list, sequence, target_id, num_samples=5):
    """
    Converts list of C1′ atom coordinates from multiple conformations into a DataFrame.
    Fills missing conformations with NaNs.

    Parameters:
        c1_coords_list: list of arrays of shape (L, 3)
        sequence: RNA sequence
        target_id: string
        num_samples: number of conformations to output columns for (e.g. 5)

    Returns:
        pd.DataFrame with columns: ID, resname, resid, x_1, y_1, z_1, ..., x_k, y_k, z_k
    """
    data = []
    L = len(sequence)
    for i in range(L):
        row = {
            "ID": f"{target_id}_{i + 1}",  # Fixed line
            "resname": sequence[i],
            "resid": i + 1,
        }
        for j in range(num_samples):
            if j < len(c1_coords_list):
                coords = c1_coords_list[j]
                if i < len(coords):
                    row[f"x_{j+1}"] = round(coords[i][0], 3)
                    row[f"y_{j+1}"] = round(coords[i][1], 3)
                    row[f"z_{j+1}"] = round(coords[i][2], 3)
                else:
                    row[f"x_{j+1}"] = row[f"y_{j+1}"] = row[f"z_{j+1}"] = np.nan
            else:
                row[f"x_{j+1}"] = row[f"y_{j+1}"] = row[f"z_{j+1}"] = np.nan
        data.append(row)

    df = pd.DataFrame(data)
    return df


def extract_c1_prime_coordinates_batched(coordinate_batch, atom_to_token_idx):
    """
    Extracts C1′ coordinates from a batch of conformations using atom_to_token_idx == 12.
    
    Parameters:
    - coordinate_batch: (N_samples, N_atoms, 3)
    - atom_to_token_idx: (N_atoms,) tensor mapping atoms to tokenized atom type indices
    
    Returns:
    - List of np.array of shape (L, 3), one per conformation
    """
    mask = (atom_to_token_idx == 12).detach().cpu().numpy()
    coordinate_batch = coordinate_batch.detach().cpu().numpy()
    
    all_c1_coords = [coords[mask] for coords in coordinate_batch]  # each is (L, 3)
    return all_c1_coords


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# --------------------------
# Your dependencies and imports here
# Make sure these are defined/imported:
# DictDataset, runner.predict, update_inference_configs, extract_c1_prime_coordinates_batched, parse_c1_coords_to_df
# --------------------------

# really infer on testset
KAGGLE_REEUN = os.getenv('KAGGLE_IS_COMPETITION_RERUN')



def kabsch_rmsd(P, Q):
    """
    Compute the Kabsch RMSD between two coordinate sets P and Q.
    """
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, W)))
    U = np.dot(V, np.dot(np.diag([1, 1, d]), W))
    P_rot = np.dot(P, U)
    return np.sqrt(np.mean(np.sum((P_rot - Q) ** 2, axis=1)))

if MODEL_TYPE == 'protenix' and not VALIDATION:
    test_df = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
    if not KAGGLE_REEUN:
       #test_df = test_df.head(1)
       test_df = test_df[test_df['target_id'] == "R1107"].reset_index(drop=True)


    
    dataset = DictDataset(
        test_df.sequence,
        dump_dir='output',
        id_list=test_df.target_id,
        use_msa=False
    )

    output_filename = 'submission_proteinx.csv'
    if os.path.exists(output_filename):
        os.remove(output_filename)  # Remove old file if exists

    long_seq_count = 0  # To keep track of how many sequences were filled with zeros

    for i, seq in tqdm(enumerate(test_df.sequence), total=len(dataset)):
        target_id = test_df.target_id[i]
        print(f"\nProcessing {target_id} | length={len(seq)}")

        if len(seq) > 820:
            print(f"⚠️ Sequence too long ({len(seq)} residues), filling with zeros for {target_id}")
            long_seq_count += 1
            num_residues = len(seq)
            dummy_data = {
                "ID": [f"{target_id}_{resid}" for resid in range(num_residues)],
                "resname": [seq[resid] for resid in range(num_residues)],
                "resid": [resid for resid in range(num_residues)],
            }
            for conf_idx in range(1, 6):
                dummy_data[f'x_{conf_idx}'] = [0.0] * num_residues
                dummy_data[f'y_{conf_idx}'] = [0.0] * num_residues
                dummy_data[f'z_{conf_idx}'] = [0.0] * num_residues

            result_gpe = pd.DataFrame(dummy_data)

        else:
            data, atom_array, data_error_message = dataset[i]
            assert data_error_message == ''
            assert target_id == data["sample_name"]

            new_configs = update_inference_configs(configs, data["N_token"].item())
            runner.update_model_configs(new_configs)

            prediction = runner.predict(data)
            coords_batch = prediction['coordinate']  # (N_conf, N_atoms, 3)
            summary_conf = prediction['summary_confidence']  # list of dicts

            c1_coords_list = extract_c1_prime_coordinates_batched(
                coordinate_batch=coords_batch,
                atom_to_token_idx=data['input_feature_dict']['atom_to_tokatom_idx']
            )

            c1_coords_list_np = [c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in c1_coords_list]

            # Step 1: Select top-1 by pLDDT
            top1_idx = np.argmax([float(conf['chain_plddt']) for conf in summary_conf])
            selected_indices = [top1_idx]

            # Step 2: Select 4 most diverse conformations
            while len(selected_indices) < 5:
                remaining = [j for j in range(len(c1_coords_list_np)) if j not in selected_indices]
                max_min_rmsd = []
                for j in remaining:
                    min_rmsd_to_selected = min(
                        kabsch_rmsd(c1_coords_list_np[j], c1_coords_list_np[k])
                        for k in selected_indices
                    )
                    max_min_rmsd.append(min_rmsd_to_selected)
                next_idx = remaining[np.argmax(max_min_rmsd)]
                selected_indices.append(next_idx)

            top_5_coords = [c1_coords_list[j] for j in selected_indices]
            
            result_gpe = parse_c1_coords_to_df(
                c1_coords_list=top_5_coords,
                sequence=seq,
                target_id=target_id,
                num_samples=5
            )

        # Write output
        result_gpe.to_csv(output_filename, index=False, mode='a', header=(i == 0))
        torch.cuda.empty_cache()

    print(f"\n✅ Completed inference. Sequences with length > 720 filled with zeros: {long_seq_count}")

    # -------------------------------
    # Final merging with sample_submission
    # -------------------------------
    my_submission = pd.read_csv(output_filename)
    sample_submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')

    non_coord_cols = ['ID', 'resname', 'resid']
    final_submission = sample_submission[non_coord_cols].merge(my_submission, on=non_coord_cols, how='left')
    final_submission = final_submission.fillna(0.0)
    final_submission.to_csv(output_filename, index=False)
    print("✅ Final submission aligned, filled, and saved as 'submission.csv'. Ready for upload!")


#!pip install --no-index /kaggle/input/boltz-dependencies/*whl --no-deps
!pip install --no-index /kaggle/input/fairscale-0413/*whl --no-deps
!pip install -q /kaggle/input/boltz-dependencies/mashumaro-3.14-py3-none-any.whl  --no-deps
!pip install -q /kaggle/input/boltz-dependencies/modelcif-1.3-py3-none-any.whl  --no-deps
!pip install -q /kaggle/input/boltz-dependencies/ihm-2.2-py3-none-any.whl --no-deps


%cd /kaggle/working/
%mkdir inputs_prediction
%mkdir outputs_prediction
%cp -rf /kaggle/input/boltz2/boltz/src/boltz .


%%writefile inference.py
import os
import random
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from pytorch_lightning import Trainer#, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1

import numpy as np
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True)


CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = (
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
)


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@rank_zero_only
def download(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz2_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(MODEL_URL, str(model))  # noqa: S310


def check_inputs(
    data: Path,
    outdir: Path,
    override: bool = False,
) -> list[Path]:
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        filtered_data = []
        for d in data:
            if d.suffix in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                filtered_data.append(d)
            elif d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

        data = filtered_data
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()}

    # Remove them from the input data
    if existing and not override:
        data = [d for d in data if d.stem not in existing]
        num_skipped = len(existing) - len(data)
        msg = (
            f"Found some existing predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing predictions, will override."
        click.echo(msg)

    return data


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.

    """
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


@rank_zero_only
def process_inputs(  # noqa: C901, PLR0912, PLR0915
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 4096,
    use_msa_server: bool = False,
) -> None:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    click.echo("Processing input data.")
    existing_records = None

    # Check if manifest exists at output path
    manifest_path = out_dir / "processed" / "manifest.json"
    if manifest_path.exists():
        click.echo(f"Found a manifest file at output directory: {out_dir}")

        manifest: Manifest = Manifest.load(manifest_path)
        input_ids = [d.stem for d in data]
        existing_records, processed_ids = zip(
            *[
                (record, record.id)
                for record in manifest.records
                if record.id in input_ids
            ]
        )

        if isinstance(existing_records, tuple):
            existing_records = list(existing_records)

        # Check how many examples need to be processed
        missing = len(input_ids) - len(processed_ids)
        if not missing:
            click.echo("All examples in data are processed. Updating the manifest")
            # Dump updated manifest
            updated_manifest = Manifest(existing_records)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")
            return

        click.echo(f"{missing} missing ids. Preprocessing these ids")
        missing_ids = list(set(input_ids).difference(set(processed_ids)))
        data = [d for d in data if d.stem in missing_ids]
        assert len(data) == len(missing_ids)

    # Create output directories
    msa_dir = out_dir / "msa"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    if existing_records is not None:
        click.echo(f"Found {len(existing_records)} records. Adding them to records")

    # Parse input data
    records: list[Record] = existing_records if existing_records is not None else []
    for path in tqdm(data):
        try:
            # Parse data
            if path.suffix in (".fa", ".fas", ".fasta"):
                target = parse_fasta(path, ccd)
            elif path.suffix in (".yml", ".yaml"):
                target = parse_yaml(path, ccd)
            elif path.is_dir():
                msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {path.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

            # Get target id
            target_id = target.record.id

            # Get all MSA ids and decide whether to generate MSA
            to_generate = {}
            prot_id = const.chain_type_ids["PROTEIN"]
            for chain in target.record.chains:
                # Add to generate list, assigning entity id
                if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                    entity_id = chain.entity_id
                    msa_id = f"{target_id}_{entity_id}"
                    to_generate[msa_id] = target.sequences[entity_id]
                    chain.msa_id = msa_dir / f"{msa_id}.csv"

                # We do not support msa generation for non-protein chains
                elif chain.msa_id == 0:
                    chain.msa_id = -1

            # Generate MSA
            if to_generate and not use_msa_server:
                msg = "Missing MSA's in input and --use_msa_server flag not set."
                raise RuntimeError(msg)

            if to_generate:
                msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
                click.echo(msg)
                compute_msa(
                    data=to_generate,
                    target_id=target_id,
                    msa_dir=msa_dir,
                    msa_server_url=msa_server_url,
                    msa_pairing_strategy=msa_pairing_strategy,
                )

            
            # Parse MSA data
            msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
            #print('msas: ', msas)
            msa_id_map = {}
            for msa_idx, msa_id in enumerate(msas):
                # Check that raw MSA exists
                msa_path = Path(msa_id)
                if not msa_path.exists():
                    msg = f"MSA file {msa_path} not found."
                    raise FileNotFoundError(msg)

                # Dump processed MSA
                processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
                msa_id_map[msa_id] = f"{target_id}_{msa_idx}"

                print('processed: ',processed)
                print('msa_path: ',msa_path)
                if not processed.exists():
                    # Parse A3M
                    if msa_path.suffix == ".a3m":
                        msa: MSA = parse_a3m(
                            msa_path,
                            taxonomy=None,
                            max_seqs=max_msa_seqs,
                        )
                    elif msa_path.suffix == ".csv":
                        msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                    else:
                        msg = f"MSA file {msa_path} not supported, only a3m or csv."
                        raise RuntimeError(msg)

                    msa.dump(processed)

            # Modify records to point to processed MSA
            for c in target.record.chains:
                if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                    c.msa_id = msa_id_map[c.msa_id]

            # Keep record
            records.append(target.record)

            # Dump structure
            struct_path = structure_dir / f"{target.record.id}.npz"
            target.structure.dump(struct_path)

        except Exception as e:
            if len(data) > 1:
                print(f"Failed to process {path}. Skipping. Error: {e}.")
            else:
                raise e

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")

def predict(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float = 1.638,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
) -> None:
    """Run predictions with Boltz-1."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set seed if desired
    if seed is not None:
       seed_everything(int(seed))

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache)

    # Validate inputs
    data = check_inputs(data, out_dir, override)
    if not data:
        click.echo("No predictions to run, exiting.")
        return

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()
        if len(data) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions."
            )
            raise ValueError(msg)

    msg = f"Running predictions for {len(data)} structure"
    msg += "s" if len(data) > 1 else ""
    click.echo(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )

    # Load model
    if checkpoint is None:
        checkpoint = cache / "boltz2_conf.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale
    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
    )
    model_module.eval()

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )

    # Compute predictions
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )



if __name__ == "__main__":
    
    predict(data="./inputs_prediction",
            out_dir="./outputs_prediction",
            cache="/kaggle/input/boltz2",
            diffusion_samples=5,
            seed=42,
            override=True)


import os
import pandas as pd
sub_file = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')

sub_file.head()

names = sub_file['target_id'].tolist()
sequences = sub_file['sequence'].tolist()

# Inference
idx = 0 
for tmp_id, tmp_sequence in zip(names, sequences):
    with open(f'/kaggle/working/inputs_prediction/{tmp_id}.yaml', 'w') as f:
        f.write("constraints: []\n")
        f.write("sequences:\n")
        f.write("- rna:\n")
        f.write("    id:\n")
        f.write("    - A1\n")
        f.write(f"    sequence: {tmp_sequence}")


%ls inputs_prediction


import torch
torch.cuda.empty_cache()
import gc
gc.collect()


!python inference.py


from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import json

def get_coords(tmp_id, idx):
    cif_file = f"outputs_prediction/boltz_results_inputs_prediction/predictions/{tmp_id}/{tmp_id}_model_{idx}.cif"

    mmcif_dict = MMCIF2Dict(cif_file)
    
    entity_poly_seq = mmcif_dict.get("_entity_poly_seq.mon_id", [])
    sequence = "".join(entity_poly_seq)
    #print("RNA sequence:", sequence)
    
    x_coords = mmcif_dict["_atom_site.Cartn_x"]
    y_coords = mmcif_dict["_atom_site.Cartn_y"]
    z_coords = mmcif_dict["_atom_site.Cartn_z"]
    atom_names = mmcif_dict["_atom_site.label_atom_id"]
    
    c1_coords = []
    for i, atom in enumerate(atom_names):
        if atom == "C1'":
            c1_coords.append((float(x_coords[i]), float(y_coords[i]), float(z_coords[i])))

    conf_file = f"outputs_prediction/boltz_results_inputs_prediction/predictions/{tmp_id}/confidence_{tmp_id}_model_{idx}.json"
    # parse json and read confidence_score from it's dict
    with open(conf_file, 'r') as f:
        conf_data = json.load(f)

    confidence_score = conf_data.get('confidence_score', -1)
        
    return c1_coords, confidence_score

all_preds = os.listdir('outputs_prediction/boltz_results_inputs_prediction/predictions')
submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')


for tmp_id in all_preds:
    print('#' * 20, f'Inferences for {tmp_id}')
    coords_with_conf = []

    for model_idx in range(5):
        coords, conf_score = get_coords(tmp_id, model_idx)
        coords_with_conf.append((coords, conf_score))

    # 置信度从高到低排序
    coords_with_conf.sort(key=lambda x: x[1], reverse=True)

    # 写入前 5 个坐标点（按置信度顺序）
    for rank_idx in range(5):
        coords = coords_with_conf[rank_idx][0]
        print(f'{rank_idx} score: ', coords_with_conf[rank_idx][1])
        submission.loc[
            submission['ID'].apply(lambda x: tmp_id in x),
            [f'x_{rank_idx+1}', f'y_{rank_idx+1}', f'z_{rank_idx+1}']
        ] = coords

    print()


%rm -rf boltz
%rm -rf inputs_prediction
%rm -rf outputs_prediction
%rm -rf inference.py


submission.to_csv("submission_boltz.csv", index=False)


import time

import pandas as pd
import numpy as np

import random
from Bio import pairwise2
from Bio.Seq import Seq

from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
from scipy.spatial import distance_matrix
import warnings
warnings.filterwarnings('ignore')

print("\nLoading data files...")
train_seqs = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
valid_seqs = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
test_seqs = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
valid_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')

print(f"Loaded {len(train_seqs)} training sequences, {len(valid_seqs)} validation sequences, and {len(test_seqs)} test sequences")


train_seqs_v2 = pd.read_csv('/kaggle/input/rna-cif-to-csv/rna_sequences.csv')
train_labels_v2 = pd.read_csv('/kaggle/input/rna-cif-to-csv/rna_coordinates.csv')


import pandas as pd
import numpy as np

# Function to extend the original dataset with new records from v2
def extend_dataset(original_df, v2_df, key_columns, dataset_name):
    print(f"Extending {dataset_name}...")
    print(f"  Original size: {len(original_df)} rows")
    print(f"  v2 size: {len(v2_df)} rows")
    
    # Create a composite key for identification if multiple key columns
    if isinstance(key_columns, list) and len(key_columns) > 1:
        original_df['temp_key'] = original_df[key_columns].astype(str).agg('_'.join, axis=1)
        v2_df['temp_key'] = v2_df[key_columns].astype(str).agg('_'.join, axis=1)
        key_for_identification = 'temp_key'
    else:
        key_for_identification = key_columns[0] if isinstance(key_columns, list) else key_columns
    
    # Identify unique records in each dataset
    original_keys = set(original_df[key_for_identification])
    v2_keys = set(v2_df[key_for_identification])
    
    # Calculate stats
    keys_only_in_original = original_keys - v2_keys
    keys_only_in_v2 = v2_keys - original_keys 
    common_keys = original_keys.intersection(v2_keys)
    
    print(f"  Keys only in original: {len(keys_only_in_original)}")
    print(f"  Keys only in v2: {len(keys_only_in_v2)}")
    print(f"  Common keys: {len(common_keys)}")
    
    # Create a mask to filter v2 records that don't exist in original
    new_records_mask = ~v2_df[key_for_identification].isin(original_keys)
    new_records = v2_df[new_records_mask].copy()
    
    # Drop temporary key if it was created
    if key_for_identification == 'temp_key':
        new_records.drop('temp_key', axis=1, inplace=True)
        original_df.drop('temp_key', axis=1, inplace=True)
    
    # Combine original with new records from v2
    extended_df = pd.concat([original_df, new_records], ignore_index=True)
    
    # Report final sizes
    print(f"  New records added: {len(new_records)}")
    print(f"  Extended dataset size: {len(extended_df)} rows")
    print(f"  Verification - All original keys in extended dataset: {set(original_df[key_columns[0] if isinstance(key_columns, list) else key_columns]).issubset(set(extended_df[key_columns[0] if isinstance(key_columns, list) else key_columns]))}")
    
    # Check for missing values in key columns
    for col in extended_df.columns:
        original_missing = original_df[col].isnull().sum()
        extended_missing = extended_df[col].isnull().sum()
        if original_missing > 0 or extended_missing > 0:
            print(f"  Column '{col}': Missing values - Original: {original_missing}, Extended: {extended_missing}")
    
    # Clean up
    if key_for_identification == 'temp_key' and 'temp_key' in v2_df.columns:
        v2_df.drop('temp_key', axis=1, inplace=True)
        
    return extended_df

# 1. Extend train_seqs with train_seqs_v2
print("\n" + "="*50)
print("EXTENDING SEQUENCE DATASETS")
print("="*50)
train_seqs_extended = extend_dataset(
    train_seqs, 
    train_seqs_v2,
    ['target_id'],  # Using target_id as the unique identifier
    "train_seqs"
)

# 2. Extend train_labels with train_labels_v2
print("\n" + "="*50)
print("EXTENDING LABELS DATASETS")
print("="*50)
# For labels, we need a composite key of ID and resid
train_labels_extended = extend_dataset(
    train_labels,
    train_labels_v2,
    ['ID', 'resid'],  # Using composite key
    "train_labels"
)

# Verify relationships between extended datasets
print("\n" + "="*50)
print("VERIFYING RELATIONSHIPS")
print("="*50)

# Check if all sequence IDs have corresponding labels
seq_ids = set(train_seqs_extended['target_id'].unique())
label_ids = set(train_labels_extended['ID'].unique())

seq_ids_with_labels = seq_ids.intersection(label_ids)
seq_ids_without_labels = seq_ids - label_ids

print(f"Total unique sequence IDs: {len(seq_ids)}")
print(f"Sequence IDs with corresponding labels: {len(seq_ids_with_labels)} ({len(seq_ids_with_labels)/len(seq_ids)*100:.2f}%)")
print(f"Sequence IDs without corresponding labels: {len(seq_ids_without_labels)} ({len(seq_ids_without_labels)/len(seq_ids)*100:.2f}%)")

if len(seq_ids_without_labels) > 0:
    print("Sample of sequence IDs without labels (up to 5):")
    print(list(seq_ids_without_labels)[:5])

# Print summary of extended datasets
print("\n" + "="*50)
print("SUMMARY OF EXTENDED DATASETS")
print("="*50)
print(f"Original train_seqs: {len(train_seqs)} rows")
print(f"Original train_labels: {len(train_labels)} rows")
print(f"Extended train_seqs: {len(train_seqs_extended)} rows (+{len(train_seqs_extended)-len(train_seqs)})")
print(f"Extended train_labels: {len(train_labels_extended)} rows (+{len(train_labels_extended)-len(train_labels)})")

# Save the extended datasets (uncomment to save)
# train_seqs_extended.to_csv('train_seqs_combined.csv', index=False)
# train_labels_extended.to_csv('train_labels_combined.csv', index=False)

print("\n" + "="*50)
print("DONE! Extended datasets created.")
print("To save the datasets, uncomment the last two lines.")
print("="*50)


def process_labels(labels_df):
    coords_dict = {}
    
    # Group by target ID and wrap with tqdm for progress tracking
    id_groups = labels_df.groupby(lambda x: labels_df['ID'][x].rsplit('_', 1)[0])
    for id_prefix, group in tqdm(id_groups, desc="Processing structures"):
        # Extract just the coordinates columns for the first structure (x_1, y_1, z_1)
        coords = []
        for _, row in group.sort_values('resid').iterrows():
            coords.append([row['x_1'], row['y_1'], row['z_1']])
        
        coords_dict[id_prefix] = np.array(coords)
    
    return coords_dict

train_coords_dict = process_labels(train_labels_extended)


from Bio.Seq import Seq
from Bio import pairwise2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, top_n=5):
    """
    Find similar RNA sequences using enhanced scoring and clustering for diversity.
    
    Improvements:
    - Multi-tier length filtering
    - Enhanced alignment scoring with multiple algorithms
    - RNA-specific structural features
    - Adaptive clustering
    """
    similar_seqs = []
    query_seq_obj = Seq(query_seq)
    query_features = _extract_enhanced_rna_features(query_seq)
    
    # Step 1: Enhanced candidate selection with multi-tier filtering
    for _, row in train_seqs_df.iterrows():
        target_id = row['target_id']
        train_seq = row['sequence']
        
        # Skip if coordinates not available
        if target_id not in train_coords_dict:
            continue
        
        # Multi-tier length filtering (more permissive for very short/long sequences)
        len_ratio = abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq))
        if len(query_seq) < 50 or len(train_seq) < 50:  # Short sequences - more permissive
            if len_ratio > 0.6:
                continue
        elif len(query_seq) > 1000 or len(train_seq) > 1000:  # Long sequences - stricter
            if len_ratio > 0.2:
                continue
        else:  # Medium sequences - original threshold
            if len_ratio > 0.4:
                continue
        
        # Calculate composite similarity score
        composite_score = _calculate_composite_similarity(query_seq, train_seq, query_features)
        
        if composite_score > 0:  # Only keep sequences with positive similarity
            similar_seqs.append((target_id, train_seq, composite_score, train_coords_dict[target_id]))
    
    # Sort by composite score and take top candidates
    similar_seqs.sort(key=lambda x: x[2], reverse=True)
    
    # Adaptive candidate selection based on score distribution
    candidate_count = min(50, len(similar_seqs))  # Increased initial pool
    if len(similar_seqs) > 10:
        # Filter out sequences with very low scores (bottom 20%)
        score_threshold = np.percentile([x[2] for x in similar_seqs], 80)
        filtered_candidates = [x for x in similar_seqs if x[2] >= score_threshold]
        candidate_count = min(candidate_count, len(filtered_candidates))
        top_candidates = filtered_candidates[:candidate_count]
    else:
        top_candidates = similar_seqs[:candidate_count]
    
    # If we have fewer sequences than requested clusters, return all
    if len(top_candidates) <= top_n:
        return top_candidates[:top_n]
    
    # Step 2: Enhanced feature matrix for better clustering
    feature_matrix = []
    for _, seq, _, _ in top_candidates:
        features = _extract_enhanced_rna_features(seq)
        feature_matrix.append(features)
    
    feature_matrix = np.array(feature_matrix)
    
    # Step 3: Adaptive clustering
    n_clusters = min(top_n, len(top_candidates))
    
    # Use different clustering approach based on dataset size
    if len(top_candidates) >= 15:
        # K-means for larger datasets
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
    else:
        # Simple diversity-based selection for smaller datasets
        cluster_labels = _diversity_based_clustering(feature_matrix, n_clusters)
    
    # Step 4: Select best representative from each cluster
    final_results = []
    for cluster_id in range(n_clusters):
        cluster_sequences = [top_candidates[i] for i in range(len(top_candidates)) 
                           if cluster_labels[i] == cluster_id]
        
        if cluster_sequences:
            # Sort by composite score and take the best one
            cluster_sequences.sort(key=lambda x: x[2], reverse=True)
            final_results.append(cluster_sequences[0])
    
    # Sort final results by similarity score
    final_results.sort(key=lambda x: x[2], reverse=True)
    
    return final_results[:top_n]

def _calculate_composite_similarity(query_seq, train_seq, query_features):
    """
    Calculate composite similarity using multiple alignment methods and features.
    """
    query_seq_obj = Seq(query_seq)
    
    # 1. Global alignment (original method)
    global_alignments = pairwise2.align.globalms(query_seq_obj, train_seq, 2.9, -1, -10, -0.5, one_alignment_only=True)
    global_score = 0
    if global_alignments:
        alignment = global_alignments[0]
        global_score = alignment.score / (2 * min(len(query_seq), len(train_seq)))
    
    # 2. Local alignment for finding similar regions
    local_alignments = pairwise2.align.localms(query_seq_obj, train_seq, 2.9, -1, -10, -0.5, one_alignment_only=True)
    local_score = 0
    if local_alignments:
        alignment = local_alignments[0]
        local_score = alignment.score / (2 * min(len(query_seq), len(train_seq)))
    
    # 3. Feature-based similarity
    train_features = _extract_enhanced_rna_features(train_seq)
    feature_similarity = cosine_similarity([query_features], [train_features])[0][0]
    
    # 4. K-mer similarity for sequence motifs
    kmer_similarity = _calculate_kmer_similarity(query_seq, train_seq, k=3)
    
    # Weighted composite score
    composite_score = (
        0.4 * global_score + 
        0.3 * local_score + 
        0.2 * feature_similarity + 
        0.1 * kmer_similarity
    )
    
    return composite_score

def _calculate_kmer_similarity(seq1, seq2, k=3):
    """Calculate k-mer based similarity between sequences."""
    def get_kmers(seq, k):
        return set(seq[i:i+k] for i in range(len(seq) - k + 1))
    
    kmers1 = get_kmers(seq1.upper(), k)
    kmers2 = get_kmers(seq2.upper(), k)
    
    if not kmers1 or not kmers2:
        return 0
    
    intersection = len(kmers1.intersection(kmers2))
    union = len(kmers1.union(kmers2))
    
    return intersection / union if union > 0 else 0

def _diversity_based_clustering(feature_matrix, n_clusters):
    """Simple diversity-based clustering for small datasets."""
    n_samples = len(feature_matrix)
    cluster_labels = np.zeros(n_samples, dtype=int)
    
    if n_samples <= n_clusters:
        return np.arange(n_samples)
    
    # Select diverse representatives
    selected_indices = [0]  # Start with first sequence
    
    for cluster_id in range(1, n_clusters):
        max_min_distance = -1
        best_idx = -1
        
        for i in range(n_samples):
            if i in selected_indices:
                continue
            
            # Find minimum distance to already selected sequences
            min_distance = min(
                np.linalg.norm(feature_matrix[i] - feature_matrix[j]) 
                for j in selected_indices
            )
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_idx = i
        
        if best_idx != -1:
            selected_indices.append(best_idx)
    
    # Assign remaining sequences to closest cluster centers
    for i in range(n_samples):
        if i not in selected_indices:
            distances = [
                np.linalg.norm(feature_matrix[i] - feature_matrix[j]) 
                for j in selected_indices
            ]
            cluster_labels[i] = np.argmin(distances)
        else:
            cluster_labels[i] = selected_indices.index(i)
    
    return cluster_labels

def _extract_enhanced_rna_features(sequence):
    """
    Extract comprehensive RNA-specific features for better clustering and similarity.
    """
    seq = sequence.upper()
    features = []
    
    # 1. Basic nucleotide frequencies
    nucleotides = ['A', 'U', 'G', 'C']
    for nuc in nucleotides:
        freq = seq.count(nuc) / len(seq) if len(seq) > 0 else 0
        features.append(freq)
    
    # 2. Dinucleotide frequencies (reduced set - most important for RNA)
    important_dinucs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG', 'AA', 'UU', 'GG', 'CC']
    for dinuc in important_dinucs:
        count = 0
        for i in range(len(seq) - 1):
            if seq[i:i+2] == dinuc:
                count += 1
        freq = count / (len(seq) - 1) if len(seq) > 1 else 0
        features.append(freq)
    
    # 3. RNA secondary structure indicators
    gc_content = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
    au_content = (seq.count('A') + seq.count('U')) / len(seq) if len(seq) > 0 else 0
    purine_content = (seq.count('A') + seq.count('G')) / len(seq) if len(seq) > 0 else 0
    pyrimidine_content = (seq.count('U') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
    
    features.extend([gc_content, au_content, purine_content, pyrimidine_content])
    
    # 4. Sequence complexity measures
    length_normalized = min(len(seq) / 1000.0, 1.0)  # Capped normalization
    
    # Simple entropy calculation
    entropy = 0
    for nuc in nucleotides:
        freq = seq.count(nuc) / len(seq) if len(seq) > 0 else 0
        if freq > 0:
            entropy -= freq * np.log2(freq)
    entropy_normalized = entropy / 2.0  # Max entropy for 4 nucleotides is 2
    
    features.extend([length_normalized, entropy_normalized])
    
    # 5. Repetitive pattern detection
    repeat_content = _calculate_repeat_content(seq)
    features.append(repeat_content)
    
    return features

def _calculate_repeat_content(sequence):
    """Calculate the proportion of repetitive content in the sequence."""
    if len(sequence) < 6:
        return 0
    
    repeat_count = 0
    window_size = 3
    
    for i in range(len(sequence) - window_size + 1):
        motif = sequence[i:i + window_size]
        # Look for the same motif in the rest of the sequence
        for j in range(i + window_size, len(sequence) - window_size + 1):
            if sequence[j:j + window_size] == motif:
                repeat_count += 1
                break
    
    return repeat_count / (len(sequence) - window_size + 1) if len(sequence) > window_size else 0


def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    # Make a copy of coordinates to refine
    refined_coords = coordinates.copy()
    n_residues = len(sequence)
    
    # Calculate constraint strength (inverse of confidence)
    # High confidence templates receive gentler constraints
    constraint_strength = 0.8 * (1.0 - min(confidence, 0.8))
    
    # 1. Sequential distance constraints (consecutive nucleotides)
    # More flexible distance range (statistical distribution from PDB)
    seq_min_dist = 5.5  # Minimum sequential distance
    seq_max_dist = 6.5  # Maximum sequential distance
    
    for i in range(n_residues - 1):
        current_pos = refined_coords[i]
        next_pos = refined_coords[i+1]
        
        # Calculate current distance
        current_dist = np.linalg.norm(next_pos - current_pos)
        
        # Only adjust if significantly outside expected range
        if current_dist < seq_min_dist or current_dist > seq_max_dist:
            # Calculate target distance (midpoint of range)
            target_dist = (seq_min_dist + seq_max_dist) / 2
            
            # Get direction vector
            direction = next_pos - current_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Apply partial adjustment based on constraint strength
            adjustment = (target_dist - current_dist) * constraint_strength
            
            # Only adjust the next position to preserve the overall fold
            refined_coords[i+1] = current_pos + direction * (current_dist + adjustment)
    
    # 2. Steric clash prevention (more conservative)
    min_allowed_distance = 3.8  # Minimum distance between non-consecutive C1' atoms
    
    # Calculate all pairwise distances
    dist_matrix = distance_matrix(refined_coords, refined_coords)
    
    # Find severe clashes (atoms too close)
    severe_clashes = np.where((dist_matrix < min_allowed_distance) & (dist_matrix > 0))
    
    # Fix severe clashes
    for idx in range(len(severe_clashes[0])):
        i, j = severe_clashes[0][idx], severe_clashes[1][idx]
        
        # Skip consecutive nucleotides and previously processed pairs
        if abs(i - j) <= 1 or i >= j:
            continue
            
        # Get current positions and distance
        pos_i = refined_coords[i]
        pos_j = refined_coords[j]
        current_dist = dist_matrix[i, j]
        
        # Calculate necessary adjustment but scale by constraint strength
        direction = pos_j - pos_i
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Calculate partial adjustment
        adjustment = (min_allowed_distance - current_dist) * constraint_strength
        
        # Move points apart
        refined_coords[i] = pos_i - direction * (adjustment / 2)
        refined_coords[j] = pos_j + direction * (adjustment / 2)
    
    # 3. Very light base-pair constraining (if confidence is low)
    if constraint_strength > 0.3:  # Only apply if template confidence is low
        # Simple Watson-Crick base pairs
        pairs = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        
        # Scan for potential base pairs
        for i in range(n_residues):
            base_i = sequence[i]
            complement = pairs.get(base_i)
            
            if not complement:
                continue
                
            # Look for complementary bases within a reasonable range
            for j in range(i + 3, min(i + 20, n_residues)):
                if sequence[j] == complement:
                    # Calculate current distance
                    current_dist = np.linalg.norm(refined_coords[i] - refined_coords[j])
                    
                    # Only consider if distance suggests potential pairing
                    if 8.0 < current_dist < 14.0:
                        # Target 10.5Å as generic base-pair C1'-C1' distance
                        target_dist = 10.5
                        
                        # Calculate very gentle adjustment (scaled by constraint_strength)
                        adjustment = (target_dist - current_dist) * (constraint_strength * 0.3)
                        
                        # Get direction vector
                        direction = refined_coords[j] - refined_coords[i]
                        direction = direction / (np.linalg.norm(direction) + 1e-10)
                        
                        # Apply very gentle adjustment to both positions
                        refined_coords[i] = refined_coords[i] - direction * (adjustment / 2)
                        refined_coords[j] = refined_coords[j] + direction * (adjustment / 2)
                        
                        # Only consider one potential pair per base (closest match)
                        break
    
    return refined_coords


def adapt_template_to_query(query_seq, template_seq, template_coords, alignment=None):
    if alignment is None:
        from Bio.Seq import Seq
        from Bio import pairwise2
        
        query_seq_obj = Seq(query_seq)
        template_seq_obj = Seq(template_seq)
        alignments = pairwise2.align.globalms(query_seq_obj, template_seq_obj, 2.9, -1, -10, -0.5, one_alignment_only=True)
        
        if not alignments:
            return generate_improved_rna_structure(query_seq)
            
        alignment = alignments[0]
    
    aligned_query = alignment.seqA
    aligned_template = alignment.seqB
    
    query_coords = np.zeros((len(query_seq), 3))
    query_coords.fill(np.nan)
    
    # Map template coordinates to query
    query_idx = 0
    template_idx = 0
    
    for i in range(len(aligned_query)):
        query_char = aligned_query[i]
        template_char = aligned_template[i]
        
        if query_char != '-' and template_char != '-':
            if template_idx < len(template_coords):
                query_coords[query_idx] = template_coords[template_idx]
            template_idx += 1
            query_idx += 1
        elif query_char != '-' and template_char == '-':
            query_idx += 1
        elif query_char == '-' and template_char != '-':
            template_idx += 1
    
    # IMPROVED GAP FILLING - maintains RNA backbone geometry
    backbone_distance = 5.9  # Typical C1'-C1' distance
    
    # Fill gaps by maintaining realistic backbone connectivity
    for i in range(len(query_coords)):
        if np.isnan(query_coords[i, 0]):
            # Find nearest valid neighbors
            prev_valid = next_valid = None
            
            for j in range(i-1, -1, -1):
                if not np.isnan(query_coords[j, 0]):
                    prev_valid = j
                    break
                    
            for j in range(i+1, len(query_coords)):
                if not np.isnan(query_coords[j, 0]):
                    next_valid = j
                    break
            
            if prev_valid is not None and next_valid is not None:
                # Interpolate along realistic RNA backbone path
                gap_size = next_valid - prev_valid
                total_distance = np.linalg.norm(query_coords[next_valid] - query_coords[prev_valid])
                expected_distance = gap_size * backbone_distance
                
                # If gap is compressed, extend it realistically
                if total_distance < expected_distance * 0.7:
                    direction = query_coords[next_valid] - query_coords[prev_valid]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                    
                    # Place intermediate points along extended path
                    for k, idx in enumerate(range(prev_valid + 1, next_valid)):
                        progress = (k + 1) / gap_size
                        base_pos = query_coords[prev_valid] + direction * expected_distance * progress
                        
                        # Add slight curvature for realism
                        perpendicular = np.cross(direction, [0, 0, 1])
                        if np.linalg.norm(perpendicular) < 1e-6:
                            perpendicular = np.cross(direction, [1, 0, 0])
                        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-10)
                        
                        curve_amplitude = 2.0 * np.sin(progress * np.pi)
                        query_coords[idx] = base_pos + perpendicular * curve_amplitude
                else:
                    # Linear interpolation for normal gaps
                    for k, idx in enumerate(range(prev_valid + 1, next_valid)):
                        weight = (k + 1) / gap_size
                        query_coords[idx] = (1 - weight) * query_coords[prev_valid] + weight * query_coords[next_valid]
            
            elif prev_valid is not None:
                # Extend from previous position
                if prev_valid > 0 and not np.isnan(query_coords[prev_valid-1, 0]):
                    direction = query_coords[prev_valid] - query_coords[prev_valid-1]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                else:
                    direction = np.array([1.0, 0.0, 0.0])
                
                steps_needed = i - prev_valid
                for step in range(1, steps_needed + 1):
                    pos_idx = prev_valid + step
                    if pos_idx < len(query_coords):
                        query_coords[pos_idx] = query_coords[prev_valid] + direction * backbone_distance * step
            
            elif next_valid is not None:
                # Work backwards from next position
                direction = np.array([-1.0, 0.0, 0.0])  # Default backward direction
                steps_needed = next_valid - i
                for step in range(steps_needed, 0, -1):
                    pos_idx = next_valid - step
                    if pos_idx >= 0:
                        query_coords[pos_idx] = query_coords[next_valid] - direction * backbone_distance * step
    
    # Final cleanup
    query_coords = np.nan_to_num(query_coords)
    return query_coords


def generate_improved_rna_structure(sequence):
    """
    Generate a more realistic RNA structure fallback based on sequence patterns
    and basic RNA structure principles.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        Array of 3D coordinates
    """
    n_residues = len(sequence)
    coordinates = np.zeros((n_residues, 3))
    
    # Analyze sequence to predict structural elements
    # Look for complementary regions that could form base pairs
    potential_stems = identify_potential_stems(sequence)
    
    # Default parameters
    radius_helix = 10.0
    radius_loop = 15.0
    rise_per_residue_helix = 2.5
    rise_per_residue_loop = 1.5
    angle_per_residue_helix = 0.6
    angle_per_residue_loop = 0.3
    
    # Assign structural classifications
    structure_types = assign_structure_types(sequence, potential_stems)
    
    # Generate coordinates based on predicted structure
    current_pos = np.array([0.0, 0.0, 0.0])
    current_direction = np.array([0.0, 0.0, 1.0])
    current_angle = 0.0
    
    for i in range(n_residues):
        if structure_types[i] == 'stem':
            # Part of a helical stem
            current_angle += angle_per_residue_helix
            coordinates[i] = [
                radius_helix * np.cos(current_angle), 
                radius_helix * np.sin(current_angle), 
                current_pos[2] + rise_per_residue_helix
            ]
            current_pos = coordinates[i]
        elif structure_types[i] == 'loop':
            # Part of a loop
            current_angle += angle_per_residue_loop
            z_shift = rise_per_residue_loop * np.sin(current_angle * 0.5)
            coordinates[i] = [
                radius_loop * np.cos(current_angle), 
                radius_loop * np.sin(current_angle), 
                current_pos[2] + z_shift
            ]
            current_pos = coordinates[i]
        else:
            # Single-stranded region
            # Add some randomness to make it look more realistic
            jitter = np.random.normal(0, 1, 3) * 2.0
            coordinates[i] = current_pos + jitter
            current_pos = coordinates[i]
            
    return coordinates

def identify_potential_stems(sequence):
    """
    Identify potential stem regions by looking for self-complementary segments.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        List of tuples (start1, end1, start2, end2) representing potentially paired regions
    """
    complementary_bases = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    min_stem_length = 3
    potential_stems = []
    
    # Simple stem identification
    for i in range(len(sequence) - min_stem_length):
        for j in range(i + min_stem_length + 3, len(sequence) - min_stem_length + 1):
            # Check if regions could form a stem
            potential_stem_len = min(min_stem_length, len(sequence) - j)
            is_stem = True
            
            for k in range(potential_stem_len):
                if sequence[i+k] not in complementary_bases or \
                   complementary_bases[sequence[i+k]] != sequence[j+potential_stem_len-k-1]:
                    is_stem = False
                    break
            
            if is_stem:
                potential_stems.append((i, i+potential_stem_len-1, j, j+potential_stem_len-1))
    
    return potential_stems

def assign_structure_types(sequence, potential_stems):
    """
    Assign each nucleotide to a structural element type.
    
    Args:
        sequence: RNA sequence string
        potential_stems: List of tuples representing stem regions
        
    Returns:
        List of structure types ('stem', 'loop', 'single')
    """
    structure_types = ['single'] * len(sequence)
    
    # Mark stem regions
    for stem in potential_stems:
        start1, end1, start2, end2 = stem
        for i in range(end1 - start1 + 1):
            structure_types[start1 + i] = 'stem'
            structure_types[end2 - i] = 'stem'
    
    # Mark loop regions (regions between paired regions)
    for i in range(len(potential_stems) - 1):
        _, end1, start2, _ = potential_stems[i]
        next_start1, _, _, _ = potential_stems[i+1]
        
        if next_start1 > end1 + 1 and start2 > next_start1:
            for j in range(end1 + 1, next_start1):
                structure_types[j] = 'loop'
    
    return structure_types


# Function to create a more realistic RNA structure when no good templates are found
def generate_rna_structure(sequence, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    n_residues = len(sequence)
    coordinates = np.zeros((n_residues, 3))
    
    # Initialize the first few residues in a helix
    for i in range(min(3, n_residues)):
        angle = i * 0.6
        coordinates[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]
    
    # Add more complex folding patterns
    current_direction = np.array([0.0, 0.0, 1.0])  # Start moving along z-axis
    
    # Define base-pairing tendencies (G-C and A-U pairs)
    for i in range(3, n_residues):
        # Check for potential base-pairing in the sequence
        has_pair = False
        pair_idx = -1
        
        # Simple detection of complementary bases (G-C, A-U)
        complementary = {'G': 'C', 'C': 'G', 'A': 'U', 'U': 'A'}
        current_base = sequence[i]
        
        # Look for potential base-pairing within a window before the current position
        window_size = min(i, 15)  # Look back up to 15 bases
        for j in range(i-window_size, i):
            if j >= 0 and sequence[j] == complementary.get(current_base, 'X'):
                # Found a potential pair
                has_pair = True
                pair_idx = j
                break
        
        if has_pair and i - pair_idx <= 10 and random.random() < 0.7:
            # Try to create a base-pair by positioning this nucleotide near its pair
            pair_pos = coordinates[pair_idx]
            
            # Create a position that's roughly opposite to the pair
            random_offset = np.random.normal(0, 1, 3) * 2.0
            base_pair_distance = 10.0 + random.uniform(-1.0, 1.0)
            
            # Calculate a vector from base-pair toward center of structure
            center = np.mean(coordinates[:i], axis=0)
            direction = center - pair_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Position new nucleotide in the general direction of the "center"
            coordinates[i] = pair_pos + direction * base_pair_distance + random_offset
            
            # Update direction for next nucleotide
            current_direction = np.random.normal(0, 0.3, 3)
            current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)
            
        else:
            # No base-pairing detected, continue with the current fold direction
            # Randomly rotate current direction to simulate RNA flexibility
            if random.random() < 0.3:
                # More significant direction change
                angle = random.uniform(0.2, 0.6)
                axis = np.random.normal(0, 1, 3)
                axis = axis / (np.linalg.norm(axis) + 1e-10)
                rotation = R.from_rotvec(angle * axis)
                current_direction = rotation.apply(current_direction)
            else:
                # Small random changes in direction
                current_direction += np.random.normal(0, 0.15, 3)
                current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)
            
            # Distance between consecutive nucleotides (3.5-4.5Å is typical)
            step_size = random.uniform(3.5, 4.5)
            
            # Update position
            coordinates[i] = coordinates[i-1] + step_size * current_direction
    
    return coordinates


def predict_rna_structures(sequence, target_id, train_seqs_df, train_coords_dict, n_predictions=5):
    predictions = []
    
    # Find similar sequences in the training data
    similar_seqs = find_similar_sequences(sequence, train_seqs_df, train_coords_dict, top_n=n_predictions)
    
    # If we found any similar sequences, use them as templates
    if similar_seqs:
        for i, (template_id, template_seq, similarity_score, template_coords) in enumerate(similar_seqs):
            # Adapt template coordinates to the query sequence
            adapted_coords = adapt_template_to_query(sequence, template_seq, template_coords)
            
            if adapted_coords is not None:
                # Apply adaptive constraints based on template similarity
                # For high similarity templates, apply very gentle constraints
                refined_coords = adaptive_rna_constraints(adapted_coords, sequence, confidence=similarity_score)
                
                # Add some randomness (less for better templates)
                random_scale = max(0.05, 0.8 - similarity_score)  # Reduced randomness
                randomized_coords = refined_coords.copy()
                randomized_coords += np.random.normal(0, random_scale, randomized_coords.shape)
                
                predictions.append(randomized_coords)
                
                if len(predictions) >= n_predictions:
                    break
    
    # If we don't have enough predictions from templates, generate de novo structures
    while len(predictions) < n_predictions:
        seed_value = hash(target_id) % 10000 + len(predictions) * 1000
        de_novo_coords = generate_rna_structure(sequence, seed=seed_value)
        
        # Apply stronger constraints to de novo structures (lower confidence)
        refined_de_novo = adaptive_rna_constraints(de_novo_coords, sequence, confidence=0.2)
        
        predictions.append(refined_de_novo)
    
    return predictions[:n_predictions]


# List to store all prediction records
all_predictions = []

# Set up time tracking
start_time = time.time()
total_targets = len(test_seqs)

# For each sequence in the test set
for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    
    # Progress tracking
    if idx % 5 == 0:
        elapsed = time.time() - start_time
        targets_processed = idx + 1
        if targets_processed > 0:
            avg_time_per_target = elapsed / targets_processed
            est_time_remaining = avg_time_per_target * (total_targets - targets_processed)
            print(f"Processing target {targets_processed}/{total_targets}: {target_id} ({len(sequence)} nt), "
                  f"elapsed: {elapsed:.1f}s, est. remaining: {est_time_remaining:.1f}s")
    
    # Generate 5 different structure predictions
    predictions = predict_rna_structures(sequence, target_id, train_seqs_extended, train_coords_dict, n_predictions=5)
    
    # For each residue in the sequence
    for j in range(len(sequence)):
        pred_row = {
            'ID': f"{target_id}_{j+1}",
            'resname': sequence[j],
            'resid': j + 1
        }
        
        # Add coordinates from all 5 predictions
        for i in range(5):
            pred_row[f'x_{i+1}'] = predictions[i][j][0]
            pred_row[f'y_{i+1}'] = predictions[i][j][1]
            pred_row[f'z_{i+1}'] = predictions[i][j][2]
        
        all_predictions.append(pred_row)

# Create DataFrame with predictions
submission_df = pd.DataFrame(all_predictions)

# Ensure the submission file has the correct format
column_order = ['ID', 'resname', 'resid']
for i in range(1, 6):
    for coord in ['x', 'y', 'z']:
        column_order.append(f'{coord}_{i}')
submission_df = submission_df[column_order]

# Save the submission file
submission_df.to_csv('submission_TBM.csv', index=False)
print(f"Generated predictions for {len(test_seqs)} RNA sequences")
print(f"Total runtime: {time.time() - start_time:.1f} seconds")


submission_df


import numpy as np
import pandas as pd
from tqdm import tqdm

# ======================================
# 1️⃣ Load Submissions
# ======================================
tbm_path = "/kaggle/working/submission_TBM.csv"
protenix_path = "/kaggle/working/submission_proteinx.csv"
nufold_path = "/kaggle/working/submission_boltz.csv"

# Read CSVs
tbm = pd.read_csv(tbm_path)
protenix = pd.read_csv(protenix_path)
nufold = pd.read_csv(nufold_path)

print("✅ Files loaded:")
print("TBM:", tbm.shape)
print("Protenix:", protenix.shape)
print("NuFold:", nufold.shape)

# ======================================
# 2️⃣ Validate and Extract Target IDs
# ======================================
assert set(tbm["ID"]) == set(protenix["ID"]) == set(nufold["ID"]), \
    "❌ Mismatch in residue IDs among models!"

def extract_target_id(resid):
    return resid.split("_")[0]

for df in [tbm, protenix, nufold]:
    df["target_id"] = df["ID"].apply(extract_target_id)

print("✅ target_id extracted successfully.")

# ======================================
# 3️⃣ Helper Functions
# ======================================
def extract_structures(df, tid):
    """Extract (5, n_res, 3) array of conformations for a target."""
    sub = df[df["target_id"] == tid].sort_values("resid")
    coords = [sub[[f"x_{i}", f"y_{i}", f"z_{i}"]].values for i in range(1, 6)]
    return np.stack(coords, axis=0)

def kabsch_rmsd(P, Q):
    """Compute RMSD between two 3D conformations after alignment."""
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    R = U @ Vt
    P_aligned = P @ R
    return np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1)))

def mean_rmsd(setA):
    """Mean pairwise RMSD for diversity scoring."""
    if len(setA) < 2: return 0
    rmsd_vals = []
    for i in range(len(setA)):
        for j in range(i+1, len(setA)):
            rmsd_vals.append(kabsch_rmsd(setA[i], setA[j]))
    return np.mean(rmsd_vals)

# ======================================
# 4️⃣ Weighted Agent Search Tree (3 models)
# ======================================
def agent_tree_search(models, tid, weights=(0.45, 0.45, 0.10), w_div=1.0, w_dist=0.5):
    tbm, protenix, nufold = models
    tbm_conf = extract_structures(tbm, tid)
    prot_conf = extract_structures(protenix, tid)
    nuf_conf  = extract_structures(nufold, tid)

    # Combine candidates (15 conformations)
    all_conf = np.concatenate([tbm_conf, prot_conf, nuf_conf], axis=0)
    model_labels = (["tbm"]*5 + ["protenix"]*5 + ["nufold"]*5)

    # Strong priors: Protenix conf0 + TBM conf0
    priors = [prot_conf[1], tbm_conf[0]]
    selected = [prot_conf[1], tbm_conf[0]]

    # Pool of remaining candidates
    pool = [(conf, label) for conf, label in zip(all_conf, model_labels)
            if not any(np.allclose(conf, s) for s in selected)]

    # Search loop until 5 conformations selected
    while len(selected) < 5:
        best_score, best_conf = -np.inf, None
        for cand, label in pool:
            diversity = mean_rmsd(selected + [cand])
            dist_to_priors = np.mean([kabsch_rmsd(cand, p) for p in priors])

            # Model-specific reliability weighting
            if label == "protenix":
                model_weight = weights[1]
            elif label == "tbm":
                model_weight = weights[0]
            else:  # nufold
                model_weight = weights[2]

            # Weighted heuristic: balance diversity and proximity
            score = model_weight * (w_div * diversity - w_dist * dist_to_priors)

            if score > best_score:
                best_score, best_conf = score, cand

        selected.append(best_conf)
        pool = [(conf, label) for conf, label in pool if not np.allclose(conf, best_conf)]

    return selected

# ======================================
# 5️⃣ Build and Save Submission
# ======================================
def build_submission(tbm, protenix, nufold, output_path="/kaggle/working/submission.csv"):
    models = [tbm, protenix, nufold]
    all_rows = []

    for tid in tqdm(tbm["target_id"].unique(), desc="Weighted Agent Search (0.45,0.45,0.10)"):
        selected = agent_tree_search(models, tid, weights=(0.15, 0.75, 0.10))
        base = tbm[tbm["target_id"] == tid].sort_values("resid")
        for j, (resid, resname) in enumerate(zip(base["resid"], base["resname"])):
            row = {"ID": f"{tid}_{resid}", "resname": resname, "resid": resid}
            for k in range(5):
                row[f"x_{k+1}"], row[f"y_{k+1}"], row[f"z_{k+1}"] = selected[k][j]
            all_rows.append(row)

    sub = pd.DataFrame(all_rows)
    sub.to_csv(output_path, index=False)
    print(f"✅ 3-model weighted agent ensemble saved to {output_path}")

# ======================================
# 6️⃣ Run Ensemble
# ======================================
build_submission(tbm, protenix, nufold)







