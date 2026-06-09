import sys
import os
from pathlib import Path

# Assuming you've mounted your dataset containing the wheels and Protenix code
DATASET_PATH = '/kaggle/input/required-presets'

# Install dependencies from wheels
wheel_path = os.path.join(DATASET_PATH, 'wheels')
!pip install -qqq --no-index --find-links {wheel_path} torch numpy pandas scipy rdkit protenix

# Add Protenix to Python path
protenix_path = os.path.join(DATASET_PATH, 'Protenix')
sys.path.append(protenix_path)




import pandas as pd 
validation_sequence = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')


test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv') 
test_sequences.head()


submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')


submission.head()


len(submission)


"""
RNA 3D Structure Prediction and Submission Generator for Kaggle Environment

This script:
1. Reads RNA sequences from test_sequences.csv
2. Creates input JSONs for each sequence
3. Runs the Protenix model to predict 3D structures
4. Extracts C1' atom coordinates from the output CIF files
5. Creates a submission.csv file in the format required
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from biotite.structure.io import pdbx

def create_input_json(sequence, target_id):
    """
    Create the input JSON for a single RNA sequence
    """
    input_json = [{
        "sequences": [
            {
                "rnaSequence": {
                    "sequence": sequence,
                    "count": 1,
                    "modifications": []
                }
            }
        ],
        "name": target_id,
        "covalent_bonds": []
    }]
    return input_json

def run_inference(input_json_path, output_dir, target_id):
    """
    Run inference using the Protenix model with kaggle-specific paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define command with kaggle-specific paths
    cmd = [
        "python", "/kaggle/input/required-presets/Protenix/runner/inference.py",
        "--seeds", "42",
        "--dump_dir", output_dir,
        "--input_json_path", input_json_path,
        "--model.N_cycle", "10",
        "--sample_diffusion.N_sample", "5",
        "--sample_diffusion.N_step", "200",
        "--load_checkpoint_path", "/kaggle/input/required-presets/Protenix/release_data/checkpoint/model_v0.2.0.pt",
        "--use_deepspeed_evo_attention", "false"
    ]
    
    print(f"Running inference for {target_id}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running inference for {target_id}: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running inference for {target_id}: {e}")
        return False

def extract_c1_coordinates(cif_file_path):
    """
    Extract C1' atom coordinates from a CIF file using biotite
    """
    try:
        # Read the CIF file using the correct biotite method
        with open(cif_file_path, 'r') as f:
            cif_data = pdbx.CIFFile.read(f)
        
        # Get structure from CIF data
        atom_array = pdbx.get_structure(cif_data, model=1)
        
        # Clean atom names and find C1' atoms
        atom_names_clean = np.char.strip(atom_array.atom_name.astype(str))
        mask_c1 = atom_names_clean == "C1'"
        c1_atoms = atom_array[mask_c1]
        
        if len(c1_atoms) == 0:
            print(f"Warning: No C1' atoms found in {cif_file_path}")
            return None
        
        # Sort by residue ID and return coordinates
        sort_indices = np.argsort(c1_atoms.res_id)
        c1_atoms_sorted = c1_atoms[sort_indices]
        c1_coords = c1_atoms_sorted.coord
        
        return c1_coords
    except Exception as e:
        print(f"Error extracting C1' coordinates from {cif_file_path}: {e}")
        return None


def process_sequence(sequence, target_id, temp_dir, output_dir):
    """
    Process a single RNA sequence and return C1' coordinates
    """
    print(f"Processing {target_id}: {sequence}")
    
    # Create input JSON
    input_json = create_input_json(sequence, target_id)
    
    # Save JSON to temporary file
    os.makedirs(temp_dir, exist_ok=True)
    input_json_path = os.path.join(temp_dir, f"{target_id}_input.json")
    with open(input_json_path, "w") as f:
        json.dump(input_json, f, indent=4)
    
    # Run inference
    success = run_inference(input_json_path, output_dir, target_id)
    
    if not success:
        print(f"Inference failed for {target_id}")
        return None
    
    # Find the CIF files for this target
    target_prediction_dir = os.path.join(output_dir, target_id, "seed_42", "predictions")
    if not os.path.exists(target_prediction_dir):
        print(f"Prediction directory not found for {target_id}")
        return None
    
    # Look for CIF files with the pattern {target_id}_seed_42_sample_*.cif
    cif_files = sorted(glob.glob(os.path.join(target_prediction_dir, f"{target_id}_seed_42_sample_*.cif")))
    
    # If no CIF files found, return None
    if not cif_files:
        print(f"No CIF files found for {target_id}")
        return None
    
    print(f"Found {len(cif_files)} CIF files for {target_id}")
    
    # Extract C1' coordinates from each CIF file
    all_coords = []
    for cif_file in cif_files:
        coords = extract_c1_coordinates(cif_file)
        if coords is not None:
            all_coords.append(coords)
    
    if not all_coords:
        print(f"No valid C1' coordinates found for {target_id}")
        return None
    
    # Ensure we have 5 models (if we have fewer, duplicate the last one)
    while len(all_coords) < 5:
        print(f"Only {len(all_coords)} models found for {target_id}, duplicating last model")
        all_coords.append(all_coords[-1])
    
    return all_coords[:5]  # Ensure we only have 5 models

def create_submission(test_sequences_df, c1_coords_dict, output_file):
    """
    Create the submission CSV file with C1' coordinates
    """
    rows = []
    
    # Process each sequence
    for _, row in test_sequences_df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        
        if target_id not in c1_coords_dict or c1_coords_dict[target_id] is None:
            print(f"No prediction found for {target_id}, using zeros")
            # Create empty predictions (all zeros)
            for i, residue in enumerate(sequence):
                row_data = {
                    'ID': f"{target_id}_{i+1}",
                    'resname': residue,
                    'resid': i+1
                }
                for model in range(1, 6):
                    row_data[f'x_{model}'] = 0.0
                    row_data[f'y_{model}'] = 0.0
                    row_data[f'z_{model}'] = 0.0
                rows.append(row_data)
        else:
            # Get the 5 models for this target
            models = c1_coords_dict[target_id]
            
            # Create a row for each residue
            for i, residue in enumerate(sequence):
                row_data = {
                    'ID': f"{target_id}_{i+1}",
                    'resname': residue,
                    'resid': i+1
                }
                
                # Add coordinates for each model
                for model_idx in range(5):
                    if model_idx < len(models) and i < len(models[model_idx]):
                        row_data[f'x_{model_idx+1}'] = models[model_idx][i][0]
                        row_data[f'y_{model_idx+1}'] = models[model_idx][i][1]
                        row_data[f'z_{model_idx+1}'] = models[model_idx][i][2]
                    else:
                        # If coordinates are not available, use zeros
                        row_data[f'x_{model_idx+1}'] = 0.0
                        row_data[f'y_{model_idx+1}'] = 0.0
                        row_data[f'z_{model_idx+1}'] = 0.0
                
                rows.append(row_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Created submission file: {output_file}")

def main():
    """
    Main function
    """
    # Set up required symlinks for CCD cache as in kaggle_inference.py
    os.makedirs("/usr/local/lib/python3.10/dist-packages/release_data/ccd_cache", exist_ok=True)
    
    source_ccd_file = "/kaggle/input/required-presets/Protenix/release_data/ccd_cache/components.v20240608.cif"
    target_ccd_file = "/usr/local/lib/python3.10/dist-packages/release_data/ccd_cache/components.v20240608.cif"
    
    source_rdkit_file = "/kaggle/input/required-presets/Protenix/release_data/ccd_cache/components.v20240608.cif.rdkit_mol.pkl"
    target_rdkit_file = "/usr/local/lib/python3.10/dist-packages/release_data/ccd_cache/components.v20240608.cif.rdkit_mol.pkl"
    
    # Create the symlinks if the source files exist
    if os.path.exists(source_ccd_file) and not os.path.exists(target_ccd_file):
        try:
            os.symlink(source_ccd_file, target_ccd_file)
            print(f"Created symlink for CCD file")
        except Exception as e:
            print(f"Error creating symlink for CCD file: {e}")
    
    if os.path.exists(source_rdkit_file) and not os.path.exists(target_rdkit_file):
        try:
            os.symlink(source_rdkit_file, target_rdkit_file)
            print(f"Created symlink for RDKIT file")
        except Exception as e:
            print(f"Error creating symlink for RDKIT file: {e}")
    
    # Create directories
    temp_dir = "./input"  # Same as in kaggle_inference.py
    output_dir = "./output"  # Same as in kaggle_inference.py
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read test sequences
    test_sequences_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
    print(f"Loaded {len(test_sequences_df)} test sequences")
    
    # Process each sequence
    c1_coords_dict = {}
    for _, row in tqdm(test_sequences_df.iterrows(), total=len(test_sequences_df)):
        target_id = row['target_id']
        sequence = row['sequence']
        
        # Check if we already have predictions for this target
        target_prediction_dir = os.path.join(output_dir, target_id, "seed_42", "predictions")
        if os.path.exists(target_prediction_dir):
            print(f"Found existing prediction for {target_id}, loading coordinates")
            # Extract coordinates from existing predictions
            cif_files = sorted(glob.glob(os.path.join(target_prediction_dir, f"{target_id}_seed_42_sample_*.cif")))
            
            all_coords = []
            for cif_file in cif_files:
                coords = extract_c1_coordinates(cif_file)
                if coords is not None:
                    all_coords.append(coords)
            
            if all_coords:
                # Ensure we have 5 models
                while len(all_coords) < 5:
                    all_coords.append(all_coords[-1])
                c1_coords_dict[target_id] = all_coords[:5]
                continue
        
        # Process the sequence if no existing prediction was found or was invalid
        c1_coords = process_sequence(sequence, target_id, temp_dir, output_dir)
        c1_coords_dict[target_id] = c1_coords
    
    # Create submission file
    create_submission(test_sequences_df, c1_coords_dict, "submission.csv")

if __name__ == "__main__":
    main()


pd.read_csv("submission.csv")


# import os

# # Check if the components.cif file exists
# ccd_file = "/kaggle/input/required-presets/Protenix/release_data/ccd_cache/components.v20240608.cif"
# rdkit_file = "/kaggle/input/required-presets/Protenix/release_data/ccd_cache/components.v20240608.cif.rdkit_mol.pkl"

# print(f"CCD file exists: {os.path.exists(ccd_file)}")
# print(f"RDKIT file exists: {os.path.exists(rdkit_file)}")

# # If they don't exist, let's look for them
# if not (os.path.exists(ccd_file) and os.path.exists(rdkit_file)):
#     print("Searching for CCD files in /kaggle/input/required-presets/Protenix/release_data/...")
#     for root, dirs, files in os.walk("/kaggle/input/required-presets/Protenix/release_data/"):
#         for file in files:
#             if "components" in file and ("cif" in file or "rdkit" in file):
#                 print(f"Found: {os.path.join(root, file)}")





# import os
# import json
# import subprocess

# # Create the directory structure needed
# os.makedirs("/usr/local/lib/python3.10/dist-packages/release_data/ccd_cache", exist_ok=True)

# # Create symlinks to the actual files
# source_ccd_file = "/kaggle/input/required-presets/Protenix/release_data/ccd_cache/components.v20240608.cif"
# target_ccd_file = "/usr/local/lib/python3.10/dist-packages/release_data/ccd_cache/components.v20240608.cif"

# source_rdkit_file = "/kaggle/input/required-presets/Protenix/release_data/ccd_cache/components.v20240608.cif.rdkit_mol.pkl"
# target_rdkit_file = "/usr/local/lib/python3.10/dist-packages/release_data/ccd_cache/components.v20240608.cif.rdkit_mol.pkl"

# # Check if the source files exist
# print(f"Source CCD file exists: {os.path.exists(source_ccd_file)}")
# print(f"Source RDKIT file exists: {os.path.exists(source_rdkit_file)}")

# # Create the symlinks if the source files exist
# if os.path.exists(source_ccd_file):
#     try:
#         os.symlink(source_ccd_file, target_ccd_file)
#         print(f"Created symlink for CCD file")
#     except FileExistsError:
#         print(f"Symlink for CCD file already exists")
#     except Exception as e:
#         print(f"Error creating symlink for CCD file: {e}")
# else:
#     print(f"Cannot create symlink, source CCD file doesn't exist")

# if os.path.exists(source_rdkit_file):
#     try:
#         os.symlink(source_rdkit_file, target_rdkit_file)
#         print(f"Created symlink for RDKIT file")
#     except FileExistsError:
#         print(f"Symlink for RDKIT file already exists")
#     except Exception as e:
#         print(f"Error creating symlink for RDKIT file: {e}")
# else:
#     print(f"Cannot create symlink, source RDKIT file doesn't exist")

# # Create RNA input JSON
# input_json = [{
#     "sequences": [
#         {
#             "rnaSequence": {
#                 "sequence": "GGGUGCUCAGUACGAGAGGAACCGCACCC",
#                 "count": 1,
#                 "modifications": []
#             }
#         }
#     ],
#     "name": "rna_prediction",
#     "covalent_bonds": []
# }]

# # Save input JSON
# os.makedirs("./input", exist_ok=True)
# with open("./input/rna_input.json", "w") as f:
#     json.dump(input_json, f, indent=4)

# # Run inference using subprocess
# cmd = [
#     "python", "/kaggle/input/required-presets/Protenix/runner/inference.py",
#     "--seeds", "42",
#     "--dump_dir", "./output",
#     "--input_json_path", "./input/rna_input.json",
#     "--model.N_cycle", "10",
#     "--sample_diffusion.N_sample", "5",
#     "--sample_diffusion.N_step", "200",
#     "--load_checkpoint_path", "/kaggle/input/required-presets/Protenix/release_data/checkpoint/model_v0.2.0.pt",
#     "--use_deepspeed_evo_attention", "false"
# ]

# # Run the command
# result = subprocess.run(cmd, capture_output=True, text=True)
# print("STDOUT:", result.stdout)
# print("STDERR:", result.stderr)


# # Check for error files
# if os.path.exists("./output/ERR"):
#     print("Error directory exists!")
#     print("Contents:")
#     for item in os.listdir("./output/ERR"):
#         print(f" - {item}")
    
#     # If there are error files, show their contents
#     error_files = os.listdir("./output/ERR")
#     if error_files:
#         with open(os.path.join("./output/ERR", error_files[0]), "r") as f:
#             print(f"Contents of {error_files[0]}:")
#             print(f.read())
            


# import os
# import json
# import pandas as pd
# import biotite.structure.io.pdbx as pdbx
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # List all the prediction files
# output_dir = "/kaggle/working/output/rna_prediction/seed_42/predictions/"
# cif_files = [f for f in os.listdir(output_dir) if f.endswith(".cif")]
# print(f"Found {len(cif_files)} prediction files:")
# for file in cif_files:
#     print(f" - {file}")

# # Read and analyze the first prediction file
# if cif_files:
#     # Read CIF file
#     cif_file = os.path.join(output_dir, cif_files[0])
#     with open(cif_file, 'r') as f:
#         cif_data = pdbx.CIFFile.read(f)

#     atom_array = pdbx.get_structure(cif_data, model=1)
    
#     print(f"\nStructure information for {cif_files[0]}:")
#     print(f"Number of atoms: {len(atom_array)}")
#     print(f"Residue count: {len(np.unique(atom_array.res_id))}")

#     # Clean and extract C1' atoms
#     atom_names_clean = np.char.strip(atom_array.atom_name.astype(str))
#     mask_c1 = atom_names_clean == "C1'"
#     c1_atoms = atom_array[mask_c1]
#     c1_coords = c1_atoms.coord

#     print(f"\nFound {len(c1_atoms)} C1' atoms")

#     # Create DataFrame
#     df = pd.DataFrame({
#         "res_name": c1_atoms.res_name,
#         "res_id": c1_atoms.res_id,
#         "chain_id": c1_atoms.chain_id,
#         "x": c1_coords[:, 0],
#         "y": c1_coords[:, 1],
#         "z": c1_coords[:, 2]
#     })

#     print("\nFirst few C1' atoms:")
#     print(df.head())

#     # Save to CSV and JSON
#     df.to_csv("c1_prime_coordinates.csv", index=False)
#     df.to_json("c1_prime_coordinates.json", orient="records", indent=2)
#     print("Saved C1' coordinates to CSV and JSON.")

#     # Plot C1' atoms with backbone
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     chain_ids = np.unique(c1_atoms.chain_id)
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(chain_ids)))

#     for i, chain_id in enumerate(chain_ids):
#         chain_mask = c1_atoms.chain_id == chain_id
#         chain_df = df[df["chain_id"] == chain_id]
#         chain_df_sorted = chain_df.sort_values("res_id")

#         ax.scatter(
#             chain_df_sorted["x"],
#             chain_df_sorted["y"],
#             chain_df_sorted["z"],
#             c=[colors[i]],
#             label=f"Chain {chain_id}",
#             alpha=0.9,
#             s=30
#         )

#         ax.plot(
#             chain_df_sorted["x"],
#             chain_df_sorted["y"],
#             chain_df_sorted["z"],
#             color=colors[i],
#             alpha=0.6,
#             linewidth=2
#         )

#     ax.set_xlabel("X (Å)")
#     ax.set_ylabel("Y (Å)")
#     ax.set_zlabel("Z (Å)")
#     ax.set_title(f"C1' atom backbone - {cif_files[0]}")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

#     # --- NEW: Plot all atoms in another image ---
#     print("\nGenerating full atom visualization...")

#     all_coords = atom_array.coord
#     all_chain_ids = np.unique(atom_array.chain_id)
#     colors_all = plt.cm.rainbow(np.linspace(0, 1, len(all_chain_ids)))

#     fig_all = plt.figure(figsize=(10, 8))
#     ax_all = fig_all.add_subplot(111, projection='3d')

#     for i, chain_id in enumerate(all_chain_ids):
#         chain_mask = atom_array.chain_id == chain_id
#         ax_all.scatter(
#             all_coords[chain_mask, 0],
#             all_coords[chain_mask, 1],
#             all_coords[chain_mask, 2],
#             c=[colors_all[i]],
#             label=f"Chain {chain_id}",
#             alpha=0.6,
#             s=5  # smaller for all atoms
#         )

#     ax_all.set_xlabel("X (Å)")
#     ax_all.set_ylabel("Y (Å)")
#     ax_all.set_zlabel("Z (Å)")
#     ax_all.set_title(f"All atoms - {cif_files[0]}")
#     ax_all.legend()
#     plt.tight_layout()
#     plt.show()



# !pip install nglview


# import json
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Path to the summary confidence JSON file
# confidence_file = "/kaggle/working/output/rna_prediction/seed_42/predictions/rna_prediction_seed_42_summary_confidence_sample_0.json"

# # Read and display the file contents
# with open(confidence_file, 'r') as f:
#     confidence_data = json.load(f)

# # Show the keys in the JSON file
# print("Keys in the confidence file:")
# for key in confidence_data.keys():
#     print(f" - {key}")

# # Function to display and visualize specific metrics
# def analyze_metric(data, metric_name):
#     if metric_name in data:
#         print(f"\n{metric_name} data:")
        
#         # Handle different data types
#         if isinstance(data[metric_name], (int, float)):
#             print(f"{metric_name}: {data[metric_name]}")
        
#         elif isinstance(data[metric_name], list):
#             print(f"{metric_name} (first 5 values): {data[metric_name][:5]}")
            
#             # Plot if it's a list of numbers
#             if data[metric_name] and isinstance(data[metric_name][0], (int, float)):
#                 plt.figure(figsize=(10, 6))
#                 plt.plot(data[metric_name])
#                 plt.title(f"{metric_name} across residues")
#                 plt.xlabel("Residue index")
#                 plt.ylabel(metric_name)
#                 plt.grid(True, alpha=0.3)
#                 plt.savefig(f"{metric_name}_plot.png")
#                 plt.close()
#                 print(f"Saved visualization to '{metric_name}_plot.png'")
        
#         elif isinstance(data[metric_name], dict):
#             # For dictionaries, show keys and sample values
#             print(f"{metric_name} contains {len(data[metric_name])} keys:")
#             for k in list(data[metric_name].keys())[:5]:
#                 print(f"  - {k}: {data[metric_name][k]}")
#     else:
#         print(f"\n{metric_name} not found in the data")

# # Analyze common confidence metrics
# analyze_metric(confidence_data, "plddt")  # Per-residue confidence scores
# analyze_metric(confidence_data, "ranking_score")  # Overall ranking score of the model
# analyze_metric(confidence_data, "gpde")  # Global predicted distance error

# # If there's PAE (Predicted Aligned Error) matrix, visualize it
# if "pae" in confidence_data and isinstance(confidence_data["pae"], list):
#     pae_data = np.array(confidence_data["pae"])
    
#     if len(pae_data.shape) == 2:
#         plt.figure(figsize=(10, 8))
#         im = plt.imshow(pae_data, cmap='viridis_r')
#         plt.colorbar(im, label="Predicted Aligned Error (Å)")
#         plt.title("PAE Matrix")
#         plt.xlabel("Residue")
#         plt.ylabel("Residue")
#         plt.savefig("pae_matrix.png")
#         plt.close()
#         print("\nSaved PAE matrix visualization to 'pae_matrix.png'")

# # Display the structure's overall quality assessment
# print("\nOverall structure quality assessment:")
# quality_metrics = ["ranking_score", "gpde", "plddt_avg", "pae_avg"]
# for metric in quality_metrics:
#     if metric in confidence_data:
#         print(f" - {metric}: {confidence_data[metric]}")


# import json
# import pprint

# # Path to the summary confidence JSON file
# confidence_file = "/kaggle/working/output/rna_prediction/seed_42/predictions/rna_prediction_seed_42_summary_confidence_sample_0.json"

# # Read and display the complete file contents
# with open(confidence_file, 'r') as f:
#     confidence_data = json.load(f)

# # Use pretty print to display formatted JSON
# print("Complete JSON content:")
# pp = pprint.PrettyPrinter(indent=2)
# pp.pprint(confidence_data)

# # Or alternatively, print it with json.dumps for more control over formatting
# print("\nAlternative JSON formatting:")
# print(json.dumps(confidence_data, indent=2))





