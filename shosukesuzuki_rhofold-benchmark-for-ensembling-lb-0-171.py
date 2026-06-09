# Requires the rhofold dependency library.
# Although it includes some unnecessary libraries, you can also use my experimental requirements.
# https://www.kaggle.com/code/shosukesuzuki/requirements
! python -m pip install --no-index --find-links=../input/requirements -r ../input/requirements/requirements.txt


import os
import gc
import math
import time
import pickle
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser

from scipy.spatial import procrustes
from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# Please download and upload the checkpoint from the RhoFold repository.
ckpt_path = "/kaggle/input/model-ckpt/RhoFold/pretrained/rhofold_pretrained_params.pt"

# The source code for RhoFold+ can be imported from Input → GitHub.
# I have named the repository 'rhofold'.
# When running, either import the source as 'rhofold' or adjust the corresponding parts accordingly.


# merge multi segments for long sequences
def align_and_merge_segments(results, original_length, overlap=20):
    results = sorted(results, key=lambda x: x[0])
    
    full_coords = np.zeros((original_length, 3))
    weights = np.zeros(original_length)
    
    ref_start, ref_names, ref_ids, ref_coords = results[0]
    
    for i in range(len(ref_ids)):
        global_idx = ref_start + i
        if global_idx < original_length:
            full_coords[global_idx] = ref_coords[i]
            weights[global_idx] = 1.0
    
    for i in range(1, len(results)):
        curr_start, curr_names, curr_ids, curr_coords = results[i]

        prev_start = results[i-1][0]
        prev_coords = results[i-1][3]

        overlap_start = max(curr_start, prev_start)
        prev_end = prev_start + len(prev_coords)
        overlap_end = min(curr_start + len(curr_coords), prev_end)

        if overlap_end > overlap_start:
            prev_overlap_indices = [j - prev_start for j in range(overlap_start, overlap_end)]
            curr_overlap_indices = [j - curr_start for j in range(overlap_start, overlap_end)]
            
            prev_overlap_coords = prev_coords[prev_overlap_indices]
            curr_overlap_coords = curr_coords[curr_overlap_indices]
            
            if len(prev_overlap_coords) >= 3 and len(curr_overlap_coords) >= 3:
                _, curr_overlap_aligned, transform_scale = procrustes(prev_overlap_coords, curr_overlap_coords)

                mtx1, mtx2, disparity = procrustes(prev_overlap_coords, curr_overlap_coords)

                aligned_coords = curr_coords.copy()
                
                curr_centroid = np.mean(curr_overlap_coords, axis=0)
                prev_centroid = np.mean(prev_overlap_coords, axis=0)
                translation = prev_centroid - curr_centroid
                
                curr_centered = curr_coords - curr_centroid

                _, _, rotation_matrix, scale = _get_procrustes_transformation(prev_overlap_coords, curr_overlap_coords)
                aligned_coords = scale * (curr_centered @ rotation_matrix.T) + prev_centroid
                
                for j in range(len(curr_ids)):
                    global_idx = curr_start + j
                    if global_idx < original_length:
                        in_overlap = global_idx >= overlap_start and global_idx < overlap_end
                        if in_overlap:
                            pos_in_overlap = (global_idx - overlap_start) / (overlap_end - overlap_start)
                            weight = 1.0 - pos_in_overlap
                            if weights[global_idx] > 0:
                                full_coords[global_idx] = full_coords[global_idx] * weights[global_idx] + aligned_coords[j] * weight
                                full_coords[global_idx] /= (weights[global_idx] + weight)
                                weights[global_idx] += weight
                            else:
                                full_coords[global_idx] = aligned_coords[j]
                                weights[global_idx] = weight
                        elif weights[global_idx] == 0:
                            full_coords[global_idx] = aligned_coords[j]
                            weights[global_idx] = 1.0
            else:
                for j in range(len(curr_ids)):
                    global_idx = curr_start + j
                    if global_idx < original_length and weights[global_idx] == 0:
                        full_coords[global_idx] = curr_coords[j]
                        weights[global_idx] = 1.0
        else:
            for j in range(len(curr_ids)):
                global_idx = curr_start + j
                if global_idx < original_length and weights[global_idx] == 0:
                    full_coords[global_idx] = curr_coords[j]
                    weights[global_idx] = 1.0
    
    missing_indices = np.where(weights == 0)[0]
    if len(missing_indices) > 0:
        valid_indices = np.where(weights > 0)[0]
        if len(valid_indices) > 0:
            for dim in range(3):
                full_coords[missing_indices, dim] = np.interp(
                    missing_indices, valid_indices, full_coords[valid_indices, dim]
                )
    
    return full_coords


def _get_procrustes_transformation(target, source):
    target_centroid = np.mean(target, axis=0)
    source_centroid = np.mean(source, axis=0)
    
    target_centered = target - target_centroid
    source_centered = source - source_centroid
    
    rotation_matrix, _ = orthogonal_procrustes(source_centered, target_centered)
    
    target_norm = np.linalg.norm(target_centered)
    source_norm = np.linalg.norm(source_centered)
    scale = target_norm / source_norm if source_norm > 0 else 1.0
    
    return target_centroid, source_centroid, rotation_matrix, scale


# run rhofold inference helper
def create_fasta_files(df, fasta_dir):
    """Create individual FASTA files for each target_id"""
    print(f"Creating FASTA files in {fasta_dir}...")
    os.makedirs(fasta_dir, exist_ok=True)
    for idx, row in df.iterrows():
        target_id = row["target_id"]
        sequence = row["sequence"]
        fasta_path = os.path.join(fasta_dir, f"{target_id}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{target_id}\n{sequence}\n")
    print(f"Created {len(df)} FASTA files.")
    gc.collect()


def run_rhofold(target_id, fasta_path, output_dir, ckpt_path):
    """Run RhoFold on a single FASTA file."""
    with open(fasta_path, 'r') as f:
        f.readline()
        sequence = f.readline().strip()
    sequence_length = len(sequence)
    device = "cpu"
    
    result_dir = os.path.join(output_dir, target_id)
    os.makedirs(result_dir, exist_ok=True)
    
    abs_fasta_path = os.path.abspath(fasta_path)
    abs_result_dir = os.path.abspath(result_dir)
    abs_checkpoint = os.path.abspath(ckpt_path)
    
    current_dir = os.getcwd()
    cmd = [
        "python", "inference.py",
        "--input_fas", abs_fasta_path,
        "--single_seq_pred", "True",
        "--output_dir", abs_result_dir,
        "--device", device,
        "--ckpt", abs_checkpoint,
        "--relax_steps", "0",
    ]
    
    print(f"Running RhoFold on {target_id} (length: {sequence_length}, device: {device})...")
    start_time = time.time()
    try:
        os.chdir("/kaggle/input/rhofold")
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"  Completed in {time.time() - start_time:.2f}s")
        print(f"  RhoFold stdout: {result.stdout[:200]}..." if len(result.stdout) > 200 else f"  RhoFold stdout: {result.stdout}")
        success = True
    except subprocess.CalledProcessError as e:
        print(f"  Error running RhoFold on {target_id}:")
        print(f"  RhoFold stderr: {e.stderr}")
        success = False
    except Exception as e:
        print(f"  Unexpected error running RhoFold on {target_id}: {str(e)}")
        success = False
    finally:
        os.chdir(current_dir)
        if 'result' in locals():
            del result
        gc.collect()
    return success


def extract_c1_coordinates(pdb_file):
    """Extract C1' coordinates from PDB file"""
    parser = PDBParser(QUIET=True)
    c1_coordinates = []
    residue_names = []
    residue_ids = []
    try:
        structure = parser.get_structure('RNA_structure', pdb_file)
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in ['A', 'U', 'G', 'C']:
                        try:
                            c1_atom = residue["C1'"]
                            residue_names.append(residue.get_resname())
                            residue_ids.append(residue.get_id()[1])
                            c1_coordinates.append(c1_atom.get_coord())
                        except KeyError:
                            print(f"C1' atom not found in residue {residue.get_resname()}{residue.get_id()[1]}")
        coords_array = np.array(c1_coordinates)
        del structure
        gc.collect()
        return residue_names, residue_ids, coords_array
    except Exception as e:
        print(f"Error parsing PDB file {pdb_file}: {e}")
        return [], [], np.array([])


def interpolate_coordinates(coords, target_length):
    """Interpolate coordinates to target length"""
    n = coords.shape[0]
    full_coords = np.empty((target_length, 3))
    old_indices = np.linspace(0, 1, n)
    new_indices = np.linspace(0, 1, target_length)
    for dim in range(3):
        full_coords[:, dim] = np.interp(new_indices, old_indices, coords[:, dim])
    return full_coords


def generate_variants(base_coords, n_variants=5, noise_scale=0.5):
    """Generate conformation variants"""
    variants = [None] * n_variants
    variants[0] = base_coords.copy()
    
    for i in range(1, n_variants):
        noise = np.random.normal(scale=noise_scale, size=base_coords.shape)
        variants[i] = base_coords + noise
    
    return variants


def run_rhofold_with_truncation(target_id, sequence, fasta_dir, output_dir, ckpt_path):
    """Run RhoFold directly without truncation, using random coordinates as fallback"""
    original_length = len(sequence)
    
    temp_fasta = os.path.join(fasta_dir, f"{target_id}_full.fasta")
    with open(temp_fasta, "w") as f:
        f.write(f">{target_id}\n{sequence}\n")
    
    print(f"Running inference for {target_id} with full sequence length {original_length}...")
    
    # Try running RhoFold once with full sequence
    if run_rhofold(target_id, temp_fasta, output_dir, ckpt_path):
        pdb_file = os.path.join(output_dir, target_id, "unrelaxed_model.pdb")
        if Path(pdb_file).exists():
            res_names, res_ids, coords = extract_c1_coordinates(pdb_file)
            if coords.size > 0:
                print(f"Successfully predicted structure for full sequence.")
                os.remove(temp_fasta)
                return res_names, res_ids, coords
    
    # If failed, use random coordinates
    print(f"Inference failed for {target_id}. Using random coordinates.")
    if os.path.exists(temp_fasta):
        os.remove(temp_fasta)
    
    # Generate random coordinates with reasonable RNA dimensions
    random_coords = np.random.rand(original_length, 3) * 20  # Scale to typical RNA dimensions
    res_names_full = list(sequence)
    res_ids_full = list(range(1, original_length + 1))
    
    return res_names_full, res_ids_full, random_coords


def run_rhofold_with_multi_segment(target_id, sequence, fasta_dir, output_dir, ckpt_path, overlap=20):
    """Split sequence into multiple segments and merge results, ensuring each segment is manageable"""
    original_length = len(sequence)
    
    # Calculate optimal number of segments to keep each segment under 100bp (including overlap)
    max_segment_size = 200
    effective_segment_size = max_segment_size - (2 * overlap)
    n_segments = max(3, math.ceil(original_length / effective_segment_size))
    
    # Recalculate segment size with fixed number of segments
    segment_length = original_length // n_segments
    
    print(f"Using multi-segment approach for {target_id} with {n_segments} segments, each ~{segment_length}bp + {overlap}bp overlap")
    
    # Calculate segment boundaries
    segments = []
    start_positions = []
    
    for i in range(n_segments):
        start = max(0, i * segment_length - overlap)
        end = min(original_length, (i + 1) * segment_length + overlap)
        current_segment = sequence[start:end]
        
        # Ensure segment is not too long
        if len(current_segment) > max_segment_size:
            print(f"Segment {i+1} is {len(current_segment)}bp, trimming to {max_segment_size}bp")
            middle = len(current_segment) // 2
            half_max = max_segment_size // 2
            current_segment = current_segment[max(0, middle-half_max):min(len(current_segment), middle+half_max)]
            # Adjust start position
            start = start + max(0, middle-half_max)
        
        segments.append(current_segment)
        start_positions.append(start)
        print(f"Segment {i+1}: positions {start}-{start+len(current_segment)} (length: {len(current_segment)}bp)")
    
    # Process each segment
    results = []
    
    for i, (seg, start_pos) in enumerate(zip(segments, start_positions)):
        seg_id = f"{target_id}_seg{i+1}"
        print(f"Processing segment {i+1}/{n_segments} for {target_id} (length: {len(seg)})")
        
        # Create segment FASTA file
        seg_fasta = os.path.join(fasta_dir, f"{seg_id}.fasta")
        with open(seg_fasta, "w") as f:
            f.write(f">{seg_id}\n{seg}\n")
        
        # Try running RhoFold on this segment
        if run_rhofold(seg_id, seg_fasta, output_dir, ckpt_path):
            pdb_file = os.path.join(output_dir, seg_id, "unrelaxed_model.pdb")
            if Path(pdb_file).exists():
                res_names, res_ids, coords = extract_c1_coordinates(pdb_file)
                if coords.size > 0:
                    results.append((start_pos, res_names, res_ids, coords))
                else:
                    print(f"No coordinates extracted for segment {i+1}")
                    # Generate random coordinates for this segment
                    random_coords = np.random.rand(len(seg), 3) * 20
                    res_names = list(seg)
                    res_ids = list(range(1, len(seg) + 1))
                    results.append((start_pos, res_names, res_ids, random_coords))
            else:
                print(f"PDB file not found for segment {i+1}")
                # Generate random coordinates for this segment
                random_coords = np.random.rand(len(seg), 3) * 20
                res_names = list(seg)
                res_ids = list(range(1, len(seg) + 1))
                results.append((start_pos, res_names, res_ids, random_coords))
        else:
            print(f"RhoFold failed for segment {i+1}")
            # Generate random coordinates for this segment
            random_coords = np.random.rand(len(seg), 3) * 20
            res_names = list(seg)
            res_ids = list(range(1, len(seg) + 1))
            results.append((start_pos, res_names, res_ids, random_coords))
        
        # Clean up segment FASTA
        if os.path.exists(seg_fasta):
            os.remove(seg_fasta)
        
        gc.collect()
    
    # integrate setments
    try:
        print(f"Aligning and merging {len(results)} segments...")
        full_coords = align_and_merge_segments(results, original_length, overlap)
        print(f"Successfully merged segments using procrustes alignment")
    except Exception as e:
        print(f"Error during segment alignment: {e}")
        print(f"Falling back to simple weighted average merging")
        
        full_coords = np.zeros((original_length, 3))
        weights = np.zeros(original_length)
        
        for start_pos, res_names, res_ids, coords in results:
            # Map coordinates to their positions in the full sequence
            for i in range(len(res_ids)):
                global_idx = start_pos + i
                if global_idx < original_length:
                    # For overlap regions, use weighted averaging
                    if weights[global_idx] > 0:
                        # Calculate weight based on distance from segment edge
                        seg_len = len(res_ids)
                        edge_dist = min(i, seg_len - i)
                        edge_weight = min(1.0, edge_dist / overlap) if overlap > 0 else 1.0
                        
                        # Update weighted average
                        full_coords[global_idx] = (full_coords[global_idx] * weights[global_idx] + 
                                                 coords[i] * edge_weight) / (weights[global_idx] + edge_weight)
                        weights[global_idx] += edge_weight
                    else:
                        full_coords[global_idx] = coords[i]
                        weights[global_idx] = 1.0
    
        # Check for missing positions
        missing_count = np.sum(weights == 0)
        if missing_count > 0:
            print(f"Warning: {missing_count} positions have no coordinates. Filling with interpolation.")
            
            # Try to fill small gaps first
            for _ in range(3):  # Repeat a few times to fill larger gaps
                valid_mask = weights > 0
                for i in range(1, original_length-1):
                    if weights[i] == 0 and weights[i-1] > 0 and weights[i+1] > 0:
                        full_coords[i] = (full_coords[i-1] + full_coords[i+1]) / 2
                        weights[i] = 0.5
            
            # Fill any remaining gaps with global interpolation
            valid_indices = np.where(weights > 0)[0]
            zero_indices = np.where(weights == 0)[0]
            
            if len(valid_indices) > 0 and len(zero_indices) > 0:
                for dim in range(3):
                    full_coords[zero_indices, dim] = np.interp(
                        zero_indices, valid_indices, full_coords[valid_indices, dim]
                    )
            else:
                print(f"Error: Cannot interpolate coordinates. Using random model as fallback.")
                full_coords = np.random.rand(original_length, 3) * 20
    
    res_names_full = list(sequence)
    res_ids_full = list(range(1, original_length + 1))
    
    return res_names_full, res_ids_full, full_coords


def create_submission_dataframe(target_ids, all_results):
    """Create submission DataFrame from results with 5 conformation variants"""
    rows = []
    for target_id in target_ids:
        if target_id not in all_results:
            print(f"Warning: No results for {target_id}")
            continue
        
        res_names, res_ids, variants = all_results[target_id]
        
        for i in range(len(res_ids)):
            row = {
                "ID": f"{target_id}_{res_ids[i]}",
                "resname": res_names[i],
                "resid": res_ids[i]
            }
            
            for model in range(1, 6):
                variant_idx = model - 1
                if variant_idx < len(variants):
                    variant_coords = variants[variant_idx]
                    if i < len(variant_coords):
                        row[f"x_{model}"] = variant_coords[i][0]
                        row[f"y_{model}"] = variant_coords[i][1]
                        row[f"z_{model}"] = variant_coords[i][2]
            
            rows.append(row)
    
    df_submission = pd.DataFrame.from_records(rows)
    return df_submission


# inference setup
fasta_dir = "test_fasta"
output_dir = "rhofold_output"

os.makedirs(fasta_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

segment_length = 400
print(f"Starting RNA folding pipeline... (segment length: {segment_length})")

df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
create_fasta_files(df, fasta_dir)

# Process all sequences
all_results = {}
checkpoint_file = "results_checkpoint.pkl"

# Load checkpoint if exists
if os.path.exists(checkpoint_file):
    try:
        with open(checkpoint_file, 'rb') as f:
            all_results = pickle.load(f)
        print(f"Loaded {len(all_results)} results from checkpoint")
    except:
        print("Failed to load checkpoint. Starting fresh.")

# Process remaining sequences
target_ids = df["target_id"].tolist()
remaining_ids = [id for id in target_ids if id not in all_results]

for idx, target_id in enumerate(remaining_ids):
    sequence = df[df["target_id"] == target_id]["sequence"].values[0]
    print(f"\n[{idx+1}/{len(remaining_ids)}] Processing {target_id} (length: {len(sequence)})...")
    
    try:
        # Select appropriate method based on sequence length
        if len(sequence) >= segment_length:
            print(f"Sequence length {len(sequence)} >= {segment_length}, using multi-segment method")
            res_names, res_ids, base_coords = run_rhofold_with_multi_segment(
                target_id, sequence, fasta_dir, output_dir, ckpt_path
            )
        else:
            print(f"Sequence length {len(sequence)} < {segment_length}, using direct method")
            res_names, res_ids, base_coords = run_rhofold_with_truncation(
                target_id, sequence, fasta_dir, output_dir, ckpt_path
            )
        
        # Generate 5 different conformation variants
        variants = generate_variants(base_coords, n_variants=5, noise_scale=2.5)
        all_results[target_id] = (res_names, res_ids, variants)
        
        print(f"Completed {target_id}: {len(res_ids)} residues")
        
        # Clean up temp files
        for f in os.listdir(fasta_dir):
            if f.startswith(f"{target_id}_") and f.endswith(".fasta"):
                try:
                    os.remove(os.path.join(fasta_dir, f))
                except:
                    pass
    except Exception as e:
        print(f"Error processing {target_id}: {e}")
        # Create fallback result in case of errors
        print(f"Using random coordinates as fallback for {target_id}")
        res_names = list(sequence)
        res_ids = list(range(1, len(sequence) + 1))
        random_coords = np.random.rand(len(sequence), 3) * 20
        variants = [random_coords.copy() + np.random.normal(scale=0.5, size=random_coords.shape) for _ in range(5)]
        all_results[target_id] = (res_names, res_ids, variants)
    
    # Save checkpoint after each sequence
    if (idx + 1) % 5 == 0:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Saved checkpoint with {len(all_results)}/{len(df)} sequences processed")
    
    gc.collect()

# Create submission file
submission_df = create_submission_dataframe(df["target_id"].tolist(), all_results)


# check predicted structures (5 conformations)
def plot_rna_structure(rna_id):
    filtered_df = submission_df[submission_df['ID'].str.startswith(f"{rna_id}_")]
    
    if len(filtered_df) == 0:
        print(f"ID '{rna_id} is not found/")
        return

    base_colors = {'A': 'red', 'U': 'blue', 'G': 'green', 'C': 'orange'}
    fig = plt.figure(figsize=(20, 15))
    for variant in range(1, 6):
        ax = fig.add_subplot(2, 3, variant, projection='3d')
        
        x = filtered_df[f'x_{variant}'].values
        y = filtered_df[f'y_{variant}'].values
        z = filtered_df[f'z_{variant}'].values
        
        colors = [base_colors.get(base, 'gray') for base in filtered_df['resname']]
        
        ax.scatter(x, y, z, c=colors, s=5, alpha=1.0)
        ax.plot(x, y, z, 'k-', alpha=1.0, linewidth=0.1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.set_title(f'Random Noise {variant} - {rna_id}')
        for base, color in base_colors.items():
            ax.scatter([], [], [], c=color, label=base)
        ax.legend()
    
    plt.suptitle(f'RNA Structure: {rna_id} (Length: {len(filtered_df)})', fontsize=16)
    plt.tight_layout()
    return fig


plot_rna_structure('R1107')
plt.show()


plot_rna_structure('R1108')
plt.show()


plot_rna_structure('R1116')
plt.show()


plot_rna_structure('R1117v2')
plt.show()


plot_rna_structure('R1126')
plt.show()


plot_rna_structure('R1128')
plt.show()


plot_rna_structure('R1136')
plt.show()


plot_rna_structure('R1138')
plt.show()


plot_rna_structure('R1149')
plt.show()


plot_rna_structure('R1156')
plt.show()


plot_rna_structure('R1189')
plt.show()


plot_rna_structure('R1190')
plt.show()


# submission data
submission_df


# Save submission
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path} with {len(submission_df)} rows")

