import os
import re
import subprocess
import pandas as pd
import numpy as np # For argmax

def parse_tmscore_output(output: str) -> float:
    """
    Extracts TM-score from USalign output, expecting the second score
    (normalized by length of reference structure).
    """
    tm_score_matches = re.findall(r'TM-score=\s+([\d.]+)', output)
    if not tm_score_matches:
        raise ValueError('No TM-score pattern found in USalign output.')
    if len(tm_score_matches) < 2:
        raise ValueError(f'Expected at least 2 TM-scores in USalign output, found {len(tm_score_matches)}. Output: {output[:500]}')
    return float(tm_score_matches[1])


def write_target_line(
    atom_name: str, atom_serial: int, residue_name: str, chain_id: str,
    residue_num: int, x_coord: float, y_coord: float, z_coord: float,
    occupancy: float = 1.0, b_factor: float = 0.0, atom_type: str = 'P'
) -> str:
    return f'ATOM  {atom_serial:>5d}  {atom_name:<5s} {residue_name:<3s} {residue_num:>3d}    {x_coord:>8.3f}{y_coord:>8.3f}{z_coord:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}           {atom_type}\n'


def write2pdb(df: pd.DataFrame, xyz_id: int, target_path: str) -> int:
    resolved_cnt = 0
    try:
        # Ensure DataFrame is not empty and required columns for the given xyz_id exist
        coord_cols = [f'x_{xyz_id}', f'y_{xyz_id}', f'z_{xyz_id}']
        if df.empty or not all(col in df.columns for col in coord_cols):
            # print(f"Warning: DataFrame is empty or missing coordinate columns for xyz_id {xyz_id} for PDB {target_path}.")
            return 0
            
        with open(target_path, 'w') as target_file:
            for _, row in df.iterrows():
                # Check if coordinate values are present and valid for the current row
                if any(pd.isna(row[col]) for col in coord_cols):
                    # print(f"Skipping row in PDB {target_path} due to NaN coordinates for xyz_id {xyz_id}: resid {row.get('resid', 'N/A')}")
                    continue

                x_coord = row[f'x_{xyz_id}']
                y_coord = row[f'y_{xyz_id}']
                z_coord = row[f'z_{xyz_id}']

                if x_coord > -1e17 and y_coord > -1e17 and z_coord > -1e17: # Check for sentinel values
                    resolved_cnt += 1
                    target_line_str = write_target_line(
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
                    target_file.write(target_line_str)
    except KeyError as e:
        # This might happen if columns like 'resid' or 'resname' are missing, or if xyz_id leads to missing x_,y_,z_ columns
        print(f"Error writing PDB {target_path}: Missing column {e}. Ensure DataFrame has required columns (resid, resname, x_{xyz_id}, y_{xyz_id}, z_{xyz_id}).")
        return 0
    except Exception as e:
        print(f"An unexpected error occurred while writing PDB {target_path}: {e}")
        return 0
    return resolved_cnt


def score(solution_df: pd.DataFrame, submission_df: pd.DataFrame, usalign_executable_path: str, verbose: bool = False) -> float:
    """
    Computes the average of best-of-5 TM-scores for predicted RNA structures against native structures.

    Args:
        solution_df (pd.DataFrame): DataFrame with native structures.
        submission_df (pd.DataFrame): DataFrame with predicted structures (5 models).
        usalign_executable_path (str): Path to the USalign executable.
        verbose (bool): If True, prints detailed scores for each target and model.

    Returns:
        float: The average of the highest TM-scores per target.
    """
    if not os.path.exists(usalign_executable_path):
        print(f"Error: USalign executable not found at '{usalign_executable_path}'. Cannot calculate scores.")
        return 0.0
    if not os.access(usalign_executable_path, os.X_OK):
        print(f"Error: USalign executable at '{usalign_executable_path}' is not executable. Please check permissions.")
        return 0.0

    # Standardize target_id column name
    solution_target_id_col = 'target_id'
    submission_target_id_col = 'target_id'

    if solution_target_id_col not in solution_df.columns:
        solution_df[solution_target_id_col] = solution_df['ID'].apply(lambda x: x.split('_')[0])
    if submission_target_id_col not in submission_df.columns:
        submission_df[submission_target_id_col] = submission_df['ID'].apply(lambda x: x.split('_')[0])

    all_target_best_tm_scores = []
    temp_dir = "_temp_pdb_scoring"
    os.makedirs(temp_dir, exist_ok=True)

    if verbose:
        print("\n--- Detailed Scoring ---")

    for target_id, group_native_full in solution_df.groupby(solution_target_id_col):
        # Filter submission data for the current target_id
        group_predicted_full = submission_df[submission_df[submission_target_id_col] == target_id]

        if group_predicted_full.empty:
            if verbose:
                print(f"\nTarget: {target_id}")
                print("  Status: No predictions found in submission data. Skipping.")
            all_target_best_tm_scores.append(0.0) # Append 0 for missing targets as per original behavior
            continue
        
        if verbose:
            print(f"\nTarget: {target_id}")

        native_pdb_path = os.path.join(temp_dir, f"{target_id}_native.pdb")
        num_native_atoms = write2pdb(group_native_full, 1, native_pdb_path) # Native uses xyz_id=1

        if num_native_atoms == 0:
            if verbose:
                print(f"  Status: Native PDB for {target_id} is empty or could not be written. TM-score: 0.0")
            all_target_best_tm_scores.append(0.0)
            if os.path.exists(native_pdb_path): os.remove(native_pdb_path)
            continue

        current_target_tm_scores = []
        for pred_model_idx in range(1, 6): # Models 1 to 5
            predicted_pdb_path = os.path.join(temp_dir, f"{target_id}_predicted_model_{pred_model_idx}.pdb")
            num_pred_atoms = write2pdb(group_predicted_full, pred_model_idx, predicted_pdb_path)
            
            tm_score_for_model = 0.0
            if num_pred_atoms == 0:
                if verbose:
                    print(f"  Model {pred_model_idx}: Predicted PDB empty. TM-score: 0.0")
            else:
                command = [usalign_executable_path, predicted_pdb_path, native_pdb_path, "-atom", " C1'"]
                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
                    if result.returncode == 0 and result.stdout:
                        tm_score_for_model = parse_tmscore_output(result.stdout)
                    elif verbose: # Print errors only in verbose mode if USalign fails
                        print(f"  Model {pred_model_idx}: USalign execution failed or no output.")
                        # print(f"    Return Code: {result.returncode}, Stdout: '{result.stdout[:100]}...', Stderr: '{result.stderr[:100]}...'")
                except subprocess.TimeoutExpired:
                    if verbose: print(f"  Model {pred_model_idx}: USalign timed out.")
                except ValueError as e:
                    if verbose: print(f"  Model {pred_model_idx}: Error parsing USalign output: {e}")
                except Exception as e:
                    if verbose: print(f"  Model {pred_model_idx}: Unexpected error: {e}")
            
            current_target_tm_scores.append(tm_score_for_model)
            if verbose:
                 print(f"  Model {pred_model_idx}: TM-score = {tm_score_for_model:.4f}")
            
            if os.path.exists(predicted_pdb_path): os.remove(predicted_pdb_path)
        
        if os.path.exists(native_pdb_path): os.remove(native_pdb_path)

        if current_target_tm_scores:
            best_tm_for_target = max(current_target_tm_scores)
            best_model_index = np.argmax(current_target_tm_scores) + 1 # 1-indexed
            all_target_best_tm_scores.append(best_tm_for_target)
            if verbose:
                print(f"  Best TM-score for {target_id}: {best_tm_for_target:.4f} (from Model {best_model_index})")
        else:
            all_target_best_tm_scores.append(0.0)
            if verbose:
                print(f"  Status: No valid TM-scores obtained for {target_id}.")
    
    # Cleanup temp_dir
    try:
        for f_name in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f_name))
        os.rmdir(temp_dir)
    except OSError as e:
        print(f"Warning: Could not completely remove temporary directory {temp_dir}: {e}")

    if not all_target_best_tm_scores:
        if verbose: print("\nNo TM-scores calculated for any target.")
        return 0.0
    
    final_average_score = sum(all_target_best_tm_scores) / len(all_target_best_tm_scores)
    if verbose:
        print(f"\n--- Scoring Summary ---")
        print(f"Number of targets processed: {len(all_target_best_tm_scores)}")
        print(f"Average Best-of-5 TM-score: {final_average_score:.4f}")
    return final_average_score



    import pandas as pd
    submission_df = pd.read_csv("/kaggle/working/validation.csv")

    solution_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv") # Or your local path
    
    os.system('cp //kaggle/input/usalign/USalign /kaggle/working/')
    os.system('sudo chmod u+x /kaggle/working//USalign')
    usalign_exe = "/kaggle/working//USalign" 

    final_score = score(solution_df, submission_df, usalign_exe, verbose=True)
    print(f"Competition Score (Average Best-of-5 TM-score): {final_score:.4f}")

