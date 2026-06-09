%%writefile alculate_arc_score.py
import json
import argparse
import sys

def load_json(filepath):
    """Loads a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}. Check for syntax errors.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

def validate_submission_format(submission_data, task_id, num_expected_outputs):
    """Checks if the submission format for a specific task is correct."""
    if task_id not in submission_data:
        print(f"Warning: Task ID '{task_id}' missing in submission file.", file=sys.stderr)
        return False # Task missing, contributes 0 correct predictions

    predictions = submission_data[task_id]

    if not isinstance(predictions, list):
        print(f"Error: Entry for task ID '{task_id}' in submission is not a list.", file=sys.stderr)
        return None # Indicates fatal format error

    if len(predictions) != num_expected_outputs:
        print(f"Error: Task ID '{task_id}' has {len(predictions)} predictions in submission, "
              f"but expected {num_expected_outputs} based on solution file.", file=sys.stderr)
        return None # Indicates fatal format error

    for i, pred_pair in enumerate(predictions):
        if not isinstance(pred_pair, dict):
            print(f"Error: Prediction item {i} for task ID '{task_id}' is not a dictionary.", file=sys.stderr)
            return None
        if "attempt_1" not in pred_pair:
            print(f"Error: 'attempt_1' missing in prediction item {i} for task ID '{task_id}'.", file=sys.stderr)
            return None
        if "attempt_2" not in pred_pair:
            print(f"Error: 'attempt_2' missing in prediction item {i} for task ID '{task_id}'.", file=sys.stderr)
            return None
        # Basic grid validation (list of lists) - enhance if needed for stricter checks
        attempt1_val = pred_pair.get('attempt_1') # Use get for safety
        attempt2_val = pred_pair.get('attempt_2')
        if not isinstance(attempt1_val, list) or \
           not all(isinstance(row, list) for row in attempt1_val):
             print(f"Warning: 'attempt_1' for task '{task_id}', output {i} is not a list of lists.", file=sys.stderr)
        if not isinstance(attempt2_val, list) or \
           not all(isinstance(row, list) for row in attempt2_val):
             print(f"Warning: 'attempt_2' for task '{task_id}', output {i} is not a list of lists.", file=sys.stderr)

    return True # Format looks correct for this task

def calculate_arc_score(submission_filepath, solution_filepath, debug_task_id=None):
    """
    Calculates the ARC competition score based on the evaluation rules.

    Args:
        submission_filepath (str): Path to the submission JSON file.
                                   Expected format: {"task_id": [{"attempt_1": grid, "attempt_2": grid}, ...], ...}
        solution_filepath (str): Path to the ground truth/solution JSON file.
                                 Expected format: {"task_id": [solution_grid_1, solution_grid_2,...], ...}
                                 The order of solution grids must match the order of test inputs
                                 and the order of predictions in the submission file.
        debug_task_id (str, optional): If provided, print detailed comparison info for this task ID.

    Returns:
        float: The calculated competition score (0.0 to 1.0).
               Returns -1.0 if fatal errors occur during validation.
    """
    print(f"Loading submission file: {submission_filepath}")
    submission_data = load_json(submission_filepath)
    print(f"Loading solution file: {solution_filepath}")
    solution_data = load_json(solution_filepath)

    total_test_outputs = 0
    total_correct_predictions = 0
    fatal_error = False

    # Iterate through the ground truth tasks to define the scope
    for task_id, ground_truth_grids in solution_data.items():
        if not isinstance(ground_truth_grids, list):
             print(f"Error: Ground truth value for task ID '{task_id}' in solution file is not a list of grids.", file=sys.stderr)
             fatal_error = True
             continue # Skip this task if solution format is wrong

        num_expected_outputs = len(ground_truth_grids)
        total_test_outputs += num_expected_outputs

        # Validate the format for this task in the submission
        validation_result = validate_submission_format(submission_data, task_id, num_expected_outputs)

        if validation_result is None: # Fatal format error detected in submission
            fatal_error = True
            continue # Skip scoring this task
        elif not validation_result: # Task missing in submission
            print(f"Info: Scoring 0 for {num_expected_outputs} outputs of missing task '{task_id}'.")
            continue # Score 0 for all outputs of this task

        # If format is valid, proceed with scoring
        predictions = submission_data[task_id]
        for i in range(num_expected_outputs):
            try:
                # --- REVERTED TO ORIGINAL LOGIC ---
                # Directly access the ground truth grid from the list
                ground_truth_grid = ground_truth_grids[i]
                # --- END OF REVERTED LOGIC ---

                # Add a check that the ground truth grid is actually a list (basic grid format check)
                if not isinstance(ground_truth_grid, list):
                     print(f"Error: Ground truth grid {i} for task ID '{task_id}' is not a list (expected grid format).", file=sys.stderr)
                     fatal_error = True
                     break # Stop processing this task's outputs

                # Check if all elements in the ground truth grid are lists (rows)
                if not all(isinstance(row, list) for row in ground_truth_grid):
                     print(f"Error: Ground truth grid {i} for task ID '{task_id}' is not a list of lists (expected grid format).", file=sys.stderr)
                     fatal_error = True
                     break # Stop processing this task's outputs

                prediction_pair = predictions[i] # Already validated structure
                attempt_1 = prediction_pair['attempt_1']
                attempt_2 = prediction_pair['attempt_2']

                # --- START DEBUG BLOCK ---
                # Only print for the task you are investigating (if specified)
                if debug_task_id is not None and task_id == debug_task_id:
                     print(f"\n--- Debugging Task: {task_id}, Output Index: {i} ---")
                     try:
                         # Basic structure check before detailed printing
                         is_attempt1_grid = isinstance(attempt_1, list) and all(isinstance(r, list) for r in attempt_1)
                         is_attempt2_grid = isinstance(attempt_2, list) and all(isinstance(r, list) for r in attempt_2)
                         is_gt_grid = isinstance(ground_truth_grid, list) and all(isinstance(r, list) for r in ground_truth_grid)

                         if not is_attempt1_grid:
                             print(f"DEBUG: attempt_1 is not a valid grid structure: {attempt_1}")
                         if not is_attempt2_grid:
                             print(f"DEBUG: attempt_2 is not a valid grid structure: {attempt_2}")
                         if not is_gt_grid:
                             print(f"DEBUG: ground_truth_grid is not a valid grid structure: {ground_truth_grid}")

                         # Proceed with detailed comparison only if structures seem valid
                         if is_attempt1_grid and is_gt_grid:
                             a1_dims = f"{len(attempt_1)}x{len(attempt_1[0]) if len(attempt_1) > 0 else 0}"
                             gt_dims = f"{len(ground_truth_grid)}x{len(ground_truth_grid[0]) if len(ground_truth_grid) > 0 else 0}"
                             print(f"DEBUG: Attempt 1 Dimensions: {a1_dims}")
                             print(f"DEBUG: Ground Truth Dimensions: {gt_dims}")

                             # Print the actual comparison result for attempt 1
                             comparison1_result = (attempt_1 == ground_truth_grid)
                             print(f"DEBUG: Comparing Attempt 1 to Ground Truth... Result: {comparison1_result}")
                             if not comparison1_result and a1_dims == gt_dims: # Only search for diff if dims match
                                 # If they look the same but aren't, find the first differing row/element
                                for row_idx, (row_a, row_g) in enumerate(zip(attempt_1, ground_truth_grid)):
                                    if row_a != row_g:
                                        print(f"DEBUG: First differing row in Attempt 1 at index: {row_idx}")
                                        print(f"DEBUG: Attempt 1 Row : {row_a}")
                                        print(f"DEBUG: Ground Truth Row: {row_g}")
                                        # Find first differing element in the row (check lengths first)
                                        if len(row_a) == len(row_g):
                                            for col_idx, (val_a, val_g) in enumerate(zip(row_a, row_g)):
                                                if val_a != val_g:
                                                    print(f"DEBUG: First differing element at col {col_idx}: Attempt={val_a} (type {type(val_a)}), GT={val_g} (type {type(val_g)})")
                                                    break # Stop after first difference in row
                                        else:
                                             print(f"DEBUG: Row lengths differ: Attempt={len(row_a)}, GT={len(row_g)}")
                                        break # Stop after first differing row
                             elif not comparison1_result and a1_dims != gt_dims:
                                 print("DEBUG: Cannot compare elements because dimensions differ.")


                         if is_attempt2_grid and is_gt_grid:
                             a2_dims = f"{len(attempt_2)}x{len(attempt_2[0]) if len(attempt_2) > 0 else 0}"
                             gt_dims = f"{len(ground_truth_grid)}x{len(ground_truth_grid[0]) if len(ground_truth_grid) > 0 else 0}"
                             print(f"DEBUG: Attempt 2 Dimensions: {a2_dims}") # Re-print GT dims for clarity

                             # Print the actual comparison result for attempt 2
                             comparison2_result = (attempt_2 == ground_truth_grid)
                             print(f"DEBUG: Comparing Attempt 2 to Ground Truth... Result: {comparison2_result}")
                             # (Could add detailed diff finding for attempt 2 as well if needed)

                         # Optionally print raw content if structures are invalid or for extra checks
                         # print(f"DEBUG: Attempt 1 Data: {attempt_1}")
                         # print(f"DEBUG: Attempt 2 Data: {attempt_2}")
                         # print(f"DEBUG: Ground Truth Data: {ground_truth_grid}")

                     except Exception as e:
                         print(f"DEBUG: Error during debug printing for {task_id}, output {i}: {e}")

                     print("--- End Debug ---")
                # --- END DEBUG BLOCK ---

                # Score calculation: Check if either attempt matches
                is_correct = False
                if attempt_1 == ground_truth_grid:
                    is_correct = True
                    if debug_task_id is not None and task_id == debug_task_id:
                        print(f"*** Match found for Task {task_id} Output {i} (Attempt 1) ***")
                elif attempt_2 == ground_truth_grid:
                     is_correct = True
                     if debug_task_id is not None and task_id == debug_task_id:
                        print(f"*** Match found for Task {task_id} Output {i} (Attempt 2) ***")

                if is_correct:
                    total_correct_predictions += 1

            except IndexError:
                 print(f"Error: Index out of bounds accessing ground truth or predictions for task {task_id}, index {i}.", file=sys.stderr)
                 fatal_error = True
                 break # Stop processing this task
            except Exception as e:
                 print(f"Error: Unexpected error processing task {task_id}, output {i}: {e}", file=sys.stderr)
                 fatal_error = True
                 break # Stop processing this task


        # If a fatal error occurred while processing this task's outputs, signal overall failure
        if fatal_error and task_id in solution_data:
            print(f"Info: Stopped scoring task '{task_id}' due to errors.")
            # fatal_error is already True, score will be -1.0


    if fatal_error:
        print("\nFatal errors encountered during processing. Cannot reliably calculate score.", file=sys.stderr)
        return -1.0

    if total_test_outputs == 0:
        print("\nWarning: No test outputs found or processed in the solution file.", file=sys.stderr)
        return 0.0

    final_score = total_correct_predictions / total_test_outputs
    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ARC competition score with optional debugging.")
    parser.add_argument("submission_file", help="Path to the submission JSON file.")
    parser.add_argument("solution_file",
                        help="Path to the ground truth (solution) JSON file (format: {'task_id': [grid1, grid2,...]}).")
    parser.add_argument("-d", "--debug", metavar="TASK_ID",
                        help="Optional: Task ID to print detailed debugging information for.", default=None)
    args = parser.parse_args()

    score = calculate_arc_score(args.submission_file, args.solution_file, debug_task_id=args.debug)

    if score >= 0.0:
        print("\n--- Scoring Summary ---")
        # Recalculate total_test_outputs cleanly for reporting
        # Need to handle potential errors loading/parsing solution file again
        total_outputs_report_val = 0
        correct_preds_report_val = 0
        try:
            solution_data_report = load_json(args.solution_file)
            # Filter out tasks with non-list values before summing lengths
            valid_tasks = {k: v for k, v in solution_data_report.items() if isinstance(v, list)}
            total_outputs_report_val = sum(len(v) for v in valid_tasks.values())
            if total_outputs_report_val > 0:
                 # Calculate correct predictions based on the final score and total outputs
                 correct_preds_report_val = round(score * total_outputs_report_val)
            else:
                 correct_preds_report_val = 0
            total_outputs_report_str = str(total_outputs_report_val)
            correct_preds_report_str = str(correct_preds_report_val)

        except Exception as e:
            print(f"\nWarning: Could not fully parse solution file for reporting: {e}", file=sys.stderr)
            total_outputs_report_str = "N/A (Error reading solution)"
            correct_preds_report_str = "N/A"


        print(f"Total number of test outputs considered: {total_outputs_report_str}")
        print(f"Total correct predictions (either attempt matched): {correct_preds_report_str}")
        print(f"Final Score: {score:.6f}") # Format score
    else:
        print("\nScore calculation aborted due to errors.", file=sys.stderr)
        sys.exit(1) # Exit with error code if score calculation failed


!python alculate_arc_score.py /kaggle/input/arc2025-demo/submission.json /kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json -d 136b0064

