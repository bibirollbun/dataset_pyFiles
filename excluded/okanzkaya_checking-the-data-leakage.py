import pandas as pd
import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
import warnings

print("--- Geology Data Leakage Detective (Notebook Edition!) ---")
print("Let's see if there are any sneaky clues hidden in this data...")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in less")


# Where's the evidence? Point me to the competition's default CSV files.
COMPETITION_TRAIN_CSV_PATH = '/kaggle/input/geology-forecast-challenge-open/data/train.csv'
COMPETITION_TEST_CSV_PATH = '/kaggle/input/geology-forecast-challenge-open/data/test.csv'

# If I find anything, I'll make some notes (and plots!) here:
OUTPUT_REPORTS_DIR = './leakage_investigation_notebook_reports' # Directory for CSV reports
os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)

# For comparing the actual squiggly lines (input sequences):
# We'll focus on the part of the input that's guaranteed to be there in the test set.
INPUT_COLS_FOR_COMPARISON = [str(i) for i in range(-49, 1)] # From X=-49 to X=0

# How close is too close? For numerical sequences to be "the same" (MSE)
CONTENT_SIMILARITY_MSE_CUTOFF = 1e-9 # Pretty strict, for nearly identical numbers.

# To keep things snappy and not run out of memory, we'll work in batches.
TEST_PROCESSING_BATCH_SIZE = 100
TRAIN_PROCESSING_BATCH_SIZE = 1000

# How many examples should I flash on the screen? And how many plots?
MAX_LEAK_EXAMPLES_DISPLAY_ONSCREEN = 5
MAX_PLOTS_TO_GENERATE_PER_CATEGORY = 3

print("\n--- My Case Brief (Configuration) ---")
print(f"  Training Data: {COMPETITION_TRAIN_CSV_PATH}")
print(f"  Test Data: {COMPETITION_TEST_CSV_PATH}")
print(f"  My reports will go here: {OUTPUT_REPORTS_DIR} (plots will be inline)")
print(f"  Comparing content columns: {INPUT_COLS_FOR_COMPARISON[0]} to {INPUT_COLS_FOR_COMPARISON[-1]}")
print(f"  Similarity MSE Cutoff: {CONTENT_SIMILARITY_MSE_CUTOFF}")
print("-" * 70)


def load_the_dossier(file_path, file_nickname, columns_i_need=None):
    """Loads a CSV file into a pandas DataFrame with some checks."""
    print(f"\nOpening the '{file_nickname}' dossier from: {file_path}...")
    if not os.path.exists(file_path):
        print(f"!!! Uh oh! Can't find the '{file_nickname}' file at {file_path}. Case cold for now.")
        return None
    try:
        dossier_df = pd.read_csv(file_path)
        print(f"  '{file_nickname}' dossier opened. Contains {dossier_df.shape[0]} entries, {dossier_df.shape[1]} details each.")
        if columns_i_need:
            missing_details = [col for col in columns_i_need if col not in dossier_df.columns]
            if missing_details:
                print(f"!!! Hold on! The '{file_nickname}' dossier is missing some key details I need: {missing_details}")
                return None
        return dossier_df
    except Exception as e:
        print(f"!!! Trouble opening the '{file_nickname}' dossier from {file_path}: {e}")
        return None

def make_sequence_fingerprint(data_record):
    """Converts a sequence (row part) into something hashable for exact checks, handling NaNs."""
    # Using a distinct placeholder for NaN helps in exact matching for tuples.
    return tuple( "NAN_FINGERPRINT_PLACEHOLDER" if pd.isna(val) else val for val in data_record)

def plot_sequence_comparison_inline(train_seq, test_seq, train_id, test_id, title_prefix, plot_number):
    """Plots a comparison of one train and one test sequence inline."""
    plt.figure(figsize=(14, 5)) # Good size for notebook inline
    x_axis_vals = np.arange(len(INPUT_COLS_FOR_COMPARISON)) - (len(INPUT_COLS_FOR_COMPARISON) - 1)

    # Make sure we're plotting numbers, NaNs become 0 for the plot.
    train_seq_numeric = pd.to_numeric(pd.Series(np.asarray(train_seq).flatten()), errors='coerce').fillna(0)
    test_seq_numeric = pd.to_numeric(pd.Series(np.asarray(test_seq).flatten()), errors='coerce').fillna(0)

    plt.plot(x_axis_vals, train_seq_numeric, label=f'Train Sample (ID: {train_id})', marker='o', linestyle='--', alpha=0.8)
    plt.plot(x_axis_vals, test_seq_numeric, label=f'Test Sample (ID: {test_id})', marker='x', linestyle='-', alpha=0.8)
    
    full_title = f"{title_prefix} - Example {plot_number}:\nTrain ({train_id}) vs. Test ({test_id})"
    plt.title(full_title, fontsize=14)
    plt.xlabel(f"Input Position Relative to X=0 (Points {INPUT_COLS_FOR_COMPARISON[0]} to {INPUT_COLS_FOR_COMPARISON[-1]})", fontsize=12)
    plt.ylabel("Normalized Z Value", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show() # This makes the plot appear inline in the notebook
    # No file saving here, as per request. You can add it back if needed.


# We need all input columns from train initially to correctly slice the comparison window
required_train_cols_to_load = ['geology_id'] + [str(i) for i in range(-299, 1)]
required_test_cols_to_load = ['geology_id'] + INPUT_COLS_FOR_COMPARISON # Test only guarantees these

train_data_df_full = load_the_dossier(COMPETITION_TRAIN_CSV_PATH, "Competition Training Data", columns_i_need=required_train_cols_to_load)
test_data_df_full = load_the_dossier(COMPETITION_TEST_CSV_PATH, "Competition Test Data", columns_i_need=required_test_cols_to_load)

if train_data_df_full is None or test_data_df_full is None:
    print("\n--- Case Closed Early: Couldn't load all necessary data. Halting investigation. ---")
    # In a notebook, you might just stop here or raise an error
    # For this script, let's make it exit if data isn't loaded.
    # In a real notebook, you'd just not run subsequent cells.
    if __name__ == "__main__": # To prevent exit if imported
        sys.exit(1)
else:
    print("\nSuccessfully loaded both training and test dossiers. The game is afoot!")


if train_data_df_full is not None and test_data_df_full is not None:
    print("\n--- Phase 1: Checking for geology_id Doppelgangers ---")
    print("Are there any IDs that appear in both the good guys' (train) and bad guys' (test) files?")
    investigation_start_time_p1 = time.time()
    
    train_set_ids = set(train_data_df_full['geology_id'].unique())
    test_set_ids = set(test_data_df_full['geology_id'].unique())
    direct_id_matches = train_set_ids.intersection(test_set_ids)

    if direct_id_matches:
        print(f"  !!! ALERT! Found {len(direct_id_matches)} geology_id(s) present in BOTH training and test sets!")
        print(f"    Here are a few of them: {list(direct_id_matches)[:MAX_LEAK_EXAMPLES_DISPLAY_ONSCREEN]}")
        direct_leaks_report_df = pd.DataFrame(list(direct_id_matches), columns=['leaked_geology_id'])
        report_path_p1 = os.path.join(OUTPUT_REPORTS_DIR, 'report_phase1_direct_id_leaks.csv')
        direct_leaks_report_df.to_csv(report_path_p1, index=False)
        print(f"    A full list of these IDs is in your report: {report_path_p1}")

        print(f"\n    Let's look at up to {MAX_PLOTS_TO_GENERATE_PER_CATEGORY} examples of these direct ID leaks...")
        for i, leaked_id_val in enumerate(list(direct_id_matches)):
            if i >= MAX_PLOTS_TO_GENERATE_PER_CATEGORY:
                print(f"    (Stopping at {MAX_PLOTS_TO_GENERATE_PER_CATEGORY} plots for brevity...)")
                break
            train_sample_for_plot = train_data_df_full[train_data_df_full['geology_id'] == leaked_id_val].iloc[0]
            test_sample_for_plot = test_data_df_full[test_data_df_full['geology_id'] == leaked_id_val].iloc[0]
            
            # Extract the actual sequence data for INPUT_COLS_FOR_COMPARISON
            train_sequence_to_plot = train_sample_for_plot[INPUT_COLS_FOR_COMPARISON].values
            test_sequence_to_plot = test_sample_for_plot[INPUT_COLS_FOR_COMPARISON].values
            
            plot_sequence_comparison_inline(train_sequence_to_plot, test_sequence_to_plot,
                                     leaked_id_val, leaked_id_val, # Same ID
                                     "Direct ID Leak", plot_number=i+1)
    else:
        print("  Good news! No geology_ids are shared directly between the train and test sets. Clean start!")
    print(f"  Phase 1 duration: {time.time() - investigation_start_time_p1:.2f} seconds.")


train_content_df_slice = None
test_content_df_slice = None

if train_data_df_full is not None and test_data_df_full is not None:
    print("\n--- Preparing Data Slices for Content Scrutiny ---")
    try:
        # From the full train_data_df, we select only the geology_id and the specific input columns we want to compare
        train_content_df_slice = train_data_df_full[['geology_id'] + INPUT_COLS_FOR_COMPARISON].copy()
        for col_name_idx in INPUT_COLS_FOR_COMPARISON: # Use col_name_idx to avoid conflict
            train_content_df_slice[col_name_idx] = pd.to_numeric(train_content_df_slice[col_name_idx], errors='coerce')
        print(f"  Prepared training data slice for content comparison. Shape: {train_content_df_slice.shape}")

        # Test data already loaded with the correct columns, just ensure type
        test_content_df_slice = test_data_df_full[['geology_id'] + INPUT_COLS_FOR_COMPARISON].copy()
        for col_name_idx in INPUT_COLS_FOR_COMPARISON:
            test_content_df_slice[col_name_idx] = pd.to_numeric(test_content_df_slice[col_name_idx], errors='raise')
        print(f"  Prepared test data slice for content comparison. Shape: {test_content_df_slice.shape}")
    except Exception as e_prep_content:
        print(f"!!! Problem during data prep for content analysis: {e_prep_content}")
        print("    Skipping detailed content-based checks (Phases 2 & 3).")
        train_content_df_slice = None # Ensure it's None to skip later cells
        test_content_df_slice = None


if train_content_df_slice is not None and test_content_df_slice is not None:
    print("\n--- Phase 2: Hunting for Identical Input Sequences (Columns -49 to 0) ---")
    print("Are any test sequences exact carbon copies of training sequences, even with different IDs?")
    investigation_start_time_p2 = time.time()

    train_sequences_for_exact_check_records = train_content_df_slice[INPUT_COLS_FOR_COMPARISON].to_records(index=False)
    test_sequences_for_exact_check_records = test_content_df_slice[INPUT_COLS_FOR_COMPARISON].to_records(index=False)

    exact_content_matches_info_p2 = []
    
    # Store index of first occurrence in train_content_df_slice
    train_sequence_fingerprints_map = {make_sequence_fingerprint(seq): i for i, seq in enumerate(train_sequences_for_exact_check_records)}
    print(f"  Cataloged {len(train_sequence_fingerprints_map)} unique sequence fingerprints from the training data slice.")

    print("  Scanning test sequences for exact matches in the training catalog...")
    test_ids_with_exact_matches_found = set() # To track for plotting unique test_ids

    for i_test_seq, test_seq_rec_val in enumerate(test_sequences_for_exact_check_records):
        current_test_fingerprint_val = make_sequence_fingerprint(test_seq_rec_val)
        if current_test_fingerprint_val in train_sequence_fingerprints_map:
            test_sample_id_val = test_content_df_slice.iloc[i_test_seq]['geology_id']
            
            # Get the index of the first matching train sample from our map
            train_sample_slice_idx = train_sequence_fingerprints_map[current_test_fingerprint_val]
            train_sample_id_val = train_content_df_slice.iloc[train_sample_slice_idx]['geology_id']

            exact_content_matches_info_p2.append({
                'test_geology_id': test_sample_id_val,
                'example_train_geology_id': train_sample_id_val,
                'test_df_slice_index': i_test_seq, 
                'train_df_slice_index': train_sample_slice_idx
            })
            test_ids_with_exact_matches_found.add(test_sample_id_val)

        if (i_test_seq + 1) % (TEST_PROCESSING_BATCH_SIZE * 10) == 0 or (i_test_seq + 1) == len(test_sequences_for_exact_check_records):
            print(f"    Scanned {i_test_seq+1}/{len(test_sequences_for_exact_check_records)} test sequences for identical twins...")

    if exact_content_matches_info_p2:
        exact_matches_report_df_p2 = pd.DataFrame(exact_content_matches_info_p2).drop_duplicates(subset=['test_geology_id', 'example_train_geology_id'])
        print(f"  !!! ALERT! Found {exact_matches_report_df_p2['test_geology_id'].nunique()} unique test_geology_id(s) whose input sequence (-49 to 0) is an EXACT MATCH to at least one training sequence!")
        print(f"    Here are some of them (Test ID, Example Train ID):")
        print(exact_matches_report_df_p2[['test_geology_id', 'example_train_geology_id']].head(MAX_LEAK_EXAMPLES_DISPLAY_ONSCREEN).to_string(index=False))
        report_path_p2 = os.path.join(OUTPUT_REPORTS_DIR, 'report_phase2_exact_content_leaks.csv')
        exact_matches_report_df_p2.to_csv(report_path_p2, index=False)
        print(f"    A full list is in your report: {report_path_p2}")

        print(f"\n    Let's look at up to {MAX_PLOTS_TO_GENERATE_PER_CATEGORY} examples of these identical sequences...")
        plotted_exact_test_ids_p2 = set()
        plot_counter_p2 = 0
        for _, match_detail in exact_matches_report_df_p2.iterrows():
            if plot_counter_p2 >= MAX_PLOTS_TO_GENERATE_PER_CATEGORY:
                print(f"    (Stopping at {MAX_PLOTS_TO_GENERATE_PER_CATEGORY} plots for brevity...)")
                break
            if match_detail['test_geology_id'] not in plotted_exact_test_ids_p2:
                train_seq_for_plot_p2 = train_content_df_slice.iloc[match_detail['train_df_slice_index']][INPUT_COLS_FOR_COMPARISON].values
                test_seq_for_plot_p2 = test_content_df_slice.iloc[match_detail['test_df_slice_index']][INPUT_COLS_FOR_COMPARISON].values
                plot_sequence_comparison_inline(train_seq_for_plot_p2, test_seq_for_plot_p2,
                                         match_detail['example_train_geology_id'], match_detail['test_geology_id'],
                                         "Exact Content Match", plot_number=plot_counter_p2+1)
                plotted_exact_test_ids_p2.add(match_detail['test_geology_id'])
                plot_counter_p2 +=1
    else:
        print("  Phew! No test input sequences (-49 to 0) were found to be identical to any training sequences.")
    print(f"  Phase 2 duration: {time.time() - investigation_start_time_p2:.2f} seconds.")


if train_content_df_slice is not None and test_content_df_slice is not None:
    print(f"\n--- Phase 3: Looking for Close Cousins (Highly Similar Input Sequences, MSE < {CONTENT_SIMILARITY_MSE_CUTOFF}) ---")
    print(f"  This part can take a while, grab a coffee! Comparing columns: {INPUT_COLS_FOR_COMPARISON[0]} to {INPUT_COLS_FOR_COMPARISON[-1]}")
    investigation_start_time_p3 = time.time()

    # Filter out rows with ANY NaNs in the comparison columns from training data to avoid NaN MSEs
    train_content_for_mse_df_no_nan = train_content_df_slice.dropna(subset=INPUT_COLS_FOR_COMPARISON)
    if len(train_content_for_mse_df_no_nan) < len(train_content_df_slice):
        print(f"    Note for MSE check: Excluded {len(train_content_df_slice) - len(train_content_for_mse_df_no_nan)} training rows that had NaNs in the -49 to 0 window.")
    
    train_np_sequences_for_mse = train_content_for_mse_df_no_nan[INPUT_COLS_FOR_COMPARISON].astype(float).values
    train_ids_for_mse_lookup = train_content_for_mse_df_no_nan['geology_id'].values
    # Store indices relative to train_content_for_mse_df_no_nan for easy plotting later
    train_indices_in_filtered_df = np.arange(len(train_content_for_mse_df_no_nan))


    test_np_sequences_for_mse = test_content_df_slice[INPUT_COLS_FOR_COMPARISON].astype(float).values # Test data should be clean here
    test_ids_all_mse = test_content_df_slice['geology_id'].values
    # Store original indices from test_content_df_slice for plotting
    test_indices_in_original_slice = test_content_df_slice.index.values


    suspiciously_similar_pairs_info_p3 = []
    num_total_test_samples_mse = len(test_np_sequences_for_mse)
    num_total_train_samples_no_nan_mse = len(train_np_sequences_for_mse)

    for i_test_batch_start_mse in range(0, num_total_test_samples_mse, TEST_PROCESSING_BATCH_SIZE):
        current_test_batch_data = test_np_sequences_for_mse[i_test_batch_start_mse : i_test_batch_start_mse + TEST_PROCESSING_BATCH_SIZE]
        current_test_batch_ids = test_ids_all_mse[i_test_batch_start_mse : i_test_batch_start_mse + TEST_PROCESSING_BATCH_SIZE]
        current_test_batch_original_indices = test_indices_in_original_slice[i_test_batch_start_mse : i_test_batch_start_mse + TEST_PROCESSING_BATCH_SIZE]
        
        batch_num_mse = i_test_batch_start_mse // TEST_PROCESSING_BATCH_SIZE + 1
        total_batches_mse = (num_total_test_samples_mse + TEST_PROCESSING_BATCH_SIZE - 1) // TEST_PROCESSING_BATCH_SIZE
        print(f"  Analyzing test batch {batch_num_mse}/{total_batches_mse} for similarity (MSE)...")

        for i_train_chunk_start_mse in range(0, num_total_train_samples_no_nan_mse, TRAIN_PROCESSING_BATCH_SIZE):
            current_train_chunk_data = train_np_sequences_for_mse[i_train_chunk_start_mse : i_train_chunk_start_mse + TRAIN_PROCESSING_BATCH_SIZE]
            current_train_chunk_ids = train_ids_for_mse_lookup[i_train_chunk_start_mse : i_train_chunk_start_mse + TRAIN_PROCESSING_BATCH_SIZE]
            current_train_chunk_filtered_indices = train_indices_in_filtered_df[i_train_chunk_start_mse : i_train_chunk_start_mse + TRAIN_PROCESSING_BATCH_SIZE]
            
            mse_values_matrix = np.mean((current_test_batch_data[:, np.newaxis, :] - current_train_chunk_data[np.newaxis, :, :])**2, axis=2)
            
            # This should now have fewer warnings due to pre-filtering NaNs from train_np_sequences_for_mse
            found_test_batch_idx_mse, found_train_chunk_idx_mse = np.where(mse_values_matrix < CONTENT_SIMILARITY_MSE_CUTOFF)
            
            for t_idx_mse, tr_idx_mse in zip(found_test_batch_idx_mse, found_train_chunk_idx_mse):
                suspiciously_similar_pairs_info_p3.append({
                    'test_geology_id': current_test_batch_ids[t_idx_mse],
                    'train_geology_id': current_train_chunk_ids[tr_idx_mse],
                    'mse': mse_values_matrix[t_idx_mse, tr_idx_mse],
                    'test_df_slice_index': current_test_batch_original_indices[t_idx_mse], # Original index in test_content_df_slice
                    'train_filtered_df_index': current_train_chunk_filtered_indices[tr_idx_mse] # Index in train_content_for_mse_df_no_nan
                })
        
        if suspiciously_similar_pairs_info_p3 and (batch_num_mse % 5 == 0 or batch_num_mse == total_batches_mse) :
             print(f"    So far, found {len(suspiciously_similar_pairs_info_p3)} suspiciously similar pairs (processed {i_test_batch_start_mse + len(current_test_batch_data)} test samples)...")

    if suspiciously_similar_pairs_info_p3:
        similar_pairs_report_df_p3 = pd.DataFrame(suspiciously_similar_pairs_info_p3).sort_values(by='mse')
        num_unique_test_with_cousins_p3 = similar_pairs_report_df_p3['test_geology_id'].nunique()
        print(f"  !!! CAUTION! Found {num_unique_test_with_cousins_p3} unique test_geology_id(s) whose input sequence is VERY SIMILAR (MSE < {CONTENT_SIMILARITY_MSE_CUTOFF}) to one or more training sequences!")
        print(f"    Total (test_id, train_id) pairs with high similarity: {len(similar_pairs_report_df_p3)}")
        print(f"    Here are some of the top suspects (lowest MSE):")
        print(similar_pairs_report_df_p3[['test_geology_id', 'train_geology_id', 'mse']].head(MAX_LEAK_EXAMPLES_DISPLAY_ONSCREEN).to_string(index=False))
        report_path_p3 = os.path.join(OUTPUT_REPORTS_DIR, 'report_phase3_highly_similar_content_leaks.csv')
        similar_pairs_report_df_p3.to_csv(report_path_p3, index=False)
        print(f"    A full list of these highly similar pairs is in your report: {report_path_p3}")

        print(f"\n    Visualizing up to {MAX_PLOTS_TO_GENERATE_PER_CATEGORY} examples of these close cousins...")
        plotted_similar_test_ids_p3 = set()
        plots_made_p3 = 0
        for _, match_detail_p3 in similar_pairs_report_df_p3.iterrows(): # Iterate through sorted (lowest MSE first)
            if plots_made_p3 >= MAX_PLOTS_TO_GENERATE_PER_CATEGORY:
                print(f"    (Stopping at {MAX_PLOTS_TO_GENERATE_PER_CATEGORY} plots for brevity...)")
                break
            if match_detail_p3['test_geology_id'] not in plotted_similar_test_ids_p3:
                # Retrieve sequences using the correct indices
                train_seq_for_plot_p3 = train_content_for_mse_df_no_nan.iloc[match_detail_p3['train_filtered_df_index']][INPUT_COLS_FOR_COMPARISON].values
                test_seq_for_plot_p3 = test_content_df_slice.loc[match_detail_p3['test_df_slice_index']][INPUT_COLS_FOR_COMPARISON].values
                
                plot_sequence_comparison_inline(train_seq_for_plot_p3, test_seq_for_plot_p3,
                                         match_detail_p3['train_geology_id'], match_detail_p3['test_geology_id'],
                                         f"Similar Content (MSE {match_detail_p3['mse']:.2e})",
                                         plot_number=plots_made_p3+1)
                plotted_similar_test_ids_p3.add(match_detail_p3['test_geology_id'])
                plots_made_p3 +=1
    else:
        print("  Looks clean! No test input sequences found to be overly similar to training sequences (based on MSE).")
    print(f"  Phase 3 duration: {time.time() - investigation_start_time_p3:.2f} seconds.")

else: # This 'else' corresponds to 'if train_content_df_slice is not None ...'
    print("\nSkipping content-based leakage checks (Phases 2 & 3) due to earlier data preparation issues.")

