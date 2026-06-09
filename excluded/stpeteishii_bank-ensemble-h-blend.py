import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MULTI-TARGET H-BLEND ENSEMBLE PIPELINE")
print("=" * 70)


def load_all_targets_from_first_model(model_paths):
    """
    Load all target columns from the first model's submission file.
    """
    try:
        first_model_path = list(model_paths.values())[0]
        df = pd.read_csv(first_model_path)
        print(f"First model columns: {list(df.columns)}")

        # Treat all columns except 'id' as targets
        if 'id' in df.columns:
            target_columns = [col for col in df.columns if col != 'id']
        else:
            target_columns = df.columns.tolist()

        print(f"Detected target columns: {target_columns}")
        return target_columns

    except Exception as e:
        print(f"Error reading first model: {e}")
        return None


def load_single_target_predictions(model_paths, target_col):
    """
    Load predictions for a single target from each model.
    """
    print(f"\nLoading predictions for target: {target_col}")

    predictions = {}

    for model_name, path in model_paths.items():
        try:
            df = pd.read_csv(path)

            # Check if the specified target column exists
            if target_col not in df.columns:
                print(f"âœ— Target column '{target_col}' not found in {model_name}")
                return None

            # Keep only id and the target column
            df = df[['id', target_col]]
            # Rename target column to the model name
            df.rename(columns={target_col: model_name}, inplace=True)
            predictions[model_name] = df
            print(f"âœ“ Loaded {model_name} predictions for {target_col}")

        except Exception as e:
            print(f"âœ— Error loading {model_name}: {e}")
            return None

    return predictions


def create_model_configuration(model_names):
    """
    Create configuration for models.
    """
    print("\nMODEL CONFIGURATION SETUP")
    print("=" * 40)

    path = './'
    file_short_names = model_names
    model_full_names = model_names

    print("Model Configuration:")
    for short, full in zip(file_short_names, model_full_names):
        print(f"  {short} -> {full}")

    return path, file_short_names, model_full_names


def calculate_weights_from_lb_scores(lb_scores, smaller_better=True):
    """
    Calculate weights from leaderboard scores.
    If smaller_better is True, invert scores so that smaller scores get larger weights.
    """
    if smaller_better:
        epsilon = 1e-10
        inverted_scores = [1 / (score + epsilon) for score in lb_scores]
        total = sum(inverted_scores)
        weights = [score / total for score in inverted_scores]
    else:
        total = sum(lb_scores)
        weights = [score / total for score in lb_scores]

    return weights


def create_optimal_params(strategy='balanced', lb_scores=None, smaller_better=False, model_names=None, target_col='target'):
    """
    Create optimal parameters for the H-Blend ensemble.
    """
    print(f"\nCreating {strategy} parameter configuration for {target_col}...")

    if lb_scores is not None and len(lb_scores) == len(model_names):
        base_weights = calculate_weights_from_lb_scores(lb_scores, smaller_better)
        print(f"Calculated weights from LB scores: {[f'{w:.4f}' for w in base_weights]}")
    else:
        # Equal weights
        base_weights = [1.0 / len(model_names)] * len(model_names)
        print("Using equal weights for all models")

    strategies = {
        'balanced': {
            'desc': 0.6,
            'asc': 0.4,
            'subwts': [0.0] * len(model_names),
            'base_weights': base_weights
        }
    }

    config = strategies.get(strategy, strategies['balanced'])

    params = {
        'path': './submission_',
        'id': 'id',
        'target': target_col,
        'desc': config['desc'],
        'asc': config['asc'],
        'subwts': config['subwts'],
        'subm': [
            {'name': model_names[i], 'weight': config['base_weights'][i]}
            for i in range(len(model_names))
        ]
    }

    print(f"Ensemble strategy: {strategy}")
    print(f"Desc weight: {config['desc']}, Asc weight: {config['asc']}")

    return params


def prepare_submission_files(predictions, target_col='target'):
    """
    Prepare individual submission files for H-Blend.
    """
    print(f"Preparing submission files for {target_col}...")

    for model_name, df in predictions.items():
        # Revert the column name back to the original target name
        df_export = df.copy()
        df_export.rename(columns={model_name: target_col}, inplace=True)

        output_file = f'submission_{model_name}_{target_col}.csv'
        df_export.to_csv(output_file, index=False)
        print(f"âœ“ Created {output_file}")

    return True


def h_blend(path, fs_names, params):
    """
    H-Blend function: Rank-based dynamic weighted ensemble.
    """
    dk = params

    def da(dk, sorting_direction):
        def read_subm(dk, i):
            tnm = dk["subm"][i]["name"]
            target_col = dk["target"]
            FiN = f"{dk['path']}{tnm}_{target_col}.csv"
            df = pd.read_csv(FiN)
            return df

        # Read all submission files
        dfs_subm = [read_subm(dk, i) for i in range(len(dk["subm"]))]

        # Merge all dataframes
        df_subms = dfs_subm[0]
        for i in range(1, len(dfs_subm)):
            df_subms = pd.merge(df_subms, dfs_subm[i], on=[dk['id']])

        cols = [col for col in df_subms.columns if col != dk['id']]
        short_name_cols = cols.copy()
        corrects = dk["subwts"]
        weights = [subm['weight'] for subm in dk["subm"]]

        def alls(x, sd=sorting_direction, cs=cols):
            reverse = True if sd == 'desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [t[0] for t in sorted(tes, key=lambda item: item[1], reverse=reverse)]
            return subms_sorted

        def correct(x, cs=cols, w=weights, cw=corrects):
            ic = [x['alls'].index(c) for c in short_name_cols]
            cS = [x[cols[j]] * (w[j] + cw[ic[j]]) for j in range(len(cols))]
            return sum(cS)

        df_subms['alls'] = df_subms.apply(lambda x: alls(x), axis=1)
        df_subms[dk["target"]] = df_subms.apply(lambda x: correct(x), axis=1)

        return df_subms[[dk['id'], dk["target"]]]

    # Use the first model's file as a template
    target_col = dk["target"]
    sample_subm = pd.read_csv(f'./submission_{fs_names[0]}_{target_col}.csv')[["id"]]

    def ensemble_da(dk, submission=sample_subm):
        _id, target, d, a = dk['id'], dk['target'], dk['desc'], dk['asc']

        print(f"  Processing Descending Direction (Weight: {d:.2f})...")
        dfs = da(dk, 'desc')
        dfD = dfs[[_id, target]]

        print(f"  Processing Ascending Direction (Weight: {a:.2f})...")
        dfs = da(dk, 'asc')
        dfA = dfs[[_id, target]]

        # Merge with submission template
        submission = pd.merge(submission, dfD, on=_id, how='left')
        submission = pd.merge(submission, dfA, on=_id, how='left')
        submission[target] = submission[target + '_x'] * d + a * submission[target + '_y']

        # Final submission
        submission = submission[[_id, target]]

        return submission

    result = ensemble_da(dk)
    return result


def analyze_predictions(predictions, target_col='target'):
    """
    Analyze individual model predictions for basic statistics.
    """
    print(f"\nAnalysis for {target_col}:")
    print("-" * 30)

    stats_data = []
    for model_name, df in predictions.items():
        pred_values = df[model_name].values
        stats = {
            'Model': model_name,
            'Count': len(pred_values),
            'Mean': np.mean(pred_values),
            'Std': np.std(pred_values),
            'Min': np.min(pred_values),
            'Max': np.max(pred_values)
        }
        stats_data.append(stats)

    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False, float_format='{:,.4f}'.format))

    return stats_df


def process_single_target(model_paths, target_col, lb_scores=None, strategy='balanced', smaller_better=True):
    """
    Process a single target end-to-end.
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING TARGET: {target_col}")
    print(f"{'='*60}")

    # Step 1: Load predictions for this target
    predictions = load_single_target_predictions(model_paths, target_col)
    if predictions is None:
        print(f"Failed to load predictions for {target_col}. Skipping.")
        return None

    # Step 2: Analyze predictions
    analyze_predictions(predictions, target_col)

    # Step 3: Prepare submission files
    prepare_submission_files(predictions, target_col)

    # Step 4: Setup configuration
    model_names = list(model_paths.keys())
    path, file_short_names, model_full_names = create_model_configuration(model_names)

    # Step 5: Create parameters
    params = create_optimal_params(
        strategy=strategy,
        lb_scores=lb_scores,
        smaller_better=smaller_better,
        model_names=model_names,
        target_col=target_col
    )

    # Step 6: Run H-Blend
    try:
        result = h_blend(path, file_short_names, params)

        # Save final result for this target
        output_filename = f'submission_{target_col}.csv'
        result.to_csv(output_filename, index=False)
        print(f"âœ“ H-Blend completed for {target_col}!")
        print(f"  Saved: {output_filename}")

        return result

    except Exception as e:
        print(f"âœ— Error in H-Blend for {target_col}: {str(e)}")
        return None


def combine_all_targets_results(target_columns, final_output_path='submission_final.csv'):
    """
    Combine results for all targets into a single submission file.
    """
    print(f"\nCombining results for all targets...")

    # Load the first target file
    first_target = target_columns[0]
    try:
        combined_df = pd.read_csv(f'submission_{first_target}.csv')
        print(f"Loaded {first_target}: {combined_df.shape}")
    except Exception:
        print(f"Failed to load {first_target}")
        return None

    # Merge the other targets
    for target_col in target_columns[1:]:
        try:
            target_df = pd.read_csv(f'submission_{target_col}.csv')
            combined_df = pd.merge(combined_df, target_df, on='id', how='left')
            print(f"Added {target_col}: {combined_df.shape}")
        except Exception:
            print(f"Failed to load {target_col}")
            continue

    # Save the final submission file
    combined_df.to_csv(final_output_path, index=False)
    print(f"âœ“ Final combined submission saved: {final_output_path}")
    print(f"  Final shape: {combined_df.shape}")
    print(f"  Columns: {list(combined_df.columns)}")

    return combined_df


# ===== MAIN PIPELINE =====

def run_multi_target_h_blend(model_paths, lb_scores=None, strategy='balanced', smaller_better=True):
    """
    Main pipeline to process all targets.
    """
    print("Starting Multi-Target H-Blend Pipeline...")

    # Step 1: Detect all target columns
    target_columns = load_all_targets_from_first_model(model_paths)
    if target_columns is None:
        print("Failed to detect target columns. Exiting.")
        return None

    print(f"\nFound {len(target_columns)} targets: {target_columns}")

    # Step 2: Process each target
    all_results = {}

    for target_col in target_columns:
        result = process_single_target(
            model_paths=model_paths,
            target_col=target_col,
            lb_scores=lb_scores,
            strategy=strategy,
            smaller_better=smaller_better
        )

        if result is not None:
            all_results[target_col] = result
            print(f"âœ… Successfully processed {target_col}")
        else:
            print(f"â�Œ Failed to process {target_col}")

    # Step 3: Combine all results
    if all_results:
        final_combined = combine_all_targets_results(list(all_results.keys()))
        return final_combined
    else:
        print("No targets were successfully processed.")
        return None


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    # Set model paths
    model_paths = {
        '001': '/kaggle/input/30-august-2025-ps-s5e8/submission 0.97772.csv',
        '002': '/kaggle/input/30-august-2025-ps-s5e8/submission 0.97771.csv',
        '003': '/kaggle/input/30-august-2025-ps-s5e8/submission 0.97770.csv',
    }

    # Leaderboard scores (optional)
    lb_scores = [0.97772, 0.97771, 0.97770]  # example

    # Run the main pipeline
    final_result = run_multi_target_h_blend(
        model_paths=model_paths,
        lb_scores=lb_scores,
        strategy='balanced',
        smaller_better=False
    )

    print(f"\n" + "=" * 70)
    print("MULTI-TARGET H-BLEND PIPELINE COMPLETE")
    print("=" * 70)

    if final_result is not None:
        print("âœ… All targets processed successfully!")
        print(f"ğŸ“Š Final results shape: {final_result.shape}")
        print(f"ğŸ“Š Final columns: {list(final_result.columns)}")

        # Show basic statistics for each target
        for target_col in final_result.columns:
            if target_col != 'id':
                values = final_result[target_col]
                print(f"ğŸ“ˆ {target_col}: {values.min():.4f} - {values.max():.4f} (mean: {values.mean():.4f})")
    else:
        print("â�Œ Pipeline failed to create final submission file")


