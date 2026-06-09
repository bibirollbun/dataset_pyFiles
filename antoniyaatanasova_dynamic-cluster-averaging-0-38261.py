import pandas as pd
from collections import defaultdict
import re
import itertools
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

def analyze_correlation_all_ranks(file_info):
    """
    Reads submission files and creates similarity matrices for the 1st, 2nd, and 3rd rank predictions.
    Returns a dictionary of similarity DataFrames, one for each rank.
    """
    all_predictions_df = pd.DataFrame()
    model_names = list(file_info.keys())

    print("\n--- Analyzing Model Correlations ---")
    try:
        for name, path in file_info.items():
            df = pd.read_csv(path)
            predictions = df['Fertilizer Name'].str.split(n=2, expand=True)
            for i in range(3):
                col_name = f'pred{i+1}_{name}'
                if i < len(predictions.columns):
                    all_predictions_df[col_name] = predictions[i].fillna('')
                else:
                    all_predictions_df[col_name] = ''
        print(f"{len(model_names)} files read and top 3 predictions parsed successfully for correlation analysis.")
    except FileNotFoundError as e:
        print(f"\nERROR in correlation analysis: File not found -> {e.filename}")
        return {}
    except Exception as e:
        print(f"\nERROR during prediction parsing for correlation: {e}")
        return {}

    all_similarity_pivots = {}
    for rank in range(1, 4):
        similarity_data = []
        for model1, model2 in itertools.combinations(model_names, 2):
            pred1_col = f'pred{rank}_{model1}'
            pred2_col = f'pred{rank}_{model2}'

            similarity = (all_predictions_df[pred1_col] == all_predictions_df[pred2_col]).mean()
            similarity_data.append({'Model 1': model1, 'Model 2': model2, 'Similarity': similarity})

        similarity_df = pd.DataFrame(similarity_data)

        all_models_sorted = sorted(model_names)
        similarity_pivot = pd.DataFrame(1.0, index=all_models_sorted, columns=all_models_sorted)

        for _, row in similarity_df.iterrows():
            m1, m2, sim = row['Model 1'], row['Model 2'], row['Similarity']
            similarity_pivot.loc[m1, m2] = sim
            similarity_pivot.loc[m2, m1] = sim

        print(f"\n--- Similarity Matrix for Rank {rank} Predictions ---")
        print(f"(Values show the % of identical predictions for rank {rank})\n")
        print(similarity_pivot.round(3))
        all_similarity_pivots[rank] = similarity_pivot.copy()

    return all_similarity_pivots

def run_cluster_averaging_ensemble():
    """
    First, combines a cluster of highly correlated models.
    Then, ensembles this representative model with other independent models.
    """
    # --- DEFINE ALL MODEL PATHS AND NAMES ---
    # Ensure this dictionary accurately lists all your models and their paths.
    all_model_info = {
        "s228": "/kaggle/input/log-reg-364948/submission_cluster_38228.csv",
        "s240": "/kaggle/input/log-reg-364948/submission_cluster_38240.csv",
        "s250": "/kaggle/input/log-reg-364948/submission_cluster_38250.csv",
        "s285": "/kaggle/input/log-reg-364948/submission_mahog_38285.csv",
        "s261": "/kaggle/input/log-reg-364948/submission_cluster_38261.csv",
        "s300": "/kaggle/input/log-reg-364948/submission_cluster_38300.csv",
        "s299": "/kaggle/input/log-reg-364948/submission_cluster_38299.csv",
        "s306": "/kaggle/input/log-reg-364948/submission_cluster_38306.csv"
    }

    # Rank points
    rank_points = {0: 4, 1: 2, 2: 1} # Current setup: 1st place gets 4 points, 2nd 2, 3rd 1

    # --- Step 1: Analyze correlations to dynamically define clusters ---
    similarity_matrices = analyze_correlation_all_ranks(all_model_info)

    cluster_model_paths = {}
    independent_model_paths = {**all_model_info}

    rank_to_cluster_on = 1 # Often Rank 1 is best for clustering

    # Hierarchical Clustering Threshold (Tune this carefully!)
    clustering_distance_threshold = 0.016 

    if rank_to_cluster_on in similarity_matrices and len(all_model_info) > 1:
        sim_matrix_rank_chosen = similarity_matrices[rank_to_cluster_on]
        model_names_sorted = list(sim_matrix_rank_chosen.index)

        dissimilarity_matrix = 1 - sim_matrix_rank_chosen.values
        if np.isnan(dissimilarity_matrix).any() or np.isinf(dissimilarity_matrix).any():
            print("WARNING: Dissimilarity matrix contains NaN or Inf. Imputing with 1.0 (max distance).")
            dissimilarity_matrix = np.nan_to_num(dissimilarity_matrix, nan=1.0, posinf=1.0, neginf=1.0)

        condensed_dist_matrix = squareform(dissimilarity_matrix)
        Z = linkage(condensed_dist_matrix, method='average')

        # --- Visualize the Dendrogram ---
        plt.figure(figsize=(12, 7))
        plt.title(f'Hierarchical Clustering Dendrogram (Rank {rank_to_cluster_on} Predictions)')
        dendrogram(
            Z,
            labels=model_names_sorted,
            leaf_rotation=45.,
            leaf_font_size=10.,
            show_leaf_counts=True
        )
        plt.axhline(y=clustering_distance_threshold, color='r', linestyle='--', label=f'Threshold: {clustering_distance_threshold:.2f}')
        plt.xlabel("Models")
        plt.ylabel("Distance (1 - Similarity)")
        plt.legend()
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()
        # --- End Visualization ---

        clusters = fcluster(Z, t=clustering_distance_threshold, criterion='distance')

        unique_clusters, counts = np.unique(clusters, return_counts=True)
        if len(unique_clusters) > 0:
            sorted_cluster_info = sorted(zip(unique_clusters, counts), key=lambda x: x[1], reverse=True)
            largest_cluster_id = sorted_cluster_info[0][0]

            for i, model_name in enumerate(model_names_sorted):
                if clusters[i] == largest_cluster_id:
                    cluster_model_paths[model_name] = all_model_info[model_name]
                    independent_model_paths.pop(model_name, None)
        else:
            print("No clusters formed with the given threshold. All models treated as independent.")
            cluster_model_paths = {}
            independent_model_paths = {**all_model_info}
    else:
        print(f"\nINFO: Cannot perform hierarchical clustering. Either only 1 model, or rank {rank_to_cluster_on} not found.")
        print("All models will be treated as independent.")
        cluster_model_paths = {}
        independent_model_paths = {**all_model_info}

    print(f"\n--- Ensemble Configuration ---")
    print(f"Cluster Models ({len(cluster_model_paths)}): {list(cluster_model_paths.keys())}")
    print(f"Independent Models ({len(independent_model_paths)}): {list(independent_model_paths.keys())}")
    print(f"Hierarchical Clustering Distance Threshold (t): {clustering_distance_threshold} (based on Rank {rank_to_cluster_on} similarity)")

    # --- CALCULATE WEIGHTS ---
    all_weights_squared = {}
    print("\nCalculating squared weights for all models...")
    for name, path in all_model_info.items():
        try:
            score_part = re.search(r'_(\d{5})\.csv', path).group(1)
            base_weight = int(score_part)
            all_weights_squared[name] = base_weight ** 2
            print(f"- {name}: {all_weights_squared[name]}")
        except (AttributeError, ValueError):
            print(f"ERROR: Could not extract score from filename '{path}'. Defaulting weight to 1.")
            all_weights_squared[name] = 1

    # --- READ FILES ---
    try:
        dfs = {name: pd.read_csv(path) for name, path in all_model_info.items()}
        print("\nAll submission files read successfully for ensembling.")
    except FileNotFoundError as e:
        print(f"\nERROR: File not found -> {e.filename}")
        return
    except Exception as e:
        print(f"\nERROR during file reading for ensembling: {e}")
        return

    sample_df = next(iter(dfs.values()))
    all_ids = sample_df['id']

    # --- STAGE 1: CREATE THE CLUSTER REPRESENTATIVE MODEL ---
    print("\nStage 1: Creating a 'Representative Model' from the highly correlated cluster...")
    cluster_predictions = {}

    # Determine the weight for the Cluster Representative Model
    cluster_representative_weight = 0
    if cluster_model_paths:
        # Option: Use the weight of the highest-scoring model in the cluster
        best_model_in_cluster = max(cluster_model_paths.keys(), key=lambda k: all_weights_squared.get(k, 0), default=None)

        if best_model_in_cluster:
            cluster_representative_weight = all_weights_squared[best_model_in_cluster]
            print(f"Cluster representative weight based on highest scoring model in cluster ('{best_model_in_cluster}'): {cluster_representative_weight}")
        else:
            # This 'else' block should ideally not be reached if cluster_model_paths is not empty
            # and weights are typically positive. It's a very unlikely edge case.
            print("Could not determine a best model with non-zero weight in cluster. Cluster representative weight set to 0.")
            cluster_representative_weight = 0
        
        # DEBUG print: check which models are in the cluster
        print(f"DEBUG: Models contributing to cluster representative: {list(cluster_model_paths.keys())}")

        for an_id in all_ids:
            fertilizer_scores = defaultdict(float)
            for name in cluster_model_paths.keys():
                model_weight = all_weights_squared.get(name, 1)
                row = dfs[name][dfs[name]['id'] == an_id]
                if not row.empty:
                    predictions_str = row['Fertilizer Name'].iloc[0]
                    predictions = [] if pd.isna(predictions_str) else predictions_str.split()
                    for rank, fertilizer in enumerate(predictions):
                        if rank in rank_points:
                            score = rank_points[rank] * model_weight
                            fertilizer_scores[fertilizer] += score

            sorted_fertilizers = sorted(fertilizer_scores.keys(), key=lambda f: fertilizer_scores[f], reverse=True)
            cluster_predictions[an_id] = sorted_fertilizers[:3]
        print("'Representative Model' predictions are complete.")
    else:
        print("No cluster models defined. Skipping Stage 1. All models will be treated independently in Stage 2.")

    # --- STAGE 2: PERFORM THE FINAL ENSEMBLE ---
    print("\nStage 2: Performing the final ensemble...")
    final_predictions_list = []

    for an_id in all_ids:
        final_scores = defaultdict(float)

        # Add votes from independent models
        for name in independent_model_paths.keys():
            model_weight = all_weights_squared.get(name, 1)
            row = dfs[name][dfs[name]['id'] == an_id]
            if not row.empty:
                predictions_str = row['Fertilizer Name'].iloc[0]
                predictions = [] if pd.isna(predictions_str) else predictions_str.split()
                for rank, fertilizer in enumerate(predictions):
                    if rank in rank_points:
                        score = rank_points[rank] * model_weight
                        final_scores[fertilizer] += score

        # Add the vote from our Cluster Representative Model (if a cluster was formed and has weight)
        if cluster_model_paths and cluster_representative_weight > 0:
            cluster_preds_for_id = cluster_predictions.get(an_id, [])
            for rank, fertilizer in enumerate(cluster_preds_for_id):
                if rank in rank_points:
                    score = rank_points[rank] * cluster_representative_weight
                    final_scores[fertilizer] += score

        # Get the final ranking
        final_sorted = sorted(final_scores.keys(), key=lambda f: final_scores[f], reverse=True)
        final_prediction_str = " ".join(final_sorted[:3])
        final_predictions_list.append({'id': an_id, 'Fertilizer Name': final_prediction_str})

    # --- SAVE RESULTS ---
    ensembled_df = pd.DataFrame(final_predictions_list)
    output_filename = "submission.csv"
    ensembled_df.to_csv(output_filename, index=False)

    print(f"\nEnsembling process complete!")
    print(f"New submission file '{output_filename}' was created successfully.")

# Run the full ensemble process
if __name__ == "__main__":
    run_cluster_averaging_ensemble()




