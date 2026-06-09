import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

def ensemble_submissions(
    submission_folder,
    use_rank=False,
    rank_axis="columns",
    output_prefix=""
):
    assert rank_axis in {"columns", "rows"}, "rank_axis must be 'columns' or 'rows'"

    pattern = re.compile(r"submission_(\d+)\.csv")
    predictions = []
    weights = []
    submission_names = []

    for fname in os.listdir(submission_folder):
        match = pattern.match(fname)
        if not match:
            continue
        score_str = match.group(1)
        score = float("0." + score_str.lstrip("0"))
        df = pd.read_csv(os.path.join(submission_folder, fname))
        pred_cols = [col for col in df.columns if col != "ID"]
        pred = df[pred_cols].copy()

        if use_rank:
            axis = 0 if rank_axis == "columns" else 1
            pred = pred.rank(axis=axis, method="average")

        predictions.append(pred)
        weights.append(score)
        submission_names.append(fname)

    pred_array = np.stack([df.values for df in predictions], axis=-1)  # (N, C, M)
    weights = np.array(weights)
    weights = weights / weights.sum()
    n_samples, n_targets, n_models = pred_array.shape
    ID = df["ID"]
    columns = predictions[0].columns

    rank_suffix = f"_ranked_{rank_axis}" if use_rank else "_raw"

    def save(df_values, name):
        df_out = pd.DataFrame(df_values, columns=columns)
        df_out.insert(0, "ID", ID)
        filename = f"{output_prefix}{name}{rank_suffix}.csv"
        df_out.to_csv(filename, index=False)

    # 1. Power Weighted Average
    power = 3
    w_pow = weights**power
    w_pow /= w_pow.sum()
    ensemble_power = np.average(pred_array, axis=-1, weights=w_pow)
    save(ensemble_power, "ensemble_power_weighted")

    # 2. Top-k Averaging
    k = 3
    top_k_idx = np.argsort(weights)[-k:]
    top_k_preds = pred_array[:, :, top_k_idx]
    top_k_weights = weights[top_k_idx]
    top_k_weights = top_k_weights / top_k_weights.sum()
    ensemble_top_k = np.average(top_k_preds, axis=-1, weights=top_k_weights)
    save(ensemble_top_k, "ensemble_top_k")

    # 3. Blend Best + Average of Others
    best_idx = np.argmax(weights)
    best_model = pred_array[:, :, best_idx]
    others = np.delete(pred_array, best_idx, axis=2)
    others_avg = np.mean(others, axis=2)
    alpha = 0.8
    ensemble_blend = alpha * best_model + (1 - alpha) * others_avg
    save(ensemble_blend, "ensemble_blend_best_avg")

    # 4. Threshold-Based Ensemble
    threshold = 0.3
    mask = weights >= threshold
    if mask.sum() > 0:
        selected_preds = pred_array[:, :, mask]
        selected_weights = weights[mask] / weights[mask].sum()
        ensemble_thresh = np.average(selected_preds, axis=-1, weights=selected_weights)
        save(ensemble_thresh, "ensemble_threshold_0.3")

    # 5. Residual-Based Per-Feature Weighting
    baseline = ensemble_power.copy()
    errors = np.zeros((n_targets, n_models))
    for i in range(n_models):
        pred_i = pred_array[:, :, i]
        errors[:, i] = ((pred_i - baseline) ** 2).mean(axis=0)
    weights_per_target = 1 / (errors + 1e-6)
    weights_per_target /= weights_per_target.sum(axis=1, keepdims=True)
    ensemble_residual = np.einsum("ncm,cm->nc", pred_array, weights_per_target)
    save(ensemble_residual, "ensemble_residual_weighted")

    # 6. Stacking with Ridge Regression
    X_all = np.transpose(pred_array, (0, 2, 1))  # (N, M, C)
    final_preds = np.zeros((n_samples, n_targets))
    y_meta = ensemble_power.copy()
    for i in range(n_targets):
        X_meta = X_all[:, :, i]
        y_target = y_meta[:, i]
        model = Ridge()
        model.fit(X_meta, y_target)
        final_preds[:, i] = model.predict(X_meta)
    save(final_preds, "ensemble_stacking")

    # 7. Greedy Ensemble (maximize Spearman vs power ensemble)
    def average_models(indices):
        return np.mean(pred_array[:, :, indices], axis=-1)

    base = [np.argmax(weights)]
    best = average_models(base)
    for i in range(n_models):
        if i in base:
            continue
        candidate = average_models(base + [i])
        score_best = np.mean([spearmanr(best[:, j], y_meta[:, j])[0] for j in range(n_targets)])
        score_candidate = np.mean([spearmanr(candidate[:, j], y_meta[:, j])[0] for j in range(n_targets)])
        if score_candidate > score_best:
            base.append(i)
            best = candidate
    save(best, "ensemble_greedy")

    # 8. Correlation-Weighted Ensemble (Spearman corr to power baseline)
    corr_weights = []
    for i in range(n_models):
        r = np.mean([spearmanr(pred_array[:, j, i], y_meta[:, j])[0] for j in range(n_targets)])
        corr_weights.append(max(0, r))  # avoid negative weights
    corr_weights = np.array(corr_weights)
    if corr_weights.sum() > 0:
        corr_weights = corr_weights / corr_weights.sum()
        ensemble_corr = np.average(pred_array, axis=-1, weights=corr_weights)
        save(ensemble_corr, "ensemble_correlation_weighted")

    # Save diagnostics
    pd.DataFrame(errors, index=columns, columns=submission_names).to_csv(f"{output_prefix}model_target_errors{rank_suffix}.csv")
    pd.DataFrame(weights_per_target, index=columns, columns=submission_names).to_csv(f"{output_prefix}residual_weights{rank_suffix}.csv")

    print("✅ Ensembles saved using " + ("ranked " if use_rank else "raw ") + f"predictions (axis={rank_axis})")



# Raw value ensembling
ensemble_submissions("/kaggle/input/elucidata-submissions", use_rank=False)




# Rank-based ensembling across columns (C1 to C36)
ensemble_submissions("/kaggle/input/elucidata-submissions", use_rank=True, rank_axis="columns")




# Rank-based ensembling across rows (per sample)
ensemble_submissions("/kaggle/input/elucidata-submissions", use_rank=True, rank_axis="rows")



import matplotlib.pyplot as plt
import seaborn as sns

def visualize_ensemble_diagnostics(errors_csv, weights_csv):
    """
    Generate heatmap visualizations from ensemble diagnostics.

    Args:
        errors_csv (str): Path to 'model_target_errors*.csv'.
        weights_csv (str): Path to 'residual_weights*.csv'.
    """
    # Load diagnostic CSVs
    errors_df = pd.read_csv(errors_csv, index_col=0)
    weights_df = pd.read_csv(weights_csv, index_col=0)

    # Heatmap: Model-Target Residual Errors (MSE)
    plt.figure(figsize=(14, 6))
    sns.heatmap(errors_df, cmap="Reds", cbar_kws={'label': 'MSE'})
    plt.title("Model-Target Errors (MSE vs Power-Weighted Ensemble)")
    plt.xlabel("Submission")
    plt.ylabel("Target Variable")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Heatmap: Residual-Based Feature Weights
    plt.figure(figsize=(14, 6))
    sns.heatmap(weights_df, cmap="Blues", cbar_kws={'label': 'Normalized Weight'})
    plt.title("Residual-Based Per-Target Model Weights")
    plt.xlabel("Submission")
    plt.ylabel("Target Variable")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Barplot: Average Model Weight (residual-weighted)
    avg_weights = weights_df.mean(axis=0).sort_values(ascending=False)
    plt.figure(figsize=(12, 4))
    sns.barplot(x=avg_weights.index, y=avg_weights.values, palette="coolwarm")
    plt.title("Average Model Weight Across Targets (Residual-Based)")
    plt.ylabel("Mean Weight")
    plt.xlabel("Submission")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()



!ls



visualize_ensemble_diagnostics(
    errors_csv="model_target_errors_raw.csv",
    weights_csv="residual_weights_raw.csv"
)


visualize_ensemble_diagnostics(
    errors_csv="model_target_errors_ranked_rows.csv",
    weights_csv="residual_weights_ranked_rows.csv"
)


visualize_ensemble_diagnostics(
    errors_csv="model_target_errors_ranked_columns.csv",
    weights_csv="residual_weights_ranked_columns.csv"
)


#visualize_ensemble_diagnostics(
#    errors_csv="model_target_errors.csv",
#    weights_csv="residual_weights.csv"
#)


import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_celltype_abundance(slide_name, image, spots, plots_per_line=4, show_image=False, plot_title=""):
    """
    Plot the slide image with spot overlays for cell-type abundances (C1 to C35).

    Parameters:
        slide_name (str): Name of the slide.
        image (np.array): 2D or 3D array representing the slide image.
        spots (np.array): Structured NumPy array containing spot information with fields:
                          'x', 'y', 'C1', 'C2', ..., 'C35'.
        plots_per_line (int): Number of subplots per row (default is 4).

    The function creates a figure with subplots arranged in a grid and plots:
      - The slide image as a background.
      - A scatter overlay at spot (x, y) positions, colored by the abundance of each cell type.
      - A common colorbar that indicates the mapping from color to abundance.
    """
    # Extract the x and y coordinates from the spots array
    x = spots["x"]
    y = spots["y"]
    num_celltypes = 35  # There are 35 cell types (C1 to C35)
    num_rows = math.ceil(num_celltypes / plots_per_line)
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(num_rows, plots_per_line, figsize=(plots_per_line * 3, num_rows * 3))
    axes = axes.flatten()  # Flatten to simplify indexing
    
    # Loop over each cell type field from C1 to C35
    for i in range(num_celltypes):
        var_name = f"C{i+1}"  # Create field name e.g. "C1", "C2", ...
        c_values = spots[var_name]  # Get the abundance values for this cell type
        
        ax = axes[i]
        if show_image is True:
            ax.imshow(image, aspect="auto")
        else:
            # Manually set the axis limits to match the image dimensions
            height, width = image.shape[:2]
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)  # invert y-axis to mimic image display

        sc = ax.scatter(x, y, c=c_values, cmap="viridis", s=2, alpha=1)
        #sc = ax.scatter(x, y, c=c_values, cmap="plasma", s=2, alpha=0.7)

        ax.set_title(var_name, fontsize=8)
        ax.axis("off")
    
    # If there are any extra subplots (in case grid has one more cell), hide them
    for j in range(num_celltypes, len(axes)):
        axes[j].axis("off")
    

    if plot_title != "":
        fig.suptitle(plot_title)
    else:
        # Set an overall title for the figure
        fig.suptitle(f"Slide {slide_name}", fontsize=14)

    # Add a common colorbar (using the last scatter object)
    #fig.colorbar(sc, ax=axes.tolist(), label="Abundance")
    #fig.subplots_adjust(right=0.85)
    #cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    #fig.colorbar(sc, cax=cbar_ax, label="Abundance")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




def plot_submission(submission_file):
    slice_name = "S_7"
    with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as h5file:
        image = np.array(h5file["images/Test"][slice_name])
        spots = np.array(h5file["spots/Test"][slice_name])
        submission_df = pd.read_csv(submission_file)
        submission_df
        spots_df = pd.DataFrame(spots).reset_index(names="ID")
        spots_merged = spots_df.drop(columns="Test_Set").merge(submission_df)
        spots_merged
        plot_celltype_abundance(slice_name, image, spots_merged, plot_title=submission_file)
    #plot_total_abundance(slice_name, image, spots, show_colorbar=True, colorbar_outside=True, show_image=False)



def plot_total_abundance(slide_name, image, spots, show_colorbar=True, colorbar_outside=True, show_image=True, show_spots=True):
    """
    Plot the slide image with spot overlays where each spot's color represents 
    the total abundance of cell types C1 to C35.

    Parameters:
        slide_name (str): Name of the slide.
        image (np.array): The slide image.
        spots (np.array): Structured NumPy array with fields 'x', 'y', and 'C1' ... 'C35'.
        show_colorbar (bool): Whether to display a colorbar (default True).
        colorbar_outside (bool): If True, place the colorbar to the right of the plot.
    """
    # Extract x and y coordinates from the spots array
    x = spots["x"]
    y = spots["y"]
    
    # Compute total abundance by summing C1 through C35 for each spot
    total_abundance = np.zeros_like(x, dtype=float)
    if slide_name == "S_7":
        total_abundance = spots["Test_Set"]
    else:
        for i in range(1, 36):  # Fields C1 to C35
            total_abundance += spots[f"C{i}"]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(6, 6))
    if show_image is True:
        ax.imshow(image, aspect="auto")
    else:
        # Manually set the axis limits to match the image dimensions
        height, width = image.shape[:2]
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # invert y-axis to mimic image display

    if show_spots:
        sc = ax.scatter(x, y, c=total_abundance, cmap="viridis", s=2, alpha=0.7)
    
        # Add a colorbar if desired
        if show_colorbar:
            if colorbar_outside:
                # Adjust the right margin to make room for the colorbar
                fig.subplots_adjust(right=0.85)
                cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
                fig.colorbar(sc, cax=cbar_ax, label="Total Abundance")
            else:
                fig.colorbar(sc, ax=ax, label="Total Abundance")
    ax.set_title(f"Total Abundance for Slide {slide_name}")
    ax.axis("off")
    
    plt.tight_layout()
    plt.show()
slice_name = "S_7"


with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as h5file:
    image = np.array(h5file["images/Test"]["S_7"])
    spots = np.array(h5file["spots/Test"]["S_7"])
    submission_df = pd.read_csv("ensemble_blend_best_avg_ranked_rows.csv") # any file works
    test_spots = pd.merge(
        pd.DataFrame(spots).reset_index(names="ID"),
        submission_df
    )
    test_spots
    plot_total_abundance("S_7", image, test_spots, show_colorbar=True, colorbar_outside=True, show_spots=False)
    plot_total_abundance("S_7", image, test_spots, show_colorbar=False, colorbar_outside=True, show_spots=True)


import glob
for submission_file in glob.glob("*ensemble*rows*"):
    plot_submission(submission_file)


plot_submission("ensemble_blend_best_avg_ranked_rows.csv")

