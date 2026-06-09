import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd

def collect_all_train_spots(train_spot_tables):
    """
    Combine all spot tables from the training slides into a single DataFrame.
    Adds a 'slide' column to indicate the source slide for each spot.

    Parameters:
        train_spot_tables (dict): Dictionary where keys are slide names (e.g. 'S_1', 'S_2', ...)
                                  and values are structured NumPy arrays with fields: x, y, C1..C35.

    Returns:
        pd.DataFrame: Combined DataFrame with columns ['x', 'y', 'C1', ..., 'C35', 'slide']
    """
    all_spots = []

    for slide_name, spot_array in train_spot_tables.items():
        # Convert structured NumPy array to DataFrame
        spot_df = pd.DataFrame(spot_array)
        # Add slide name column
        spot_df.insert(0, "slide", slide_name)
        all_spots.append(spot_df)

    # Concatenate all into a single DataFrame
    combined_df = pd.concat(all_spots, ignore_index=True)
    return combined_df



with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as h5file:
    train_spot_tables = h5file["spots/Train"]
    # Convert HDF5 datasets to NumPy arrays for each slide
    train_spot_arrays = {slide: np.array(train_spot_tables[slide]) for slide in train_spot_tables}

combined_df = collect_all_train_spots(train_spot_arrays)
combined_df.head()



combined_df.describe()


ab_long = pd.wide_to_long(
        combined_df,
        stubnames="C",
        i=['slide', 'x', 'y'],
        j='cell_type',
        sep='',
        suffix='\\d+'
    ).reset_index()

ab_long.rename(columns={'C': 'abundance'}, inplace=True)
ab_long["rank"] = ab_long.groupby(["slide", "x", "y"])["abundance"].rank(method="dense", ascending=True)
ab_long


composition_df = ab_long.groupby(["slide", "cell_type"])["abundance"].mean().unstack(fill_value=0)
composition_df


# Count how often each cell type was ranked #1 or #35
rank_counts = ab_long[ab_long["rank"].isin([1, 35])].groupby(["cell_type", "rank"]).size().unstack(fill_value=0)

# Bar plot
rank_counts.plot(kind="bar", figsize=(12, 5))
plt.title("How Often Each Cell Type is Ranked First or Last")
plt.xlabel("Cell Type")
plt.ylabel("Frequency")
plt.legend(title="Rank", labels=["Top (1)", "Bottom (35)"])
plt.show()



# Define top and bottom rank ranges
top_ranks = list(range(1, 6))       # 1 to 5
bottom_ranks = list(range(31, 36))  # 31 to 35

# Label ranks as 'Top 5' or 'Bottom 5'
ab_long["rank_group"] = ab_long["rank"].apply(
    lambda r: "Top 5" if r in top_ranks else ("Bottom 5" if r in bottom_ranks else None)
)

# Filter to just Top 5 and Bottom 5
rank_subset = ab_long[ab_long["rank_group"].notnull()]

# Count occurrences per cell type and rank group
rank_counts = rank_subset.groupby(["cell_type", "rank_group"]).size().unstack(fill_value=0)

# Bar plot
rank_counts.plot(kind="bar", figsize=(12, 6), color=["#ff7f0e","#1f77b4"])
plt.title("How Often Each Cell Type is Ranked in Top 5 or Bottom 5")
plt.xlabel("Cell Type")
plt.ylabel("Frequency")
plt.legend(title="Rank Group")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



plt.figure(figsize=(14, 6))
sns.boxplot(data=ab_long, x="cell_type", y="rank", order=sorted(ab_long["cell_type"].unique()))
plt.xticks(rotation=90)
plt.title("Rank Distribution per Cell Type")
plt.ylabel("Rank (1 = highest)")
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

plt.figure(figsize=(12, 6))
plot = sns.histplot(
    data=ab_long,
    x="rank",
    hue="cell_type",
    bins=35,
    multiple="stack",
    palette="tab20",
    shrink=0.9
)
plt.title("Distribution of Ranks per Cell Type")
plt.xlabel("Rank (1 = highest abundance)")
plt.ylabel("Count across all spots")

## Access the legend from the plot object
#legend = plot.get_legend()

sns.move_legend(
    plot, "lower center",
    bbox_to_anchor=(.5, -0.5), ncol=12, title="Cell Type", frameon=False,
)

plt.subplots_adjust(bottom=0.35)
plt.show()


import seaborn as sns
sns.clustermap(composition_df, metric="cosine", cmap="viridis", figsize=(8, 6))
plt.suptitle("Slide Similarity by Cell Type Composition")
plt.show()



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
coords = pca.fit_transform(composition_df)

plt.figure(figsize=(6, 5))
for i, slide in enumerate(composition_df.index):
    plt.scatter(coords[i, 0], coords[i, 1], label=slide)
plt.legend()
plt.title("Slide Similarity via PCA of Composition")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()



rank_matrix = ab_long.groupby(["slide", "cell_type"])["rank"].mean().unstack(fill_value=np.nan)
rank_matrix


import seaborn as sns
sns.clustermap(rank_matrix, metric="cosine", cmap="viridis", figsize=(8, 6))
plt.suptitle("Slide Similarity by Cell Type Rank")
plt.show()



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
coords = pca.fit_transform(rank_matrix)

plt.figure(figsize=(6, 5))
for i, slide in enumerate(composition_df.index):
    plt.scatter(coords[i, 0], coords[i, 1], label=slide)
plt.legend()
plt.title("Slide Similarity via PCA of Ranks")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()



from scipy.spatial import KDTree
import pandas as pd
import numpy as np
from tqdm import tqdm

def smooth_by_neighbors(df, value_col="rank", radius=100):
    """
    For each spot, compute the average `value_col` (e.g. 'rank' or 'abundance') among neighbors within radius.
    
    Parameters:
        df: long-format DataFrame with columns: ['slide', 'x', 'y', 'cell_type', value_col]
        value_col: the name of the column to average over neighbors
        radius: radius for neighbor search (same units as x, y coords)
        
    Returns:
        A copy of the DataFrame with a new column: 'smoothed_<value_col>'
    """
    df = df.copy()
    df[f"smoothed_{value_col}"] = np.nan

    for slide in df["slide"].unique():
        for cell_type in df["cell_type"].unique():
            subset = df[(df["slide"] == slide) & (df["cell_type"] == cell_type)].copy()
            coords = subset[["x", "y"]].values
            values = subset[value_col].values
            tree = KDTree(coords)

            smoothed = []
            for i in range(len(coords)):
                idx = tree.query_ball_point(coords[i], r=radius)
                if idx:
                    avg = np.mean(values[idx])
                else:
                    avg = np.nan
                smoothed.append(avg)

            df.loc[subset.index, f"smoothed_{value_col}"] = smoothed

    return df



ab_long_smoothed = smooth_by_neighbors(ab_long, value_col="rank", radius=100)



ab_long_smoothed


import matplotlib.pyplot as plt

def plot_smoothed_celltype_all_slides(df, cell_type, value_col="smoothed_rank", cmap="plasma", cols=3, vmin=1, vmax=35):
    """
    Plot smoothed values for a single cell type across all slides, with a consistent color scale.

    Parameters:
        df (pd.DataFrame): DataFrame with ['slide', 'x', 'y', 'cell_type', value_col']
        cell_type (int): Cell type ID to visualize (e.g. 17)
        value_col (str): Column to color by ('smoothed_rank' or 'abundance')
        cmap (str): Matplotlib colormap
        cols (int): Number of plots per row
        vmin, vmax (float): Fixed color scale range (e.g. 1–35 for ranks)
    """
    slides = df["slide"].unique()
    n = len(slides)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    axes = axes.flatten()

    for i, slide in enumerate(slides):
        ax = axes[i]
        data = df[(df["slide"] == slide) & (df["cell_type"] == cell_type)]

        sc = ax.scatter(data["x"], data["y"], c=data[value_col], cmap=cmap, s=10, alpha=0.9, vmin=vmin, vmax=vmax)
        ax.set_title(f"{slide}")
        ax.invert_yaxis()
        ax.axis("off")

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add a shared colorbar
    cbar = fig.colorbar(sc, ax=axes.tolist(), orientation="horizontal", fraction=0.05, pad=0.05)
    cbar.set_label(f"{value_col} (rank 1 = most abundant)")

    fig.suptitle(f"Cell Type {cell_type} – {value_col}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()



plot_smoothed_celltype_all_slides(ab_long_smoothed, cell_type=1, value_col="smoothed_rank")



plot_smoothed_celltype_all_slides(ab_long_smoothed, cell_type=2, value_col="smoothed_rank")



plot_smoothed_celltype_all_slides(ab_long_smoothed, cell_type=3, value_col="smoothed_rank")



plot_smoothed_celltype_all_slides(ab_long_smoothed, cell_type=4, value_col="smoothed_rank")



plot_smoothed_celltype_all_slides(ab_long_smoothed, cell_type=5, value_col="smoothed_rank")



plot_smoothed_celltype_all_slides(ab_long_smoothed, cell_type=6, value_col="smoothed_rank")



plot_smoothed_celltype_all_slides(ab_long_smoothed, cell_type=30, value_col="smoothed_rank")



avg_rank_by_cell = ab_long_smoothed.groupby("cell_type")["smoothed_rank"].mean()



from scipy.spatial import KDTree
import numpy as np
import pandas as pd
from tqdm import tqdm

def predict_test_from_train(train_df, test_spot_table, radius=100, seed=None, noise_std=0.5):
    """
    Generate a realistic test submission using spatial neighbors from training data
    with added randomness via Gaussian noise.

    Parameters:
        train_df (pd.DataFrame): long-format train data with ['x', 'y', 'cell_type', 'smoothed_rank']
        test_spot_table (pd.DataFrame): Test spot DataFrame with 'x', 'y' columns
        radius (float): Radius in pixels for neighbor search
        seed (int, optional): Random seed for reproducibility
        noise_std (float): Standard deviation of the Gaussian noise to add

    Returns:
        pd.DataFrame: Submission DataFrame with predicted ranks (columns C1 to C35), indexed by test spot ID
    """
    if seed is not None:
        np.random.seed(seed)

    # Pivot to wide format: (x, y) → smoothed_rank for each cell type
    rank_table = train_df.pivot_table(index=["x", "y"], columns="cell_type", values="smoothed_rank")
    rank_table = rank_table.dropna(how="any")  # drop incomplete rows

    # Build KDTree
    train_coords = np.array(rank_table.index.tolist())
    tree = KDTree(train_coords)

    # Get cell type column names C1 to C35
    cell_types = [f"C{i}" for i in range(1, 36)]
    cell_type_ids = list(range(1, 36))

    # Init output
    test_indices = test_spot_table.index
    pred_df = pd.DataFrame(index=test_indices, columns=cell_types, dtype=float)

    # Fallback: average smoothed rank per cell type
    avg_rank_by_cell = train_df.groupby("cell_type")["smoothed_rank"].mean().reindex(cell_type_ids)

    # Iterate through test spots
    for idx in tqdm(test_spot_table.index, desc="Generating Test Predictions"):
        x, y = test_spot_table.loc[idx, ["x", "y"]]
        neighbors = tree.query_ball_point([x, y], r=radius)

        if neighbors:
            neighbor_ranks = rank_table.iloc[neighbors]
            pred = neighbor_ranks.mean().to_numpy()
        else:
            pred = avg_rank_by_cell.to_numpy()

        # Introduce randomness by adding Gaussian noise
        pred += np.random.normal(0, noise_std, size=pred.shape)

        # Clip and round to stay within valid rank range
        pred = np.clip(np.round(pred), 1, 35)

        pred_df.loc[idx] = pred

    return pred_df



# Prepare test spot DataFrame from HDF5
with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as h5file:
    test_spot_table = pd.DataFrame(h5file["spots/Test/S_7"][:])

submission_df = predict_test_from_train(ab_long_smoothed, test_spot_table, radius=120, seed=6, noise_std=1)

# Format for submission
submission_df.index.name = "ID"
submission_df.reset_index().to_csv("submission.csv", index=False)



submission_df


def score_submission_with_global_consistency(
    submission_df,
    test_spot_table,
    train_df,
    radius=100,
    method="spearman",
    alpha=0.7,
    show_plot=False,
    cmap="plasma",
    plots_per_row=5
):
    """
    Score a submission using local spatial similarity and global rank trends,
    and optionally plot predicted ranks for all 35 cell types in a grid.

    Parameters:
        submission_df (pd.DataFrame): Predicted ranks (C1–C35) per test spot
        test_spot_table (pd.DataFrame): Test spot table with 'x', 'y' per index
        train_df (pd.DataFrame): Long-format training data (ab_long_smoothed)
        radius (float): Radius for neighbor search
        method (str): "spearman" or "cosine"
        alpha (float): Weight for local (alpha) vs global (1 - alpha)
        show_plot (bool): Whether to show 35 cell type plots
        cmap (str): Colormap
        plots_per_row (int): Number of plots per row in grid
    """
    from scipy.stats import spearmanr
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt

    # Wide format for Train: (x, y) -> [C1 ... C35]
    rank_table = train_df.pivot_table(index=["x", "y"], columns="cell_type", values="smoothed_rank")
    rank_table = rank_table.dropna()

    coords_train = np.array(rank_table.index.tolist())
    tree = KDTree(coords_train)

    expected_avg_rank = train_df.groupby("cell_type")["smoothed_rank"].mean().sort_index().to_numpy()

    scores = []

    # For plotting
    if show_plot:
        coords = test_spot_table.loc[submission_df.index, ["x", "y"]].copy()
        submission_with_coords = submission_df.copy()
        submission_with_coords["x"] = coords["x"].values
        submission_with_coords["y"] = coords["y"].values

    for idx in tqdm(submission_df.index, desc="Scoring with Global Consistency"):
        pred_vector = submission_df.loc[idx].to_numpy()
        x, y = test_spot_table.loc[idx, ["x", "y"]]

        neighbors = tree.query_ball_point([x, y], r=radius)
        local_sim = np.nan
        if neighbors:
            neighbor_vec = rank_table.iloc[neighbors].mean(axis=0).to_numpy()
            if method == "cosine":
                local_sim = cosine_similarity(pred_vector.reshape(1, -1), neighbor_vec.reshape(1, -1))[0, 0]
            elif method == "spearman":
                local_sim, _ = spearmanr(pred_vector, neighbor_vec)

        global_sim = np.nan
        if method == "cosine":
            global_sim = cosine_similarity(pred_vector.reshape(1, -1), expected_avg_rank.reshape(1, -1))[0, 0]
        elif method == "spearman":
            global_sim, _ = spearmanr(pred_vector, expected_avg_rank)

        # Weighted average
        if np.isnan(local_sim) and np.isnan(global_sim):
            continue
        elif np.isnan(local_sim):
            score = (1 - alpha) * global_sim
        elif np.isnan(global_sim):
            score = alpha * local_sim
        else:
            score = alpha * local_sim + (1 - alpha) * global_sim

        scores.append(score)

    # ---------- Plotting all 35 cell types ----------
    if show_plot:
        cell_types = [f"C{i}" for i in range(1, 36)]
        rows = (len(cell_types) + plots_per_row - 1) // plots_per_row

        fig, axes = plt.subplots(rows, plots_per_row, figsize=(plots_per_row * 3, rows * 3))
        axes = axes.flatten()

        for i, cell_type in enumerate(cell_types):
            ax = axes[i]
            data = submission_with_coords.copy()
            sc = ax.scatter(data["x"], data["y"], c=data[cell_type], cmap=cmap, s=5, vmin=1, vmax=35)
            ax.set_title(cell_type, fontsize=8)
            ax.axis("off")
            ax.invert_yaxis()

        for j in range(len(cell_types), len(axes)):
            axes[j].axis("off")

        cbar = fig.colorbar(sc, ax=axes.tolist(), orientation="horizontal", fraction=0.03, pad=0.04)
        cbar.set_label("Predicted Rank (1 = most abundant)")
        plt.suptitle("Predicted Ranks for All 35 Cell Types", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()

    return np.nanmean(scores)



submission_df


def plot_total_abundance(slide_name, image, spots, show_colorbar=True, colorbar_outside=True, show_image=True):
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

    sc = ax.scatter(x, y, c=total_abundance, cmap="viridis", s=2, alpha=0.7)
    ax.set_title(f"Total Abundance for Slide {slide_name}")
    ax.axis("off")
    
    # Add a colorbar if desired
    if show_colorbar:
        if colorbar_outside:
            # Adjust the right margin to make room for the colorbar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            fig.colorbar(sc, cax=cbar_ax, label="Total Abundance")
        else:
            fig.colorbar(sc, ax=ax, label="Total Abundance")
    
    plt.tight_layout()
    plt.show()
    slice_name = "S_7"


with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as h5file:
    image = np.array(h5file["images/Test"]["S_7"])
    spots = np.array(h5file["spots/Test"]["S_7"])
    test_spots = pd.merge(
        pd.DataFrame(spots).reset_index(names="ID").drop(columns="Test_Set"),
        submission_df.reset_index(names="ID")
    )
    test_spots
    plot_total_abundance("Test S_7", image, test_spots, show_colorbar=True, colorbar_outside=True)



score = score_submission_with_global_consistency(
    submission_df=submission_df,
    test_spot_table=test_spot_table,
    train_df=ab_long_smoothed,
    show_plot=True
)

print(f"Feasibility Score: {score:.3f}")


