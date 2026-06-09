
# ============================================================
# Blending Script - Optimized & Clean Version 
# Author: Daniyal Atta
# ============================================================

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.1)

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def load_predictions(file_path: str, label: str) -> pd.DataFrame:
    """Load a submission file and rename prediction column."""
    df = pd.read_csv(file_path)
    df = df.rename(columns={"y": f"y_{label}"})
    return df

def blend_predictions(df1: pd.DataFrame, df2: pd.DataFrame, weight: float = 0.5) -> pd.DataFrame:
    """Blend two prediction columns using probability clipping + normal transform."""
    df = pd.merge(df1, df2, on="id", how="inner")

    eps = 1e-6
    y1_safe = df["y_1"].clip(eps, 1 - eps)
    y2_safe = df["y_2"].clip(eps, 1 - eps)

    # Gaussian transformation
    norm1 = stats.norm.ppf(y1_safe)
    norm2 = stats.norm.ppf(y2_safe)

    # Weighted average in normal space
    blended = weight * norm1 + (1 - weight) * norm2

    # Back-transform to probability
    df["y"] = stats.norm.cdf(blended)
    return df[["id", "y"]]

def plot_distributions(df: pd.DataFrame, cols: list, title: str):
    """Visualize probability distributions of predictions."""
    plt.figure(figsize=(8, 4))
    for col in cols:
        sns.kdeplot(df[col], label=col, fill=True, alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    # Paths (update as needed)
    path1 = "/kaggle/input/ps-s5e8-blend-xgb-lgb/submission.csv"
    path2 = "/kaggle/input/train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv"
    output_path = Path("/kaggle/working/submission.csv")

    # Load files
    df1 = load_predictions(path1, "1")
    df2 = load_predictions(path2, "2")

    # Blend predictions
    final_df = blend_predictions(df1, df2, weight=0.5)

    # Save final submission
    final_df.to_csv(output_path, index=False)
    print(f"âœ… Submission saved at: {output_path}")

    # Optional visualization
    try:
        merged_df = pd.merge(df1, df2, on="id")
        plot_distributions(merged_df, ["y_1", "y_2"], "Original Model Distributions")
        plot_distributions(final_df, ["y"], "Blended Distribution")
    except Exception as e:
        print("âš ï¸� Plotting skipped:", e)




