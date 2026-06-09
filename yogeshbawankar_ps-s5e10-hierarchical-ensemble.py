import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from bokeh.io import output_notebook
from bokeh.resources import INLINE

output_notebook(resources=INLINE)

CONFIG = {
    # Path to the main competition data (train.csv, test.csv, etc.)
    "competition_data_path": "/kaggle/input/playground-series-s5e10",
    
    # List of directories where submission files are located
    "dataset_paths": [
        "/kaggle/input/1-october-2025-ps-s5-e10",
        "/kaggle/input/2-october-2025-ps-s5e10",
    ],
    
    # Path for generated placeholder files
    "path_oof_preds": "/kaggle/working/oof_",
    "path_ground_truth": "/kaggle/working/y_true.csv",
    
    "id_col": "id",
    "target_col": "accident_risk",
    
    # Models to be blended
    "models": [
        '0.05547.a', '0.05550.c', '0.05551.a', '0.05552.a', 
        '0.05552.b', '0.05553.c', '0.05553.b'
    ]
}

# This part is now fixed to use the reliable sample_submission.csv.
print("Creating placeholder OOF and ground truth files for demonstration...")
try:
    sample_submission = pd.read_csv(f"{CONFIG['competition_data_path']}/sample_submission.csv")
    y_true = np.random.rand(len(sample_submission))
    pd.DataFrame({CONFIG['id_col']: sample_submission[CONFIG['id_col']], CONFIG['target_col']: y_true}).to_csv(CONFIG['path_ground_truth'], index=False)

    for model_name in CONFIG["models"]:
        oof_preds = np.random.rand(len(sample_submission))
        pd.DataFrame({CONFIG['id_col']: sample_submission[CONFIG['id_col']], CONFIG['target_col']: oof_preds}).to_csv(f"{CONFIG['path_oof_preds']}{model_name}.csv", index=False)
    print("Placeholder files created successfully.")
except FileNotFoundError:
    print("Could not find sample_submission.csv. Skipping placeholder file creation.")
    


def load_submissions(model_names, dataset_paths, oof_path, id_col, target_col):
    """
    Loads and validates OOF and test submission files from multiple possible directories.
    """
    oof_dfs, test_dfs = [], []

    print("Loading and validating submission files...")
    for name in model_names:
        found_test = False
        
        for path in dataset_paths:
            filepath = os.path.join(path, f"submission_{name}.csv")
            if os.path.exists(filepath):
                test_df = pd.read_csv(filepath).sort_values(by=id_col).reset_index(drop=True)
                test_dfs.append(test_df)
                found_test = True
                break
        if not found_test:
            raise FileNotFoundError(f"Could not find submission_{name}.csv in any of the specified dataset paths.")


        oof_filepath = f"{oof_path}{name}.csv"
        if not os.path.exists(oof_filepath):
            raise FileNotFoundError(f"OOF file not found: {oof_filepath}. Please ensure placeholder files were created.")
        oof_df = pd.read_csv(oof_filepath).sort_values(by=id_col).reset_index(drop=True)
        oof_dfs.append(oof_df)


    base_ids = oof_dfs[0][id_col]
    for i, df in enumerate(oof_dfs[1:], 1):
        assert df[id_col].equals(base_ids), f"ID misalignment in OOF file for model {model_names[i]}"
    for i, df in enumerate(test_dfs):
        assert df[id_col].equals(base_ids), f"ID misalignment in test file for model {model_names[i]}"

    oof_predictions = np.stack([df[target_col].values for df in oof_dfs], axis=1)
    test_predictions = np.stack([df[target_col].values for df in test_dfs], axis=1)
    
    print("Data loaded and validated successfully.")
    return oof_predictions, test_predictions, base_ids.values

def optimize_weights(oof_preds, y_true, model_names):
    """
    Finds optimal blending weights by minimizing RMSE on OOF predictions.
    """
    def rmse_objective(weights, preds, true_vals):
        ensembled_preds = np.dot(preds, weights)
        return np.sqrt(np.mean((true_vals - ensembled_preds)**2))

    print("Optimizing ensemble weights...")
    initial_weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(oof_preds.shape[1])]

    result = minimize(
        fun=rmse_objective, x0=initial_weights, args=(oof_preds, y_true),
        method='SLSQP', bounds=bounds, constraints=constraints
    )

    if not result.success:
        print("Warning: Optimization failed. Using equal weights as a fallback.")
        return initial_weights

    optimized_weights = result.x
    print(f"Optimal weights found with OOF RMSE: {result.fun:.6f}")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=optimized_weights, palette="viridis")
    plt.title("Optimized Ensemble Weights")
    plt.ylabel("Weight")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return optimized_weights


# --- 1. Load Data ---
oof_preds, test_preds, ids = load_submissions(
    model_names=CONFIG["models"],
    dataset_paths=CONFIG["dataset_paths"],
    oof_path=CONFIG["path_oof_preds"],
    id_col=CONFIG["id_col"],
    target_col=CONFIG["target_col"]
)
y_true_df = pd.read_csv(CONFIG["path_ground_truth"]).sort_values(by=CONFIG["id_col"])
y_true = y_true_df[CONFIG["target_col"]].values

# --- 2. Optimize Weights ---
optimal_weights = optimize_weights(oof_preds, y_true, CONFIG["models"])

# --- 3. Blend Test Predictions ---
print("Blending test predictions with optimal weights...")
final_predictions = np.dot(test_preds, optimal_weights)

# --- 4. Generate Submission File ---
submission_df = pd.DataFrame({
    CONFIG["id_col"]: ids,
    CONFIG["target_col"]: final_predictions
})
submission_df.to_csv("submission.csv", index=False)

print("\nâœ… Submission file 'submission.csv' created successfully!")
display(submission_df.head())




