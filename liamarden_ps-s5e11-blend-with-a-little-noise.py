import pandas as pd


import numpy as np

def blend_submissions(weight_dict, output_path):
    dataframes = []

    for path, weight in weight_dict.items():
        df = pd.read_csv(path)
        df["weighted_pred"] = df["loan_paid_back"] * weight
        dataframes.append(df[["id", "weighted_pred"]])

    merged = dataframes[0]
    for df in dataframes[1:]:
        merged = merged.merge(df, on="id", how="inner", suffixes=("", "_dup"))
        if "weighted_pred_dup" in merged.columns:
            merged["weighted_pred"] += merged["weighted_pred_dup"]
            merged.drop(columns=["weighted_pred_dup"], inplace=True)

    total_weight = sum(weight_dict.values())
    merged["loan_paid_back"] = merged["weighted_pred"] / total_weight

    # ðŸ”¥ Add small random noise (safe)
    merged["loan_paid_back"] += np.random.uniform(-0.002, 0.002, size=len(merged))
    merged["loan_paid_back"] = merged["loan_paid_back"].clip(0, 1)

    blended = merged[["id", "loan_paid_back"]]
    blended.to_csv(output_path, index=False)
    print(f"âœ… Blended submission saved to {output_path}")



# Define the main function
def main():
    # Define file paths and their respective weights
    weight_dict = {
        "/kaggle/input/predicting-loan-payback-vault/submission.csv": 3.7,
        "/kaggle/input/predicting-loan-payback-vault/submission (1).csv": 0.9,
    }

    # Call blend function
    blend_submissions(weight_dict, output_path="submission.csv")



# Call the main function
if __name__ == "__main__":
    main()

