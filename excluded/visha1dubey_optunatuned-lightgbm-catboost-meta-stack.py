import pandas as pd
import numpy as np
import os


sub1 = pd.read_csv("/kaggle/input/optunatuned-lightgbm-catboost-with-meta-stacking/submission.csv")
sub1 = sub1.sort_values("id").reset_index(drop=True)
print("Main Optuna model loaded:", sub1.shape)


secondary_path = "/kaggle/working/submission_v1.csv"
if os.path.exists(secondary_path):
    sub2 = pd.read_csv(secondary_path).sort_values("id").reset_index(drop=True)
    print("Secondary model found:", sub2.shape)
    

    corr = sub1["loan_paid_back"].corr(sub2["loan_paid_back"])
    mean_diff = abs(sub1["loan_paid_back"] - sub2["loan_paid_back"]).mean()
    print(f"Correlation between models: {corr:.6f} | MeanDiff: {mean_diff:.5f}")
    
    if corr < 0.9 or mean_diff > 0.05:
        print("Secondary model appears random or too noisy — skipping blend.")
        sub_blend = sub1.copy()
    else:
        # Safe blend with 2.0 : 0.07 ratio
        w1, w2 = 2.0, 0.07
        sub_blend = sub1.copy()
        sub_blend["loan_paid_back"] = (w1 * sub1["loan_paid_back"] + w2 * sub2["loan_paid_back"]) / (w1 + w2)
        print("✅ Safe blended submission created (2.0 : 0.07 ratio).")
else:
    print("No secondary model found — using Optuna model only.")
    sub_blend = sub1.copy()


output_path = "/kaggle/working/submission.csv"
sub_blend.to_csv(output_path, index=False)

print(f"\nFinal submission file saved to: {output_path}")
print(sub_blend.head())
print("\nShape:", sub_blend.shape)
print("Range of predictions:",
      round(sub_blend['loan_paid_back'].min(), 6), "→",
      round(sub_blend['loan_paid_back'].max(), 6))


