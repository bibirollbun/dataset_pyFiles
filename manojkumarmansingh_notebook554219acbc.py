import os, pandas as pd

# Find the dataset folder automatically
base = "/kaggle/input"
test_path = train_path = None
for root, dirs, files in os.walk(base):
    if "test.csv" in files:
        test_path  = os.path.join(root, "test.csv")
    if "train.csv" in files:
        train_path = os.path.join(root, "train.csv")
    if test_path and train_path:
        break

print("Using test.csv:", test_path)
print("Using train.csv:", train_path)

# Safety check
assert test_path is not None,  "Couldn't find test.csv under /kaggle/input"
assert train_path is not None, "Couldn't find train.csv under /kaggle/input"

# Simple baseline: constant probability
test = pd.read_csv(test_path)
test["loan_paid_back"] = 0.5

# Make the submission file
submission = test[["id", "loan_paid_back"]]
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)

