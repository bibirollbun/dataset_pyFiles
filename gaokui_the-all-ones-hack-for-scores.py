import pandas as pd

test_csv_path = "/kaggle/input/detect-ai-vs-human-generated-images/test.csv"
test_df = pd.read_csv(test_csv_path)

test_df["label"] = 1

submission_csv_path = "submission.csv"
test_df.to_csv(submission_csv_path, index=False)
print(f"Submission file saved at {submission_csv_path}")

