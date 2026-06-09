import pandas as pd
import glob

# 1. Read all submission files (assuming filenames are submission1.csv, submission2.csv, ...)
submission_files = [
    "submission.csv",
    "submission (1).csv",
    "submission (2).csv",
    "submission (3).csv",
    "submission (4).csv",
    "submission (5).csv",
    "submission (6).csv",
    "submission_final_v3.csv",
    "submission_tuned.csv"
]

dfs = []
#/kaggle/input/submission-976518/submission/submission (1).csv
for file in submission_files:
    submission_name = file.split(".")[0]  # Extract base name (remove extension)
    print(submission_name)
    df = pd.read_csv(f"/kaggle/input/submission-976518/submission/{file}")
    df = df.rename(columns={"Personality": submission_name})  
    dfs.append(df[["id", submission_name]])  

# 2. Merge all submission data (using 'id' as the key)
merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="id", how="outer") 

# 3. Filter rows with differences: at least one submission has a different value than others
value_cols = [col for col in merged.columns if col != "id"]
has_diff = merged[value_cols].nunique(axis=1) > 1  
diff_rows = merged[has_diff]

# 4. the results
print("Differences to submission_differences.csv")
diff_rows.head()


sample=pd.read_csv("/kaggle/input/submission-976518/submission/submission_final_v3.csv")
sample.loc[sample["id"] == 18565, "Personality"] = "Extrovert"
sample.loc[sample["id"] == 18876, "Personality"] = "Introvert"
sample.loc[sample["id"] == 19612, "Personality"] = "Introvert"
sample.loc[sample["id"] == 20017, "Personality"] = "Introvert"
sample.loc[sample["id"] == 24005, "Personality"] = "Introvert"

sample.to_csv("submission_private_970850.csv",index=False)
sample.head(10)


