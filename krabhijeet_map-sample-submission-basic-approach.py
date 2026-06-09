import pandas as pd
import numpy as np


test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
test.head()


sample_submission = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
sample_submission.head()


sample_submission.iloc[0,1]


submit_df = pd.DataFrame()
submit_df["row_id"] = test["row_id"]
submit_df["Category:Misconception"] = sample_submission.iloc[0,1]
submit_df.head()


submit_df.to_csv("submission.csv", index=False)

