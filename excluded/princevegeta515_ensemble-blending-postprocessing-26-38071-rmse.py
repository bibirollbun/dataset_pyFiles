import pandas as pd


submission= pd.read_csv("/kaggle/input/1moregain/FinalRun_V2_1.csv")
print(submission.head(10))


submission.to_csv("submission.csv",index=False)
print("Submission Saved As submission.csv")

