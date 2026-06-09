import pandas as pd
results = pd.read_csv("/kaggle/input/submission/submission.csv")
results.head()
results.to_csv("submission.csv", index=False)



