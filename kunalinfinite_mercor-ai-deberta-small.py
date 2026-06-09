import pandas as pd
import os
import numpy as np


train_df=pd.read_csv("/kaggle/input/mercor-ai-detection/train.csv")
print(len(train_df))
display(train_df)


test_df=pd.read_csv("/kaggle/input/mercor-ai-detection/test.csv")
print(len(test_df))
display(test_df)


test_df["is_cheating"]=0.5


test_df


# # Keep only the required columns for submission
# submission = test_df[['id', 'is_cheating']]

# # Save to CSV
# submission.to_csv("submission.csv", index=False)



submission_df=pd.read_csv("/kaggle/input/mercor-ai-deberta-small-fivefolds-sevenepochs/submission_deberta_small_7epochs.csv")


submission_df.to_csv("submission.csv", index=False)


