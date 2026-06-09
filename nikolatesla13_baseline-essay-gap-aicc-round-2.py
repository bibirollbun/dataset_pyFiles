import pandas as pd


root_path = "/kaggle/input/essay-gap-aicc-round-2/essay-gap"


train_df = pd.read_csv(f"{root_path}/train.csv")
train_df.head()


test_df = pd.read_csv(f"{root_path}/test.csv")


submission = pd.DataFrame({"sampleID": test_df["sampleID"], "answer": 0})
submission.head()


submission.to_csv("submission.csv", index=False)

