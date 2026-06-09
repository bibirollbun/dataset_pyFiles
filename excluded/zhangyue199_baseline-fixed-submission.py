import numpy as np
import pandas as pd


train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")


train["Misconception"] = train["Misconception"].fillna("NA")
train["strata"] = train.apply(lambda row: row["Category"]+":"+row["Misconception"], axis=1)


values, counts = np.unique(train["strata"], return_counts=True)


sorted_idx = np.argsort(-counts)
values_sorted = values[sorted_idx]
counts_sorted = counts[sorted_idx]
print(values_sorted, counts_sorted)


p = " ".join(values_sorted[:3])
p = 'True_Correct:NA False_Neither:NA False_Misconception:Incomplete'
print(p)

preds = p * len(test.row_id.values)
sub = pd.DataFrame({"row_id": test.row_id.values, "Category:Misconception": preds})
sub.to_csv("submission.csv", index=False)
sub




