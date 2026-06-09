import glob
import pandas as pd
from sklearn.metrics import log_loss
import numpy as np 


submission_files = glob.glob("/kaggle/input/tabular-playground-series-nov-2022/submission_files/*.csv")
# print(submission_files)

# dfs = [pd.read_csv(f).set_index("id") for f in submission_files]


submission_format = pd.read_csv("/kaggle/input/tabular-playground-series-nov-2022/sample_submission.csv", index_col='id')


sub_ids = submission_format.index
print(f"sub_ids:{len(sub_ids)}")
print(sub_ids)


dfs = [pd.read_csv(f).set_index("id") for f in submission_files]


model_names = [f"model_{i}" for i in range(len(dfs))]


labels = pd.read_csv("/kaggle/input/tabular-playground-series-nov-2022/train_labels.csv").set_index("id")
print(labels.head())
gt_ids = labels.index
print(f"sub_ids:{len(gt_ids)}")


missing_ids = dfs[0].index.difference(labels.index)
print(missing_ids)


common_index = dfs[0].index.intersection(labels.index)
y_true = labels.loc[common_index, "label"].values


pred_matrix = np.column_stack([df["pred"].values for df in dfs])


results = {}


simple_avg = np.mean(pred_matrix, axis=1)
print(simple_avg)


print(len(y_true))


print(y_true)


subset = simple_avg[2000 : 2000 + 2000]
print(subset)


y_true_subset = y_true[0:2000]


y_true_subset = y_true[0:2000]

subset_loss = log_loss(y_true_subset, np.clip(subset, 1e-5, 1 - 1e-5))
results["Subset Loss"] = subset_loss


print(results)


pred_series = pd.Series(simple_avg,index=dfs[0].index)
clipped_pred = np.clip(pred_series.loc[gt_ids], 1e-5, 1 - 1e-5)
print(clipped_pred)


final_score = log_loss(y_true, clipped_pred)
print(f"log Loss = {final_score:.6f}")


final_submission = pd.DataFrame({'pred':simple_avg}, index=dfs[0].index)


print(final_submission)


print(final_submission.index)
print(final_submission.head())


final_submission.loc[sub_ids].to_csv('submission.csv')

