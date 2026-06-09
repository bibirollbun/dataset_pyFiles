import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train.head()


# 1. Helper to grab the top N items from a pandas Series:
def top_n_list(series, n=3):
    return series.value_counts().index[:n].tolist()


train_split, valid_split = train_test_split(train, test_size=0.2, random_state=42)


df_counts = (
    train_split
    .groupby(["Crop Type", "Fertilizer Name"])
    .size()
    .reset_index(name="count")
)


df_sorted = df_counts.sort_values(
    by=["Crop Type", "count"], 
    ascending=[True, False]
)


top3_rows = df_sorted.groupby("Crop Type", as_index=False).head(3)


top3_rows.head()


result = top3_rows[["Crop Type", "Fertilizer Name"]].reset_index(drop=True)


result.head()


result


result.rename(columns={'Fertilizer Name': 'Fertilizer Name_pred'}, inplace=True)


train_split = pd.merge(train_split,result,on='Crop Type')


mapk(train_split['Fertilizer Name'],train_split['Fertilizer Name_pred'])


valid_split = pd.merge(valid_split,result,on='Crop Type')


valid_split


mapk(valid_split['Fertilizer Name'],valid_split['Fertilizer Name_pred'])


sample_submission = pd.merge(test,result,on='Crop Type')


sample_submission


sample_submission.rename(columns={'Fertilizer Name_pred': 'Fertilizer Name'}, inplace=True)



collapsed = (
    sample_submission
    .groupby("id")["Fertilizer Name"]
    .apply(lambda names: " ".join(names))
    .reset_index(name="Fertilizer Name")
)


sample_submission


collapsed


submission = collapsed.copy()
submission.to_csv("submission.csv", index=False, header=True)
print("✅ submission.csv збережено.")


submission

