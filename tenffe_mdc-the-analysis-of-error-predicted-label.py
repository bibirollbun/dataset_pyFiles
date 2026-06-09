import pandas as pd


train_label_path = "/kaggle/input/make-data-count-finding-data-references/train_labels.csv"
train_label_df = pd.read_csv(train_label_path)
submission_df = pd.read_csv("/kaggle/input/lb-llm-baseline/submission.csv")

train_label_df.shape, submission_df.shape


train_label_df.sample(10)


# Validation scoring (keeping your existing evaluation)
def f1_score(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    
pred_df = submission_df
label_df = train_label_df
label_df = label_df[label_df['type'] != 'Missing'].reset_index(drop=True)

hits_df = label_df.merge(pred_df, on=["article_id", "dataset_id", "type"])

tp = hits_df.shape[0]
fp = pred_df.shape[0] - tp
fn = label_df.shape[0] - tp

print("\nValidation Results:")
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("F1 Score:", round(f1_score(tp, fp, fn), 3))


analysis_df = label_df.merge(pred_df, on=["article_id"], how="outer")
analysis_df.shape


analysis_df


def df2dict(df):
    to_dict = {}
    for row_id in range(df.shape[0]):
        row = df.iloc[row_id]
        article_id = row["article_id"]
        dataset_id = row["dataset_id"]
        dataset_type = row["type"]
    
        if article_id in to_dict.keys():
            to_dict[article_id].append(
                [dataset_id, dataset_type]
            )
        else:
            to_dict[article_id] = [
                [dataset_id, dataset_type]
            ]
    return to_dict


label_dict = df2dict(label_df)
pred_dict = df2dict(submission_df)

len(label_dict.keys()), len(pred_dict.keys())


len(label_dict.keys()), len(label_df)


len(pred_dict.keys()), len(submission_df)




