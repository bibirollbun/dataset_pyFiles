import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#train.csv
train_csv = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
print(train_csv.shape)
print(train_csv.columns)


#test.csv
test_csv = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
print(test_csv.shape)
print(test_csv.columns)


#sample_submission.csv
sample_submission_csv = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")
print(sample_submission_csv.shape)
print(sample_submission_csv.columns)


def show_train_data(i):
    row_id = train_csv.loc[i, "row_id"]
    body = train_csv.loc[i, "body"]
    rule = train_csv.loc[i, "rule"]
    subreddit = train_csv.loc[i, "subreddit"]
    positive_example_1 = train_csv.loc[i, "positive_example_1"]
    positive_example_2 = train_csv.loc[i, "positive_example_2"]
    negative_example_1 = train_csv.loc[i, "negative_example_1"]
    negative_example_2 = train_csv.loc[i, "negative_example_2"]
    rule_violation = train_csv.loc[i, "rule_violation"]
    
    print(f"row_id : {row_id}")
    print(f"body : {body}")
    print(f"rule : {rule}")
    print(f"subreddit : {subreddit}")
    print(f"positive_example_1 : {positive_example_1}")
    print(f"positive_example_2 : {positive_example_2}")
    print(f"negative_example_1 : {negative_example_1}")
    print(f"negative_example_2 : {negative_example_2}")
    print(f"rule_violation : {rule_violation}")


train_csv[["rule", "rule_violation"]].value_counts().reset_index().sort_values(["rule", "rule_violation"])


train_csv["subreddit"].value_counts().reset_index()


train_csv[["subreddit", "rule", "rule_violation"]].value_counts().reset_index().sort_values(["subreddit", "rule", "rule_violation"])


NoAdv_0_list = train_csv.loc[((train_csv["subreddit"] == "legaladvice")
                            &(train_csv["rule"].str.contains("No Advertising"))
                            &(train_csv["rule_violation"] == 0)),
                            "row_id"].values
NoAdv_1_list = train_csv.loc[((train_csv["subreddit"] == "legaladvice")
                            &(train_csv["rule"].str.contains("No Advertising"))
                            &(train_csv["rule_violation"] == 1)),
                            "row_id"].values
Noleg_0_list = train_csv.loc[((train_csv["subreddit"] == "legaladvice")
                            &(train_csv["rule"].str.contains("No legal advice"))
                            &(train_csv["rule_violation"] == 0)),
                            "row_id"].values
Noleg_1_list = train_csv.loc[((train_csv["subreddit"] == "legaladvice")
                            &(train_csv["rule"].str.contains("No legal advice"))
                            &(train_csv["rule_violation"] == 1)),
                            "row_id"].values
print(len(NoAdv_0_list))
print(len(NoAdv_1_list))
print(len(Noleg_0_list))
print(len(Noleg_1_list))


show_train_data(NoAdv_0_list[0])


show_train_data(NoAdv_1_list[0])


show_train_data(Noleg_0_list[0])


show_train_data(Noleg_1_list[0])







