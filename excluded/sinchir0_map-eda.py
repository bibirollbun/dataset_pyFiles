import pandas as pd
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sample_submission = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")


train.head()


train.shape


train["QuestionId"].value_counts()


train["QuestionText"][0]


train["QuestionText"][10000]


train["QuestionText"].value_counts()


train["QuestionText"].value_counts(normalize=True)


train["MC_Answer"][0]


train["MC_Answer"].value_counts()


train["MC_Answer"].value_counts(normalize=True)


train["StudentExplanation"][0]


train["StudentExplanation"].value_counts()
# Is the response open-ended? It feels a bit strange that there are multiple identical answers.


train["Category"][0]


train["Category"].value_counts()

# [A]_[B]
# A: Determines whether the selected answer is correct. (True or False in Category; e.g., True_Correct)
# B: Assesses whether the explanation contains a misconception. (Correct, Misconception, or Neither in Category; e.g., True_Correct)


train_true_correct = train[train["Category"] == "True_Correct"].reset_index(drop=True)
train_false_misconception = train[train["Category"] == "False_Misconception"].reset_index(drop=True)
train_false_neither = train[train["Category"] == "False_Neither"].reset_index(drop=True)
train_true_neither = train[train["Category"] == "True_Neither"].reset_index(drop=True)
train_true_misconception = train[train["Category"] == "True_Misconception"].reset_index(drop=True)
train_false_correct = train[train["Category"] == "False_Correct"].reset_index(drop=True)


def check_df(df: pd.DataFrame, idx: int) -> None:
    row = df.iloc[idx, :]

    print("QuestionId")
    print(row["QuestionId"])
    print("\n")
    
    print("QuestionText")
    print(row["QuestionText"])
    print("\n")

    print("MC_Answer")
    print(row["MC_Answer"])
    print("\n")

    print("StudentExplanation")
    print(row["StudentExplanation"])
    print("\n")

    print("Category")
    print(row["Category"])
    print("\n")

    print("Misconception")
    print(row["Misconception"])


check_df(train_true_correct, 0)


check_df(train_true_correct, 1000)


check_df(train_false_misconception, 0)


check_df(train_false_neither, 0)


check_df(train_false_neither, 100)


check_df(train_true_neither, 0)


check_df(train_true_neither, 10)


check_df(train_true_misconception, 0)


check_df(train_true_misconception, 10)


check_df(train_false_correct, 10)


train["Misconception"].value_counts()


test


sample_submission


sample_submission["Category:Misconception"][0]

