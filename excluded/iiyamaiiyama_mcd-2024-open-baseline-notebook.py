import pandas as pd


data_dir = "/kaggle/input/mcd-data-science-competition-2024-open/"


article_path = data_dir + "article.csv"
sample_submission_path = data_dir + "sample_submission.csv"
test_path = data_dir + "test.csv"
train_path = data_dir + "train.csv"


article_df = pd.read_csv(article_path)
sample_submission_df = pd.read_csv(sample_submission_path)
test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)


article_df.shape,sample_submission_df.shape,test_df.shape,train_df.shape


train_df.head(3)


row = train_df.iloc[0]
print(row.quiz_text)
print("-"*30)
print(row.choice_0)
print(row.choice_1)
print(row.choice_2)
print(row.choice_3)

print("-"*30)
print(row.answer)


article_df.head(3)


print(article_df.query("article_id == 37494513").iloc[0].article_text)


test_df.head(3)


sample_submission_df.head()


sample_submission_df.answer.value_counts()


sample_submission_df.to_csv("submission.csv",index=None)

