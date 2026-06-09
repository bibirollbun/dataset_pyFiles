import pandas as pd
import matplotlib.pyplot as plt
import re


train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")


train.head()


# Check nulls and dtype
train.info()


# How many distinct questions are there in the data?

print(f" -> There are {train['QuestionId'].nunique()} unique question in the data.")
print(f" -> And the count for each question id is:\n{train['QuestionId'].value_counts()}")


# Does the question id match the question text everytime?

is_unique = train[["QuestionId", "QuestionText"]].drop_duplicates()
print("-> The count should be 15(total unique questions)")
print(len(is_unique))
print(is_unique)


# Are there more than 4 unique answers for each question?

print("-> The unique counts of answer associated with each question id must be 4")
train.groupby("QuestionId")["MC_Answer"].nunique()


# What does the distribution of answers for each question look like?

question_ids = train['QuestionId'].unique()

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(18, 20))
axes = axes.flatten()

for i, qid in enumerate(question_ids):
    ax = axes[i]
    counts = train[train['QuestionId'] == qid]['MC_Answer'].value_counts()
    counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(f'QuestionId: {qid}', fontsize=12)
    ax.set_xlabel('Answer')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=10) 

plt.tight_layout()
plt.show()


# Are there any non-alphanumeric characters in the data?

non_alnum = []
for i in train['StudentExplanation']:
    if pd.notnull(i):
        chars = re.findall(r'[^a-zA-Z0-9]', str(i))
        non_alnum.extend(chars)

unique_non_alnum = list(set(non_alnum))

print("-> Unique non-alphanumeric chars are", unique_non_alnum)

# I was not expecting emojis lol


# What is the distribution of categories?

train['Category'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# What is the split between true and false?

print(train['Category'].str.extract(r'^(True|False)', expand=False).value_counts())


# What are the misconception types and it's counts?

print(train["Misconception"].value_counts())
print("\n-> Total of unique misconception types is", train["Misconception"].nunique())


# What is the percent share of misconceptions out of all the records?

share = train["Misconception"].count()/train["QuestionId"].count()*100
print(round(share, 2),"%")


# Please add any more data checks that I missed

