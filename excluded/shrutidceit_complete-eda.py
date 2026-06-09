import pandas as pd, numpy as np, re, textwrap, string
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from pathlib import Path



train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
sample_submission = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")


pd.set_option("display.max_colwidth", 200)


data = Path("../data")


train.columns = [c.strip().lower() for c in train.columns]
target = "rule_violation"
text_cols = ["body", "rule", "positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]
cat_cols = ["subreddit"]
id_cols = ["row_id"]


train.info()


train.sample(5, random_state=42)


train[target].value_counts(dropna=False).to_frame("count").assign(pct=lambda t: t["count"]/len(train))


train.isna().mean().sort_values(ascending=False)


train.duplicated().sum()


train[id_cols[0]].duplicated().sum()


matches = 0

for col in ["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]:
    matches += train.merge(
        train[["row_id", col]].rename(columns = {col:"x"}),
        left_on = "body",
        right_on = "x",
        how = "inner"
    ).shape[0]

matches


# top_ngrams() is defined to find the most frequent n-grams
def top_ngrams(texts, ngram_range = (2,2), topk=2000): 
    v = CountVectorizer(ngram_range=ngram_range, min_df=2) # min_df=2 means it only include phrases that appear in at least 2 documents
    X = v.fit_transform(texts)
    freqs = np.asarray(X.sum(0)).ravel()
    order = freqs.argsort()[::-1][:topk]
    vocab = np.array(sorted(v.vocabulary_.items(), key=lambda kv:kv[1]))[:, 0]
    return set(vocab[order])

body_bi = top_ngrams(train["body"].fillna(""),(2,2), 3000)
ex_all = pd.Series(train[text_cols[1:]].fillna("").agg(" ".join, axis=1))
ex_bi = top_ngrams(ex_all, (2,2), 3000)
len(body_bi & ex_bi )



# How often each text field is empty/whitespace

def empty_rate(s): 
    return s.fillna("").str.strip().eq("").mean()
pd.Series({c: empty_rate(train[c]) for c in text_cols + cat_cols})


# normalize() creates Simple normalized versions for analysis

def normalize(s):
    s = s.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()                           
                    # Replace NaN/None with empty string 
                    # Replace multiple whitespace (spaces, tabs, newlines) with a single space
                    # Remove leading and trailing spaces
    return s

for c in text_cols:
    train[c+"_clean"] = normalize(train[c])

train["subreddit"] = train["subreddit"].astype("category")


sub_counts = train["subreddit"].value_counts().to_frame("count").assign(pct=lambda t: t["count"]/len(train))
sub_counts.head(20)



# Calculating how often rule violations happen in each subreddit - only for subreddits that have enough data to be meaningful

tmp = (train.groupby("subreddit")[target]
         .agg(["count","mean"])
         .rename(columns={"mean":"violation_rate"})
         .query("count >= 30")
         .sort_values("violation_rate", ascending=False))

tmp.head(20)


rule_counts = (train["rule"].fillna("NA").value_counts()
               .to_frame("count")
               .assign(pct=lambda t: t["count"]/len(train)))

rule_counts.head(20)





# Violation rate per rule (for rules with support)
rule_stats = (train.assign(rule=train["rule"].fillna("NA"))
                .groupby("rule")[target]
                .agg(["count","mean"])
                .rename(columns={"mean":"violation_rate"})
                .query("count >= 20")
                .sort_values("violation_rate", ascending=False))
rule_stats.head(20)


sample_submission.to_csv("submission.csv", index=False)


# /kaggle/input/jigsaw-agile-community-rules/sample_submission.csv
# /kaggle/input/jigsaw-agile-community-rules/train.csv
# /kaggle/input/jigsaw-agile-community-rules/test.csv

