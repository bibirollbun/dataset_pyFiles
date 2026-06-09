import numpy as np
import pandas as pd

df= pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df.head(10)


df = df.drop(columns=['row_id','rule', 'subreddit', 'positive_example_1','positive_example_2','negative_example_1','negative_example_2'])	
df.head()


df['clean'] = df['body'].str.replace(r"<[^>]+>", " ", regex=True)
df['clean'] = df['body'].str.lower()
df['clean'] = df['clean'].str.replace(r'http\S+|www.\S+', '', regex=True)
df['clean'] = df['clean'].str.replace(r'\s+', ' ', regex=True).str.strip()
df['clean'] = df['clean'].str.replace(r'\[.*?\]\(.*?\)', '', regex=True)
df['clean'] = df['clean'].str.replace(r'[^a-z\s]', ' ', regex=True)
df['clean'] = df['clean'].str.replace(r'\s+', ' ', regex=True).str.strip()

df = df[['clean', 'rule_violation']]

df.head(20)


from tqdm import tqdm
import spacy

# Enable GPU
spacy.prefer_gpu()
print(f"Using GPU: {spacy.require_gpu()}")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])



tqdm.pandas()

def remove_stopwords(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])

# Apply stopword removal
df['clean'] = df['clean'].progress_apply(remove_stopwords)

# Drop old review column, keep clean text
df = df[['clean', 'rule_violation']]

df.head()

