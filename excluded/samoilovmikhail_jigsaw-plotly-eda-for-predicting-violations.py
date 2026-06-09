import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "kaggle" 

# Text Analysis
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

# For semantic similarity and toxicity
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
submission_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')


print(f'Shape of train_df: {train_df.shape}')
print(f'Shape of test_df: {test_df.shape}')
print(f'Shape of submission_df: {submission_df.shape}')


display(train_df.head(1))
display(test_df.head(1))
display(submission_df.head(1))


dataframes = {
    'train_df': train_df,
    'test_df': test_df,
    'submission_df': submission_df
}

all_columns = sorted(set().union(*[df.columns for df in dataframes.values()]))
presence_matrix = pd.DataFrame(
    index=dataframes.keys(),
    columns=all_columns
)

for df_name, df in dataframes.items():
    for col in all_columns:
        presence_matrix.loc[df_name, col] = col in df.columns

presence_numeric = presence_matrix.astype(int)

plt.figure(figsize=(12, 4))
sns.heatmap(
    presence_numeric,
    annot=presence_matrix,
    cmap=['red', 'green'],
    cbar=False,
    fmt=''
)

plt.title('The presence of columns in dataframes', fontsize=16)
plt.xlabel('Columns')
plt.ylabel('Datasets')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


train_df['body_char_len'] = train_df['body'].apply(len)
train_df['body_word_count'] = train_df['body'].apply(lambda x: len(x.split()))

print("Descriptive statistics for new features:")
print(train_df[['body_char_len', 'body_word_count']].describe())

fig = make_subplots(rows=1, cols=2, subplot_titles=('Length distribution of comments (characters)', 'Distribution of the number of words'))

fig.add_trace(go.Histogram(x=train_df['body_char_len'], name='Length in characters', marker_color='#330C73'), row=1, col=1)
fig.add_trace(go.Histogram(x=train_df['body_word_count'], name='Number of words', marker_color='#7A4F9D'), row=1, col=2)

fig.update_layout(title_text='Basic characteristics of the comment text', showlegend=False)
fig.show()


fig = px.pie(train_df, 
             names='rule_violation', 
             title='Distribution of the target variable (0 - does not violate, 1 - violates)',
             hole=.3)
fig.show()


fig = px.bar(train_df['subreddit'].value_counts(),
             x=train_df['subreddit'].value_counts().index,
             y=train_df['subreddit'].value_counts().values,
             title='Distribution of comments by subreddits')
fig.show()


violation_rate_by_subreddit = train_df.groupby('subreddit')['rule_violation'].mean().sort_values(ascending=False)

fig = px.bar(violation_rate_by_subreddit,
             x=violation_rate_by_subreddit.index,
             y=violation_rate_by_subreddit.values,
             title='Average percentage of violations by subreddits',
             labels={'x': 'The subreddit', 'y': 'Percentage of violations'})
fig.show()


train_df['word_count'] = train_df['body'].apply(lambda x: len(x.split()))

fig = px.violin(train_df, x='rule_violation', y='word_count', 
                box=True, points="all",
                title='Distribution of the number of words for infringing and non-infringing comments')
fig.show()


stop_words = list(stopwords.words('english')) + list(stopwords.words('russian'))

def plot_top_ngrams(corpus, title, n_gram_range=(1,1), n=20):
    """A function for extracting and plotting the top N-grams."""
    vec = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    top_df = pd.DataFrame(words_freq[:n], columns=['N-gram', 'Frequency'])
    
    fig = px.bar(top_df, x='Frequency', y='N-gram', orientation='h',
                 title=title, color='Frequency', color_continuous_scale='Viridis')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.show()

# Separate into violating and non-violating
violating_comments = train_df[train_df['rule_violation'] == 1]['body']
non_violating_comments = train_df[train_df['rule_violation'] == 0]['body']

# Unigrams (single words)
plot_top_ngrams(violating_comments, 'Top 20 unigrams in offending comments', n_gram_range=(1,1))
plot_top_ngrams(non_violating_comments, 'Top 20 unigrams in NOT offending comments', n_gram_range=(1,1))

# Bigrams (pairs of words)
plot_top_ngrams(violating_comments, 'Top 20 bigrams in infringing comments', n_gram_range=(2,2))
plot_top_ngrams(non_violating_comments, 'Top 20 bigrams in NOT infringing comments', n_gram_range=(2,2))


train_df['polarity'] = train_df['body'].apply(lambda x: TextBlob(x).sentiment.polarity)
train_df['subjectivity'] = train_df['body'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

fig = make_subplots(rows=1, cols=2, subplot_titles=('Polarity (-1 to 1)', 'Subjectivity (0 to 1)'))
fig.add_trace(go.Violin(y=train_df['polarity'], x=train_df['rule_violation'],
                        legendgroup='0', scalegroup='0', name='Does not violate',
                        side='negative', box_visible=True, meanline_visible=True,
                        points='all', line_color='blue'), 1, 1)                      
fig.add_trace(go.Violin(y=train_df['polarity'], x=train_df['rule_violation'],
                        legendgroup='1', scalegroup='1', name='Violates',
                        side='positive', box_visible=True, meanline_visible=True,
                        points='all', line_color='red'), 1, 1)

fig.add_trace(go.Violin(y=train_df['subjectivity'], x=train_df['rule_violation'],
                        legendgroup='0', scalegroup='0', name='Does not violate',
                        side='negative', box_visible=True, meanline_visible=True,
                        points='all', line_color='blue', showlegend=False), 1, 2)
fig.add_trace(go.Violin(y=train_df['subjectivity'], x=train_df['rule_violation'],
                        legendgroup='1', scalegroup='1', name='Violates',
                        side='positive', box_visible=True, meanline_visible=True,
                        points='all', line_color='red', showlegend=False), 1, 2)

fig.update_layout(title_text="Analysis of tonality and subjectivity (TextBlob)", violingap=0, violinmode='overlay')
fig.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"The device is being used: {device}")

model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_toxicity = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def get_toxicity_score(texts, batch_size=32):
    scores = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
            outputs = model_toxicity(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            toxicity_prob = probabilities[:, -1].cpu().numpy()
            scores.extend(toxicity_prob)
    return scores

print("Calculating the toxicity of comments... It may take some time.")
train_df['toxicity_score'] = get_toxicity_score(train_df['body'].tolist())

fig = px.violin(train_df, 
                y="toxicity_score", 
                x="rule_violation", 
                color="rule_violation",
                box=True, 
                points="all", 
                title="The distribution of toxicity of comments",
                labels={"toxicity_score": "Toxicity assessment", "rule_violation": "Violation of the rule"})
fig.show()


print("Loading the SentenceTransformer model...")
model_st = SentenceTransformer('all-MiniLM-L6-v2', device=device)

cols_to_encode = ['body', 'rule', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']
embeddings = {}

for col in cols_to_encode:
    print(f"Column encoding: {col}...")
    embeddings[col] = model_st.encode(train_df[col].tolist(), show_progress_bar=True, convert_to_tensor=True)

def calculate_cosine_similarity(emb1, emb2):
    return util.cos_sim(emb1, emb2)

print("Calculation of cosine proximity...")
train_df['sim_body_rule'] = [calculate_cosine_similarity(emb1, emb2).item() for emb1, emb2 in zip(embeddings['body'], embeddings['rule'])]
train_df['sim_body_pos1'] = [calculate_cosine_similarity(emb1, emb2).item() for emb1, emb2 in zip(embeddings['body'], embeddings['positive_example_1'])]
train_df['sim_body_pos2'] = [calculate_cosine_similarity(emb1, emb2).item() for emb1, emb2 in zip(embeddings['body'], embeddings['positive_example_2'])]
train_df['sim_body_neg1'] = [calculate_cosine_similarity(emb1, emb2).item() for emb1, emb2 in zip(embeddings['body'], embeddings['negative_example_1'])]
train_df['sim_body_neg2'] = [calculate_cosine_similarity(emb1, emb2).item() for emb1, emb2 in zip(embeddings['body'], embeddings['negative_example_2'])]

train_df['sim_body_positive_mean'] = train_df[['sim_body_pos1', 'sim_body_pos2']].mean(axis=1)
train_df['sim_body_negative_mean'] = train_df[['sim_body_neg1', 'sim_body_neg2']].mean(axis=1)
train_df['diff_sim_pos_neg'] = train_df['sim_body_positive_mean'] - train_df['sim_body_negative_mean']


fig = px.violin(train_df, y='diff_sim_pos_neg', x='rule_violation', color='rule_violation',
                box=True, points='all',
                title='The difference is in similarity (Positive examples - Negative examples)',
                labels={'diff_sim_pos_neg': 'The difference of similarity', 'rule_violation': 'Violation'})
fig.show()

fig = px.scatter(train_df,
                 x='sim_body_positive_mean',
                 y='sim_body_negative_mean',
                 color=train_df['rule_violation'].astype(str),
                 opacity=0.7,
                 title='Similarity Space: Commentary vs Examples',
                 labels={
                     'sim_body_positive_mean': 'Similarity to POSITIVE examples',
                     'sim_body_negative_mean': 'Similarity to NEGATIVE examples'
                 },
                 hover_data=['body'])
fig.show()


numeric_cols_for_corr = [
    'rule_violation', 'body_char_len', 'body_word_count', 
    'polarity', 'subjectivity', 'toxicity_score',
    'sim_body_rule', 'sim_body_positive_mean', 'sim_body_negative_mean', 'diff_sim_pos_neg'
]
existing_cols = [col for col in numeric_cols_for_corr if col in train_df.columns]
corr_matrix = train_df[existing_cols].corr()

fig = px.imshow(corr_matrix,
                text_auto=True, 
                aspect="auto",
                color_continuous_scale='RdBu_r', 
                title='Heat map of correlations of generated features')
fig.show()

fig = px.violin(train_df, x='rule_violation', y='body_char_len', 
                color='rule_violation',
                facet_col='subreddit',
                facet_col_wrap=4,
                box=True,
                title='The length of the comment depends on the subreddit and the fact of the violation.',
                facet_row_spacing=0.02
               )

fig.update_layout(height=2000)
fig.show()

