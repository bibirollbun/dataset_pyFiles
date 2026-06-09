import pandas as pd
import os

df_forum = pd.read_csv("/kaggle/input/meta-kaggle/ForumMessages.csv")


# df_forum.info()
# df_forum.describe(include='all')
df_forum.isnull().sum()


df_forum['PostDate'] = pd.to_datetime(df_forum['PostDate'])
df_forum['year'] = df_forum['PostDate'].dt.year

# Messages over time
df_forum['year'].value_counts().sort_index().plot(kind='bar', title="Messages per Year")



top_users = df_forum['PostUserId'].value_counts().head(10)
print(top_users)

# Activity timeline for a prolific user
user_id = top_users.index[0]
df_forum[df_forum['PostUserId'] == user_id]['PostDate'].dt.date.value_counts().sort_index().plot()



df_forum['message_len'] = df_forum['Message'].str.len()
df_forum['message_len'].hist(bins=50)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = tfidf.fit_transform(df_forum['Message'].iloc[:500:].dropna())
X_pca = PCA(n_components=2).fit_transform(X_tfidf.toarray())



!pip install bertopic -q


from bertopic import BERTopic

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(df_forum['Message'].iloc[-1000:].tolist())
topic_model.visualize_topics(top_n_topics=5)



import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load and prepare
df = pd.read_csv("/kaggle/input/meta-kaggle/ForumMessages.csv")
df = df.dropna(subset=["Message", "ForumTopicId"])  # Drop missing
grouped = df.groupby('ForumTopicId')['Message'].apply(lambda msgs: ' '.join(msgs)).reset_index()


def generate_wordcloud(text, title=None):
    wc = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16)
    plt.show()



# Top 5 most active topics
top_topics = df['ForumTopicId'].value_counts().head(5).index.tolist()

for topic_id in top_topics:
    text = grouped[grouped['ForumTopicId'] == topic_id]['Message'].values[0]
    generate_wordcloud(text, title=f"ForumTopicId: {topic_id}")




