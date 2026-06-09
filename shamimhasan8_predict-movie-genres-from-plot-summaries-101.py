import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

train = pd.read_csv('/kaggle/input/predict-movie-genres-from-plot-summaries/train.csv')
test = pd.read_csv('/kaggle/input/predict-movie-genres-from-plot-summaries/test.csv')
movies_genres = pd.read_csv('/kaggle/input/predict-movie-genres-from-plot-summaries/movies_genres.csv')

train.head()


train.info()
train.describe(include='all')
train.isnull().sum()


movies_genres.head()
print(f"Total unique genres: {movies_genres['id'].nunique()}")
movies_genres['name'].value_counts().plot(kind='barh', figsize=(10,6))
plt.title("Available Genres in Dataset")
plt.show()



train['genre_ids'] = train['genre_ids'].apply(lambda x: ast.literal_eval(x))
all_genres = [g for sublist in train['genre_ids'] for g in sublist]

genre_counts = pd.Series(all_genres).value_counts().reset_index()
genre_counts.columns = ['genre_id', 'count']
genre_counts = genre_counts.merge(movies_genres, left_on='genre_id', right_on='id', how='left')

plt.figure(figsize=(12,6))
sns.barplot(x='count', y='name', data=genre_counts.sort_values('count', ascending=False))
plt.title("Most Common Genres in Training Data")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")
plt.show()



train['overview_length'] = train['overview'].astype(str).apply(len)
plt.figure(figsize=(8,5))
sns.histplot(train['overview_length'], bins=40, kde=True)
plt.title("Distribution of Overview Lengths")
plt.xlabel("Overview Length (characters)")
plt.show()



from wordcloud import WordCloud, STOPWORDS

text = " ".join(train['overview'].dropna().tolist())
wc = WordCloud(width=1200, height=600, background_color='white', stopwords=STOPWORDS).generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Movie Overviews")
plt.show()



train['num_genres'] = train['genre_ids'].apply(len)
sns.countplot(x='num_genres', data=train)
plt.title("Number of Genres per Movie")
plt.xlabel("Number of Genres")
plt.ylabel("Count")
plt.show()



print(f"Train movies: {len(train)}, Test movies: {len(test)}")

train['title_length'] = train['title'].astype(str).apply(len)
test['title_length'] = test['title'].astype(str).apply(len)

plt.figure(figsize=(8,5))
sns.kdeplot(train['title_length'], label='Train')
sns.kdeplot(test['title_length'], label='Test')
plt.legend()
plt.title("Distribution of Title Lengths (Train vs Test)")
plt.show()


