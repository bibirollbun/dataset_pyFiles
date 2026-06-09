# -------------------------------------------------------------
# MOVIE RECOMMENDATION SYSTEM (TF-IDF + COSINE SIMILARITY)
# -------------------------------------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
# You can upload a CSV or use Kaggle's TMDB dataset
df = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")

# Clean the 'overview' column (replace NaN)
df['overview'] = df['overview'].fillna('')

# Step 1: TF-IDF Vectorizer (NLP)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Step 2: Calculate Cosine Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 3: Helper function — recommend movies
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend_movie(title, num_recommendations=10):
    if title not in indices:
        return f"Movie '{title}' not found in dataset."

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Test the system
movie_name = "The Dark Knight"
print("Recommended movies for:", movie_name)
print(recommend_movie(movie_name))


