# --- 1. Import Essential Libraries ---
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')


# --- 2. Load and Preprocess Data ---
print("Loading data... This may take a moment.")

# Configuration
BASE_PATH = '/kaggle/input/meta-kaggle/'

# Load raw data
competitions_df = pd.read_csv(BASE_PATH + 'Competitions.csv')
teams_df = pd.read_csv(BASE_PATH + 'Teams.csv')
team_memberships_df = pd.read_csv(BASE_PATH + 'TeamMemberships.csv')
users_df = pd.read_csv(BASE_PATH + 'Users.csv')


# Merge tables to create a master dataframe of user participation history
print("Merging user participation data...")
user_teams = pd.merge(teams_df, team_memberships_df, on='Id')
user_competitions = pd.merge(user_teams, competitions_df, left_on='CompetitionId', right_on='Id')
user_participation_history_df = pd.merge(user_competitions, users_df[['Id', 'UserName']], left_on='UserId', right_on='Id')

# For clarity, select and rename key columns
user_participation_history_df = user_participation_history_df[[
    'UserId', 'UserName', 'CompetitionId', 'Title', 'HostSegmentTitle'
]].rename(columns={'HostSegmentTitle': 'CompetitionType'})

print("Data loading and merging complete.")


# --- 3. Feature Engineering & Profiling ---
# Create a numerical profile (vector) for every competition using TF-IDF.

print("Creating competition profiles using TF-IDF...")

# Handle potential missing values in competition type
competitions_df['CompetitionType'] = competitions_df['HostSegmentTitle'].fillna('General')

# Initialize the vectorizer. TF-IDF is robust for converting categorical text to numerical features.
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Create the matrix where each row is a competition and each column is a competition type score.
competition_tfidf_matrix = tfidf_vectorizer.fit_transform(competitions_df['CompetitionType'])

print("TF-IDF matrix created. Shape:", competition_tfidf_matrix.shape)


# --- 4. The Recommender Function ---
def recommend_competitions(user_id: int, top_n: int = 10) -> pd.DataFrame:
    """
    Recommends active Kaggle competitions for a given user.

    Args:
        user_id (int): The ID of the user to generate recommendations for.
        top_n (int): The number of recommendations to return.

    Returns:
        pd.DataFrame: A DataFrame containing the top N recommended competitions,
                      including their title, type, and similarity score.
    """
    
    # --- 4.1. User Profile Creation ---
    # The user's interest profile is the centroid (mean) of their past competition vectors.
    
    # Check if user exists in our history
    if user_id not in user_participation_history_df['UserId'].values:
        print(f"User with ID {user_id} not found in participation history. Cannot generate recommendations.")
        return pd.DataFrame()

    user_name = user_participation_history_df[user_participation_history_df['UserId'] == user_id]['UserName'].iloc[0]
    print(f"\nGenerating recommendations for user: '{user_name}' (ID: {user_id})")
    
    # Get the indices of competitions the user participated in
    past_competition_ids = user_participation_history_df[user_participation_history_df['UserId'] == user_id]['CompetitionId']
    competition_indices = competitions_df.index[competitions_df['Id'].isin(past_competition_ids)]
    
    # Retrieve the TF-IDF vectors for those competitions
    user_past_vectors = competition_tfidf_matrix[competition_indices]
    
    # Calculate the user's average interest vector
    user_profile_vector = user_past_vectors.mean(axis=0)
    
    # Convert from np.matrix to np.ndarray to prevent sklearn TypeError
    user_profile_array = np.asarray(user_profile_vector)

    # --- 4.2. Recommendation Generation ---
    # Compare the user's profile to all currently active competitions.

    # Identify active competitions
    competitions_df['DeadlineDate'] = pd.to_datetime(competitions_df['DeadlineDate'], errors='coerce')
    active_competitions = competitions_df[competitions_df['DeadlineDate'] > pd.to_datetime(datetime.now())].copy()
    
    # Get TF-IDF vectors for active competitions
    active_indices = active_competitions.index
    active_competition_vectors = competition_tfidf_matrix[active_indices]

    # Calculate cosine similarity between the user and all active competitions
    similarity_scores = cosine_similarity(user_profile_array.reshape(1, -1), active_competition_vectors)
    
    # --- 4.3. Format and Return Results ---
    active_competitions['SimilarityScore'] = similarity_scores[0]
    
    # Sort by score to get the top recommendations
    recommendations_df = active_competitions.sort_values(by='SimilarityScore', ascending=False)
    
    final_output = recommendations_df[['Title', 'CompetitionType', 'SimilarityScore']]
    
    return final_output.head(top_n)


# --- 5. Example Usage ---
# This block demonstrates how to use the function.

if __name__ == '__main__':
    # Using the most active user as our example
    EXAMPLE_USER_ID = user_participation_history_df['UserId'].value_counts().idxmax()
    
    # Get and print the recommendations
    top_recommendations = recommend_competitions(user_id=EXAMPLE_USER_ID, top_n=10)
    
    print("\n--- Top 10 Recommendations ---")
    print(top_recommendations)




