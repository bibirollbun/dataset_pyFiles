# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install spotipy ytmusicapi --quiet
!pip install scikit-surprise --quiet



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

print("ğŸ�µ Music Recommendation AI Agent")
print("<="*50)




import pandas as pd
import random

# ------------------------------
# Step 1: Generate the songs dataset
# ------------------------------

genres = ["Pop", "Rock", "Hip-Hop", "Jazz", "EDM", "Country", "R&B", "Metal"]
artists_by_genre = {
    "Pop": ["Taylor Swift", "Ed Sheeran", "Ariana Grande"],
    "Rock": ["Imagine Dragons", "Queen", "Coldplay"],
    "Hip-Hop": ["Drake", "Kendrick Lamar", "Cardi B"],
    "Jazz": ["Miles Davis", "John Coltrane", "Ella Fitzgerald"],
    "EDM": ["Calvin Harris", "Marshmello", "David Guetta"],
    "Country": ["Luke Bryan", "Kacey Musgraves", "Carrie Underwood"],
    "R&B": ["The Weeknd", "Alicia Keys", "Bruno Mars"],
    "Metal": ["Metallica", "Iron Maiden", "Slipknot"]
}

def generate_songs():
    songs = []
    song_id = 1
    
    for genre in genres:
        for artist in artists_by_genre[genre]:
            for i in range(random.randint(3, 5)):
                songs.append({
                    'song_id': song_id,
                    'title': f"Song {song_id}",
                    'artist': artist,
                    'genre': genre,
                    'tempo': random.randint(60, 180),
                    'energy': round(random.uniform(0.1, 1.0), 2),
                    'danceability': round(random.uniform(0.1, 1.0), 2),
                    'popularity': random.randint(1, 100),
                    'year': random.randint(2000, 2023)
                })
                song_id += 1
    return pd.DataFrame(songs)

df_songs = generate_songs()
print("Sample of generated songs:")
print(df_songs.head(10))



user_energy = 0.7
user_danceability = 0.6


recommended_songs = df_songs[
    (df_songs['energy'] >= user_energy) &
    (df_songs['danceability'] >= user_danceability)
]


recommended_artists = recommended_songs.groupby('artist')['popularity'].mean().sort_values(ascending=False).head(3)
print("\nRecommended artists for you:")
print(recommended_artists)



import pandas as pd
import random

genres = ["Pop", "Rock", "Hip-Hop", "Jazz", "EDM", "Country", "R&B", "Metal"]
artists_by_genre = {
    "Pop": ["Taylor Swift", "Ed Sheeran", "Ariana Grande"],
    "Rock": ["Imagine Dragons", "Queen", "Coldplay"],
    "Hip-Hop": ["Drake", "Kendrick Lamar", "Cardi B"],
    "Jazz": ["Miles Davis", "John Coltrane", "Ella Fitzgerald"],
    "EDM": ["Calvin Harris", "Marshmello", "David Guetta"],
    "Country": ["Luke Bryan", "Kacey Musgraves", "Carrie Underwood"],
    "R&B": ["The Weeknd", "Alicia Keys", "Bruno Mars"],
    "Metal": ["Metallica", "Iron Maiden", "Slipknot"]
}

def generate_songs():
    songs = []
    song_id = 1
    for genre in genres:
        for artist in artists_by_genre[genre]:
            for i in range(random.randint(3,5)):
                songs.append({
                    'song_id': song_id,
                    'title': f"Song {song_id}",
                    'artist': artist,
                    'genre': genre,
                    'tempo': random.randint(60, 180),
                    'energy': round(random.uniform(0.1,1.0),2),
                    'danceability': round(random.uniform(0.1,1.0),2),
                    'popularity': random.randint(1,100),
                    'year': random.randint(2000,2023)
                })
                song_id += 1
    return pd.DataFrame(songs)

music_df = generate_songs()
print("âœ… Songs dataset created")
print(music_df.head())

def create_user_profiles():
    """Simulated user profiles with genre preferences"""
    user_types = {
        'rock_lover': {'Rock': 0.8, 'Pop': 0.1, 'Hip-Hop': 0.1},
        'pop_enthusiast': {'Pop': 0.7, 'R&B': 0.2, 'EDM': 0.1},
        'hiphop_fan': {'Hip-Hop': 0.7, 'R&B': 0.2, 'Pop': 0.1},
        'eclectic_listener': {genre: 1/len(music_df['genre'].unique()) for genre in music_df['genre'].unique()},
        'jazz_connoisseur': {'Jazz': 0.9, 'Classical': 0.1}
    }
    
    users = []
    user_id = 1
    for user_type, genre_prefs in user_types.items():
        for _ in range(random.randint(2,3)):
            users.append({
                'user_id': user_id,
                'user_type': user_type,
                'age': random.randint(18,40),
                'genre_preferences': genre_prefs
            })
            user_id += 1
    return pd.DataFrame(users)

users_df = create_user_profiles()
print(f"\nâœ… Created {len(users_df)} user profiles")
print(users_df.head())

def recommend_songs_for_user(user_id, top_n=3):
    user = users_df[users_df['user_id']==user_id].iloc[0]
    prefs = user['genre_preferences']
    
   
    music_df['score'] = music_df['genre'].apply(lambda g: prefs.get(g,0))
    
   
    recommended = music_df.sort_values('score', ascending=False).head(top_n)
    return recommended[['title','artist','genre','score']]


print(f"\nğŸ�µ Recommendations for user_id 1:")
print(recommend_songs_for_user(1))




from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import random


music_df = pd.DataFrame({
    'song_id': range(1, 21),
    'title': [f"Song {i}" for i in range(1, 21)],
    'artist': ["Artist A", "Artist B", "Artist C", "Artist D"]*5,
    'genre': ["Pop", "Rock", "Hip-Hop", "Jazz", "EDM"]*4,
    'tempo': [random.randint(60,180) for _ in range(20)],
    'energy': [round(random.uniform(0.1,1.0),2) for _ in range(20)],
    'danceability': [round(random.uniform(0.1,1.0),2) for _ in range(20)],
    'popularity': [random.randint(1,100) for _ in range(20)],
    'year': [random.randint(2000,2023) for _ in range(20)]
})


users_df = pd.DataFrame({
    'user_id': [1,2,3],
    'user_type': ['pop_enthusiast','rock_lover','hiphop_fan'],
    'age': [25,30,22],
    'genre_preferences': [
        {'Pop':0.7,'R&B':0.2,'EDM':0.1},
        {'Rock':0.8,'Pop':0.1,'Hip-Hop':0.1},
        {'Hip-Hop':0.7,'R&B':0.2,'Pop':0.1}
    ]
})


class MusicRecommendationAgent:
    def __init__(self, music_data, users_data):
        """Initialize the music recommendation agent"""
        self.music_df = music_data
        self.users_df = users_data
        self.recommendation_history = defaultdict(list)
        
        print("\nğŸ¤– Music Recommendation AI Agent Initialized!")
        print(f"   â€¢ Songs in database: {len(music_data)}")
        print(f"   â€¢ Users in system: {len(users_data)}")
    
    def get_user_mood(self):
        """Ask user for their current mood"""
        print("\nğŸ˜Š How are you feeling right now?")
        moods = {
            '1': ('Happy/Upbeat', {'energy': 0.8, 'danceability': 0.7, 'tempo': (120, 180)}),
            '2': ('Relaxed/Chill', {'energy': 0.3, 'danceability': 0.4, 'tempo': (60, 100)}),
            '3': ('Energetic/Workout', {'energy': 0.9, 'danceability': 0.8, 'tempo': (140, 180)}),
            '4': ('Focused/Study', {'energy': 0.5, 'danceability': 0.3, 'tempo': (80, 120)}),
            '5': ('Romantic', {'energy': 0.4, 'danceability': 0.6, 'tempo': (70, 110)})
        }
        
        for key, (mood_name, _) in moods.items():
            print(f"   {key}. {mood_name}")
        
        while True:
            choice = input("\nEnter your choice (1-5): ").strip()
            if choice in moods:
                mood_name, mood_filters = moods[choice]
                print(f"ğŸ�¯ Selected mood: {mood_name}")
                return mood_name, mood_filters
            else:
                print("â�Œ Please enter a number between 1 and 5")
    
    def get_user_preferences(self):
        """Get user's genre preferences"""
        print("\nğŸ�­ What type of music do you generally enjoy?")
        all_genres = sorted(self.music_df['genre'].unique())
        
        for i, genre in enumerate(all_genres, 1):
            print(f"   {i}. {genre}")
        print(f"   {len(all_genres)+1}. Mix of everything")
        
        while True:
            try:
                choice = int(input(f"\nEnter your choice (1-{len(all_genres)+1}): "))
                if 1 <= choice <= len(all_genres):
                    return [all_genres[choice-1]]
                elif choice == len(all_genres) + 1:
                    return all_genres
                else:
                    print(f"â�Œ Please enter a number between 1 and {len(all_genres)+1}")
            except ValueError:
                print("â�Œ Please enter a valid number")
    
    def filter_by_mood(self, mood_filters):
        """Filter songs based on mood"""
        filtered_songs = self.music_df.copy()
        if 'energy' in mood_filters:
            filtered_songs = filtered_songs[filtered_songs['energy'] >= mood_filters['energy']]
        if 'danceability' in mood_filters:
            filtered_songs = filtered_songs[filtered_songs['danceability'] >= mood_filters['danceability']]
        if 'tempo' in mood_filters:
            min_tempo, max_tempo = mood_filters['tempo']
            filtered_songs = filtered_songs[(filtered_songs['tempo']>=min_tempo)&(filtered_songs['tempo']<=max_tempo)]
        return filtered_songs
    
    def recommend_songs(self, genres, mood_filters, n_recommendations=10):
        """Generate song recommendations"""
        mood_filtered = self.filter_by_mood(mood_filters)
        genre_filtered = mood_filtered[mood_filtered['genre'].isin(genres)]
        if len(genre_filtered)==0:
            print("âš   Not enough songs matching your criteria. Expanding search...")
            genre_filtered = mood_filtered
        genre_filtered = genre_filtered.copy()
        genre_filtered['score'] = genre_filtered['popularity']*0.4 + genre_filtered['energy']*100*0.3 + genre_filtered['danceability']*100*0.3
        recommendations = genre_filtered.sort_values('score',ascending=False).head(n_recommendations)
        return recommendations
    
    def create_playlist(self, recommendations, mood_name):
        """Format and display the playlist"""
        print(f"\nğŸ�µ Your '{mood_name}' Playlist")
        print("="*60)
        playlist = []
        for i, (_, song) in enumerate(recommendations.iterrows(), 1):
            playlist.append({
                'Track': i,
                'Song': song['title'],
                'Artist': song['artist'],
                'Genre': song['genre'],
                'Tempo': song['tempo'],
                'Energy': song['energy'],
                'Danceability': song['danceability']
            })
            print(f"{i:2d}. {song['artist']} - {song['title']} | {song['genre']} | Tempo: {song['tempo']} | Energy: {song['energy']:.2f} | Dance: {song['danceability']:.2f}")
        return pd.DataFrame(playlist)
    
    def analyze_recommendations(self, playlist_df):
        """Analyze and visualize the playlist"""
        if playlist_df.empty:
            return
        fig, axes = plt.subplots(2,2,figsize=(15,10))
        genre_counts = playlist_df['Genre'].value_counts()
        axes[0,0].pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Playlist Genre Distribution')
        scatter = axes[0,1].scatter(playlist_df['Energy'], playlist_df['Danceability'], c=playlist_df['Tempo'], cmap='viridis', s=100)
        axes[0,1].set_xlabel('Energy'); axes[0,1].set_ylabel('Danceability'); axes[0,1].set_title('Energy vs Danceability')
        plt.colorbar(scatter, ax=axes[0,1], label='Tempo (BPM)')
        axes[1,0].hist(playlist_df['Tempo'], bins=10, edgecolor='black', alpha=0.7)
        axes[1,0].set_xlabel('Tempo (BPM)'); axes[1,0].set_ylabel('Number of Songs'); axes[1,0].set_title('Tempo Distribution')
        axes[1,0].axvline(playlist_df['Tempo'].mean(), color='red', linestyle='--', label=f"Mean: {playlist_df['Tempo'].mean():.0f} BPM")
        axes[1,0].legend()
        axes[1,1].axis('off')
        summary_text = f"""
        ğŸ“Š PLAYLIST ANALYSIS
        
        Total Songs: {len(playlist_df)}
        Average Tempo: {playlist_df['Tempo'].mean():.0f} BPM
        Average Energy: {playlist_df['Energy'].mean():.2f}
        Average Danceability: {playlist_df['Danceability'].mean():.2f}
        Top Genre: {playlist_df['Genre'].mode()[0]}
        Most Frequent Artist: {playlist_df['Artist'].mode()[0]}
        """
        axes[1,1].text(0.1,0.5,summary_text, fontsize=12, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        plt.tight_layout()
        plt.show()
    
    def run_recommendation_engine(self):
        """Main method to run the recommendation engine"""
        print("\n" + "="*60)
        print("ğŸ�¶ WELCOME TO MUSIC RECOMMENDATION AI AGENT")
        print("="*60)
        mood_name, mood_filters = self.get_user_mood()
        preferred_genres = self.get_user_preferences()
        print(f"\nğŸ”� Finding perfect songs for your {mood_name} mood...")
        recommendations = self.recommend_songs(preferred_genres, mood_filters)
        playlist_df = self.create_playlist(recommendations, mood_name)
        print("\nğŸ“Š Analyzing your playlist...")
        self.analyze_recommendations(playlist_df)
        self.recommendation_history[mood_name].append({
            'genres': preferred_genres,
            'num_songs': len(playlist_df),
            'avg_energy': playlist_df['Energy'].mean(),
            'avg_tempo': playlist_df['Tempo'].mean()
        })
        return playlist_df

agent = MusicRecommendationAgent(music_df, users_df)
playlist = agent.run_recommendation_engine()



def main():
    """Main function to run the music recommendation agent"""
    
    agent = MusicRecommendationAgent(music_df, users_df)
    
    # Run recommendation engine
    playlist = agent.run_recommendation_engine()
    
    print("\n" + "="*60)
    print("ğŸ“ˆ RECOMMENDATION STATISTICS")
    print("="*60)
    
    print(f"\nâœ¨ Based on your preferences, we recommended:")
    print(f"   â€¢ {len(playlist)} songs total")
    print(f"   â€¢ Featuring {playlist['Artist'].nunique()} different artists")
    print(f"   â€¢ Across {playlist['Genre'].nunique()} music genres")
    
    top_artists = playlist['Artist'].value_counts().head(3)
    print(f"\nğŸ�¤ Top artists in your playlist:")
    for artist, count in top_artists.items():
        print(f"   â€¢ {artist}: {count} songs")
    
    avg_tempo = playlist['Tempo'].mean()
    if avg_tempo > 140:
        tempo_desc = "high-energy workout"
    elif avg_tempo > 120:
        tempo_desc = "upbeat and energetic"
    elif avg_tempo > 90:
        tempo_desc = "moderate pace"
    else:
        tempo_desc = "relaxed and chill"
    
    print(f"\nğŸ’¡ Fun fact: Your playlist has a {tempo_desc} vibe with average tempo of {avg_tempo:.0f} BPM!")
    
    return playlist

if __name__ == "__main__":
    playlist_df = main()



if __name__ == "__main__":
    
    print("ğŸ”§ Running in simulation mode for Kaggle...")
    print("   (In a local environment, you'd get interactive prompts)\n")
    
    simulated_mood = 'Happy/Upbeat'
    simulated_mood_filters = {'energy': 0.8, 'danceability': 0.7, 'tempo': (120, 180)}
    simulated_genres = ['Pop', 'Rock']
    
   
    agent = MusicRecommendationAgent(music_df, users_df)
    
    print(f"ğŸ�¯ Simulated User Preferences:")
    print(f"   â€¢ Mood: {simulated_mood}")
    print(f"   â€¢ Preferred Genres: {', '.join(simulated_genres)}")
    
   
    print(f"\nğŸ”� Finding perfect songs for your {simulated_mood} mood...")
    recommendations = agent.recommend_songs(simulated_genres, simulated_mood_filters)
    
    
    playlist_df = agent.create_playlist(recommendations, simulated_mood)
    
   
    print("\nğŸ“Š Analyzing your playlist...")
    agent.analyze_recommendations(playlist_df)
    
    print("\nâœ… Music Recommendation Complete!")
    print("ğŸ�§ Enjoy your personalized playlist!")


