#Import necessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Preparing Data 
# This dataset now includes a 'home_court' feature.
data = {
    'team_a_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'team_b_id': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    'team_a_rating': [85, 92, 78, 88, 95, 80, 89, 75, 91, 84, 93, 79, 87, 90, 83, 96],
    'team_b_rating': [70, 85, 80, 90, 82, 75, 92, 70, 88, 86, 77, 81, 84, 89, 76, 74],
    'home_court_advantage': [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0], # 1 if team A is home, 0 otherwise
    'team_a_won': [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1] # 1 if team A won, 0 otherwise
}
df = pd.DataFrame(data)
df


#Showing Data 

print("Original Data with Home Court Advantage:")
print(df)
print("\n" + "="*50 + "\n")


# Feature Engineering 

# We now use both rating difference and home court advantage as features. 


df['rating_diff'] = df['team_a_rating'] - df['team_b_rating']

df['rating_diff'] # # Instead of giving the model two separate ratings (team_a_rating and team_b_rating) and forcing it to learn how to compare them, you provide a single feature that directly represents the difference in their strengths
# Our features now include rating difference and home court.
X = df[['rating_diff', 'home_court_advantage']]
y = df['team_a_won'] 




# the Final data will be 

# X for Features and y for Output 

X , y


# Split the data into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_test, y_train, y_test


# Train a Machine Learning Model ---
# We'll use Logistic Regression for this example.
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model training complete.")
print("Model coefficients:", model.coef_)
print("\n" + "="*50 + "\n")



#  Make predictions and evaluate the model ---
# For competition-style scoring, log loss is often used.
y_pred_proba = model.predict_proba(X_test)[:, 1] # Get probabilities for Team A winning
logloss_score = log_loss(y_test, y_pred_proba)

y_pred_proba
print(f"Model Log Loss on Test Data: {logloss_score:.4f}")
print("\n" + "="*50 + "\n")


# Function to simulate a tournament ---
def simulate_tournament(teams, model):
    """
    Simulates a tournament bracket for a given set of teams using the trained model.
    teams: a list of dictionaries, where each dict has 'id' and 'rating'.
    model: the trained Logistic Regression model.
    """
    if len(teams) % 2 != 0 or len(teams) == 0:
        print("Invalid number of teams. Must be a power of 2.")
        return []

    print(f"Simulating a tournament with {len(teams)} teams...")

    round_teams = list(teams)
    while len(round_teams) > 1:
        winners = []
        # Pair up teams for the current round
        for i in range(0, len(round_teams), 2):
            team_a = round_teams[i]
            team_b = round_teams[i+1]
            
            # Predict the winner using our model (we assume no home-court advantage in a neutral tournament)
            rating_diff = team_a['rating'] - team_b['rating']
            matchup_features = pd.DataFrame({'rating_diff': [rating_diff], 'home_court_advantage': [0]})
            
            prob_a_wins = model.predict_proba(matchup_features)[0][1]

            # The winner is the team with the higher probability of winning
            winner = team_a if prob_a_wins > 0.5 else team_b

            print(f"Matchup: Team {team_a['id']} (rating: {team_a['rating']}) vs Team {team_b['id']} (rating: {team_b['rating']})")
            print(f"  -> Predicted probability of Team {team_a['id']} winning: {prob_a_wins:.2f}")
            print(f"  -> Winner: Team {winner['id']}")
            
            winners.append(winner)

        round_teams = winners
        print("-" * 20)
    
    return round_teams[0]



# Example Tournament Simulation ---
# Define a set of teams for a small, 8-team tournament
tournament_teams = [
    {'id': 1, 'rating': 95}, {'id': 2, 'rating': 80},
    {'id': 3, 'rating': 90}, {'id': 4, 'rating': 85},
    {'id': 5, 'rating': 75}, {'id': 6, 'rating': 88},
    {'id': 7, 'rating': 92}, {'id': 8, 'rating': 70},
]

champion = simulate_tournament(tournament_teams, model)
print(f"\nTournament Champion: Team {champion['id']} with a rating of {champion['rating']}")





