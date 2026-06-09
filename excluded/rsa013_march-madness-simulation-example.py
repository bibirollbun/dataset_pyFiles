from march_madness_simulator import start_tournament
import pandas as pd

submission = pd.read_csv("/kaggle/input/march-ml-mania-2025-brier-score-prediction/submission.csv")

# initialize the Tournament class
tournament = start_tournament(season=2024, mw="M", submission=submission,)


from march_madness_simulator import simulate_n_tournaments

summary = simulate_n_tournaments(tournament, 1000)
summary



from march_madness_simulator import graph_games

graph_games(tournament)




