!pip install autoviz


!pip install autogluon


%matplotlib inline


from autoviz.AutoViz_Class import AutoViz_Class
atv = AutoViz_Class()

atv.AutoViz(
    filename="/kaggle/input/playground-series-s5e4/train.csv",
    sep=",",
    depVar="Listening_Time_minutes",
    dfte=None, 
    # lowess=False,
    max_rows_analyzed=750000,
    max_cols_analyzed=30,
    chart_format="svg",
    verbose=1,
)   


import pandas as pd
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
print(train_df.shape)


import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os
from datetime import datetime

# Create paths
MODEL_DIR = "models/autogluon"
LOG_DIR = "logs"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Define features and target
target = 'Listening_Time_minutes'
features = [col for col in train_df.columns if col not in [target, 'id']]


# Initialize predictor
print("Initializing predictor...")
predictor = TabularPredictor(
    label=target,
    path=MODEL_DIR,
    eval_metric='root_mean_squared_error'  # Common metric for regression
)
# Train with default parameters
print("Training model...")
predictor.fit(
    train_data=train_df[features + [target]],
    presets="best_quality",
    time_limit= 1800 # 0.5 hour time limit
)
# Get model performance
print("\nModel Performance:")
leaderboard = predictor.leaderboard()
print(leaderboard)
# Save leaderboard
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
leaderboard.to_csv(f"{LOG_DIR}/autogluon_leaderboard_{timestamp}.csv", index=False)
print(f"\nTraining completed. Models saved in {MODEL_DIR}")
print(f"Leaderboard saved in {LOG_DIR}/autogluon_leaderboard_{timestamp}.csv")


test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_pred = predictor.predict(test_df[features])
df_subm = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv', index_col='id')
df_subm['Listening_Time_minutes'] = test_pred.values
df_subm.to_csv('submission.csv')
df_subm.head()

