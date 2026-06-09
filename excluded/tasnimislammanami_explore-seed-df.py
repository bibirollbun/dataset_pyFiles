import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_squared_error


# Submission file
submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv')


w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)


def extract_seed_value(seed_str):
    # Extract seed value
    try:
        return int(seed_str[1:])
    # Set seed to 16 for unselected teams and errors
    except ValueError:
        return 16


seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)


seed_df.head()


seed_df['TeamID'].value_counts()


seed_df['Conference'] = seed_df['Seed'].str[0]


seed_df.head()


seed_df['Conference'].value_counts()


seed_df['SeedValue'].value_counts()

