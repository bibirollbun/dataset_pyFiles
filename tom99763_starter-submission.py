import numpy as np
import pandas as pd
import os


train_data = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
test_data = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
submit_sample = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/sample_submission.csv')


submit_df = pd.DataFrame()


mices = ['mouse1', 'mouse2']
actions = ['submit', 'avoid', 'attack', 'chase', 'sniff', 'mount']


submit_df['row_id'] = np.arange(0, test_data.shape[0])
submit_df['video_id'] = test_data.video_id
submit_df['agent_id'] = np.random.choice(mices, size = (test_data.shape[0],),  replace=True)
submit_df['target_id'] = np.random.choice(mices, size = (test_data.shape[0], ),  replace=True)
submit_df['action'] = 'attack' #np.random.choice(actions, size = (test_data.shape[0], ),  replace=True)
submit_df['start_frame'] = 0
submit_df['stop_frame'] = 10


submit_df


submit_df.to_csv('submission.csv')




