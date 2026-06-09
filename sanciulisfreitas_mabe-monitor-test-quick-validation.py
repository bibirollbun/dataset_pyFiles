# MABe Test - Quick Validation
import time
import pandas as pd
from datetime import datetime

print('Starting MABe test - ' + str(datetime.now()))
print('Testing basic functionality...')

# Simulate processing
for i in range(5):
    print(f'Processing step {i+1}/5...')
    time.sleep(2)

# Create MABe submission
submission_data = {
    'row_id': [0, 1, 2],
    'video_id': [101686631, 101686631, 101686631],
    'agent_id': ['mouse1', 'mouse2', 'mouse1'],
    'target_id': ['mouse2', 'mouse1', 'mouse2'],
    'action': ['sniff', 'approach', 'chase'],
    'start_frame': [100, 250, 400],
    'stop_frame': [150, 280, 450]
}

df = pd.DataFrame(submission_data)
df.to_csv('submission.csv', index=False)

print('Success! Submission created:')
print(df)
print('MABe test completed successfully!')
print('File submission.csv saved')

