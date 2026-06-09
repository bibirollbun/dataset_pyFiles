import pandas as pd
import os
import numpy as np
import librosa

np.random.seed(42)


ROOT = '/kaggle/input/birdclef-2025/'
sub = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
train = pd.read_csv(ROOT+'train.csv')


'''
row_id: A slug of soundscape_[soundscape_id]_[end_time] for the prediction; e.g.,
Segment 00:15-00:20 of 1-minute test soundscape soundscape_12345.ogg has row ID soundscape_12345_20.
'''


def generate_sub():
    classes = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
    df = pd.DataFrame(columns = ['row_id'] + classes)
    row_id = []
    aux_count = 0
    for f in os.listdir(ROOT+'test_soundscapes'):   
        if os.path.isfile(os.path.join(ROOT+'test_soundscapes', f)):
            for i in range(1, 60//5+1):
                row_id.append('soundscape_'+f.split('.')[0]+f'_{i*5}')
                df.loc[aux_count, 'row_id'] = row_id[-1]
                df.loc[aux_count, classes[0]:] = [np.random.rand(len(classes)) for _ in classes]
                aux_count += 1
                
    df.to_csv('/kaggle/working/submission.csv', index=False)


'''
from Stefan Kahl's Notebook
https://www.kaggle.com/code/stefankahl/birdclef-2025-sample-submission/notebook
'''

import os
import librosa
import numpy as np
import pandas as pd


# Set seed
np.random.seed(42)

# Class labels from train audio
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))

# List of test soundscapes (only visible during submission)
test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]

# Open each soundscape and make predictions for 5-second segments
# Use pandas df with 'row_id' plus class labels as columns
predictions = pd.DataFrame(columns=['row_id'] + class_labels)
for soundscape in test_soundscapes:

    # Load audio
    sig, rate = librosa.load(path=soundscape, sr=None)

    # Split into 5-second chunks
    chunks = []
    for i in range(0, len(sig), rate*5):
        chunk = sig[i:i+rate*5]
        chunks.append(chunk)
        
    # Make predictions for each chunk
    for i, chunk in enumerate(chunks):
        
        # Get row id  (soundscape id + end time of 5s chunk)      
        row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'
        
        # Make prediction (let's use random scores for now)
        # scores = model.predict...
        scores = np.random.rand(len(class_labels))
        
        # Append to predictions as new row
        new_row = pd.DataFrame([[row_id] + list(scores)], columns=['row_id'] + class_labels)
        predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
        
# Save prediction as csv
predictions.to_csv('submission.csv', index=False)
predictions.head()

