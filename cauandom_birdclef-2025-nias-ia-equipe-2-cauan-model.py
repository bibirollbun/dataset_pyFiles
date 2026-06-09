#Library importing

import os
import librosa
import numpy as np
import pandas as pd

#File importing

train_audio_path = '/kaggle/input/birdclef-2025/train_audio'
test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes'

#General configs

np.random.seed(0)

print('Setup completed :)')


# Class labels from train audio
class_labels = sorted(os.listdir(train_audio_path))

# List of test soundscapes
test_soundscapes = [os.path.join(test_soundscape_path, afile)
 for afile in sorted(os.listdir(test_soundscape_path))
 if afile.endswith('.ogg')]

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
        # Note to self: that f is a diffeent way of using format
        row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'

        # Make prediction (let's use random scores for now)
        # scores = model.predict...
        # Note to self: this is a dummy to be substituted by the actual model predctions
        scores = np.random.rand(len(class_labels))

        # Append to predictions as new row
        new_row = pd.DataFrame([[row_id] + list(scores)], columns=['row_id'] + class_labels)

        # Create a the final, made to submit, dataframe
        predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)

# Save prediction as csv
predictions.to_csv('submission.csv', index=False)
display(predictions)

