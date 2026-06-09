import time
START = time.time()

!pip install /kaggle/input/bc25-lib/wheel/resampy-*
!pip install /kaggle/input/bc25-lib/wheel/watchdog-*
!pip install /kaggle/input/bc25-lib/wheel/birdnetlib-*

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from concurrent.futures import ThreadPoolExecutor
import glob
import librosa
import numpy as np
import os
import pandas as pd
import re
from scipy.interpolate import CubicSpline
import sys
import torch
import torchaudio

TERMINATE_TIME = START + 5300


primary_labels = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv').columns[1:].to_list()
primary_labels_indices = range(len(primary_labels))

primary_labels_map = dict(zip(primary_labels, primary_labels_indices))

taxonomy = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv', index_col='common_name')['primary_label']
taxonomy_map = taxonomy.map(primary_labels_map)

common_names = taxonomy.index.to_list()


def get_oggs(max_oggs=10):
    if len(glob.glob('/kaggle/input/birdclef-2025/test_soundscapes/*.ogg')) > 0:
        oggs = glob.glob('/kaggle/input/birdclef-2025/test_soundscapes/*.ogg')
    else:
        oggs = sorted(glob.glob(f'/kaggle/input/birdclef-2025/train_soundscapes/*.ogg'))[:max_oggs]
    return [(n, ogg, re.search(r'/([^/]+)\.ogg$', ogg).group(1)) for n, ogg in enumerate(oggs)]

oggs = get_oggs()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


analyzer = Analyzer()

row_ids = []
result = []
for _, fname, ss_id in oggs:
    print(f'{ss_id}')
    with HiddenPrints():
        recording = Recording(analyzer,
                              fname,
                              #lat=6.763345368718646, 
                              #lon=-74.20911748873883,
                              min_conf=1e-10
                             )
        recording.analyze()
    
    x1 = np.arange(1.5, 60, 3)
    x2 = np.arange(2.5, 60, 5)
    
    # Zero-filled resulting array of size [duration // 3; num_species]
    bn_result = np.zeros((20, len(common_names)))
    
    # Fill the result with BirdNet prediction
    for rec in recording.detections:
        if rec['common_name'] in common_names:
            species_idx = taxonomy_map[rec['common_name']]
            time_idx = int(rec['start_time'] // 3)
            bn_result[time_idx, species_idx] = rec['confidence']
    
    # Reshape the resulting array to the size of [duration // 5; num_species]
    bn_result = CubicSpline(x1, bn_result)(x2)

    row_ids += [f'{ss_id}_{n}' for n in range(5, 65, 5)]
    result.append(bn_result)


submission = pd.DataFrame(np.concatenate(result), columns=primary_labels)
submission['row_id'] = row_ids
submission = submission[['row_id'] + primary_labels]


# Write CSV
submission.to_csv('submission.csv', index=False)

# Display submission DataFrame
display(submission.head(20))

display(submission.tail(20))




