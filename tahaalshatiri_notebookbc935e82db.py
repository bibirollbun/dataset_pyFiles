# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



from collections import Counter
sub1 = pd.read_csv('/kaggle/input/notebooke6e26fef2f/submission_confidence_based.csv')
sub2 = pd.read_csv('/kaggle/input/kachallenges-ensemble/submission.csv')
sub3 = pd.read_csv('/kaggle/input/x-small-deberta-baseline-924d8b/submission.csv')

# Initialize a DataFrame using one of the submissions
sub = sub1.copy()

# Combine predictions by majority voting
ensemble_labels = []
for i in range(len(sub)):
    labels = [sub1.loc[i, 'label'], sub2.loc[i, 'label'], sub3.loc[i, 'label']]
    vote_counts = Counter(labels)
    
    if vote_counts.most_common(1)[0][1] >= 2:
        # Majority exists
        final_label = vote_counts.most_common(1)[0][0]
    else:
        # No majority, fallback to sub1
        final_label = labels[0]
    
    ensemble_labels.append(final_label)

sub['label'] = ensemble_labels
sub.to_csv('submission_ensemble.csv', index=False)

