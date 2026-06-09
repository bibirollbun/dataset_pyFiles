# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ðŸ“‚ Load submissions
sub1 = pd.read_csv("/kaggle/input/prediction-1/submission.csv")
sub2 = pd.read_csv("/kaggle/input/notebook095b527234/submission_threshold_0.52.csv")
sub3 = pd.read_csv("/kaggle/input/notebook127da17237/submission.csv")

# ðŸ§  Encode 'Personality' to numeric
le = LabelEncoder()
le.fit(["Extrovert", "Introvert"])

sub1["num"] = le.transform(sub1["Personality"])
sub2["num"] = le.transform(sub2["Personality"])
sub3["num"] = le.transform(sub3["Personality"])

# ðŸ”— Simple average
blended_probs = (sub1["num"] + sub2["num"] + sub3["num"]) / 3

# ðŸŽ¯ Apply threshold
best_threshold = 0.5  # You can tune this based on validation
final_preds = (blended_probs > best_threshold).astype(int)

# ðŸ§¾ Final blended submission
submission = sub1[["id"]].copy()
submission["Personality"] = le.inverse_transform(final_preds)

# ðŸ’¾ Save result
submission.to_csv("submission_blended.csv", index=False)
print("âœ… Final blended submission saved as 'submission_blended.csv'")
print(submission.head())





