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

df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")

# Convert 'Personality' to numerical for correlation calculation
df["Personality_numeric"] = df["Personality"].apply(lambda x: 1 if x == "Extrovert" else 0)

numerical_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]

print("Correlation with Personality (Extrovert=1, Introvert=0):")
for col in numerical_cols:
    correlation = df[col].corr(df["Personality_numeric"])
    print(f"  {col}: {correlation:.4f}")


