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

# example DataFrame to write to CSV
df = pd.DataFrame([[1, 'a'], [2, 'b']], columns=['index', 'letter'])

# set an index to avoid a 'blank' row number column and write to a file named 'submission.csv'
df.set_index('index').to_csv('submission.csv')


import random

# List of essay topics
topics = [
    "The impact of technology on education",
    "The role of government in healthcare",
    "Climate change and its effects on society",
    "The importance of art in culture",
    "The future of artificial intelligence",
]

# Function to generate an essay that maximizes disagreement
def generate_disagreement_essay(topic):
    # Create a list of conflicting statements
    conflicting_statements = [
        f"On one hand, {topic} has greatly improved accessibility to information and learning resources.",
        f"On the other hand, {topic} has led to a decline in critical thinking skills among students.",
        f"Many argue that {topic} fosters collaboration and communication among students.",
        f"Conversely, some believe that {topic} creates a sense of isolation and reduces face-to-face interactions.",
        f"Supporters of {topic} claim it prepares students for the future job market.",
        f"Critics argue that {topic} is overemphasized and traditional methods are being neglected.",
        f"Some studies suggest that {topic} enhances engagement and motivation in learning.",
        f"However, others point out that it can lead to distractions and decreased attention spans."
    ]

    # Randomly select statements to create an essay
    essay = f"Essay on: {topic}\n\n"
    essay += "Introduction:\n"
    essay += f"The topic of {topic} is highly debated in contemporary society. \n\n"
    
    essay += "Body:\n"
    for _ in range(4):  # Generate 4 conflicting points
        essay += random.choice(conflicting_statements) + "\n"
    
    essay += "\nConclusion:\n"
    essay += f"In conclusion, {topic} presents a complex array of perspectives that highlight the ongoing debate surrounding its implications. It is clear that opinions vary widely, and further discussion is necessary to reach a consensus."

    return essay

# Generate essays for each topic
for topic in topics:
    essay = generate_disagreement_essay(topic)
    print(essay)
    print("\n" + "="*50 + "\n")

