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


import numpy as np
import pandas as pd

from IPython.display import Latex, Markdown, display, display_latex

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set_palette('crest')


train_df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-prize/train.csv')
external_df = pd.read_csv('/kaggle/input/aimo-external-dataset/external_df.csv')


train_df.head()


external_df.head()


def display_train_problem(row):
    display(Markdown("**Problem Statement:**"))
    display(Latex(row['problem']))
    display()
    display(Markdown("**Solution:**"))
    display(Markdown(f"${row['answer']}$"))

for i, row in train_df.iterrows():
    display(Markdown(f"### Problem {row['id']}"))
    display_train_problem(row)


def display_problem(row):
    display(Markdown("**Problem Statement:**"))
    display(Latex(row['problem']))
    display()
    display(Markdown("**Solution:**"))
    display(Latex(row['solution']))
    display()
    display(Markdown(f"**Level:** {row['level']}"))
    display(Markdown(f"**Type:** {row['type']}"))
    display(Markdown(f"**Stage:** {row['stage']}"))
    display(Markdown(f"**Source:** {row['source']}"))


display_problem(external_df.iloc[0])


external_df.info()


fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))

sns.countplot(data = external_df, x = 'source', ax = axes[0])
axes[0].set_title('Countplot of source')
sns.countplot(data = external_df, x = 'stage', ax = axes[1])
axes[1].set_title('Countplot of stage')

plt.show()


### Explore the MATH dataset

ax = sns.countplot(data = external_df, x = 'level')
ax.set_title('Countplot of level')
plt.show()


plt.figure(figsize=(12, 6))
ax = sns.countplot(data = external_df, x = 'type')
ax.set_title('Countplot of type')
plt.show()


### Display some Sample Problems From Each Type and Level

levels = [f'Level {i}' for i in range(1,6)]
types = ['Prealgebra', 'Algebra', 'Geometry', 'Number Theory', 'Counting & Probability', 'Intermediate Algebra', 'Precalculus']

def display_topic(topic, num_to_display = 2):
    display(Markdown(f"# Problems for {topic}"))
    for level in levels:
        display(Markdown(f"## {level}"))
        sample = external_df[(external_df['level'] == level) & (external_df['type'] == topic)].sample(num_to_display)
        for i, row in sample.iterrows():
            display(Markdown(f"### Problem {i+1}"))
            display_problem(row)


display_topic(types[0])


display_topic(types[1])


display_topic(types[2])


display_topic(types[3])


display_topic(types[4])


display_topic(types[4])


display_topic(types[5])


display_topic(types[6])

