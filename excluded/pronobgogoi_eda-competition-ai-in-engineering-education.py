# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# warnings
import warnings

# suppress all warnings
warnings.filterwarnings("ignore")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


location = "/kaggle/input/eda-competition-by-ashok/AI in Engineering Education.csv"
df = pd.read_csv(location)
df.head()


# shape of dataframe
df.shape


df.columns


# numerical description
df.describe()


# categorical description
df.describe(include="O")


# Look at presence of null values
df.isna().sum()


# Look at percentage of null values
df.isna().mean() * 100

# NOTE: So, the last two columns are about 60-70 % filled with null values.
# Also, the columns are filled with strings, so, we can simply ignore them.
# Even can remove the two columns if some calculation demands it. This we will judge down the line.


# Check "Timestamp" column
# df["Timestamp"].dtypes
# NOTE: Timestamp column is in 'object' type so we will convert it to datetime format for further analysis.
df["Timestamp"] = pd.to_datetime(df["Timestamp"])


print(f'Datatype of "Timestamp" column is: {df["Timestamp"].dtypes}')
df.info()



# Getting day, months, time column from "Timestamp"
# day
df["day"] = df["Timestamp"].dt.day_name()

# time
df["time"] = df["Timestamp"].dt.time

# look at total dataframe
df.head()


df.columns


df.columns


# histogram of all the relevant columns
column_list = [# 'Timestamp', 'Username',
       'Which department of engineering are you affiliated with? ',
       '  How would you rate your current knowledge of AI?   ',
       'Which AI tools do you use?(dalle, midjourney, chapgpt, claude,gemini, grammarly...)',
       'How often do you use AI tools?',
       'For what purposes do you use AI tools?',
       'On a scale of 0 to 10, how much have AI tools increased your productivity? (0 = not at all, 10 = significantly)',
       'On a scale of 0 to 10, how much have AI tools increased your laziness? (0 = not at all, 10 = significantly)',
       'Which tasks do you think are not well-solved by AI tools in your experience?(EG. writing original stories)',
       'Do you think AI will take your job in the future?',
       'In your opinion, will AI ever take over humans?',
       'Do you want to work on developing the best AI? ',
       'What do you wish AI could do better to make your life easier?',
       'Anything you want to share about AI.',
        'day', 
        # 'time'
        ]
for column in df[column_list]:
    plt.figure(figsize=(12,6))
    plt.title(f"Histogram(with KDE) of '{column}'.")
    sns.histplot(data=df, x=column, kde=True, label=None)
    plt.xticks(rotation=90)
    plt.grid(axis="y")
    plt.show()


# Based on departments:
column_list = [#'Which department of engineering are you affiliated with? ',
       '  How would you rate your current knowledge of AI?   ',
       #'Which AI tools do you use?(dalle, midjourney, chapgpt, claude,gemini, grammarly...)',
       'How often do you use AI tools?',
       #'For what purposes do you use AI tools?',
       'On a scale of 0 to 10, how much have AI tools increased your productivity? (0 = not at all, 10 = significantly)',
       'On a scale of 0 to 10, how much have AI tools increased your laziness? (0 = not at all, 10 = significantly)',
       'Which tasks do you think are not well-solved by AI tools in your experience?(EG. writing original stories)',
       'Do you think AI will take your job in the future?',
       'In your opinion, will AI ever take over humans?',
       'Do you want to work on developing the best AI? ',
       'What do you wish AI could do better to make your life easier?',
       'Anything you want to share about AI.', 'day']
for column in df[column_list]:
    grp_df = df.groupby("Which department of engineering are you affiliated with? ")[column].value_counts().sort_values(ascending=False).reset_index()
    grp_df
    
    plt.figure(figsize=(12,6))
    plt.title(f"Cluster barplot of '{column}' vs. Departments.")
    sns.barplot(data=grp_df, x="Which department of engineering are you affiliated with? ", y="count", hue=column)
    plt.xticks(rotation=90)
    plt.grid(axis="y")
    plt.show()




