# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')



new_data = pd.read_csv("/kaggle/input/math-problem-classification-data/train_augmented.csv")
train = pd.concat([train, new_data]).reset_index(drop=True).sample(frac=1).reset_index(drop=True)



test= pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')


pseudo = pd.read_csv("/kaggle/input/notebook7922746d36/submission.csv")  # should have columns: "id", "label"

# Step 2: Merge pseudo labels with test questions
pseudo_labeled_test = test.merge(pseudo, on="id")  # now has columns: id, Question, label

# Step 3: Drop 'id' column to match train format
pseudo_labeled_test = pseudo_labeled_test.drop(columns=["id"])

# Step 4: Concatenate with the original training data
train = pd.concat([train, pseudo_labeled_test], ignore_index=True)


df = train


df


import pandas as pd

# 1. Define your mapping from text labels to numbers
label_map = {
    "Algebra": 0,
    "Geometry and Trigonometry": 1,
    "Calculus and Analysis": 2,
    "Probability and Statistics": 3,
    "Number Theory": 4,
    "Combinatorics and Discrete Math": 5,
    "Linear Algebra": 6,
    "Abstract Algebra and Topology": 7,
}

# 2. Define the prompt template
prompt_template = """You are an expert math problem classifier.

Task: Classify the following math problem into one of the categories below.
Instruction: Respond with the correct number (0-7) only. Do NOT include any explanation.

Math Problem:
{text}

Categories:
0. Algebra  
1. Geometry and Trigonometry  
2. Calculus and Analysis  
3. Probability and Statistics  
4. Number Theory  
5. Combinatorics and Discrete Math  
6. Linear Algebra  
7. Abstract Algebra and Topology

Answer (number only):"""

# 3. Build the new column
def make_chat_record(row):
    question = row["Question"]
    label_num = row["label"]
    
    prompt     = prompt_template.format(text=question)
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt",   "value": str(label_num)}
    ]

df["chat_record"] = df.apply(make_chat_record, axis=1)

# 4. (Optional) If you want just the single-column dataset:
result_df = df[["chat_record"]]



result_df.to_csv('conversations.csv',index=False)




