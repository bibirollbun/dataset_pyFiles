import pandas as pd

# Replace with your actual filenames
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
sample_submission = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')

# Preview
train.head()



import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

df = train

# Keep only the necessary columns
df = df[['row_id', 'QuestionId', 'QuestionText', 'MC_Answer', 'StudentExplanation', 'Category', 'Misconception']]

# Group by Category and sample 2 examples each
samples = df.groupby('Category').apply(lambda x: x.sample(2, random_state=42)).reset_index(drop=True)


# Display results nicely grouped
for cat in samples['Category'].unique():
    print(f"\nğŸŸ¦ Category: {cat}\n{'-'*70}")
    cat_df = samples[samples['Category'] == cat]
    for i, row in cat_df.iterrows():
        print(f"Question ID: {row['QuestionId']}")
        print(f"Question: {row['QuestionText']}")
        print(f"MC Answer: {row['MC_Answer']}")
        print(f"Student Explanation: {row['StudentExplanation']}")
        print(f"Misconception: {row['Misconception'] if pd.notna(row['Misconception']) else 'NA'}")
        print("-" * 70)


import pandas as pd

# Group by Misconception and get two examples for each
examples_per_misconception = (
    df.dropna(subset=['Misconception'])  # remove rows with no misconception label
    .groupby('Misconception')
    .apply(lambda x: x.head(2))
    .reset_index(drop=True)
)

# Display examples nicely
for misconception in examples_per_misconception['Misconception'].unique():
    print(f"\n{'='*60}")
    print(f"Misconception: {misconception}")
    print(f"{'='*60}")
    
    subset = examples_per_misconception[examples_per_misconception['Misconception'] == misconception]
    for i, row in subset.iterrows():
        print(f"\nExample {i % 2 + 1}:")
        print(f"Question ID   : {row['QuestionId']}")
        print(f"Question      : {row['QuestionText']}")
        print(f"MC Answer     : {row['MC_Answer']}")
        print(f"Student Answer: {row['StudentExplanation']}")
        print(f"Category: {row['Category']}")



