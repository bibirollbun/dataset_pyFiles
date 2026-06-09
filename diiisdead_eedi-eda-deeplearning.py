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


import matplotlib.pyplot as plt
import seaborn as sns


train_path = '/kaggle/input/eedi-mining-misconceptions-in-mathematics/'
misconception_mapping = pd.read_csv(train_path + 'misconception_mapping.csv')
sample_submission = pd.read_csv(train_path + 'sample_submission.csv')
test = pd.read_csv(train_path + 'test.csv')
train = pd.read_csv(train_path + 'train.csv')


# Train file
train.head()


train.columns


# test file
test.head()


test.columns


misconception_mapping


misTable = {}
for i in range(misconception_mapping.shape[0]):
    misTable[misconception_mapping.MisconceptionId.values[i]] = misconception_mapping.MisconceptionName.values[i]


sample_submission


train.isnull().sum()


stacked = []
for i in ['MisconceptionAId','MisconceptionBId','MisconceptionCId','MisconceptionDId']:
    stacked.extend(train[train[i].notna()][i].tolist())

    from collections import Counter

def sort_by_frequency_with_counts(input_list):
    freq_count = Counter(input_list)
    sorted_elements_with_counts = sorted(freq_count.items(), key=lambda x: (-x[1], x[0]))
    
    return sorted_elements_with_counts

sorted_elements_with_counts = sort_by_frequency_with_counts(stacked)

# Print the sorted elements along with their frequencies
print("Most frequent MisconceptionIds accross the Options : ")
print("-"*100,'\n')
for element, count in sorted_elements_with_counts[:10]:
    print("Misconception : ",misTable[int(element)])
    print("MisconceptionId : ",int(element))
    print("MisconceptionId count : ",count)
    print("-"*100,'\n')


value_counts = train['CorrectAnswer'].value_counts()

plt.figure(figsize=(10, 6))
value_counts.plot(kind='bar', color='skyblue')

plt.title(f'Value Counts for CorrectAnswer', fontsize=16)
plt.xlabel('Values', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.show()


for misconception in misconception_cols:
    df_misconception = train[train[misconception].notnull()]
    print(f"Answer choices with misconception {misconception}:")
    answer_counts = df_misconception[['CorrectAnswer', misconception]].groupby('CorrectAnswer').size()
    plt.figure(figsize=(10, 6))
    plt.bar(answer_counts.index, answer_counts.values, color='skyblue')
    plt.xlabel('Correct Answer', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Answer Choices for {misconception}')
    plt.show()


# Count the NaN values in each misconception column
nan_counts = train[misconception_cols].isna().sum()
plt.figure(figsize=(10, 6))
nan_counts.plot(kind='bar', color='skyblue')

plt.title(f'NaN values in each misconception column', fontsize=16)
plt.xlabel('Values', fontsize=12)
plt.ylabel('Column', fontsize=12)
plt.xticks(rotation=0) 
plt.show()


# Verify that the correct answer has NaN for the respective Misconception column
for correct_answer in ['A', 'B', 'C', 'D']:
    correct_misconception_col = f'Misconception{correct_answer}Id'
    
    # Check if the correct answer has NaN in its corresponding misconception column
    correct_answer_nan = train[train['CorrectAnswer'] == correct_answer][correct_misconception_col].isna().sum()
    total_correct_answer = train[train['CorrectAnswer'] == correct_answer].shape[0]
    
    print(f"For correct answer {correct_answer}:")
    print(f"NaN in {correct_misconception_col}: {correct_answer_nan} out of {total_correct_answer} rows")

# Optional: Check if there's any inconsistency where the correct answer doesn't have NaN
for index, row in train.iterrows():
    correct_answer = row['CorrectAnswer']
    correct_misconception_col = f'Misconception{correct_answer}Id'
    if not pd.isna(row[correct_misconception_col]):
        print(f"Inconsistency found in row {index}: correct answer {correct_answer} should have NaN in {correct_misconception_col}")


train['misconception_count'] = train[misconception_cols].notna().sum(axis=1)

#Display disribution of the number of misconceptions
misconception_distribution = train['misconception_count'].value_counts().sort_index()
print(train['misconception_count'])
print(misconception_distribution)


!pip install -qq pylatexenc


from IPython.display import display, HTML
from pylatexenc.latex2text import LatexNodes2Text

def map_misconception_id(value, misTable):
    # Check if value is NaN
    if pd.isna(value):
        return "No misconception/NaN"
    else:
        # Map the value to the misTable, if the value exists
        return misTable.get(int(value), "Unknown Misconception")

def generate_question_html(df, index,lat2text=1):
    global misTable
    # Fetch the question, options, and additional fields from the DataFrame
    question = df.iloc[index].QuestionText
    if lat2text==1:
        question = LatexNodes2Text().latex_to_text(question)
    options = []
    for i in ['A', 'B', 'C', 'D']:
        options.append(df['Answer' + i + 'Text'].iloc[index])
    
    # Fetch the correct answer and additional fields
    correct_answer = df.iloc[index].CorrectAnswer
    construct_name = df.iloc[index].ConstructName
    subject_name = df.iloc[index].SubjectName
    
    # Map misconception IDs with the hash table, considering NaN values
    misconception_a = map_misconception_id(df.iloc[index].MisconceptionAId, misTable)
    misconception_b = map_misconception_id(df.iloc[index].MisconceptionBId, misTable)
    misconception_c = map_misconception_id(df.iloc[index].MisconceptionCId, misTable)
    misconception_d = map_misconception_id(df.iloc[index].MisconceptionDId, misTable)

    # Start the HTML structure
    html = f"""
    <div style='font-family: Arial, sans-serif; border: 2px solid #007bff; padding: 15px; border-radius: 10px; width: 60%; margin: 0 auto; background-color: #f4f9ff;'>
        <p style='font-size: 16px; color: #007bff;'><strong>Construct Name:</strong> {construct_name}</p>
        <p style='font-size: 16px; color: #007bff;'><strong>Subject Name:</strong> {subject_name}</p>
        <hr style='border: 1px solid #007bff; margin: 10px 0;'>
        <p style='font-size: 18px; font-weight: bold; color: #ff6f61;'>Problem:</p>
        <p style='font-size: 20px; color: #333; font-weight: bold;'>{question}</p>
        <ul style='list-style-type: none; padding: 0;'>
    """
    
    # Option letters
    option_letters = ['A', 'B', 'C', 'D']
    
    # Add each option with the corresponding letter and styled nicely
    for i, option in enumerate(options):
        html += f"""
        <li style='background-color: #e0f2ff; padding: 10px; margin: 5px 0; border-radius: 5px;'>
            <span style='font-weight: bold; color: #007bff;'>{option_letters[i]}.</span> {option}
        </li>
        """
    
    # Close the list and add the correct answer at the bottom
    html += f"""
        </ul>
        <p style='font-size: 18px; font-weight: bold; color: #28a745;'>Correct answer: {correct_answer}</p>
        <hr style='border: 1px solid #007bff; margin: 20px 0;'>
        <p style='font-size: 16px; color: #333;'><strong>Misconception A:</strong> {misconception_a}</p>
        <p style='font-size: 16px; color: #333;'><strong>Misconception B:</strong> {misconception_b}</p>
        <p style='font-size: 16px; color: #333;'><strong>Misconception C:</strong> {misconception_c}</p>
        <p style='font-size: 16px; color: #333;'><strong>Misconception D:</strong> {misconception_d}</p>
    </div>
    """
    
    # Display the HTML
    display(HTML(html))


index = 2
generate_question_html(train,index,lat2text=0)


# Count the occurrences of misconceptions for each incorrect answer
misconception_distribution = {}

for misconception in misconception_cols:
    misconception_distribution[misconception] = train[misconception].value_counts()

# Show the top misconceptions for each incorrect answer choice
for misconception, counts in misconception_distribution.items():
    plt.figure(figsize=(10, 6))
    counts.head(10).plot(kind='bar', color='skyblue')

    plt.title(f"Most common misconceptions for {misconception}:", fontsize=16)
    plt.xticks(rotation=0) 
    plt.show()

