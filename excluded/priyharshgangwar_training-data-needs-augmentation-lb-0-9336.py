import os
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.metrics import accuracy_score
import numpy as np


def read_texts_from_dir(dir_path):
  """
  Reads the texts from a given directory and saves them in the pd.DataFrame with columns ['id', 'file_1', 'file_2'].

  Params:
    dir_path (str): path to the directory with data
  """
  # Count number of directories in the provided path
  dir_count = sum(os.path.isdir(os.path.join(root, d)) for root, dirs, _ in os.walk(dir_path) for d in dirs)
  data=[0 for _ in range(dir_count)]
  print(f"Number of directories: {dir_count}")

  # For each directory, read both file_1.txt and file_2.txt and save results to the list
  i=0
  for folder_name in sorted(os.listdir(dir_path)):
    folder_path = os.path.join(dir_path, folder_name)
    if os.path.isdir(folder_path):
      try:
        with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
          text1 = f1.read().strip()
        with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
          text2 = f2.read().strip()
        index = int(folder_name[-4:])
        data[i]=(index, text1, text2)
        i+=1
      except Exception as e:
        print(f"Error reading directory {folder_name}: {e}")

  # Change list with results into pandas DataFrame
  df = pd.DataFrame(data, columns=['id', 'file_1', 'file_2']).set_index('id')
  return df


# Use the above function to load both train and test data
train_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
df_train=read_texts_from_dir(train_path)
test_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
df_test=read_texts_from_dir(test_path)
df_train_gt=pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")
df_train_gt.head(2)


df = df_train.merge(df_train_gt, on="id", how="left")
df['real_text_id'] = df['real_text_id'].astype(int)
print(df.head())


def get_real(row):
    return row['file_1'] if row['real_text_id'] == 1 else row['file_2']
def get_fake(row):
    return row['file_2'] if row['real_text_id'] == 1 else row['file_1']

df['real_text'] = df.apply(get_real, axis=1)
df['fake_text'] = df.apply(get_fake, axis=1)

df[['id', 'real_text', 'fake_text']].head()



import pandas as pd

# Define the simple rule: begins with capital and ends with '.'
def follows_rule(text):
    return isinstance(text, str) and len(text) > 0 and text[0].isupper() and text.strip().endswith(".")

# Apply rule to both columns
df['real_rule'] = df['real_text'].apply(follows_rule)
df['fake_rule'] = df['fake_text'].apply(follows_rule)

# Categorize into 4 cases
def categorize(row):
    if row['real_rule'] and row['fake_rule']:
        return "Both follow rule"
    elif row['real_rule'] and not row['fake_rule']:
        return "Real follows, Fake doesn't"
    elif not row['real_rule'] and row['fake_rule']:
        return "Fake follows, Real doesn't"
    else:
        return "Neither follows"

df['rule_case'] = df.apply(categorize, axis=1)

# Count cases
rule_counts = df['rule_case'].value_counts().reset_index()
rule_counts.columns = ['Case', 'Count']

# Display results
rule_counts


def classify_pair(row):
    f1_rule = follows_rule(row['file_1'])
    f2_rule = follows_rule(row['file_2'])
    
    if f1_rule and not f2_rule:
        return 1
    elif f2_rule and not f1_rule:
        return 2
    else:
        return 0

df_test['classify_id'] = df_test.apply(classify_pair, axis=1)
df_test[['file_1', 'file_2', 'classify_id']].head()


df_test["classify_id"].value_counts()


sub_avg = pd.read_csv("/kaggle/input/submission-file-by-torch/submission_average_deberta.csv")
sub_avg.set_index("id", inplace=True)


df_test['real_text_id'] = df_test['classify_id'].where(df_test['classify_id'] != 0,
                                                    sub_avg['real_text_id'])



df_test[["real_text_id"]].to_csv("submission.csv")




