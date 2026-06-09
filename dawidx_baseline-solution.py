!pip install langdetect


import os
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import unicodedata

import string
from sklearn.metrics import accuracy_score
import numpy as np
DetectorFactory.seed = 42


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
train_path="/kaggle/input/fake-or-real-impostor-hunt/data/train"
df_train=read_texts_from_dir(train_path)
test_path="/kaggle/input/fake-or-real-impostor-hunt/data/test"
df_test=read_texts_from_dir(test_path)


df_train.head()


df_test.head()


# Load ground truth for train data
df_train_gt=pd.read_csv("/kaggle/input/fake-or-real-impostor-hunt/data/train.csv")
df_train_gt


def baseline_method_english_word(df):
  """
  This baseline method predicts which of the texts is Real, based on the percentage of English words in each text.
  It returns list with predictions.

  Params:
    df (pd.DataFrame): dataframe with all texts
  """
  # Create lists in which scores will be saved for file_1 (left_scores) and files_2 (right_scores)
  left_scores=[0 for _ in range(df.shape[0])]
  right_scores=[0 for _ in range(df.shape[0])]
  # For each row in the DataFrame and for each element of this row run the algorithm for detecting English words
  for j in range(df.shape[0]):
    for z in range(df.shape[1]):
      sum_english=0
      n=10
      delete=str.maketrans('', '', string.punctuation+'\n')
      cleaned=df.iloc[j].iloc[z].translate(delete)
      text_to_check=cleaned.split(" ")
      text_to_check=[' '.join(text_to_check[i:i+n]) for i in range(0, len(text_to_check),n)]

      # Run algorithm for detecting English words
      for i in range(len(text_to_check)):
        try:
          language=detect(text_to_check[i])
        except LangDetectException as e:
          pass
        if language=='en':
          sum_english+=1
      result=sum_english/len(text_to_check)
      if z==0:
        left_scores[j]=result
      elif z==1:
        right_scores[j]=result
      else:
        print('Wrong')
  # Create list with predictions by setting value in list to 1 if the first text is `Real` or 2 when the second seems to be better
  predictions=[1 if left_scores[k]>right_scores[k] else 2 for k in range(len(left_scores))]
  return predictions


def evaluate_baseline(predictions, gt_list, text='Score with english detection:'):
  """
  Evaluates the predictions for train data, when the ground truth is provided.

  Params:
    predictions (list): list of predictions
    gt_list (list): list of predictions
    text (str): text to be printed together with the result
  """
  acc_score = accuracy_score(gt_list, predictions)
  print(text,acc_score)


# Use the algorithm for the train data and check accuracy
predictions_train=baseline_method_english_word(df_train)
gt_train=list(df_train_gt['real_text_id'])
evaluate_baseline(predictions_train, gt_train)


# Use the algorithm for the test data
predictions_test=baseline_method_english_word(df_test)


# Change the format of predictions into requested format, as described in Overview section of this competition
df_results_test=pd.DataFrame(predictions_test)
output_df = df_results_test.copy()
output_df.columns = ['real_text_id']
output_df.reset_index(inplace=True)
output_df.rename(columns={'index': 'id'}, inplace=True)
output_df


output_df.to_csv('sample_submission_1.csv', index=False)


def is_latin_char(char):
  """
  Detect if given character is from Latin alphabet.

  Params:
    char (str): given character
  """
  char=str(char)
  try:
    name=unicodedata.name(char)
    return 'LATIN' in name
  except ValueError:
    return False

def baseline_chars_method(df):
  """
  This baseline method predicts which of the texts is Real, based on the percentage of Lating letters words in each text.
  It returns list with predictions.

  Params:
    df (pd.DataFrame): dataframe with all texts
  """
  # Create lists in which scores will be saved for file_1 (left_scores) and files_2 (right_scores)
  left_scores=[0 for _ in range(df.shape[0])]
  right_scores=[0 for _ in range(df.shape[0])]
  # For each row in the DataFrame and for each element of this row run the algorithm for detecting Latin chars
  for j in range(df.shape[0]):
    for z in range(df.shape[1]):
      sum_latin=0
      count_spaces=0
      delete=str.maketrans('', '', string.punctuation+'\n')
      cleaned=df.iloc[j].iloc[z].translate(delete)
      
      # Run algorithm for detecting Latin chars
      for i in range(len(cleaned)):
        if cleaned[i] !=' ':
          if is_latin_char(cleaned[i]):
            sum_latin+=1
        else:
          count_spaces+=1
      if len(cleaned)==0:
        result=0
      else:
        result=sum_latin/(len(cleaned)-count_spaces)
      if z==0:
        left_scores[j]=result
      elif z==1:
        right_scores[j]=result
      else:
        print('Wrong')
  # Create list with predictions by setting value in list to 1 if the first text is `Real` or 2 when the second seems to be better
  predictions=[1 if left_scores[k]>right_scores[k] else 2 for k in range(len(left_scores))]
  return predictions


# Use the algorithm for the train data and check accuracy
predictions_train_char=baseline_chars_method(df_train)
gt_train=list(df_train_gt['real_text_id'])
evaluate_baseline(predictions_train_char, gt_train, text='Score with latin detection:')


# Use the algorithm for the test data
preds_test_char=baseline_chars_method(df_test)


# Change the format of predictions into requested format, as described in Overview section of this competition
df_results_test_char=pd.DataFrame(preds_test_char)
output_df_char = df_results_test_char.copy()
output_df_char.columns = ['real_text_id']
output_df_char.reset_index(inplace=True)
output_df_char.rename(columns={'index': 'id'}, inplace=True)
output_df_char


output_df_char.to_csv('sample_submission_2.csv', index=False)

