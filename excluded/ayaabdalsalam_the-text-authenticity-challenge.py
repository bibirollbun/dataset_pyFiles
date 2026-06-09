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


import os
import pandas as pd
def read_texts_from_dir(dir_path):
    """
    Reads the texts from a given directory and saves them in the pd.DataFrame with columns ['id', 'file_1', 'file_2'].
    Params:
      dir_path (str): path to the directory with data
    """
    data = []
    
    for folder_name in sorted(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
                    text1 = f1.read().strip()
                with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
                    text2 = f2.read().strip()
                
                index = int(folder_name[-4:])  # Extract last 4 characters as ID
                data.append((index, text1, text2))
                
            except Exception as e:
                print(f"Error reading directory {folder_name}: {e}")
    
    print(f"Successfully read {len(data)} directories")
    df = pd.DataFrame(data, columns=['id', 'file_1', 'file_2'])
    return df


# Use the above function to load both train and test data
train_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
df_train=read_texts_from_dir(train_path)
test_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
df_test=read_texts_from_dir(test_path)


df_train.head()


df_test.head()


# Load ground truth for train data
df_train_gt=pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")
df_train_gt


def calculate_english_percentage(text):
    """Helper function to calculate percentage of English content in text"""
    delete = str.maketrans('', '', string.punctuation + '\n')
    cleaned = text.translate(delete)
    
    # Split into words and group into chunks of 10
    words = cleaned.split(" ")
    chunks = [' '.join(words[i:i+10]) for i in range(0, len(words), 10)]
    
    # Count English chunks
    english_count = 0
    for chunk in chunks:
        try:
            if detect(chunk) == 'en':
                english_count += 1
        except LangDetectException:
            continue
    
    return english_count / len(chunks) if chunks else 0

def baseline_method_english_word(df):
    
    file1_english_scores = []
    file2_english_scores = []
    
    for row_idx in range(df.shape[0]):
        # Process file_1 (column index 1)
        file1_text = df.iloc[row_idx, 1]  # More explicit column access
        file1_score = calculate_english_percentage(file1_text)
        file1_english_scores.append(file1_score)
        
        # Process file_2 (column index 2)  
        file2_text = df.iloc[row_idx, 2]
        file2_score = calculate_english_percentage(file2_text)
        file2_english_scores.append(file2_score)
    
    # Make predictions: 1 if file_1 more English, 2 if file_2 more English
    predictions = []
    for i in range(len(file1_english_scores)):
        if file1_english_scores[i] > file2_english_scores[i]:
            predictions.append(1)  # file_1 is "Real"
        else:
            predictions.append(2)  # file_2 is "Real"
    
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


output_df.to_csv('submission.csv', index=False)
















