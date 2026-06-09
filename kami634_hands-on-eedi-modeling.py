data = {'QuestionId': 101,
 'ConstructId': 579,
 'ConstructName': 'Express one quantity as a percentage of another mentally',
 'SubjectId': 233,
 'SubjectName': 'Percentages of an Amount',
 'CorrectAnswer': 'B',
 'QuestionText': 'What is \\( 8 \\) out of \\( 40 \\) as a percentage?',
 'AnswerAText': '\\( 8.4 \\% \\)',
 'AnswerBText': '\\( 20 \\% \\)',
 'AnswerCText': '\\( 16 \\% \\)',
 'AnswerDText': '\\( 24 \\% \\)',
 'MisconceptionAId': 1786,
 'MisconceptionBId': -1,
 'MisconceptionCId': 658,
 'MisconceptionDId': -1,
 'MisconceptionAName': 'Converts a fraction to a percentage by writing the numerator followed by the denominator',
 'MisconceptionBName': None,
 'MisconceptionCName': 'Thinks they double the numerator to turn a fraction into a percentage',
 'MisconceptionDName': None}














!pip install -q bertviz


from bertviz import head_view
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
inputs = tokenizer("The astronaut looked out the window and saw a beautiful blue planet.", return_tensors="pt")
out = model(**inputs)
head_view(out.attentions, tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))








# ここのコードは変えないようにしてください

# 設定によっては学習が始まらないケースがあるのでその対応
import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd

columns = ['x1', 'source','labels','message']
train_df = pd.read_csv('/kaggle/input/twitter-entity-sentiment-analysis/twitter_training.csv', header=None, names=columns).dropna()
# データ量が多いので今回は一部だけ使うことにします
train_df['message'] = train_df['source'] + ': ' + train_df['message']
train_df = train_df.sample(frac=0.05, random_state=42)

test_df = pd.read_csv('/kaggle/input/twitter-entity-sentiment-analysis/twitter_validation.csv', header=None, names=columns)

display(train_df)
display(test_df)





data = {'QuestionId': 101,
 'ConstructId': 579,
 'ConstructName': 'Express one quantity as a percentage of another mentally',
 'SubjectId': 233,
 'SubjectName': 'Percentages of an Amount',
 'CorrectAnswer': 'B',
 'QuestionText': 'What is \\( 8 \\) out of \\( 40 \\) as a percentage?',
 'AnswerAText': '\\( 8.4 \\% \\)',
 'AnswerBText': '\\( 20 \\% \\)',
 'AnswerCText': '\\( 16 \\% \\)',
 'AnswerDText': '\\( 24 \\% \\)',
 'MisconceptionAId': 1786,
 'MisconceptionBId': -1,
 'MisconceptionCId': 658,
 'MisconceptionDId': -1,
 'MisconceptionAName': 'Converts a fraction to a percentage by writing the numerator followed by the denominator',
 'MisconceptionBName': None,
 'MisconceptionCName': 'Thinks they double the numerator to turn a fraction into a percentage',
 'MisconceptionDName': None}











import pandas as pd
import numpy as np

df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')

# データ数を取得
data_num = len(df)

# train と test の分割比率を指定
train_rate = 0.8
train_num = int(data_num * train_rate)

# index を分割
np.random.seed(42)
train_index = np.random.choice(data_num, train_num, replace=False)
valid_index = list(set(range(data_num)) - set(train_index))

# train と test に分割
train_df = df.iloc[train_index]
valid_df = df.iloc[valid_index]

print("train_df の形状:", train_df.shape)
print("valid_df の形状:", valid_df.shape)


def preprocess(df):    
    result = []
    for i, row in df.iterrows():
        for option in ['A', 'B', 'C', 'D']:
            if pd.isnull(row[f'Misconception{option}Id']):
                continue
            result.append(
                {
                    'ConstructId': row['ConstructId'],
                    'ConstructName': row['ConstructName'],
                    'SubjectId': row['SubjectId'],
                    'SubjectName': row['SubjectName'],
                    'CorrectAnswer': row['CorrectAnswer'],
                    'IsCorrect': row['CorrectAnswer']==option,
                    'Option': option,
                    'AnswerText': row[f'Answer{option}Text'],
                    'MisconceptionId': int(row[f'Misconception{option}Id']),
                }
            )
    df = pd.DataFrame(result)

    misconception_mapping_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')
    df = df.merge(
        misconception_mapping_df,
        on="MisconceptionId",
        how="left"
    )

    return df

train_df = preprocess(train_df)
valid_df = preprocess(valid_df)

print("train_df の形状:", train_df.shape)
display(train_df.head())


print("valid_df の形状:", valid_df.shape)
display(train_df.head())



import numpy as np
def apk(actual, predicted, k=25):
    """
    Computes the average precision at k.
    
    This function computes the average prescision at k between two lists of
    items.
    
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    """
    Computes the mean average precision at k.
    
    This function computes the mean average prescision at k between two lists
    of lists of items.
    
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])





