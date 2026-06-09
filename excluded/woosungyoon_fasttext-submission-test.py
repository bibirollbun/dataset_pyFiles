from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import numpy as np
from typing import List, Tuple
import textwrap
import re
from cleantext import clean
from scipy.stats import rankdata
import fasttext


def cleaner(text):
    return clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        lang="en",
    )


def clean_text_series(series: pd.Series) -> pd.Series:
    PATTERN =  r'[\w]+'
    cleaned = series.replace('\n', ' ', regex=True)
    cleaned = cleaned.str.lower()
    
    return cleaned.apply(lambda x: " ".join(re.findall(PATTERN, str(x))))


def write_postive_negative_label(df):
    df['label'] = df['rule_violation'].apply(lambda x: 'positive' if x == 1 else 'negative')
    return df


def predict(sub_train_df, sub_valid_df, texts, name, train_time, items):
    try:
        train_path = f'train_{name}.txt'
        test_path = f'test_{name}.txt'
       
        with open(train_path, 'w', encoding='utf-8') as f:
            for _, row in sub_train_df.iterrows():
                f.write(f"__label__{row['label']} {row['body']}\n")
       
        with open(test_path, 'w', encoding='utf-8') as f:
            for _, row in sub_valid_df.iterrows():
                f.write(f"__label__{row['label']} {row['body']}\n")
       
        model = fasttext.train_supervised(
            input=train_path,
            autotuneValidationFile=test_path,
            autotuneMetric='f1:__label__positive',
            autotuneDuration=train_time,
        )
       
        labels, probs = model.predict(texts)

        POSITIVE_LABEL = '__label__positive'
        NEGATIVE_LABEL = '__label__negative'
        
        logits = []
        for label, prob in zip(labels, probs):
            try:
                if label[0] == POSITIVE_LABEL:
                    logits.append(prob[0])
                elif label[0] == NEGATIVE_LABEL:
                    logits.append(1 - prob[0])
                else:
                    logits.append(np.nan)
            except (IndexError, TypeError, ValueError):
                logits.append(np.nan)
                
        items.append((name, np.array(logits).flatten()))
    except Exception as e:
        items.append((name, None))
        print(f"{name} 처리 오류: {e}")
    return None


def process_example_data(test_df):
    def create_example_df(df, column, is_positive):
        example_df = df[['row_id', column, 'rule', 'subreddit']].copy()
        example_df = example_df.rename(columns={column: 'body'})
        example_df['rule_violation'] = 1 if is_positive else 0
        return example_df

    original_test_df = test_df[['row_id', 'body', 'rule', 'subreddit']].copy()
    
    dataframes = []
    for df, prefix in [(test_df, 'test')]:
        for col in ['positive_example_1', 'positive_example_2']:
            dataframes.append(create_example_df(df, col, True))
        for col in ['negative_example_1', 'negative_example_2']:
            dataframes.append(create_example_df(df, col, False))

    new_train_df = pd.concat(dataframes, ignore_index=True)
    new_train_df = new_train_df.sample(frac=1, random_state=None).reset_index(drop=True)
    return new_train_df, original_test_df


test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

new_train_df, original_test_df = process_example_data(test_df)
new_train_df['body'] = new_train_df.body.apply(cleaner)
new_train_df['body'] = clean_text_series(new_train_df['body'])
new_train_df = write_postive_negative_label(new_train_df)

original_test_df['body'] = original_test_df.body.apply(cleaner)
original_test_df['body'] = clean_text_series(original_test_df['body'])


submission = pd.DataFrame(columns=['row_id', 'rule_violation'])
submission['row_id'] = original_test_df['row_id']
rule_items = list(test_df.rule.unique())

if len(test_df) == 10:
    max_time = 30
    default_fold_size = 4
else:
    max_time = 600
    default_fold_size = 8

for rule in rule_items:

    sub_test_df = original_test_df.loc[original_test_df.rule==rule].copy()
    sub_train_df = new_train_df.loc[new_train_df.rule==rule].copy()

    if len(sub_train_df) == 0:
        continue
    
    items = []
    texts = original_test_df.body.tolist()

    fold_size = min(len(sub_train_df), default_fold_size)
    kf = KFold(n_splits=fold_size, shuffle=True)
    
    all_folds = list(kf.split(sub_train_df))
    
    texts = sub_test_df['body'].values.tolist()
    positions = [i for i in sub_test_df['row_id'].values.tolist()]
    
    for i, (train_idx, val_idx) in enumerate(all_folds):
        cur_train_df = sub_train_df.iloc[train_idx]
        cur_valid_df = sub_train_df.iloc[val_idx]
        predict(cur_train_df, cur_valid_df, texts, f'cls_{i}', max_time, items)
        
    logits = [it[1] for it in items if it[1] is not None]
    logits_rank = [rankdata(it) for it in logits]
    prediction = np.nanmean(logits_rank, axis=0)
    prediction = prediction/len(prediction)
    
    
    for k, row_id in enumerate(positions):
        submission.loc[submission.row_id == row_id, 'rule_violation'] = prediction[k]



submission.to_csv("submission.csv", index=None)
submission

