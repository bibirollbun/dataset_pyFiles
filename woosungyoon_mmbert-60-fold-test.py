import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from multiprocessing import Process, Queue
import textwrap
from scipy.stats import rankdata
from cleantext import clean
from multiprocessing import Process, set_start_method


def reconstruct_dataframes(df):
    n = len(df)
    df1 = df.copy()
    df2 = df.copy()
    
    is_positive_body = np.random.choice([True, False], size=n)
    body_idx = np.random.randint(0, 2, size=n)

    df1['row_id'] = df1['row_id'].astype(str) + '_gen'
    df1['body'] = np.where(
        is_positive_body,
        np.where(body_idx == 0, df['positive_example_1'], df['positive_example_2']),
        np.where(body_idx == 0, df['negative_example_1'], df['negative_example_2'])
    )
    
    df1['positive_example'] = np.where(
        is_positive_body,
        np.where(body_idx == 0, df['positive_example_2'], df['positive_example_1']),
        np.where(body_idx == 0, df['positive_example_2'], df['positive_example_1'])
    )
    df1['negative_example'] = np.where(
        is_positive_body,
        np.where(body_idx == 0, df['negative_example_2'], df['negative_example_1']),
        np.where(body_idx == 0, df['negative_example_2'], df['negative_example_1'])
    )
    
    df1['rule_violation'] = np.where(is_positive_body, 1, 0)
    df1 = df1.sample(frac=1, random_state=None).reset_index(drop=True)
    
    #df2
    df2['positive_example'] = np.where(
        is_positive_body,
        df1['body'].values, 
        np.where(body_idx == 0, df['positive_example_1'], df['positive_example_2'])
    )
    df2['negative_example'] = np.where(
        is_positive_body,
        np.where(body_idx == 0, df['negative_example_1'], df['negative_example_2']),
        df1['body'].values  
    )

    return df1, df2


def format_input(row):
    PROMPT_TEMPLATE = textwrap.dedent("""\
        Rule: {rule}
        Is Safe?
        Comment: {body}""")
    formatted = PROMPT_TEMPLATE.format(
        body=row['body'],
        rule=row['rule'],
    )
    return formatted

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

def process_df(df):
    df['body'] = df.body.apply(cleaner)
    df['body'] = clean_text_series(df['body'])
    df['text'] = df.apply(format_input, axis=1)
    if 'rule_violation' in df.columns:
        df['label'] = df['rule_violation']
    return df


def kfold_with_sparse(df_train: pd.DataFrame,
                      n_splits: int = 5,
                      target_col: str = "rule",
                      min_count: int = 100,
                      random_state: int = None
                     ):
    
    # Count occurrences
    token_counts = df_train[target_col].value_counts()

    # Separate dense / sparse
    dense_tokens = token_counts[token_counts >= min_count].index
    df_dense = df_train[df_train[target_col].isin(dense_tokens)]
    df_sparse = df_train[~df_train[target_col].isin(dense_tokens)]

    if len(df_dense) < min_count:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return [df_train.iloc[idx].copy() for idx, _ in kf.split(df_train)]

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=None
    )

    folds: List[pd.DataFrame] = []
    for train_idx, _ in skf.split(df_dense, df_dense[target_col]):
        fold_dense = df_dense.iloc[train_idx]
        fold = pd.concat([fold_dense, df_sparse], ignore_index=True)
        folds.append(fold)

    return folds


def train_classification_model(model_name, train, test, gid, sid, max_steps=60):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)

    import gc
    import pickle
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import numpy as np
    import pandas as pd
    from datasets import Dataset

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        device_map = {"":torch.cuda.current_device()},
    )
    
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=180)
    
    train_ds = Dataset.from_pandas(train[['text','label']])
    test_ds = Dataset.from_pandas(test[['text']])
    
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    eval_strategy="no",
    save_strategy="no",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    report_to="none",
    fp16=True,
    max_steps=max_steps, 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.eval()
    device = model.device
    predictions = trainer.predict(test_ds)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

    with open(f"bert-solution-gid{str(gid)}-sid{str(sid)}.pkl", 'wb') as f:
        pickle.dump(probs, f)

    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return    


def train_multiple_models_parallel(models_config, max_steps=4000):
    processes = []
    for config in models_config:
        p = Process(
            target=train_classification_model,
            args=(
                config["model_name"],
                config["train"].copy(),
                config["test"].copy(),
                config["gid"],
                config["sid"],
                max_steps
            )
        )
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()


test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

if len(test_df) == 10:
    max_steps = 3
else:
    max_steps = 4000


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 1},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 2},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 1},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 2},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 3},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 4},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 3},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 4},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 5},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 6},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 5},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 6},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 7},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 8},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 7},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 8},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 9},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 10},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 9},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 10},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 11},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 12},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 11},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 12},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 13},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 14},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 13},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 14},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 15},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 16},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 15},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 16},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 17},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 18},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 17},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 18},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 19},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 20},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 19},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 20},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 21},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 22},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 21},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 22},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 23},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 24},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 23},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 24},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 25},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 26},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 25},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 26},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 27},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 28},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 27},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 28},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


new_test_df, original_test_df = reconstruct_dataframes(test_df)
new_test_df = process_df(new_test_df)
original_test_df = process_df(original_test_df)

folds = kfold_with_sparse(new_test_df, n_splits=4)

models_config = [
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[0], "test": original_test_df, "gid": 0, "sid": 29},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[1], "test": original_test_df, "gid": 0, "sid": 30},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[2], "test": original_test_df, "gid": 1, "sid": 29},
    {"model_name": "/kaggle/input/mmbert-small/mmBERT-small", "train": folds[3], "test": original_test_df, "gid": 1, "sid": 30},
]

train_multiple_models_parallel(models_config, max_steps=max_steps)


import os
import pickle
import numpy as np
import time
from scipy.stats import rankdata

def load_file(path):
    with open(path, 'rb') as f:
        arr = pickle.load(f)
    return arr    


arrays = ( 
            [load_file(f"/kaggle/working/bert-solution-gid0-sid{i}.pkl") for i in range(1, 31)] + 
            [load_file(f"/kaggle/working/bert-solution-gid1-sid{i}.pkl") for i in range(1, 31)]
         )
arrays = [rankdata(arr[:, 1]) for arr in arrays]
pred  = np.nanmean(arrays, axis=0)
pred = pred/len(pred)


submission = pd.DataFrame(columns=['row_id', 'rule_violation'])
submission['row_id'] = original_test_df['row_id']
submission['rule_violation'] = pred

submission.to_csv("/kaggle/working/submission.csv", index=None)
submission

