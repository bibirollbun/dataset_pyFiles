import os
import gc
import random
from typing import Tuple

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

from cleantext import clean

import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


IS_KAGGLE = os.path.exists("/kaggle/input")
TRAIN_DATA = "/kaggle/input/jigsaw-agile-community-rules/train.csv" if IS_KAGGLE else "data/train.csv"
TEST_DATA = "/kaggle/input/jigsaw-agile-community-rules/test.csv" if IS_KAGGLE else "data/test.csv"

SHORT_RULE = True
CLEAN = True
COMMENT_FIRST = False


def load_model() -> Tuple[DistilBertTokenizer, DistilBertForSequenceClassification]:
    model_name = "distilbert-base-uncased"
    model_path = "/kaggle/input/transformers/pytorch/default/1/distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)

    set_seed(42)
    
    return (tokenizer, model)


# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text_pair"], truncation=True)


def generate_text(sep_token, rule, comment):
    if COMMENT_FIRST:
        return comment + sep_token + rule + sep_token
    else:
        return rule + sep_token + comment + sep_token


def load_train_data(tokenizer: DistilBertTokenizer) -> Tuple[Dataset, Dataset]:
    # Train Data Load
    data_path = TRAIN_DATA
    df = pd.read_csv(data_path)

    if SHORT_RULE:
        df["rule"] = df["rule"].apply(lambda x: x.split(":")[0])

    if CLEAN:
        df["body"] = df["body"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        df["positive_example_1"] = df["positive_example_1"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        df["positive_example_2"] = df["positive_example_2"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        df["negative_example_1"] = df["negative_example_1"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        df["negative_example_2"] = df["negative_example_2"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
    
    # Data Process
    expanded_data = []
  
    for _, row in df.iterrows():
        # Original example
        original_text = generate_text(tokenizer.sep_token, row["rule"], row["body"])
        expanded_data.append({
            'text_pair': original_text,
            'labels': row['rule_violation'],
        })
        
        # Positive examples (rule_violation = 1)
        for i in range(1, 3):
            if pd.notna(row[f'positive_example_{i}']) and row[f'positive_example_{i}'].strip():
                pos_example = generate_text(tokenizer.sep_token, row["rule"], row[f'positive_example_{i}'])
                expanded_data.append({'text_pair': pos_example, 'labels': 1})
        
        # Negative examples (rule_violation = 0)
        for i in range(1, 3):
            if pd.notna(row[f'negative_example_{i}']) and row[f'negative_example_{i}'].strip():
                neg_example = generate_text(tokenizer.sep_token, row["rule"], row[f'negative_example_{i}'])
                expanded_data.append({'text_pair': neg_example, 'labels': 0})

    # Test Data도 함께 로드하여 positive/negative example 활용
    test_df = pd.read_csv(TEST_DATA)
    
    if SHORT_RULE:
        test_df["rule"] = test_df["rule"].apply(lambda x: x.split(":")[0])
        
    if CLEAN:
        test_df["positive_example_1"] = test_df["positive_example_1"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        test_df["positive_example_2"] = test_df["positive_example_2"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        test_df["negative_example_1"] = test_df["negative_example_1"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        test_df["negative_example_2"] = test_df["negative_example_2"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
    
    for _, row in test_df.iterrows():
        # Test data의 positive examples (rule_violation = 1로 가정)
        for i in range(1, 3):
            if pd.notna(row[f'positive_example_{i}']) and row[f'positive_example_{i}'].strip():
                pos_example = generate_text(tokenizer.sep_token, row["rule"], row[f'positive_example_{i}'])
                expanded_data.append({'text_pair': pos_example, 'labels': 1})
        
        # Test data의 negative examples (rule_violation = 0으로 가정)
        for i in range(1, 3):
            if pd.notna(row[f'negative_example_{i}']) and row[f'negative_example_{i}'].strip():
                neg_example = generate_text(tokenizer.sep_token, row["rule"], row[f'negative_example_{i}'])
                expanded_data.append({'text_pair': neg_example, 'labels': 0})

    df = pd.DataFrame(expanded_data)
    text_length = df["text_pair"].apply(len)
    print(f"max: {text_length.max()}, min: {text_length.min()}, mean: {text_length.mean()}")
    print(df.iloc[0].text_pair)
    
    # Data split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    return (tokenized_train, tokenized_val)


def compute_metrics(eval_pred):
    """
    평가 시 사용할 메트릭을 계산합니다.
    
    Args:
        eval_pred: (predictions, labels) 튜플
        
    Returns:
        dict: 계산된 메트릭들
    """
    predictions, labels = eval_pred
    
    # 예측 확률 계산 (softmax 적용)
    probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=1)
    
    # 각 Column별 AUC 계산
    auc_scores = {}
    # TODO
    
    # 전체 AUC (클래스 1에 대한)
    try:
        overall_auc = roc_auc_score(labels, probabilities[:, 1])
        auc_scores['overall_auc'] = overall_auc
    except ValueError:
        auc_scores['overall_auc'] = 0.0
    
    return auc_scores


def train_model(model: DistilBertForSequenceClassification,
                tokenized_train: Dataset, 
                tokenized_val: Dataset, 
                tokenizer: DistilBertTokenizer) -> Trainer:
    """
    모델을 훈련합니다.
    
    Args:
        model: DistilBERT 모델
        tokenized_train: 토크나이징된 훈련 데이터셋
        tokenized_val: 토크나이징된 검증 데이터셋
        tokenizer: DistilBERT 토크나이저
        
    Returns:
        Trainer: 훈련된 트레이너 객체
    """
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,              # 훈련 에포크 수
        per_device_train_batch_size=8,   # GPU당 훈련 배치 크기
        per_device_eval_batch_size=8,    # GPU당 평가 배치 크기
        # early_stopping_patience=3,
        # early_stopping_threshold=0.001,
        learning_rate=2e-5,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",                # 로깅 비활성화
        metric_for_best_model="overall_auc",  # 최고 모델 선택 기준
        greater_is_better=True,          # AUC는 높을수록 좋음
        load_best_model_at_end=True      # 훈련 끝에 최고 모델 로드
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # 여기에 추가
    )

    print("Training started...")
    trainer.train()
    print("Training finished.")

    return trainer


def load_test_data(tokenizer: DistilBertTokenizer) -> Tuple[Dataset, pd.DataFrame]:
    test_df = pd.read_csv(TEST_DATA)

    if SHORT_RULE:
        test_df["rule"] = test_df["rule"].apply(lambda x: x.split(":")[0])
    if CLEAN:
        test_df["body"] = test_df["body"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        test_df["positive_example_1"] = test_df["positive_example_1"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        test_df["positive_example_2"] = test_df["positive_example_2"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        test_df["negative_example_1"] = test_df["negative_example_1"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        test_df["negative_example_2"] = test_df["negative_example_2"].apply(lambda x: clean(x, lower=True, no_urls=True, no_emails=True, no_phone_numbers=True))
        
    # test_df['text_pair'] = test_df['body'] + tokenizer.sep_token + test_df['rule'] + tokenizer.sep_token
    text_pairs = generate_text(tokenizer.sep_token, test_df['body'], test_df['rule'])
    test_df['text_pair'] = text_pairs

    
    test_dataset = Dataset.from_pandas(test_df)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    tokenized_test = tokenized_test.remove_columns(["body", "rule", "subreddit", "positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"])

    return (tokenized_test, test_df)


def predict_model(model: DistilBertForSequenceClassification, trainer: Trainer, tokenized_test: Dataset, test_df: pd.DataFrame):
    # Predict
    model.eval()

    # 예측 수행
    predictions = trainer.predict(tokenized_test)
    probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions), dim=1)[:, 1].numpy()

    # 제출 파일 생성
    submission_df = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': probabilities})
    submission_df.to_csv('submission.csv', index=False)

    print("Submission file created successfully.")


# Run
tokenizer, model = load_model()
print(tokenizer)

tokenized_train, tokenized_val = load_train_data(tokenizer)
tokenized_test, test_df = load_test_data(tokenizer)
trainer = train_model(model, tokenized_train, tokenized_val, tokenizer)
predict_model(model, trainer, tokenized_test, test_df)


del model
del tokenizer
del trainer
torch.cuda.empty_cache()
gc.collect()

