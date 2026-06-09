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


import torch
import torch.nn as nn

# TransformerEncoderLayerのパラメータ設定
embedding_dim = 16  # 埋め込み次元（embedding dimension）
nhead = 4           # マルチヘッドの数
hidden_dim = 64     # 内部FFNの次元
dropout = 0.1       # ドロップアウト率

# TransformerEncoderLayerを1層構築
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embedding_dim,
    nhead=nhead,
    dim_feedforward=hidden_dim,
    dropout=dropout
)

# ランダムな入力テンソル (sequence_length, batch_size, embedding_dim)
sequence_length = 10
batch_size = 32
src = torch.randn(sequence_length, batch_size, embedding_dim)

# 入力のshapeと中身
print("Input shape:", src.shape)

# TransformerEncoderLayerを通した結果
output = encoder_layer(src)

# 出力のshapeと中身
print("\nOutput shape:", output.shape)


from transformers import AutoTokenizer

# Tokenizerのロード
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# トークン化
tokens = tokenizer.tokenize(data['QuestionText'])
print("Tokens:", tokens)

# トークンIDの取得
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", token_ids)


# テキストをトークン化 (モデルへ入力可能な形式に変換)
inputs = tokenizer(data['QuestionText'], return_tensors="pt", add_special_tokens=True)

# トークナイザーの中身を確認
print("トークン化された結果:")
print(inputs)

# トークン列として確認 (IDだけの形式で表示)
token_ids = inputs['input_ids'][0]
print("トークンID列:", token_ids)

# トークン列としてのトークン数を確認
print("トークン数:", len(token_ids))

# トークン列を元の形式にデコード (テキストに戻す)
decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
print("元のテキストに戻した結果:")
print(decoded_text)


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print("📏  語彙サイズ =", tok.vocab_size)


!pip install -q bertviz


from bertviz import head_view
from transformers import AutoModel
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
inputs = tokenizer("The astronaut looked out the window and saw a beautiful blue planet.", return_tensors="pt")
out = model(**inputs)
head_view(out.attentions, tok.convert_ids_to_tokens(inputs["input_ids"][0]))


from transformers import pipeline

# 感情分析パイプラインの作成
sentiment_analysis = pipeline("sentiment-analysis", model="tabularisai/robust-sentiment-analysis")

# 分析する文章のリスト
sentences = [
    "I'm so grateful for all the amazing people in my life!",
    "I love this product!",
    "The movie was okay, nothing special.",
    "I feel sad.",
    "I feel really sad and lonely today.",
]

# 感情分析の実行と結果の表示
for sentence in sentences:
    result = sentiment_analysis(sentence)[0]
    print(f"Sentence: {sentence}")
    print(f"Label: {result['label']}")
    print(f"Score: {result['score']:.4f}")
    print("-" * 20)


from transformers import AutoTokenizer, AutoModelForSequenceClassification

sentences = [
    "I'm so grateful for all the amazing people in my life!",
    "I love this product!",
    "The movie was okay, nothing special.",
    "I feel sad.",
    "I feel really sad and lonely today.",
]

model_name = "tabularisai/robust-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
probabilities = outputs.logits.softmax(dim=-1)

label_mapping = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}

for i, probs in enumerate(probabilities):
    print(f"Sentence: {sentences[i]}")
    for index, score in enumerate(probs):
        print(f"  {label_mapping[index]}: {score:.4f}")
    print("-" * 20)


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


import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

model_name = "google-bert/bert-base-cased"

# ——— pandas DataFrame から 🤗datasets Dataset に変換 ———
train_ds = Dataset.from_pandas(train_df)
test_ds  = Dataset.from_pandas(test_df)

# ——— ラベル文字列 → 整数 ID マッピング ———
label2id = {"Negative": 0, "Neutral": 1, "Positive": 2, "Irrelevant": 3}
id2label = {v: k for k, v in label2id.items()}

def map_labels(example):
    example["labels"] = label2id[example["labels"]]
    return example

train_ds = train_ds.map(map_labels)
test_ds = test_ds.map(map_labels)

# ——— トークナイザー準備 ———
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(
        text=examples["message"], 
        truncation=True,
        padding="max_length", 
        max_length=128,
    )

train_ds = train_ds.map(preprocess_function, batched=True)
test_ds  = test_ds.map(preprocess_function,  batched=True)

# 4) PyTorch Tensor フォーマットに
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format( "torch", columns=["input_ids", "attention_mask", "labels"])

# ——— モデル準備 ———
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id))

# ——— 評価指標 ———
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ——— トレーニング設定 ———
training_args = TrainingArguments(
    output_dir="outputs",
    report_to='none',
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=20,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)


# ——— 学習実行 ———
trainer.train()

# ——— テストセットで評価 ———
metrics = trainer.evaluate(eval_dataset=test_ds)
print(f"Test Accuracy: {metrics['eval_accuracy']:.4f}")


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


from transformers import AutoTokenizer, AutoModel
import torch

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 入力のトークン化（return_tensorsでPyTorchのテンソルを取得）
inputs = tokenizer(data['QuestionText'], return_tensors='pt')

# モデルに入力を渡す
with torch.no_grad():
    outputs = model(**inputs)

# 最終層の出力 (last_hidden_state) を取得
last_hidden_state = outputs.last_hidden_state

# 出力の内容と形状を表示
print("last_hidden_state:")
print(last_hidden_state)
print("形状 (shape):", last_hidden_state.shape)


# Mean Poolingの定義: attention maskを考慮した平均化
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # モデル出力の最初の要素がトークンごとの埋め込み
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# テキストをトークン化
inputs = tokenizer(data['QuestionText'], return_tensors="pt", padding=True, truncation=True, max_length=512)

# モデルに入力を渡して計算
with torch.no_grad():
    outputs = model(**inputs)

# Mean Poolingを適用して文章全体のembeddingを取得
sentence_embedding = mean_pooling(outputs, inputs['attention_mask'])

# 出力の結果を確認
print("文章全体のembedding:")
print(sentence_embedding)
print("埋め込みベクトルの形状 (shape):", sentence_embedding.shape)


import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

# コサイン類似度の計算
def calculate_cosine_similarity(embedding1, embedding2):
    # scipy を利用しコサイン類似度を計算
    return 1 - cosine(embedding1.cpu().numpy(), embedding2.cpu().numpy())

texts = [data['QuestionText'], data['MisconceptionAName'], "Hello world!"]

# それぞれのテキストから埋め込みを取得
embeddings = []
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean Poolingを適用して文章全体のembeddingを取得
    sentence_embedding = mean_pooling(outputs, inputs['attention_mask'])
    embeddings.append(sentence_embedding)

# コサイン類似度を計算
similarity_q_misconception = calculate_cosine_similarity(embeddings[0][0], embeddings[1][0])  # QuestionText vs MisconceptionAName
similarity_q_hello = calculate_cosine_similarity(embeddings[0][0], embeddings[2][0])         # QuestionText vs Hello world!
similarity_misconception_hello = calculate_cosine_similarity(embeddings[1][0], embeddings[2][0])  # MisconceptionAName vs Hello world!

# 結果の出力
print("コサイン類似度 (QuestionText vs MisconceptionAName):", similarity_q_misconception)
print("コサイン類似度 (QuestionText vs Hello world!):", similarity_q_hello)
print("コサイン類似度 (MisconceptionAName vs Hello world!):", similarity_misconception_hello)


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





