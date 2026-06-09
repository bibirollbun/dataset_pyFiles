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

# パラメータ
sequence_length = 10
batch_size = 2
embedding_dim = 384
nhead = 8  # embedding_dim が 384 なら 8 ヘッドが一般的

# 1 層分の定義を作って、それを 2 層重ねる
single_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
encoder = nn.TransformerEncoder(single_layer, num_layers=1)

# ランダムな入力テンソルを生成 (sequence_length, batch_size, embedding_dim)
x = torch.rand(sequence_length, batch_size, embedding_dim)

# 順伝播
output = encoder(x)

# shape を確認
print("入力  shape:", x.shape)       # -> torch.Size([10, 2, 384])
print("出力  shape:", output.shape)  # -> torch.Size([10, 2, 384])

# サンプル表示
print("\n入力テンソルサンプル:\n", x[:2, :, :2])
print("\n出力テンソルサンプル:\n", output[:2, :, :2])


from transformers import AutoTokenizer

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


# トークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# トークン化
question = data['QuestionText']
tokens = tokenizer.tokenize(question)
ids = tokenizer.convert_tokens_to_ids(tokens)

# 結果の表示
for tok, idx in zip(tokens, ids):
    print(f"{tok:15} → {idx}")


from transformers import AutoTokenizer

# サンプルデータ
data = {
    'QuestionId': 101,
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
    'MisconceptionDName': None
}

# 1. トークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# 2. モデル入力フォーマットへの変換（自動で [CLS] と [SEP] が追加される）
encoding = tokenizer(
    data['QuestionText'],
    return_tensors='pt',      # PyTorch tensor 形式で返す
    add_special_tokens=True,  # 明示的に特別トークン追加（デフォルト True）
)

input_ids = encoding['input_ids'][0]  # バッチサイズ 1 の場合、最初の要素を取り出す

# 3. トークン列とトークン数を確認
tokens = tokenizer.convert_ids_to_tokens(input_ids)
print("=== トークン一覧 ===")
print(tokens)
print(f"トークン数: {len(tokens)}")  # 先頭 [CLS] と末尾 [SEP] を含む件数

# 4. トークン ID 列を表示
print("\n=== トークン ID 列 ===")
print(input_ids.tolist())

# 5. 元のトークン列に戻す（デコード）
decoded_with_special = tokenizer.decode(input_ids, skip_special_tokens=False)
decoded_without_special = tokenizer.decode(input_ids, skip_special_tokens=True)
print("\n=== デコード結果 ===")
print("特殊トークン含む:", decoded_with_special)
print("特殊トークン除外:", decoded_without_special)


print(f"Vocabulary size: {tokenizer.vocab_size}")


!pip install -q bertviz


from bertviz import head_view
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
inputs = tokenizer("The astronaut looked out the window and saw a beautiful blue planet.", return_tensors="pt")
out = model(**inputs)
head_view(out.attentions, tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))


from transformers import pipeline

# 感情分析パイプラインを、tabularisai/robust-sentiment-analysis モデルで読み込む
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="tabularisai/robust-sentiment-analysis"
)

sentences = [
    "I'm so grateful for all the amazing people in my life!",
    "I love this product!",
    "The movie was okay, nothing special.",
    "I feel sad.",
    "I feel really sad and lonely today.",
]

# 推論
results = sentiment_analyzer(sentences)

# 結果を表示
for sent, res in zip(sentences, results):
    print(f"文: {sent!r}")
    print(f"  → ラベル: {res['label']}, 確信度: {res['score']:.3f}")


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# モデル名
model_name = "tabularisai/robust-sentiment-analysis"

# トークナイザ & モデルのロード
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# テスト文
sentences = [
    "I'm so grateful for all the amazing people in my life!",
    "I love this product!",
    "The movie was okay, nothing special.",
    "I feel sad.",
    "I feel really sad and lonely today.",
]

# 1. トークン化 & テンソル化
encoding = tokenizer(
    sentences,
    padding=True, 
    truncation=True,
    return_tensors="pt"
)

# 2. モデル推論（勾配不要モード）
with torch.no_grad():
    outputs = model(**encoding)

# 3. ロジット → 確率
logits = outputs.logits           # shape: (batch_size, num_labels)
probs  = F.softmax(logits, dim=-1)  # shape: (batch_size, num_labels)

# 4. ラベルマッピング
label_map = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

# 5. 結果表示
for sent, prob, logit in zip(sentences, probs, logits):
    pred_idx  = prob.argmax().item()
    pred_lbl  = label_map[pred_idx]
    confidence = prob[pred_idx].item()
    print(f"文: {sent!r}")
    print(f"  → 予測ラベル: {pred_lbl} ({pred_idx}), 確信度: {confidence:.3f}")
    print(f"    各クラス確率: " + ", ".join(f"{label_map[i]}: {p:.3f}" for i,p in enumerate(prob.tolist())))
    print()


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


!pip install -q evaluate

import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import evaluate
import numpy as np

# ── 以下、変更しないでください ────────────────────
columns = ['x1', 'source','labels','message']
train_df = pd.read_csv(
    '/kaggle/input/twitter-entity-sentiment-analysis/twitter_training.csv',
    header=None, names=columns
).dropna()
train_df['message'] = train_df['source'] + ': ' + train_df['message']
train_df = train_df.sample(frac=0.05, random_state=42)

test_df = pd.read_csv(
    '/kaggle/input/twitter-entity-sentiment-analysis/twitter_validation.csv',
    header=None, names=columns
)

# ラベルの数値化
label_list = ["Positive", "Negative", "Neutral", "Irrelevant"]
label2id = {label: i for i, label in enumerate(label_list)}
train_df["label"] = train_df["labels"].map(label2id)
test_df["label"]  = test_df["labels"].map(label2id)

# HF Dataset に変換
train_ds = Dataset.from_pandas(train_df[["message", "label"]])
eval_ds  = Dataset.from_pandas(test_df[["message", "label"]])

# モデル＆トークナイザをロード
model_name = "microsoft/deberta-v3-xsmall"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label={i:l for i,l in enumerate(label_list)},
    label2id=label2id
)

# トークン化関数
def tokenize_fn(example):
    return tokenizer(
        example["message"],
        truncation=True,
        padding=False,
    )

train_ds = train_ds.map(tokenize_fn, batched=False)
eval_ds  = eval_ds.map(tokenize_fn, batched=False)

# 不要列を削除＆フォーマット設定
train_ds = train_ds.remove_columns(["message"])
eval_ds  = eval_ds.remove_columns(["message"])
train_ds.set_format("torch")
eval_ds.set_format("torch")

# バッチごとに動的パディング
data_collator = DataCollatorWithPadding(tokenizer)

# Accuracy メトリクス
metric = evaluate.load("accuracy")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)

# ── ↓ここから Trainer まで同じセルにまとめます↓ ────────────────────

# TrainingArguments の定義
training_args = TrainingArguments(
    output_dir="./deberta-xsmall-finetuned",
    evaluation_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    learning_rate=2e-5,
    logging_steps=50,
    load_best_model_at_end=False,
)

# Trainer インスタンス作成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 学習＆評価
trainer.train()
eval_result = trainer.evaluate()

print("\n=== Validation Accuracy ===")
print(f"Accuracy: {eval_result['eval_accuracy']:.4f}")


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


# 必要ライブラリのインポート
from transformers import AutoTokenizer, AutoModel
import torch

# 1) テキスト抽出
text = data['QuestionText']

# 2) Tokenizer と Model のロード
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model     = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# 3) トークン化
inputs    = tokenizer(text, return_tensors='pt')
input_ids = inputs['input_ids']           # shape: [1, seq_len]
tokens    = tokenizer.convert_ids_to_tokens(input_ids[0])

# 4) 埋め込みを計算
with torch.no_grad():
    outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
# last_hidden_state の shape は torch.Size([1, seq_len, hidden_size])

# 5) 中身と shape の確認
print("Tokens:", tokens)
print("Number of tokens:", len(tokens))
print("last_hidden_state shape:", last_hidden_state.shape)
print("\n-- sample embedding vector for first token --")
print(last_hidden_state[0, 0])  # 1 トークン目の 384 次元ベクトルを表示


# Mean Pooling - attention mask を考慮して正しい平均を取る
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
    # attention_mask を [batch_size, seq_len, 1] に拡張
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # マスクされた部分を除いたトークンのベクトルを足し合わせ
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    # トークン数で割って平均を取る（ゼロ除算防止の clamp）
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask  # [batch_size, hidden_size]

# ── セットアップ ──
data = {
    'QuestionText': 'What is \\( 8 \\) out of \\( 40 \\) as a percentage?'
}
text = data['QuestionText']

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model     = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# トークン化
inputs = tokenizer(text, return_tensors='pt')
# 埋め込み計算（勾配不要）
with torch.no_grad():
    outputs = model(**inputs)

# Mean pooling で文章全体のベクトルを得る
sentence_embedding = mean_pooling(outputs, inputs['attention_mask'])

# 確認
print("Sentence embedding shape:", sentence_embedding.shape)  # torch.Size([1, 384])
print("Sentence embedding vector:", sentence_embedding[0])   # 384次元ベクトル


# 1. ライブラリのインポート
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

# 2. Mean Pooling 関数
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [batch, seq_len, hidden]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_emb = torch.sum(token_embeddings * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    return sum_emb / sum_mask  # [batch, hidden]

# 3. 対象の文章
data = {
    'QuestionText': 'What is \\( 8 \\) out of \\( 40 \\) as a percentage?',
    'MisconceptionAName': 'Converts a fraction to a percentage by writing the numerator followed by the denominator'
}
texts = {
    'QuestionText': data['QuestionText'],
    'MisconceptionAName': data['MisconceptionAName'],
    'Hello world!': 'Hello world!'
}

# 4. モデルとトークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model     = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# 5. 埋め込み取得
embeddings = {}
with torch.no_grad():
    for name, txt in texts.items():
        inputs = tokenizer(txt, return_tensors='pt')
        out    = model(**inputs)
        emb    = mean_pooling(out, inputs['attention_mask'])
        embeddings[name] = emb[0]  # [hidden]

# 6. コサイン類似度行列の計算
keys = list(embeddings.keys())
sim = []
for k1 in keys:
    row = []
    for k2 in keys:
        cos = torch.cosine_similarity(
            embeddings[k1].unsqueeze(0),
            embeddings[k2].unsqueeze(0),
            dim=1
        ).item()
        row.append(cos)
    sim.append(row)

# 7. 結果を DataFrame で表示
df = pd.DataFrame(sim, index=keys, columns=keys)
print(df)


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


# 3. misconception_mappingの内容を全てembeddingに変換

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Mean Pooling - attention mask を考慮して平均化
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state            # [batch_size, seq_len, hidden_size]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask                            # [batch_size, hidden_size]

# 1) misconception_mapping.csv を読み込む
mapping_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')

# 2) トークナイザーとモデルをロード
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model     = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# 3) 各 MisconceptionName を embedding に変換
embeddings = []
with torch.no_grad():
     for text in mapping_df['MisconceptionName'].fillna(''):
        inputs = tokenizer(text,
                           return_tensors='pt',
                           padding=True,
                           truncation=True,
                           max_length=128)
        outputs = model(**inputs)
        emb = mean_pooling(outputs, inputs['attention_mask'])   # shape: [1, hidden_size]
        embeddings.append(emb.squeeze(0).cpu().numpy())

# 4) DataFrame に埋め込みを追加
mapping_df['embedding'] = embeddings

# 5) 埋め込み次元の確認用カラム
mapping_df['emb_dim'] = mapping_df['embedding'].apply(lambda x: x.shape[0])

# 6) 結果を確認
print(mapping_df[['MisconceptionId', 'MisconceptionName', 'emb_dim']].head())

# 7) 必要に応じて保存
mapping_df.to_pickle('/kaggle/working/miscon_mapping_with_emb.pkl')


# 4. validationデータの QuestionTextとMisconceptionName を結合して embeddingに変換
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ====================================================
# 1) データ読み込み＆分割 (#1 のコードをそのまま利用)
# ====================================================
base_path   = '/kaggle/input/eedi-mining-misconceptions-in-mathematics'
df_all      = pd.read_csv(f'{base_path}/train.csv')

data_num    = len(df_all)
train_rate  = 0.8
train_num   = int(data_num * train_rate)

np.random.seed(42)
train_index = np.random.choice(data_num, train_num, replace=False)
valid_index = list(set(range(data_num)) - set(train_index))

train_raw   = df_all.iloc[train_index].reset_index(drop=True)
valid_raw   = df_all.iloc[valid_index].reset_index(drop=True)

# ====================================================
# 2) 前処理 (#2 の preprocess をそのまま利用)
#    （valid_raw → valid_df に変形し、MisconceptionName をマージ済みと仮定）
# ====================================================
def preprocess(df):
    result = []
    for _, row in df.iterrows():
        for option in ['A','B','C','D']:
            mid_col = f'Misconception{option}Id'
            if pd.isnull(row[mid_col]):
                continue
            result.append({
                'QuestionText'    : row['QuestionText'],
                'ConstructId'     : row['ConstructId'],
                'ConstructName'   : row['ConstructName'],
                'SubjectId'       : row['SubjectId'],
                'SubjectName'     : row['SubjectName'],
                'CorrectAnswer'   : row['CorrectAnswer'],
                'IsCorrect'       : (row['CorrectAnswer'] == option),
                'Option'          : option,
                'AnswerText'      : row[f'Answer{option}Text'],
                'MisconceptionId' : int(row[mid_col]),
            })
    flat = pd.DataFrame(result)
    mapping_df = pd.read_csv(f'{base_path}/misconception_mapping.csv')
    flat = flat.merge(mapping_df, on='MisconceptionId', how='left')
    return flat

valid_df = preprocess(valid_raw)
# この時点で valid_df に 'QuestionText' と 'MisconceptionName' が入っている

# ====================================================
# 3) 埋め込みモデルのロード (#3 のまま)
# ====================================================
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model     = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state               # [batch, seq, dim]
    mask_expanded    = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings   = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask         = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask                                # [batch, dim]

# ====================================================
# 4) QuestionText + MisconceptionName を結合して埋め込み算出
# ====================================================
# 欠損値があれば空文字に置換→
texts = valid_df['QuestionText'].fillna('')
batch_size = 32
all_embs   = []

with torch.no_grad():
    for start in range(0, len(texts), batch_size):
        batch_texts = texts.iloc[start:start+batch_size].tolist()
        inputs      = tokenizer(
                          batch_texts,
                          padding=True,
                          truncation=True,
                          max_length=128,
                          return_tensors='pt'
                      )
        outputs     = model(**inputs)
        embs        = mean_pooling(outputs, inputs['attention_mask'])
        all_embs.append(embs.cpu().numpy())

# numpy 配列にまとめて DataFrame に格納
emb_matrix = np.vstack(all_embs)
valid_df['combo_embedding'] = list(emb_matrix)

# ====================================================
# 5) 結果の保存
# ====================================================
valid_df.to_pickle('/kaggle/working/valid_with_combo_emb.pkl')
print("完了：", valid_df.shape)


# 5. ベクトル同士の検索をして、validationのそれぞれについて候補を25個作成
import pandas as pd
import numpy as np

# ----------------------------------------------------
# 1) pickle から前段階で作成した DataFrame を読み込み
# ----------------------------------------------------
mapping_df = pd.read_pickle('/kaggle/working/miscon_mapping_with_emb.pkl')
valid_df   = pd.read_pickle('/kaggle/working/valid_with_combo_emb.pkl')

# ----------------------------------------------------
# 2) 埋め込み行列を抽出・正規化
# ----------------------------------------------------
# mapping：MisconceptionName の embedding
M = np.vstack(mapping_df['embedding'].values)      # shape: [num_map, dim]
# validation：QuestionText + MisconceptionName の embedding
V = np.vstack(valid_df['combo_embedding'].values)  # shape: [num_val, dim]

# L2 正規化
M_norm = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
V_norm = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)

# ----------------------------------------------------
# 3) コサイン類似度行列を計算
#    （V_norm × M_norm^T で各 valid サンプル × 各 mapping サンプルの cos 類似度）
# ----------------------------------------------------
sim_matrix = V_norm.dot(M_norm.T)  # shape: [num_val, num_map]

# ----------------------------------------------------
# 4) top-25 のインデックスを取得し、MisconceptionId を候補リスト化
# ----------------------------------------------------
top_k    = 25
# 各行について類似度の降順ソート → 上位 top_k の列インデックスを取得
top_idx  = np.argsort(-sim_matrix, axis=1)[:, :top_k]      # shape: [num_val, top_k]
# インデックスを MisconceptionId に変換
top_preds = mapping_df['MisconceptionId'].values[top_idx]  # shape: [num_val, top_k]

# ----------------------------------------------------
# 5) 結果を DataFrame にまとめる
# ----------------------------------------------------
pred_cols = [f'pred_{i+1}' for i in range(top_k)]
preds_df   = pd.DataFrame(top_preds, columns=pred_cols)

# 元の valid_df から識別用カラムを持ってくる
preds_df['ConstructId']         = valid_df['ConstructId'].values
preds_df['Option']              = valid_df['Option'].values
preds_df['TrueMisconceptionId'] = valid_df['MisconceptionId'].values

# カラム順を整理
preds_df = preds_df[
    ['ConstructId', 'Option', 'TrueMisconceptionId'] + pred_cols
]

# ----------------------------------------------------
# 6) 保存
# ----------------------------------------------------
preds_df.to_pickle('/kaggle/working/valid_top25_preds.pkl')
print("Finished: predictions for", len(preds_df), "validation samples.")


import pandas as pd

# 1) 予測結果とマッピングを読み込む
preds_df    = pd.read_pickle('/kaggle/working/valid_top25_preds.pkl')
mapping_df  = pd.read_csv(
    '/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv'
)

# 2) MisconceptionId → MisconceptionName の辞書を作成
id2name = dict(zip(mapping_df['MisconceptionId'], mapping_df['MisconceptionName']))

# 3) pred_1～pred_25 の列名リスト
top_k     = 25
pred_cols = [f'pred_{i}' for i in range(1, top_k+1)]

# 4) 先頭5件だけを出力
for idx, row in preds_df.head(5).iterrows():
    sample_id = f"{row['ConstructId']}_{row['Option']}"
    id_list   = [row[col] for col in pred_cols]
    name_list = [id2name.get(m_id, '') for m_id in id_list]
    
    print(f"Sample {sample_id} → Top{top_k} Misconceptions:")
    for rank, (m_id, m_name) in enumerate(zip(id_list, name_list), start=1):
        print(f"  {rank:2d}. ID={m_id}  Name={m_name}")
    print('-' * 80)


import pandas as pd
import numpy as np

# ----------------------------------------
# 1) MAP@K の実装（与えられた関数をそのまま利用）
# ----------------------------------------
def apk(actual, predicted, k=25):
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# ----------------------------------------
# 2) 予測結果の読み込み
# ----------------------------------------
preds_df = pd.read_pickle('/kaggle/working/valid_top25_preds.pkl')

# ----------------------------------------
# 3) actual（正解ID のリスト化）と predicted（上位25件のリスト化）を作成
# ----------------------------------------
# 各行の TrueMisconceptionId を長さ1のリストに
actual    = preds_df['TrueMisconceptionId'].apply(lambda x: [x]).tolist()

# pred_1～pred_25 の列をリスト化
pred_cols = [f'pred_{i}' for i in range(1, 26)]
predicted = preds_df[pred_cols].values.tolist()

# ----------------------------------------
# 4) MAP@25 を計算して出力
# ----------------------------------------
score = mapk(actual, predicted, k=25)
print(f"MAP@25 = {score:.4f}")


#1.`sentence-transformers/all-MiniLM-L6-v2'をtrainデータを使ってfine tuningしてください

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,  # モデル本体を扱うクラス
    InputExample,         # トレーニングデータのフォーマット
    losses                # 損失関数（学習の「採点基準」）が入っているモジュール
)

# ----------------------------------------
# 1) データ読み込み＆前処理
# ----------------------------------------

# Kaggle 上のデータが置かれているディレクトリ
base_path = '/kaggle/input/eedi-mining-misconceptions-in-mathematics'

# train.csv を pandas の DataFrame に読み込む
# DataFrame は Excel の表みたいなもの。行と列でデータを扱える。
df_all = pd.read_csv(f'{base_path}/train.csv')

def preprocess(df):
    """
    DataFrame の各行について、
      ・QuestionText（設問文）
      ・MisconceptionName（誤解の名前）
    のペアを作る関数です。
    """
    rows = []
    # DataFrame.iterrows() で 1 行ずつ取り出す
    for _, row in df.iterrows():
        qtext = row['QuestionText'] or ""  # 設問文。もし NaN（欠損）なら空文字に置換
        # A～D の選択肢それぞれについて MisconceptionId があれば処理
        for opt in ['A', 'B', 'C', 'D']:
            mid = row.get(f'Misconception{opt}Id')
            if pd.isnull(mid):
                # None や空ならスキップ
                continue
            # 1つの dict (行データ) にまとめる
            rows.append({
                'QuestionText': qtext,
                'MisconceptionId': int(mid),  # float → int に変換
            })
    # リスト of dict を DataFrame に変換
    flat = pd.DataFrame(rows)

    # misconception_mapping.csv を読み込んで「ID → 名前」のマッピングを取得
    mapping = pd.read_csv(f'{base_path}/misconception_mapping.csv')
    # merge() で flat と mapping を MisconceptionId が一致する行同士で結合
    flat = flat.merge(mapping, on='MisconceptionId', how='left')
    # これで flat に 'MisconceptionName' も入る
    return flat

# train/valid 分割：まず全行数を取得
data_num = len(df_all)
# ランダムで 80% を train、残り 20% を valid に
rng = np.random.RandomState(42)
train_idx = rng.choice(data_num, size=int(0.8 * data_num), replace=False)
train_raw = df_all.iloc[train_idx].reset_index(drop=True)  # train 用の DataFrame
# 前処理関数で必要なカラムだけにフラット化
train_df = preprocess(train_raw)

# ----------------------------------------
# 2) Sentence-Transformers 用トレーニングデータ作成
# ----------------------------------------

train_samples = []
# train_df の各行を InputExample に変換
for _, row in train_df.iterrows():
    question = row['QuestionText']
    misconception = row['MisconceptionName'] or ""  # NaN → 空文字
    # InputExample(texts=[text1, text2], label=1.0) は
    # 「text1 と text2 は似ている（正例）」という情報を伝えるフォーマット
    train_samples.append(
        InputExample(texts=[question, misconception], label=1.0)
    )
    # （必要に応じて）ネガティブ例を同じバッチ内で自動生成する損失関数もある

# ----------------------------------------
# 3) モデル＆トレーニング設定
# ----------------------------------------

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# 事前学習済みモデルをロード
model = SentenceTransformer(model_name)

train_batch_size = 16   # 一度に GPU に通すサンプル数
num_epochs = 3         # データセットを何周するか
# ウォームアップステップ数（学習率を徐々に上げるステップ数）
warmup_steps = int(len(train_samples) * num_epochs / train_batch_size * 0.1)

# DataLoader は PyTorch のバッチ生成器
train_dataloader = DataLoader(
    train_samples,
    shuffle=True,       # 毎エポックごとに順番をシャッフル
    batch_size=train_batch_size
)

# MultipleNegativesRankingLoss は、一つのテキストと他のネガティブテキストの組み合わせを利用する
train_loss = losses.MultipleNegativesRankingLoss(model)

# ----------------------------------------
# 4) Fine‑tuning 実行
# ----------------------------------------

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,        # エポック数
    warmup_steps=warmup_steps,# ウォームアップステップ
    use_amp=True,             # GPU の場合は混合精度(AMP)を使うと速い
    show_progress_bar=True,   # 学習の進捗バーを表示
    checkpoint_path='./checkpoints',      # 中間チェックポイントの保存先
    checkpoint_save_steps=1000            # 何ステップごとにチェックポイントを保存するか
)

# ----------------------------------------
# 5) Fine‑tuned モデルの保存
# ----------------------------------------

# 学習後のモデルを指定パスに保存
model.save('/kaggle/working/fine-tuned-all-MiniLM-L6-v2')
print("Fine-tuned model has been saved.")


#2. 学習したモデルを使って、再度 misconception_mapping, validationデータのembeddingを作成してください

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------------------
# 1) ファインチューニング済みモデルのロード
# ----------------------------------------
# 先ほど学習したモデルが保存されているディレクトリを指定
ft_model_path = '/kaggle/working/fine-tuned-all-MiniLM-L6-v2'
model = SentenceTransformer(ft_model_path)

# ----------------------------------------
# 2) misconception_mapping の埋め込みを再作成
# ----------------------------------------
base_path   = '/kaggle/input/eedi-mining-misconceptions-in-mathematics'
mapping_df  = pd.read_csv(f'{base_path}/misconception_mapping.csv')

# テキストリスト作成（欠損は空文字に）
map_texts = mapping_df['MisconceptionName'].fillna('').tolist()

# model.encode を使って一括エンコード
# convert_to_numpy=True で NumPy 配列を直接取得
map_embs = model.encode(
    map_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# DataFrame に埋め込みと次元情報を格納
mapping_df['embedding_ft'] = list(map_embs)
mapping_df['emb_ft_dim']   = [emb.shape[0] for emb in map_embs]

# 必要なら保存
mapping_df.to_pickle('/kaggle/working/miscon_mapping_ft.pkl')
print("Finished: mapping embeddings (fine-tuned)")

# ----------------------------------------
# 3) validation データの埋め込みを再作成
# ----------------------------------------
# 前処理関数をそのまま再利用して valid_df を再構築
def preprocess(df):
    rows = []
    for _, row in df.iterrows():
        qtext = row['QuestionText'] or ""
        for opt in ['A','B','C','D']:
            mid = row.get(f'Misconception{opt}Id')
            if pd.isnull(mid):
                continue
            rows.append({
                'QuestionText'    : qtext,
                'MisconceptionId' : int(mid),
            })
    flat = pd.DataFrame(rows)
    mapping = mapping_df[['MisconceptionId','MisconceptionName']]
    return flat.merge(mapping, on='MisconceptionId', how='left')

# train/valid split（#1 と同じ乱数設定を再現）
df_all     = pd.read_csv(f'{base_path}/train.csv')
data_num   = len(df_all)
np.random.seed(42)
train_idx  = np.random.choice(data_num, int(0.8*data_num), replace=False)
valid_raw  = df_all.iloc[list(set(range(data_num)) - set(train_idx))].reset_index(drop=True)
valid_df   = preprocess(valid_raw)

# QuestionText と MisconceptionName を結合
valid_texts = valid_df['QuestionText'].fillna('').tolist()

# model.encode で埋め込み取得
valid_embs = model.encode(
    valid_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# DataFrame に格納
valid_df['combo_embedding_ft'] = list(valid_embs)
valid_df['combo_emb_ft_dim']   = [emb.shape[0] for emb in valid_embs]

# 保存
valid_df.to_pickle('/kaggle/working/valid_with_combo_emb_ft.pkl')
print("Finished: validation embeddings (fine-tuned)")


#3. ベクトル同士の検索をして、validationのそれぞれについて候補を25個作成してください

import pandas as pd
import numpy as np

# ----------------------------------------------------
# 1) fine‑tuned 埋め込みを保存した Pickle ファイルを読み込む
# ----------------------------------------------------
mapping_df = pd.read_pickle('/kaggle/working/miscon_mapping_ft.pkl')
valid_df   = pd.read_pickle('/kaggle/working/valid_with_combo_emb_ft.pkl')

# ----------------------------------------------------
# 2) 埋め込み行列を NumPy 配列として取り出し
#    – mapping_df['embedding_ft']: MisconceptionName のファインチューニング後埋め込み
#    – valid_df['combo_embedding_ft']: QuestionText+MisconceptionName の埋め込み
# ----------------------------------------------------
M = np.vstack(mapping_df['embedding_ft'].values)      # shape: [num_map, dim]
V = np.vstack(valid_df['combo_embedding_ft'].values)  # shape: [num_val, dim]

# ----------------------------------------------------
# 3) L2 正規化（コサイン類似度計算の前処理）
# ----------------------------------------------------
M_norm = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
V_norm = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)

# ----------------------------------------------------
# 4) 類似度行列を計算 → 各検証サンプルごとに上位25インデックスを取得
# ----------------------------------------------------
sim_matrix = V_norm.dot(M_norm.T)                  # [num_val, num_map]
top_k      = 25
top_idx    = np.argsort(-sim_matrix, axis=1)[:, :top_k]  # [num_val, top_k]

# ----------------------------------------------------
# 5) MisconceptionId の候補リストに変換
# ----------------------------------------------------
top_preds = mapping_df['MisconceptionId'].values[top_idx]  # [num_val, top_k]

# ----------------------------------------------------
# 6) DataFrame にまとめる
# ----------------------------------------------------
pred_cols = [f'pred_{i+1}' for i in range(top_k)]
preds_df  = pd.DataFrame(top_preds, columns=pred_cols)

# （必要であれば）各検証サンプルの正解IDも追加
preds_df['TrueMisconceptionId'] = valid_df['MisconceptionId'].values

# ----------------------------------------------------
# 7) 保存 or 出力
# ----------------------------------------------------
preds_df.to_pickle('/kaggle/working/valid_top25_preds_ft.pkl')
print("Finished: generated top-25 candidates for", len(preds_df), "validation samples.")


#4. map@25を計算して性能を評価してください
import pandas as pd
import numpy as np

# ----------------------------------------
# 1) MAP@K を計算する関数定義（前と同じ）
# ----------------------------------------
def apk(actual, predicted, k=25):
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=25):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# ----------------------------------------
# 2) Fine‑tuned モデルの予測結果を読み込み
# ----------------------------------------
preds_df = pd.read_pickle('/kaggle/working/valid_top25_preds_ft.pkl')

# ----------------------------------------
# 3) actual（正解ID のリスト化）と predicted（上位25件のリスト化）を作成
# ----------------------------------------
# 各行の TrueMisconceptionId を長さ1のリストに
actual    = preds_df['TrueMisconceptionId'].apply(lambda x: [x]).tolist()

# pred_1～pred_25 の列をリスト化
pred_cols = [f'pred_{i}' for i in range(1, 26)]
predicted = preds_df[pred_cols].values.tolist()

# ----------------------------------------
# 4) MAP@25 を計算して出力
# ----------------------------------------
score = mapk(actual, predicted, k=25)
print(f"Fine‑tuned MAP@25 = {score:.4f}")


# 99. リークの検証
import pandas as pd
import numpy as np

# 1) 埋め込み済みデータの読み込み
#    （元の埋め込み or Fine‑tuned埋め込み、どちらでも可）
mapping_df = pd.read_pickle('/kaggle/working/miscon_mapping_with_emb.pkl')
valid_df   = pd.read_pickle('/kaggle/working/valid_with_combo_emb.pkl')

# 2) 埋め込み行列の準備＆正規化
M = np.vstack(mapping_df['embedding'].values)
V = np.vstack(valid_df['combo_embedding'].values)
M_norm = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
V_norm = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)

# 3) コサイン類似度行列
sim = V_norm.dot(M_norm.T)   # shape: [num_val, num_map]

# 4) 各サンプルで「真の MisconceptionId が類似度何位か」を算出
ranks = []
for i, true_id in enumerate(valid_df['MisconceptionId'].values):
    # マッピング側でのインデックスを取得
    j = mapping_df.index[mapping_df['MisconceptionId']==true_id][0]
    # 真の類似度
    true_sim = sim[i, j]
    # 真の類似度より大きいものがいくつあるか → これがランク
    rank = int((sim[i] > true_sim).sum())  # 0なら1位
    ranks.append(rank)

ranks = np.array(ranks)

# 5) 結果の確認
print("真のラベルが“1位” (rank=0) の割合:", np.mean(ranks==0))
print("ランク分布 (上位3位まで):")
for r in [0,1,2]:
    print(f"  Rank={r+1}: {(ranks==r).mean():.3f}")

# 6) 分布を可視化（任意）
import matplotlib.pyplot as plt
plt.hist(ranks, bins=50)
plt.xlabel("True label rank (0=1位)")
plt.ylabel("Count")
plt.title("Distribution of true-label ranks")
plt.show()


#4.1 ハートネガティブ・マイニングの実装

