!pip install unidecode --quiet


import pandas as pd
import numpy as np
import tiktoken
# import re
# import unicodedata
# import unidecode

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset # PyTorch에서 Dataset을 정의할 때 쓰는 기본 클래스
# Deep Learning에서는 Data를 효율적? 으로 불러오고 가공해야 한다네.
# 필수 method: __len__(self), Dataset의 전체크기를 반환
# 필수 method: __getitem__(self, idx), 인덱스 idx에 해당하는 데이터를 반환
from torch.utils.data import DataLoader
# 데이터의 가공과 어떤 전처리를 할지가 DataSet의 역할이라면
# 단일 행인 샘플을 반환하는 DataSet을 batch 즉 여러 묶음으로 반환시키기 위한게 Loader
# 또 매 epoch마다 순서를 섞어서 모델이 학습시 순서와 상관없도록 함
# 게다가 iterator 객체라서 반복작업(값을 한번에 하나씩 꺼내주는) 가능
    # iterator(반복자): 그냥 list라고 해서 값을 차례대로 꺼낼 수 있는게 아니라 순회하면서 값을 꺼내주는 역할을 하는 녀석이 있음
    # 그게 이 iterator. DataLoader에도 존재.
# DataSet이 책 가공 후 책장에 놓는 역할이라면 DataLoader는 사서

from torchmetrics.classification import BinaryAUROC
from tqdm.auto import tqdm


DATA_PATH1 = '/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/'
DATA_PATH2 = '/kaggle/input/jigsaw-preprocessed-data/'

# df_train = pd.read_csv(DATA_PATH1+"train.csv")
# df_test = pd.read_csv(DATA_PATH1+"test.csv")
df_train_prep_dl = pd.read_csv(DATA_PATH2+"train_prep/train_prep.csv", index_col=0)
test_df = pd.read_csv(DATA_PATH2+"test_prep/test_prep.csv", index_col=0)
df_sample = pd.read_csv(DATA_PATH1+"sample_submission.csv")


# def missing_count(df: pd.DataFrame) -> pd.Series:
#     """
#     각 컬럼별 결측치 개수를 반환하는 함수
#     df.count() 기반으로 계산
#     """
#     n_rows = len(df)
#     return n_rows - df.count()
#     # df.count()는 non-null만 세기에 isnull.sum()보다 빠름

# missing = missing_count(df_train)
# print(missing)



# ASCII 변환
def ascii_text(text):    
    # 보이지 않는(invisible) 유니코드 문자 탐지 및 제거
    INVISIBLE = (
        r"[\u00AD\u200B\u200C\u200D\u200E\u200F"
        r"\u202C\u202D\u202E\u2060\u2066\u2067\u2068\u2069"
        r"\uFEFF]"
    )
    text = re.sub(INVISIBLE, "", text)
    # re.sub(pattern, replacement, text): text에서 pattern(정규식)을 찾아 대체
    
    # 결합문자 분해
    text = unicodedata.normalize("NFKD", text)
    # ASCII로 변환
    text = unidecode.unidecode(text)
    return text

# 소문자화 + 업체
def lower_clean_text(text):
    # text = str(text)
    text = text.lower() # 같은 단어의 대,소문자 버전을 동일하게 인식하기 위해
    text = " ".join(text.split()) # 공백을 한칸으로만 깔끔하게 정리
    # TF-IDF나 Word2Vec 같은 임베딩은 공백(" ")을 기준으로 단어를 쪼갬(tokenize)
    # 공백이 2칸 이상이면 빈 문자열 생김
    return text

# 전처리
def preprocess(df):
    # 결측치 제거
    # df['comment_text'] = df['comment_text'].fillna("")로 채워도 의미가 없음
    df = df.dropna(subset=["comment_text"]).copy()
    # 문자열 변환
    df['comment_text'] = df['comment_text'].astype(str)
    # ASCII 변환
    df['comment_text'] = df['comment_text'].apply(ascii_text)
    # url 제거
    url_pattern = r"https?://\S+|www\.\S+"
    df['comment_text'] = df['comment_text'].str.replace(url_pattern, " ", regex=True)
    # lower case & cleaning
    df['comment_text'] = df['comment_text'].apply(lower_clean_text)

    return df

# 예시
# df_train = preprocess(df_train)


# df_train_prep = preprocess(df_train)
# df_test_prep = preprocess(df_test)


# df_train_prep.to_csv('df_train_prep.csv')
# df_test_prep.to_csv('df_test_prep.csv')


# print(df_train_prep["comment_text"].apply(type).value_counts())


# print(df_test_prep["comment_text"].apply(type).value_counts())


# df_train_prep_dl.sample(3).T


df_sample = df_train_prep_dl[['target', 'comment_text']].sample(n=50000, random_state=42)
# df_sample = df_train_prep_dl[['target', 'comment_text']].sample(frac=0.8, random_state=42)

train_df, valid_df = train_test_split(
    df_sample,
    test_size=0.2, # 나머지 20%를 valid로
    # stratify=df_sample["target"],  # 비율 유지
    random_state=42
)

# valid_df, test_df = train_test_split(
#     temp_df,
#     test_size=0.5,          # 40% 중 절반은 test, 절반은 valid
#     # stratify=temp_df["target"],
#     random_state=42
# )


train_df.shape[0], valid_df.shape[0]


pos_ratio = (train_df["target"] >= 0.5).mean()
print("target>=0.5 비율:", pos_ratio)


# df_test_prep_dl["comment_text"] = df_test_prep_dl["comment_text"].fillna("")

# def missing_count(df: pd.DataFrame) -> pd.Series:
#     """
#     각 컬럼별 결측치 개수를 반환하는 함수
#     df.count() 기반으로 계산
#     """
#     n_rows = len(df)
#     return n_rows - df.count()
#     # df.count()는 non-null만 세기에 isnull.sum()보다 빠름

# missing = missing_count(df_test_prep_dl)
# print(missing)



tokenizer = tiktoken.get_encoding("gpt2") # gpt2의 vocabulary
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=None, pad_token_id=50256):
        self.data = df
        # 토큰 ID로 변환
        self.encoded_texts = [tokenizer.encode(comment) for comment in self.data["comment_text"]]

        # if max_length is None: # max_length 설정
        #     self.max_length = self._longest_encoded_length()
        # else:                  # 이후 >=max_length 이면 자름 
        #     self.max_length = max_length
        #     self.encoded_texts = [ids[: self.max_length] for ids in self.encoded_texts]
        #     # self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        self.max_length = max_length

        self.input_ids = []
        self.attention_masks = []
        for ids in self.encoded_texts:
            # 길이가 max_length보다 길면 자르기
            if len(ids) > self.max_length:
                ids = ids[: self.max_length]

            pad_len = self.max_length - len(ids)
            padded = ids + [pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len

            self.input_ids.append(padded)
            self.attention_masks.append(mask)
        # self.encoded_texts = [
        #     encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
        #     for encoded_text in self.encoded_texts
        # ]
    
    def _longest_encoded_length(self):
        # max_length = 0
        # for encoded_text in self.encoded_texts:
        #     encoded_length = len(encoded_text)
        #     if encoded_length > max_length:
        #         max_length = encoded_length
        # return max_length
        
        # 위 코드를 한 줄로 표현하면,
        return max(len(ids) for ids in self.encoded_texts)

    
    def __getitem__(self, index):
        # encoded = self.encoded_texts[index]
        # label = self.data.iloc[index]['target']
        # return (
        #     torch.tensor(encoded, dtype=torch.long),
        #     torch.tensor(label, dtype=torch.long)
        # )
        input_ids = torch.tensor(self.input_ids[index], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_masks[index], dtype=torch.long)
        label = torch.tensor(self.data.iloc[index]["target"], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }
        # 신기하게 index에 해당하는 값 반환시 comment 그대로 반환하는게 아니라 target값이랑 함께 tensor변환 상태로 반환하네

    def __len__(self):
        return len(self.data)


train_dataset = JigsawDataset(
    df = train_df,
    tokenizer = tokenizer,
    max_length = 128,
)


print(train_dataset.max_length)
# 참고로 GPT2는 문맥 길이 한도에 해당하는 1024개 토큰까지 처리할 수 있음
# 더 긴 문장이 존재하는 경우 max_length=1024로 설정


# sequence 길이에 맞춰 valid, test에도 패딩을 추가
val_dataset = JigsawDataset(
    df = valid_df,
    tokenizer = tokenizer,
    max_length = 128,
    # max_length = train_dataset.max_length,
    # max_length = 453
    
)

# test_dataset = JigsawDataset(
#     df = test_df,
#     tokenizer = tokenizer,
#     # max_length = train_dataset.max_length,
#     max_length = 453,
# )


num_workers = 0
# batch_size = 8
batch_size = 32
torch.manual_seed(123)

# Iterator가 동작할 때마다 (for문같은거 순회시)
# Dataset.__getitem__반복해서 batch 단위로 꺼냄

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    drop_last = True,
    # 데이터 개수가 딱 batch의 배수가 아니면 마지막 batch는 작은 배치가 됨
    # 작은 배치는 BatchNorm같은 레이어에서 통계가 불안정해질 수 있음
    # 그래서 훈련에서는 보통 버림
)

valid_loader = DataLoader(
    dataset = val_dataset,
    batch_size = batch_size,
    num_workers = num_workers,
    drop_last = False,
)

# test_loader = DataLoader(
#     dataset = test_dataset,
#     batch_size = batch_size,
#     num_workers = num_workers,
#     drop_last = False,
# )


for batch in train_loader:
    # input_ids = batch["input_ids"]
    # attention_mask = batch["attention_mask"]
    # labels = batch["labels"]
    pass

print("Input batch dimension:", batch["input_ids"].shape)
print("Attention batch dimension:", batch["attention_mask"].shape)
print("Label batch dimension:", batch["labels"].shape)


# 마지막 배치는 배치 size 8로 동일. 토큰은 max_length 453개로 잘 맞춰져있음


from transformers import GPT2Model

gpt2 = GPT2Model.from_pretrained("gpt2")

class GPT2TocixClassifier(torch.nn.Module):
    def __init__(self, model, emb_dim=None, num_classes=1):
        super().__init__()
        self.model = model
        emb_dim = model.config.hidden_size
        self.final_norm = torch.nn.LayerNorm(emb_dim)
        self.out_head = torch.nn.Linear(emb_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, L, H) ← 공통으로 먼저 정의
        # h_last = out.last_hidden_state[:, -1, :]
        if attention_mask is not None:
            # mask 기반 평균 pooling
            mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
            hidden = hidden * mask  # pad 위치는 0
            lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
            h = hidden.sum(dim=1) / lengths  # (B, H)
        else:
            # 그냥 전체 평균 (mean pooling)
            h = hidden.mean(dim=1)
        
        h = self.final_norm(h)
        logits = self.out_head(h).squeeze(-1) 
        return logits

model = GPT2TocixClassifier(model=gpt2)


# 모든 층을 훈련되지 않도록 모델동결
for param in model.parameters():
    param.requires_grad = False

# GPT2 출력층과 마지막 transformer block, LayerNorm을 훈련 가능하도록 설정
    # HuggingFace GPT2에서는 block 리스트가 model.model.h 에 있음
for param in model.model.h[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
for param in model.out_head.parameters():
    param.requires_grad = True


inputs = tokenizer.encode("what a moron. this imposter must go!")
inputs = torch.tensor(inputs).unsqueeze(0)
print("입력:", inputs)
print("입력 차원:", inputs.shape)


with torch.no_grad():
    outputs = model(inputs)

print("출력:", outputs)
print("출력 텐서:", outputs.shape)


# 원래는 vocabulary 개수만큼 출력 → softmax로 확률화 → argmax로 가장 높은 확률 반환 = 생성할 다음단어 토큰 ID
# 지금은 toxicity 아님/맞음 2개의 2차원 출력
# 둘 중 높은 쪽으로 선택됨 [0.65, 0.35] → toxicity 아님

# probas = torch.softmax(outputs, dim=-1)
# label = torch.argmax(probas)
# print("클래스 레이블:", label.item())


# 분류 auc를 계산
def calc_auc_loader(data_loader, model, device, num_batches=None):
    model.eval()
    auroc = BinaryAUROC().to(device)

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)

            bin_targets = (labels >= 0.5).long()
            auroc.update(probs, bin_targets)
        else:
            break

    auc = auroc.compute().item()
    auroc.reset() # 다음에 또 사용시 초기화

    return auc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)

train_auc = calc_auc_loader(
    train_loader, model, device, num_batches=100
)
valid_auc = calc_auc_loader(
    valid_loader, model, device, num_batches=100
)
# test_auc = calc_auc_loader(
#     test_loader, model, device, num_batches=100
# )

print(f"훈련 auc: {train_auc*100:.2f}%")
print(f"검증 auc: {valid_auc*100:.2f}%")
# print(f"테스트 auc: {test_auc*100:.2f}%")


# fine-tuning 없이는 랜덤예측인 50%수준으로 나옴


# loss function으로 cross-entropy 사용
def calc_loss_batch(batch, model, device, pos_weight):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    targets = batch['labels'].to(device).float()
    
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = logits.squeeze(-1)

    if pos_weight is not None:
        # pos_weight를 텐서로 변환
        pos_weight_tensor = torch.as_tensor(
            pos_weight, dtype=torch.float32, device=device
        )

        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight_tensor,
        )
    else:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    return loss

# 분류손실 계산함수
def calc_loss_loader(data_loader, model, device, pos_weight, num_batches=None):
    total_loss = 0
    
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                batch, model, device, pos_weight
            )
            total_loss += loss.item()

        else:
            break

    return total_loss / num_batches


pos_ratio = (train_df["target"] >= 0.5).mean()
neg_ratio = 1 - pos_ratio
pos_weight_value = neg_ratio / pos_ratio

# 초기 손실 계산
with torch.no_grad(): # no_grad가 그레디언트 추적을 비활성화한다는 건데 뭐 효율성 증대랑 뭔 관련이 있는건지는 모르겠네
    train_loss = calc_loss_loader(
        train_loader, model, device, pos_weight_value, num_batches=5
    )
    valid_loss = calc_loss_loader(valid_loader, model, device, pos_weight_value, num_batches=5)
    # test_loss = calc_loss_loader(test_loader, model, device, pos_weight_value, num_batches=5)
    

print(f"훈련 손실: {train_loss:.2f}")
print(f"검증 손실: {valid_loss:.2f}")
# print(f"테스트 손실: {test_loss:.2f}")


from datetime import datetime, timedelta, timezone

def train_classifier_simple(
        model, train_loader, valid_loader, optimizer, device, pos_weight, 
        max_steps, eval_freq, eval_iter, 
        start_step=-1, start_epoch=0):
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    # examples_seen, global_step = 0, -1
    examples_seen = 0
    global_step = start_step
    epoch = start_epoch

    # train_loader를 계속 순회하기 위한 iterator
    train_iter = iter(train_loader)
    running_train_losses = []  # eval 사이 구간에서의 train loss들을 임시로 모음
    
    # 가속을 위한 AMP 사용
    use_cuda_amp = (device.type == "cuda")
    if use_cuda_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None
    # scaler = torch.amp.GradScaler('cuda')
    # for epoch in range(num_epochs): # epoch 단위가 아니라 step 단위로 진행

    pbar = tqdm(
        range(start_step, max_steps),
        initial=start_step,
        total=max_steps,
        desc=f"Epoch {epoch+1}",
        dynamic_ncols=True,
    )
    for _ in pbar:
    # for step in range(start_step, max_steps):
        model.train() # 모델을 훈련모드로 설정

        # epoch 점검
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1                   # 에폭 하나 끝남
            train_iter = iter(train_loader)
            batch = next(train_iter)     # 새 에폭의 첫 배치

        optimizer.zero_grad() # 이전 배치반복에서 얻은 손실 그레디언트를 초기화
        if use_cuda_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = calc_loss_batch(batch, model, device, pos_weight)
        else:
            loss = calc_loss_batch(batch, model, device, pos_weight)
        # loss.backward() # 손실 그레디언트를 계산
        # optimizer.step() # 손실 그레디언트를 사용해 가중치 업데이트
        if use_cuda_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        examples_seen += batch["input_ids"].shape[0] # sample개수 추적
        global_step += 1

        # 훈련과정 추적 & 모니터링?
        # if global_step % eval_freq == 0: # 일정주기 eval_freq마다 확인
        #     # train, valid loss 계산
        #     train_loss, val_loss = evaluate_model(
        #         model, train_loader, valid_loader, device, pos_weight, eval_iter)
        #     # 
        #     train_losses.append(train_loss)
        #     val_losses.append(val_loss)
            
        #     print(f"에포크 {epoch+1} (Step {global_step:06d}): " 
        #           f"훈련손실 {train_loss:.3f}/검증 손실 {val_loss:.3f}"
        #           f" [KST {datetime.now(timezone(timedelta(hours=9))).strftime('%H:%M')}]"
        #     )

        # 훈련과정 추적 간소화
        running_train_losses.append(loss.item()) # train loss는 매 step에서 나온 loss를 모아두기만 한다
        
        if global_step % eval_freq == 0: # 일정주기 eval_freq마다 확인
            # train loss는 최근 eval_freq 구간의 평균으로 대표값을 만든다
            avg_train_loss = sum(running_train_losses) / len(running_train_losses)
            running_train_losses = []  # 초기화

            # val loss는 valid_loader에서 eval_iter 만큼만 평가
            val_loss = calc_loss_loader(
                valid_loader, model, device, pos_weight, num_batches=eval_iter
            )

            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            print(f"에포크 {epoch+1} (Step {global_step:06d}): " 
                  f"훈련손실 {avg_train_loss:.3f}/검증 손실 {val_loss:.3f}"
                  f" [KST {datetime.now(timezone(timedelta(hours=9))).strftime('%H:%M')}]"
            )

    train_auc = calc_auc_loader(
        train_loader, model, device, num_batches=eval_iter
    )
    val_auc = calc_auc_loader(
        valid_loader, model, device, num_batches=eval_iter
    )

    print(f"훈련 auc: {train_auc*100:.2f}% | ", end="")
    print(f"검증 auc: {val_auc*100:.2f}%")
    train_accs.append(train_auc)
    val_accs.append(val_auc)
        
    return train_losses, val_losses, train_accs, val_accs, examples_seen, global_step, epoch


def evaluate_model(model, train_loader, valid_loader, device, pos_weight, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, pos_weight, num_batches = eval_iter
        )
        val_loss = calc_loss_loader(
            valid_loader, model, device, pos_weight, num_batches = eval_iter
        )

    model.train()
    return train_loss, val_loss


# 이어하기

# CHECKPOINT_PATH = "~/~"
# checkpoint = torch.load(CHECKPOINT_PATH+"sample.pt", map_location=device)

# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# start_step = checkpoint["global_step"]
# start_epoch = checkpoint["epoch"]
# pos_weight_value = checkpoint["pos_weight"]
# train_losses_total = checkpoint.get("train_losses", [])
# val_losses_total = checkpoint.get("val_losses", [])
# train_accs_total = checkpoint.get("train_accs", [])
# val_accs_total = checkpoint.get("val_accs", [])


import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
# num_epochs = 1

pos_ratio = (train_df["target"] >= 0.5).mean()
neg_ratio = 1 - pos_ratio
pos_weight_value = neg_ratio / pos_ratio

'''
학습할 step수 지정
'''
TRIAL_STEPS = 40000

# 학습하면서 나온 여러가지 값들 반환
train_losses, val_losses, train_accs, val_accs, examples_seen, global_step, epoch = \
    train_classifier_simple(
        model, train_loader, valid_loader, optimizer, device, pos_weight_value, 
        max_steps=TRIAL_STEPS, eval_freq=10000, eval_iter=500,
        start_step=0, start_epoch=0,
        # start_step=start_step, start_epoch=start_epoch,
    )

# 소요시간 체크
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"\n훈련 소요시간: {execution_time_minutes:.2f}분")


# 실행결과 중간저장
start_time = time.time()

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "global_step": global_step,
    "epoch": epoch,
    "pos_weight": pos_weight_value,
    # "train_losses":train_losses_total.extend(train_losses),
    # "val_losses":val_losses_total.extend(val_losses),
    # "train_accs":train_accs_total.extend(train_accs),
    # "val_accs":val_accs_total.extend(val_accs),
    "train_losses":train_losses,
    "val_losses":val_losses,
    "train_accs":train_accs,
    "val_accs":val_accs,
}
time = datetime.now(timezone(timedelta(hours=9))).strftime('%H:%M')
torch.save(checkpoint, f"gpt2_weighted_251209_{time}_step{TRIAL_STEPS//1000}k.pt")

# 소요시간 체크
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"걸린시간: {execution_time_minutes:.2f}분")


import matplotlib.pyplot as plt

def plot_values(
    epochs_seen, examples_seen, train_values, val_values,
    label="loss"):
    fig, ax1 = plt.subplots(figsize=(5,3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-",
            label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.show()

num_epochs = 1
epochs_tensor = torch.linspace(0, num_epoch, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)


epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(
    epochs_tensor, examples_seen_tensor, train_accs, val_accs,
    label='auc'
)


def predict_toxicity(
    texts, model, tokenizer, device, max_length=None,
    pad_token_id=50256, batch_size=32):
    model.eval()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = []
        
        for text in batch_texts:
            input_ids = tokenizer.encode(text)
            input_ids = input_ids[:max_length]
            pad_len = max_length - len(input_ids)
            input_ids += [pad_token_id] * (max_length - len(input_ids))

            batch_ids.append(input_ids)

        input_tensor = torch.tensor(batch_ids, device=device)

        with torch.no_grad():
            logits = model(input_tensor)
        logits = logits.view(-1)

        probs = torch.sigmoid(logits)
        all_probs.append(probs.detach().cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)  # shape [N]
    return all_probs

# 사용예시
# probs = predict_toxicity_batch(
#     df_test_prep['comment_text'], model, tokenizer, device, max_length=train_dataset.max_length
# )


torch.save(model.state_dict(), "gpt2_baseline.pth")


%%time

texts = df_test_prep['comment_text'].tolist()
probs = predict_toxicity(
    texts, model, tokenizer, device, max_length=453, batch_size=32
)


len(texts)


len(probs)


probs.min(), probs.max()


with torch.no_grad():
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device).float()
        logits = model(x).squeeze(-1)
        probs = torch.sigmoid(logits)

        print("targets (first 10):", y[:10].cpu().numpy())
        print("bin_targets:", (y[:10] >= 0.5).long().cpu().numpy())
        print("probs:", probs[:10].cpu().numpy())
        break


submission = pd.DataFrame({
    'id': df_test['id'],
    'prediction': probs
})

submission.to_csv('submission.csv', index=False)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # idx는 (batch, num_tokens)크기의 인덱스
        # batch는 전부 선택 -context_size:-1이니까 마지막 n개만 사용 (-5:-1이면 마지막 5개만 쓰는 상황)
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

text1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

text2 = (
    "Evaluate the toxicity of the following sentence on a scale from 0 (low) to 1 (high):"
    " 'what a moron. this imposter must go!'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text, tokenizer),
    max_new_token=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_dis_to_text(token_ids, tokenizer))


model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=2
)

# 마지막 LayerNorm과 마지막 transformer 블록을 훈련가능하도록 설정
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True




