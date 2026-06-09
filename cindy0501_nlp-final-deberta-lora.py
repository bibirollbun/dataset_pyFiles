# ----------------------------------------------------
#  é›¢ç·šå¥—ä»¶å®‰è£�
# ----------------------------------------------------

# æ›¿æ�›æˆ�æ‚¨å¯¦éš›çš„æ•¸æ“šé›†è·¯å¾‘
DEPS_PATH = "/kaggle/input/dependencies-offline/offline_packages" 

# ä½¿ç”¨ --no-index å’Œ --find-links å¼·åˆ¶å¾�æœ¬åœ°æ–‡ä»¶å®‰è£�
!pip install \
    --no-index \
    --find-links {DEPS_PATH} \
    transformers \
    peft \
    datasets \
    evaluate \
    accelerate \
    scipy \
    'protobuf==4.21.0'


## Cell 1: å®‰è£�èˆ‡å°�å…¥ PyTorch ç›¸é—œåº«

#!pip install torch transformers accelerate peft datasets scikit-learn

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from sklearn.model_selection import train_test_split

OFFLINE_MODEL_PATH = "/kaggle/input/deberta-v3-base-offline/deberta-v3-base-offline"




## Configuration
class CFG:
    seed = 42
    # å°‡é �è¨­æ¨¡å�‹ä¿®æ”¹ç‚º Hugging Face çš„æ¨™æº–å‘½å��
    # preset = "microsoft/deberta-v3-base" 
    # preset = "google/electra-base-discriminator"
    sequence_length = 512
    epochs = 3
    batch_size = 8  # 16
    scheduler = 'cosine'
    label2name = {0: 'winner_model_a', 1: 'winner_model_b', 2: 'winner_tie'}
    name2label = {v:k for k, v in label2name.items()}
    class_labels = list(label2name.keys())
    class_names = list(label2name.values())

# è¨­ç½®éš¨æ©Ÿç¨®å­� (PyTorch å°ˆæ¡ˆçš„æ¨™æº–å�šæ³•)
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(CFG.seed)

# PyTorch/Hugging Face è¨“ç·´æ™‚ï¼Œæ··å�ˆç²¾åº¦ (fp16) æ˜¯åœ¨ TrainingArguments ä¸­è¨­ç½®çš„ï¼Œ
# ä½†æˆ‘å€‘åœ¨é€™è£¡ç¢ºä¿� CUDA å�¯ç”¨ã€‚
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"è¨“ç·´è¨­å‚™: {device}")


## Cell 2: è¨“ç·´æ•¸æ“šè¼‰å…¥èˆ‡æ¨™ç±¤è½‰æ�›

# å�‡è¨­æ•¸æ“šè·¯å¾‘å·²è¨­å®š
BASE_PATH = '/kaggle/input/lmsys-chatbot-arena'
df = pd.read_csv(f'{BASE_PATH}/train.csv')

# --- æ•¸æ“šè§£æ��èˆ‡æ¸…æ´— (èˆ‡ Keras é‚�è¼¯ä¿�æŒ�ä¸€è‡´) ---
# æ��å�–ç¬¬ä¸€çµ„ prompt å’Œ response (è™•ç�†æ½›åœ¨çš„ JSON/List æ ¼å¼�)
df["prompt"] = df.prompt.map(lambda x: eval(x)[0])
df["response_a"] = df.response_a.map(lambda x: eval(x.replace("null","''"))[0])
df["response_b"] = df.response_b.map(lambda x: eval(x.replace("null", "''"))[0])

# æ¨™ç±¤è½‰æ�› (å¾� winner_model_a/b/tie æ¬„ä½�è½‰æ�›ç‚ºå–®ä¸€æ•¸å­—æ¨™ç±¤ 0, 1, 2)
# idxmax(axis=1) æ‰¾å‡ºå€¼ç‚º 1 çš„é‚£ä¸€åˆ—çš„æ¬„ä½�å��ç¨±
df["class_name"] = df[["winner_model_a", "winner_model_b" , "winner_tie"]].idxmax(axis=1)
df["class_label"] = df.class_name.map(CFG.name2label)

# æº–å‚™ PyTorch / Hugging Face çš„å–®ä¸€æ–‡æœ¬è¼¸å…¥æ ¼å¼�
df["text"] = df["prompt"] + " " + df["response_a"] + " " + df["response_b"]

print("è¨“ç·´æ•¸æ“šè¼‰å…¥å®Œæˆ�ï¼Œå·²æº–å‚™ 'text' å’Œ 'class_label' æ¬„ä½�ã€‚")
df.head(2)

# ===== Swap Augmentation =====

def swap_augmentation(df):
    df_swap = df.copy()

    # äº¤æ�› response_a èˆ‡ response_b
    df_swap["response_a"], df_swap["response_b"] = (
        df["response_b"].values,
        df["response_a"].values
    )

    # å°�æ‡‰èª¿æ•´ label
    df_swap["class_label"] = df_swap["class_label"].map({
        0: 1,  # A win â†’ B win
        1: 0,  # B win â†’ A win
        2: 2   # tie ä¸�è®Š
    })

    return df_swap


# å�Ÿå§‹è³‡æ–™
df_original = df.copy()

# ç”¢ç”Ÿ swap è³‡æ–™
df_swapped = swap_augmentation(df_original)

# å�ˆä½µå�Ÿå§‹ + swap
df = pd.concat([df_original, df_swapped], ignore_index=True)

print("After swap augmentation:", df.shape)



# ===== Relation Feature: response length difference =====
df["len_a"] = df["response_a"].map(len)
df["len_b"] = df["response_b"].map(len)

df["rel_feature"] = (
    df["len_a"] - df["len_b"]
) / (df["len_a"] + df["len_b"] + 1e-6)





## Cell 3: æ¸¬è©¦æ•¸æ“šè¼‰å…¥

test_df = pd.read_csv(f'{BASE_PATH}/test.csv')

# --- æ•¸æ“šè§£æ��èˆ‡æ¸…æ´— ---
test_df["prompt"] = test_df.prompt.map(lambda x: eval(x)[0])
test_df["response_a"] = test_df.response_a.map(lambda x: eval(x.replace("null","''"))[0])
test_df["response_b"] = test_df.response_b.map(lambda x: eval(x.replace("null", "''"))[0])

# æº–å‚™ PyTorch / Hugging Face çš„å–®ä¸€æ–‡æœ¬è¼¸å…¥æ ¼å¼�
test_df["text"] = test_df["prompt"] + " " + test_df["response_a"] + " " + test_df["response_b"]

print("æ¸¬è©¦æ•¸æ“šè¼‰å…¥å®Œæˆ�ï¼Œå·²æº–å‚™ 'text' æ¬„ä½�ã€‚")
test_df.head(2)

test_df["len_a"] = test_df["response_a"].map(len)
test_df["len_b"] = test_df["response_b"].map(len)

test_df["rel_feature"] = (test_df["len_a"] - test_df["len_b"]) / (test_df["len_a"] + test_df["len_b"] + 1e-6)



## Cell 4: å‰µå»ºé�¸é …å°�èˆ‡æ–‡æœ¬æ¸…ç�† (èˆ‡ Keras é‚�è¼¯ç›¸å�Œ)

def make_pairs(row):
    row["encode_fail"] = False
    
    # ä¿®æ­£ï¼šåœ¨ encode() ä¸­åŠ å…¥ errors='ignore'
    # é€™æœƒè·³é��æ‰€æœ‰é€ æˆ�éŒ¯èª¤çš„ç„¡æ•ˆä»£ç�†å­—ç¬¦ã€‚

    try:
        # å°� prompt é€²è¡Œæ¸…ç�†
        prompt = row.prompt.encode("utf-8", errors='ignore').decode("utf-8")
    except Exception:
        prompt = ""
        row["encode_fail"] = True
    
    try:
        # å°� response_a é€²è¡Œæ¸…ç�†
        response_a = row.response_a.encode("utf-8", errors='ignore').decode("utf-8")
    except Exception:
        response_a = ""
        row["encode_fail"] = True

    try:
        # å°� response_b é€²è¡Œæ¸…ç�†
        response_b = row.response_b.encode("utf-8", errors='ignore').decode("utf-8")
    except Exception:
        response_b = ""
        row["encode_fail"] = True
        
    # å‰µå»ºé�¸é …å°� (é€™éƒ¨åˆ†é‚�è¼¯ä¸�è®Š)
    row['options'] = [
        f"Prompt: {prompt}\n\nResponse: {response_a}", # Option 0: P + R_A
        f"Prompt: {prompt}\n\nResponse: {response_b}"  # Option 1: P + R_B
    ]
    return row

# æ‡‰ç”¨å‡½æ•¸
df = df.apply(make_pairs, axis=1)
test_df = test_df.apply(make_pairs, axis=1)

print("è¨“ç·´é›†é�¸é …å°� (Options) å‰µå»ºå®Œæˆ�ã€‚")
display(df[['options', 'class_label']].head(2))


## Cell 5: æ•¸æ“šå“�è³ªæª¢æŸ¥èˆ‡åˆ†å±¤æŠ½æ¨£åˆ†å‰²

from sklearn.model_selection import train_test_split

# æ•¸æ“šå“�è³ªæª¢æŸ¥
print("è¨“ç·´æ•¸æ“šç·¨ç¢¼å¤±æ•—çµ±è¨ˆï¼š")
print(df.encode_fail.value_counts(normalize=False))

# åˆ†å±¤æŠ½æ¨£åˆ†å‰² (Stratified Split)
# å°‡ df åˆ†å‰²æˆ� train_data å’Œ valid_data
train_data, valid_data = train_test_split(
    df, 
    test_size=0.2, 
    random_state=CFG.seed, 
    stratify=df["class_label"] # ç¢ºä¿�è¨“ç·´é›†å’Œé©—è­‰é›†çš„é¡�åˆ¥åˆ†ä½ˆä¸€è‡´
)

# ç‚ºäº† Hugging Face Dataset çš„å…¼å®¹æ€§ï¼Œå°‡ Pandas Index é‡�ç½®
train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)

print(f"\nè¨“ç·´é›†å¤§å°�: {len(train_data)}")
print(f"é©—è­‰é›†å¤§å°�: {len(valid_data)}")


## Cell 6: è½‰æ�›ç‚º Hugging Face Dataset (æ‰¿æ�¥ PyTorch æµ�ç¨‹)

from datasets import Dataset

# å�ªä¿�ç•™ options èˆ‡ label
train_dataset = Dataset.from_pandas(
    train_data[['options', 'rel_feature', 'class_label']]
)
valid_dataset = Dataset.from_pandas(
    valid_data[['options', 'rel_feature', 'class_label']]
)
test_dataset = Dataset.from_pandas(
    test_df[['options', 'rel_feature']]
)

train_dataset = train_dataset.rename_column("class_label", "labels")
valid_dataset = valid_dataset.rename_column("class_label", "labels")

print("æ•¸æ“šå·²æº–å‚™å¥½é€²å…¥åˆ†è©�å™¨ (Tokenizer)ã€‚")


## Cell 7: æ¨¡å�‹è¼‰å…¥èˆ‡ LoRA é…�ç½® (PEFT)

import torch
import torch.nn as nn
from transformers import AutoModel

class DebertaWithRelFeature(nn.Module):
    def __init__(self, model_path, num_labels=3):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            model_path,
            local_files_only=True
        )
        hidden = self.base.config.hidden_size

        # +1 æ˜¯ rel_feature
        self.classifier = nn.Linear(hidden + 1, num_labels)

    def forward(self, input_ids, attention_mask, rel_feature, labels=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs.last_hidden_state[:, 0]  # CLS
        rel_feature = rel_feature.unsqueeze(1).to(pooled.device)

        x = torch.cat([pooled, rel_feature], dim=1)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}





# å¥—ç”¨lora
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

model = AutoModelForSequenceClassification.from_pretrained(
    OFFLINE_MODEL_PATH,
    num_labels=3,
    local_files_only=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query_proj", "key_proj", "value_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)




## Cell 8: åŸ·è¡Œåˆ†è©�

tokenizer = AutoTokenizer.from_pretrained(
    OFFLINE_MODEL_PATH,
    local_files_only=True
)

def tokenize_function(examples):
    encoding = tokenizer(
        examples["options"],
        truncation=True,
        padding="max_length",
        max_length=CFG.sequence_length
    )
    encoding["rel_feature"] = examples["rel_feature"]
    return encoding



# æ‡‰ç”¨åˆ†è©�å™¨
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# æ¸…ç�†æ¬„ä½�ï¼Œå�ªç•™ä¸‹æ¨¡å�‹éœ€è¦�çš„
tokenized_train = tokenized_train.remove_columns(['options'])
tokenized_valid = tokenized_valid.remove_columns(['options'])
tokenized_test = tokenized_test.remove_columns(['options'])

# â­� ç¢ºä¿� test set æ²’æœ‰ labels
if "labels" in tokenized_test.column_names:
    tokenized_test = tokenized_test.remove_columns(["labels"])

# è¨­ç½®æ ¼å¼�
tokenized_train.set_format("torch")
tokenized_valid.set_format("torch")
tokenized_test.set_format("torch")

# â­� DeBERTa ä¸�éœ€è¦� token_type_idsï¼Œç§»é™¤é�¿å…� NoneType å•�é¡Œ
for ds in [tokenized_train, tokenized_valid, tokenized_test]:
    if "token_type_ids" in ds.column_names:
        ds = ds.remove_columns(["token_type_ids"])

print("æ•¸æ“šåˆ†è©�å®Œæˆ�ï¼Œå·²æº–å‚™å¥½è¨“ç·´ã€‚")
print(tokenized_test.column_names)



## Cell 9: å®šç¾©è¨“ç·´å�ƒæ•¸ä¸¦é–‹å§‹è¨“ç·´ (ä¿®æ­£ TrainingArguments å�ƒæ•¸å��ç¨±)

from transformers import TrainingArguments, Trainer
import evaluate 

OFFLINE_ACCURACY_PATH = "/kaggle/input/dependencies-offline/offline_packages/accuracy.py"

# 1. å®šç¾©è©•ä¼°å‡½æ•¸ (Compute Metrics) (ç¨‹å¼�ç¢¼ä¸�è®Šï¼Œç‚ºç°¡æ½”çœ�ç•¥)
def compute_metrics(eval_pred):
    # ä½¿ç”¨æœ¬åœ°è·¯å¾‘è¼‰å…¥æŒ‡æ¨™è…³æœ¬
    accuracy_metric = evaluate.load(OFFLINE_ACCURACY_PATH) 
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # æª¢æŸ¥ labels æ˜¯å�¦ç‚º NumPy æ•¸çµ„ï¼Œå¦‚æ�œæ˜¯ PyTorch å¼µé‡�éœ€è¦�å…ˆè½‰æ�›
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
    
    return {"accuracy": accuracy}


# 2. å®šç¾©è¨“ç·´å�ƒæ•¸ (TrainingArguments)
training_args = TrainingArguments(
    output_dir="./lora_results",
    num_train_epochs=CFG.epochs,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size,
    warmup_ratio=0.05, 
    weight_decay=0.01,
    learning_rate=3e-5,
    logging_steps=50,
    
    # å°‡ evaluation_strategy æ›¿æ�›ç‚º eval_strategy ***
    eval_strategy="epoch",       
    save_strategy="epoch",       
    
    load_best_model_at_end=True, 
    fp16=True,                   
    report_to="none",            
)

# 3. å®šç¾© Trainer (ç¨‹å¼�ç¢¼ä¸�è®Š)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid, 
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 4. é–‹å§‹è¨“ç·´
print("é–‹å§‹è¨“ç·´ PyTorch / LoRA æ¨¡å�‹...")
trainer.train()


## Cell 10: é �æ¸¬ (Prediction)

# ä½¿ç”¨è¨“ç·´å¥½çš„ Trainer å°�æ¸¬è©¦é›†é€²è¡Œé �æ¸¬
# Trainer æœƒè‡ªå‹•è™•ç�†æ‰¹æ¬¡è™•ç�†å’Œ GPU è¨˜æ†¶é«”
predictions_output = trainer.predict(tokenized_test)

# predictions_output æ˜¯ä¸€å€‹ PredictionOutput å°�è±¡
logits = predictions_output.predictions

# 1. å°‡ logits è½‰æ�›ç‚ºæ¦‚ç�‡ (Softmax)
probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()

# 2. å°‡æ¦‚ç�‡è½‰æ�›ç‚ºä¸‰å€‹ç›®æ¨™é¡�åˆ¥çš„ DataFrame
pred_df = pd.DataFrame(probabilities, columns=CFG.class_names)
print("é �æ¸¬æ¦‚ç�‡ç¯„ä¾‹ï¼š")
display(pred_df.head())


## Cell 11: å‰µå»ºæ��äº¤æ–‡ä»¶ (Submission)

# è¼‰å…¥å�Ÿå§‹æ¸¬è©¦é›† ID
test_df_original = pd.read_csv(f'{BASE_PATH}/test.csv')

# ç¢ºä¿�é �æ¸¬çµ�æ�œçš„è¡Œæ•¸èˆ‡æ¸¬è©¦é›†ä¸€è‡´
if len(pred_df) != len(test_df_original):
    print("ğŸš¨ è­¦å‘Šï¼šé �æ¸¬çµ�æ�œè¡Œæ•¸èˆ‡æ¸¬è©¦é›†ä¸�åŒ¹é…�ï¼�")

# 3. æº–å‚™æ��äº¤æ–‡ä»¶
submission_df = pd.DataFrame({
    'id': test_df_original['id'],
    'winner_model_a': pred_df['winner_model_a'],
    'winner_model_b': pred_df['winner_model_b'],
    'winner_tie': pred_df['winner_tie'],
})

# 4. ä¿�å­˜ç‚º CSV æ–‡ä»¶
submission_df.to_csv("submission.csv", index=False)

print("æ��äº¤æ–‡ä»¶ submission.csv å·²ç”Ÿæˆ�ï¼�")

