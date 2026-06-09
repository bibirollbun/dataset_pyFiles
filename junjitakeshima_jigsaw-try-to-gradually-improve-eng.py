import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
df_test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
df


df_train = pd.DataFrame(df[['body', 'rule', 'subreddit', 'rule_violation']].copy())

#X = df_train['body'].tolist()


#df_train['positive'] = df_train['positive_example_1'] + df_train['positive_example_2']
#df_train['negative'] = df_train['negative_example_1'] + df_train['negative_example_2']

#df_train['val_pos_ex'] = 1
#df_train['val_neg_ex'] = 0


df_test['positive'] = df_test['positive_example_1'] + df_test['positive_example_2']
df_test['negative'] = df_test['negative_example_1'] + df_test['negative_example_2']

df_test['val_pos_ex'] = 1
df_test['val_neg_ex'] = 0
df_test.head()


df_add_pos = pd.DataFrame(df_test[['positive', 'rule', 'subreddit', 'val_pos_ex']])
df_add_pos.columns = ['body', 'rule', 'subreddit', 'rule_violation']
df_add_neg = pd.DataFrame(df_test[['negative', 'rule', 'subreddit', 'val_neg_ex']])
df_add_neg.columns = ['body', 'rule', 'subreddit', 'rule_violation']

df_add = pd.concat([df_add_pos, df_add_neg], axis=0)

df_add = df_add.sample(frac=0.1, random_state=2) #<----- use 10% of test data records

df_train = pd.concat([df_train, df_add], axis=0)


#df_add['rule_descr'] = df_add['rule'].str[:12]
#df_add = df_add[df_add['rule_descr'] != "No Advertisi"]
#df_add = df_add[df_add['rule_descr'] != "No legal adv"]


df_train['input'] =  "rule:" + df_train['rule'] \
                     +"subreddit:" + df_train['subreddit'] \
                     +"body:"      + df_train['body']


X = df_train['input'].tolist()
y = df_train['rule_violation'].tolist()


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


tokenizer = RobertaTokenizer.from_pretrained("/kaggle/input/roberta-base/transformers/default/1")
model = RobertaForSequenceClassification.from_pretrained("/kaggle/input/roberta-base/transformers/default/1")

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=512)


class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = RedditDataset(train_encodings, y_train)
val_dataset = RedditDataset(val_encodings, y_val)


os.environ["WANDB_DISABLED"] = "true"

model = RobertaForSequenceClassification.from_pretrained("/kaggle/input/roberta-base/transformers/default/1").to("cuda")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    logging_dir='./logs',
    report_to=[],
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)


trainer.train()


preds = trainer.predict(val_dataset)
probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()

# Evaluate
auc = roc_auc_score(y_val, probs)
print(f"Validation AUC: {auc:.4f}")


df_test['input'] =  "rule:" + df_test['rule'] \
                     +"subreddit:" + df_test['subreddit'] \
                     +"body:"      + df_test['body']

test_encodings = tokenizer(df_test['input'].tolist(), truncation=True, padding=True, max_length=512)

dummy_labels = [0] * len(df_test)
test_dataset = RedditDataset(test_encodings, dummy_labels)

test_outputs = trainer.predict(test_dataset)
probs = torch.nn.functional.softmax(torch.tensor(test_outputs.predictions), dim=1)[:, 1].numpy()


submission_df = pd.DataFrame({
    "row_id": df_test["row_id"],
    "rule_violation": probs
})
submission_df.to_csv("submission.csv", index=False)
submission_df

