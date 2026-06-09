import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/huggingfacedebertav3variants/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/huggingfacedebertav3variants/deberta-v3-base")


ds = load_dataset('csv', data_files='/kaggle/input/jigsaw-agile-community-rules/train.csv')
print(ds)


def tokenizer_function(data):
   return tokenizer(data["body"], padding="max_length")

def add_token(text, category):
    text = f'[CATEGORY: {category}] {text}'
    return text

def preprocessing(data):
    category = {"No legal advice: Do not offer or request legal advice.": "legal_advice",
                "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.": "advertising"}
    data['category'] = category.get(data["rule"])
    return data

ds = ds.map(preprocessing)
ds = ds.map(lambda x: {"text": add_token(x["body"], x['rule'])})
ds = ds.select_columns(["body", "rule_violation"])
ds = ds.map(tokenizer_function, batched=True)
ds = ds.rename_column("rule_violation", "label")

split = ds["train"].train_test_split(
    test_size=0.2,
    seed=42
)

train_ds = split["train"]
test_ds = split["test"]


training_args = TrainingArguments(
   "model",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=5,
   weight_decay=0.01,
   save_strategy="epoch",
    report_to='none'
)


trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=train_ds,
   eval_dataset=test_ds,
   tokenizer=tokenizer
)


trainer.train()


device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

category = {"No legal advice: Do not offer or request legal advice.": "legal_advice",
            "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.": "advertising"}

df["category"] = df["rule"].map(category).fillna(-1)
df["body"] = [add_token(text, token) for text, token in zip(df["body"], df["category"])]
data = df['body'].tolist()

batch_pred = 64
predict_result=[]

model.eval()
for i in range(0, len(data), batch_pred):
    with torch.no_grad():
        batch_data=data[i:i+batch_pred]
        enc = tokenizer(batch_data,padding=True,truncation=True,return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        output_val = model(**enc).logits
        res = torch.softmax(output_val, dim=-1).cpu().numpy()
        res = res[:, 1]
        predict_result.extend(res)


test_pd = pd.DataFrame({'row_id':df['row_id'],'rule_violation':predict_result})
test_pd.to_csv("submission.csv",index=False)

