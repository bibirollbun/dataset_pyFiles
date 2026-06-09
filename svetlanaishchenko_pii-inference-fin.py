import pandas as pd 
import json
from pathlib import Path
from spacy.lang.en import English
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from scipy.special import softmax


test_path = "/kaggle/input/pii-detection-removal-from-educational-data/test.json"
df = pd.read_json(test_path)


MAX_LENGTH = 0
for x in df['tokens']:
    if len(x) > MAX_LENGTH:
        MAX_LENGTH = len(x)
MAX_LENGTH = int(MAX_LENGTH * 1.2)
print(MAX_LENGTH )


MODEL_PATH = '/kaggle/input/deberta-orig-data'


def tokenize(example, tokenizer, max_length):
    text = []
    token_map = []
    idx = 0
    for t, ws in zip(example["tokens"], example["trailing_whitespace"]):
        text.append(t)
        token_map.extend([idx] * len(t))
        if ws:
            text.append(" ")
            token_map.append(-1)
        idx += 1
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)
    return {**tokenized, "token_map": token_map}


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
ds = Dataset.from_dict({
    "full_text": [x for x in df["full_text"]],
    "document": [x for x in df["document"]],
    "tokens": [x for x in df["tokens"]],
    "trailing_whitespace": [x for x in df["trailing_whitespace"]],
})
ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH}, num_proc=4)


model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
args = TrainingArguments(
       ".",
       per_device_eval_batch_size=1,
       report_to="none") 
trainer = Trainer(
          model=model,
          args=args,
          data_collator=collator,
          tokenizer=tokenizer)


predictions = softmax(trainer.predict(ds).predictions, axis=-1).argmax(-1)


config = json.load(open('/kaggle/input/deberta-orig-data/config.json')) #json.load(open(Path(MODEL_PATH) / "config.json"))
id2label = config["id2label"]


pairs = set() 
processed = []
for p, token_map, offsets, tokens, doc in zip(predictions, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]):
    for token_pred, (start_idx, end_idx) in zip(p, offsets): 
        label_pred = id2label[str(token_pred)]  # Predicted label from token
        if start_idx + end_idx == 0:
            continue
        if token_map[start_idx] == -1:
            start_idx += 1
        while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace(): # Ignore leading whitespace tokens ("\n\n")
            start_idx += 1
        if start_idx >= len(token_map): # If start index exceeds the length of token mapping, break the loop
            break
        token_id = token_map[start_idx]  # Token ID at start index
 
        if label_pred in ("O", "B-EMAIL", "B-PHONE_NUM", "I-PHONE_NUM", "B-URL_PERSONAL", "I-URL_PERSONAL") or token_id == -1: # Ignore "O" predictions and whitespace tokens
            continue
        pair = (doc, token_id)

        if pair not in pairs:
            processed.append({"document": doc, "token": token_id, "label": label_pred, "token_str": tokens[token_id]})
            pairs.add(pair)


nlp = English()

def find_span(target: list[str], document: list[str]) -> list[list[int]]:
    idx = 0
    spans = []
    span = []
    for i, token in enumerate(document):
        if token != target[idx]:
            idx = 0
            span = []
            continue
        span.append(i)
        idx += 1
        if idx == len(target):
            spans.append(span)
            span = []
            idx = 0
            continue  
    return spans

data = json.load(open("/kaggle/input/pii-detection-removal-from-educational-data/test.json"))

email_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
phone_num_regex = re.compile(r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")

url_regex = re.compile(
    r'http[s]?://'  # http or https
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)', re.IGNORECASE)

emails = []
phone_nums = []
urls = []
for _data in data:
    # email
    for token_idx, token in enumerate(_data["tokens"]):
        if re.fullmatch(email_regex, token) is not None:
            emails.append({"document": _data["document"], "token": token_idx, "label": "B-EMAIL", "token_str": token})
      
    # url
    matches = url_regex.findall(_data["full_text"])
    if not matches:
        continue
    for match in matches:
        target = [t.text for t in nlp.tokenizer(match)]
        matched_spans = find_span(target, _data["tokens"])    
    for matched_span in matched_spans:
        for intermediate, token_idx in enumerate(matched_span):
            prefix = "I" if intermediate else "B"
            phone_nums.append({"document": _data["document"], "token": token_idx, "label": f"{prefix}-URL_PERSONAL", 
                               "token_str": _data["tokens"][token_idx]})
    # phone number
    matches = phone_num_regex.findall(_data["full_text"])
    if not matches:
        continue
    for match in matches:
        target = [t.text for t in nlp.tokenizer(match)]
        matched_spans = find_span(target, _data["tokens"])    
    for matched_span in matched_spans:
        for intermediate, token_idx in enumerate(matched_span):
            prefix = "I" if intermediate else "B"
            phone_nums.append({"document": _data["document"], "token": token_idx, "label": f"{prefix}-PHONE_NUM", 
                               "token_str": _data["tokens"][token_idx]})


df = pd.DataFrame(processed + emails + phone_nums + urls)
df["row_id"] = list(range(len(df))) # Assign each row a unique 'row_id'
# display(df.head(100))
df[["row_id", "document", "token", "label"]].to_csv("submission.csv", index=False)
df['label'].unique()


