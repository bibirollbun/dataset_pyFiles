import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def load_data(base_path):
    rows, labels = [], []
    for label, cls in enumerate(["clean", "malware"]):
        for fpath in tqdm(glob.glob(f"{base_path}/{cls}/*.jsonl")):
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        rqs = obj[2].get('rqs', {})
                        rsp = obj[2].get('rsp', {})
                        text = " ".join([
                            " ".join([f"{k}:{v}" for k, v in rqs.items()]),
                            " ".join([f"{k}:{v}" for k, v in rsp.items()])
                        ])
                        rows.append(text)
                        labels.append(label)
                    except Exception:
                        continue
    return pd.DataFrame({"text": rows, "label": labels})


def predict_file(file_path, vectorizer, clf):
    preds = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            rqs = obj[2].get('rqs', {})
            rsp = obj[2].get('rsp', {})
            text = " ".join([
                " ".join([f"{k}:{v}" for k, v in rqs.items()]),
                " ".join([f"{k}:{v}" for k, v in rsp.items()])
            ])
            X = vectorizer.transform([text])
            preds.append(clf.predict(X)[0])
    return int(any(pred == 1 for pred in preds))


df = load_data("/kaggle/input/http-malware-detection/train")


train_texts, val_texts, y_train, y_val = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)


vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 5))
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)


clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_val)


print("Validation F1:", f1_score(y_val, y_pred))


test = glob.glob("/kaggle/input/http-malware-detection/test/*.jsonl")


result = []

for path in test:
    predict = predict_file(path, vectorizer, clf)
    result.append({'id': Path(path).stem, 'target': predict})


result_df = pd.DataFrame(result)


result_df.to_csv('/kaggle/working/submission.csv', index=False)

