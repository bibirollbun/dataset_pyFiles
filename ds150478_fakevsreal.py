train = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"

import os
import pandas as pd


data = []
for folder in os.listdir(train):
    folder_path = os.path.join(train, folder)
    idx = int(folder.rsplit("_")[-1])
    with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
          text1 = f1.read().strip()
    with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
          text2 = f2.read().strip()
    data.append([idx, text1, text2])


df = pd.DataFrame(data, columns=['id','Text1', 'Text2'])
df.head()


result_train = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")
result_train.head()


dataframe = pd.merge(df, result_train, on = 'id')
dataframe.head()


#Let's take the first example and analyze it


real = df[df['id']==26]['Text2'].values[0]   #real
fake = df[df['id']==26]['Text1'].values[0]



print(f"REAL text : {real} \n\n\n FAKE text: {fake}")


#Let's take the first example and analyze it


real = df[df['id']==75]['Text1'].values[0]   #real
fake = df[df['id']==75]['Text2'].values[0]
print(f"REAL text : {real} \n\n\n FAKE text: {fake}")


from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

# Load GPT-2 LM (language modeling head is important here!)
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

model.eval()  # set to evaluation mode

def calculate_perplexity(text):
    # Encode text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    
    if input_ids.size(1) == 0:  # No tokens after encoding
        return None

    with torch.no_grad():
        # Important: `labels=input_ids` makes the model compute loss
        outputs = model(input_ids, labels=input_ids)
        loss = outputs["loss"]  # or outputs.loss works too

    perplexity = torch.exp(loss)
    return perplexity.item()



print("Real text perplexity:", calculate_perplexity(real))
print("AI text perplexity:", calculate_perplexity(fake))


dataframe["Text1_PS"] = dataframe["Text1"].apply(calculate_perplexity)
dataframe["Text2_PS"] = dataframe["Text2"].apply(calculate_perplexity)


dataframe.head()


import numpy as np
dataframe["predicted"] = np.where(dataframe["Text1"] < dataframe["Text2"], 2, 1)


dataframe


mismatches = dataframe[dataframe["real_text_id"] != dataframe["predicted"]]
count_mismatches = mismatches.shape[0]

print("Mismatches count:", count_mismatches)



dataframe.shape


!pip install transformers torch scikit-learn



from transformers import AutoTokenizer, AutoModel
import torch

# Load pretrained model & tokenizer
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Set model to evaluation mode
model.eval()


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()



train_df = dataframe[["Text1", "Text2", "real_text_id"]]
train_df.head()


X, y = [], []
for _, row in train_df.iterrows():
    emb1 = get_embedding(row["Text1"])
    emb2 = get_embedding(row["Text2"])
    features = np.hstack([np.abs(emb1 - emb2), emb1, emb2])
    X.append(features)
    y.append(row["real_text_id"])  # label = 1 or 2

X = np.vstack(X)
y = np.array(y)


X


y


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

clf = LogisticRegression(max_iter=2000)

# Cross-validation is important
scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print("CV Accuracy:", scores.mean())

clf.fit(X, y)


test = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

data = []
for folder in os.listdir(test):
    folder_path = os.path.join(test, folder)
    idx = int(folder.rsplit("_")[-1])
    with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
          text1 = f1.read().strip()
    with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
          text2 = f2.read().strip()
    data.append([idx, text1, text2])


test_df = pd.DataFrame(data, columns=['id','Text1', 'Text2'])
test_df.head()


test_df.shape


test_X, test_y = [], []
for _, row in test_df.iterrows():
    emb1 = get_embedding(row["Text1"])
    emb2 = get_embedding(row["Text2"])
    features = np.hstack([np.abs(emb1 - emb2), emb1, emb2])
    test_X.append(features)

test_X = np.vstack(test_X)


test_X


pred = clf.predict([test_X[0]])[0]
print("Predicted real text:", 1 if pred == 1 else 2)



preds = clf.predict(test_X)

# Map predictions back to "Text1" or "Text2"
test_df["real_text_id"] = preds


preds


test_df


result = test_df[["id", "real_text_id"]]


result


result.to_csv("Submission_File.csv", index=False)


result.shape




