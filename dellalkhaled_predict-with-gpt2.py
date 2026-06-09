import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


for col in train.columns:
    if train[col].dtypes == 'object':
        train[pd.get_dummies(train[col]).columns] = pd.get_dummies(train[col])
        train = train.drop(columns = col)


for col in test.columns:
    if test[col].dtypes == 'object':
        test[pd.get_dummies(test[col]).columns] = pd.get_dummies(test[col])
        test = test.drop(columns = col)


train['y']


train['y']


cols = list(train.columns)
cols.remove('y')
cols.append('y')


train = train[cols]


for col in train.columns:
    train[col] = train[col].astype(int)

for col in test.columns:
    test[col] = test[col].astype(int)

for col in train.columns:
    train[col] = train[col].astype(str)

for col in test.columns:
    test[col] = test[col].astype(str)


train["inputs"] = train["age"]


for i in range(2, len(train.columns)-2):
    train["inputs"] = train["inputs"] + " " + train[train.columns[i]]


train['inputs'][4]


#train['y'] = train['y'].astype(int)


df_terms = train[['inputs', 'y']]


import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Chargement du modèle et tokenizer
model_name = "gpt2"#'tiiuae/falcon-rw-1b'#"gpt2"#"bert-base-multilingual-cased"#"tiiuae/falcon-rw-1b"#"bert-base-multilingual-cased"  # Remplacer par un modèle causal adapté (GPT-style)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fonction de génération
def generate_accounting_response(prompt, max_length=100):
    # Remplacer les termes par leurs explications (même plusieurs fois si besoin)
    for _, row in df_terms.iterrows():
        prompt = prompt.replace(row["inputs"], row["y"])
    
    # Tokenisation
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Génération de texte
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Décodage
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Nettoyage final (optionnel)
    if not generated_text.endswith(('.', '!', '?')):
        last_period = generated_text.rfind('.')
        if last_period != -1:
            generated_text = generated_text[:last_period+1]
    
    return generated_text


train['inputs'][0]


# Exemple
prompt = "42 7 25 117 3 -1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 1 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0"
response = generate_accounting_response(prompt)
print(response[0])


test


test["inputs"] = test["age"]


for i in range(2, len(test.columns)):
    test["inputs"] = test["inputs"] + " " + test[test.columns[i]]


test['inputs'][0]


import random
# Generate a random integer between 1 and 250000 (inclusive)
random_number = random.randint(1, 250000)
random_number


y, N = [], len(test)
for i in [230600, 147297, 155675, 153917]:
    p = test['inputs'][i]
    r = generate_accounting_response(p)
    y.append(r[0])


y


r


#test['y'] = y


#test['y'] = test['y'].astype(int)


#test[['id', 'y']].to_csv("submission.csv", index = False)

