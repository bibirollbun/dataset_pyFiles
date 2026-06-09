# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


#By Abhishek Thakur https://www.kaggle.com/code/abhishek/autonlp-for-toxic-ratings/notebook  

class Dataset:
    def __init__(self, text, tokenizer, max_len):
        self.text = text
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        inputs = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


#By Abhishek Thakur https://www.kaggle.com/code/abhishek/autonlp-for-toxic-ratings/notebook

def generate_predictions(model_path, max_len):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to("cuda")
    model.eval()
    
    df = pd.read_csv("../input/jigsaw-agile-community-rules/test.csv")
    
    dataset = Dataset(text=df.positive_example_2.values, tokenizer=tokenizer, max_len=max_len)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, num_workers=2, pin_memory=True, shuffle=False #Suggested number of worker 2 Original 4
    )

    final_output = []

    for b_idx, data in enumerate(data_loader):
        with torch.no_grad():
            for key, value in data.items():
                data[key] = value.to("cuda")
            output = model(**data)
            output = output.logits.detach().cpu().numpy()[:, 1].ravel().tolist()
            final_output.extend(output)
    
    torch.cuda.empty_cache()
    return np.array(final_output)


#By Abhishek Thakur https://www.kaggle.com/code/abhishek/autonlp-for-toxic-ratings/notebook
preds = generate_predictions("../input/autonlp-toxic-1/", max_len=192)


#By Abhishek Thakur https://www.kaggle.com/code/abhishek/autonlp-for-toxic-ratings/notebook
sub = pd.read_csv("../input/jigsaw-agile-community-rules/test.csv")
sub["positive_example_2"] = preds
sub = sub[["row_id", "positive_example_2"]]
sub.to_csv("submission.csv", index=False)


sub.head()

