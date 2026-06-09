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


import warnings
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer
from peft import LoraConfig,TaskType,get_peft_model
import torch
warnings.simplefilter('ignore')
print('sabki ma ki chut')


df_train=pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df_test=pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")


df_train.info()


# Add prompt-formatted input text
df_train['text'] = (
    'SUBREDDIT: ' + df_train['subreddit'] +
    ' RULE: ' + df_train['rule'] +
    ' Positive Examples: ' + df_train['positive_example_1'] + " " + df_train['positive_example_2'] +
    ' Negative Examples: ' + df_train['negative_example_1'] + " " + df_train['negative_example_2'] +
    ' COMMENT: ' + df_train['body']
)

# Set the labels
df_train['labels'] = df_train['rule_violation']

# Drop unused columns
df_train = df_train.drop([
    'row_id', 'body', 'rule', 'subreddit',
    'positive_example_1', 'positive_example_2',
    'negative_example_1', 'negative_example_2',
    'rule_violation'
], axis=1)



model_path = "/kaggle/input/deberta-v3-base/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)


train_encodings = tokenizer(list(df_train['text']), truncation=True, padding=True, max_length=512)


import torch
from torch.utils.data import Dataset

class RedditDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = RedditDataset(train_encodings, df_train['labels'].values)




import os
os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir="./results",
    save_steps=500,
    logging_steps=100,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=2,
    # logging_level='info', 
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)


trainer.train()



import pandas as pd
import matplotlib.pyplot as plt


log_history = pd.DataFrame(trainer.state.log_history)


train_loss = log_history[log_history["loss"].notna()][["epoch", "step", "loss"]]


plt.figure(figsize=(10, 6))
plt.plot(train_loss["step"], train_loss["loss"], label="Training Loss", color='royalblue', linewidth=2)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss over Steps")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
df_test['text'] = (
    'SUBREDDIT: ' + df_test['subreddit'] +
    ' RULE: ' + df_test['rule'] +
    ' Positive Examples: ' + df_test['positive_example_1'] + " " + df_test['positive_example_2'] +
    ' Negative Examples: ' + df_test['negative_example_1'] + " " + df_test['negative_example_2'] +
    ' COMMENT: ' + df_test['body']
)




# from datasets import Dataset
# test_dataset = Dataset.from_pandas(df_test[['text']])  # only need 'text'


test_encodings = tokenizer(list(df_test['text']), truncation=True, padding=True, max_length=512)



class RedditTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

test_dataset = RedditTestDataset(test_encodings)



predictions = trainer.predict(test_dataset)



import numpy as np
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()



sample_submission = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")
sample_submission["rule_violation"] = probs
sample_submission.to_csv("submission.csv", index=False)





