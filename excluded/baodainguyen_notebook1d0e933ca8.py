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


# =======================================
# 1. Cài đặt & import thư viện
# =======================================
import pandas as pd
import torch
import more_itertools
from transformers import AutoTokenizer, AutoModelForCausalLM
import kagglehub   # để tải mô hình Gemma từ KaggleHub



# 2. Tải model & tokenizer Gemma
# =======================================
# Tải về Gemma 3 (1B instruct) từ Kaggle
GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-1b-it")

# Load tokenizer & model
processor = AutoTokenizer.from_pretrained(GEMMA_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH).to(device)

print("✅ Model loaded on", device)


import os
print(os.listdir("/kaggle/input"))


print(os.listdir("/kaggle/input/jigsaw-agile-community-rules"))
# ['train.csv', 'test.csv', 'sample_submission.csv']


# 3. Đọc dữ liệu
# =======================================
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
sample_submission = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")

print("Train:", train_df.shape, "Test:", test_df.shape)



def prompt(input: pd.Series):
    return """<start_of_turn>user
You are a really experienced moderator for the subreddit /r/%s. Your job
is to determine if the following reported comments violates the rule:
%s

%s
Decision:
True

%s
Decision:
False

%s
Decision:
False

%s
Decision:
True

%s
<end_of_turn>
<start_of_turn>model\n""" % (
    input['subreddit'],
    input['rule'],
    "\n".join(["| " + x for x in input['positive_example_1'].split('\n')]),
    "\n".join(["| " + x for x in input['negative_example_1'].split('\n')]),
    "\n".join(["| " + x for x in input['negative_example_2'].split('\n')]),
    "\n".join(["| " + x for x in input['positive_example_2'].split('\n')]),
    "\n".join(["| " + x for x in input['body'].split('\n')])    
)


token_ids = [processor.get_vocab()[word] for word in ['True', 'False']]
if any(token_id == processor.get_vocab()['<unk>'] for token_id in token_ids):
      raise ValueError('One of the target classes is not in the vocabulary.')


responses = []
for batch in more_itertools.batched(test_df.iterrows(), 4):
    prompts = [prompt(x) for _, x in batch]
    pre = processor(text=prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=512).to(device)
    with torch.no_grad():
      outputs = model(**pre)
    logits = outputs.logits[:, -1, token_ids]  
    probabilities = torch.softmax(logits, dim=-1)
    responses.extend(probabilities[:, 0].tolist())  


#7. Xuất file submission
# =======================================
submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "rule_violation": responses
})

submission.to_csv("submission.csv", index=False, float_format="%.6f")

print("✅ Done! File submission.csv đã sẵn sàng.")
print(submission.head(10))

