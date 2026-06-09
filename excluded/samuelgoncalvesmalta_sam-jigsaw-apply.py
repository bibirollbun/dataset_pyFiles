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


import kagglehub
import more_itertools
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-1b-it")
processor = AutoTokenizer.from_pretrained(GEMMA_PATH)

# Determine if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH).to(device)
print(model)


test_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
train_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')


test_data.head(5)


model


def prompt(input: pd.Series):
    return """<start_of_turn>user
You are a really experienced moderator for the subreddit /r/%s. You will be provided 
with two positive examples and two negative examples to help you determine if a 
violation occurs. Your job is to determine if the following reported comments 
violates the rule: %s. 

%s
Decision:
True

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
<end_of_turn>
<start_of_turn>model\n""" % (
    input['subreddit'],
    input['rule'],
    "\n".join(["| " + x for x in input['positive_example_1'].split('\n')]),
    "\n".join(["| " + x for x in input['positive_example_2'].split('\n')]),
    "\n".join(["| " + x for x in input['negative_example_1'].split('\n')]),
    "\n".join(["| " + x for x in input['negative_example_2'].split('\n')]),
    "\n".join(["| " + x for x in input['body'].split('\n')])    
)


token_ids = [processor.get_vocab()[word] for word in ['True', 'False']]
if any(token_id == processor.get_vocab()['<unk>'] for token_id in token_ids):
    raise ValueError('One of the target classes is not in the vocabulary.')


responses = []
for batch in more_itertools.batched(test_data.iterrows(), 4):
    prompts = [prompt(x) for _, x in batch]
    pre = processor(text=prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=512).to(device)
    with torch.no_grad():
      outputs = model(**pre)
    logits = outputs.logits[:, -1, token_ids]  
    probabilities = torch.softmax(logits, dim=-1)
    responses.extend(probabilities[:, 0].tolist())


test_data_final = test_data.copy()


test_data_final["model"] = "gemma-3-1b-it"
test_data_final["responses"] = responses


test_data_final[["body", "rule", "responses"]]


my_submission = pd.DataFrame({
    'row_id': test_data['row_id'],
    'rule_violation': responses
})


my_submission




