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


import torch._dynamo
torch._dynamo.config.suppress_errors = True


# Guidelines
# Source: https://www.kaggle.com/code/sorenj/batch-gemma3-sample-rules-classification/notebook


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used.")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU instead.")


GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-1b-it")

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH).to(device)
print(model)


prompt_test = "Write a short poem about nature."

inputs = tokenizer(prompt_test, return_tensors="pt").to(device)


inputs


tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
tokens


inputs["input_ids"]


prompt_length = inputs['input_ids'].shape[1]
prompt_length


output_ids = model.generate(**inputs, max_new_tokens=50)
output_ids


# LLM steps
# 1. human prompt
# 2. prompt tokenized
# 3. prompt encoded
# 4. model reads prompt encoded
# 5. model answers prompt encoded
# 6. answer decoded (human answer)


output_ids.shape[1]


output_ids[0]


answer_token = output_ids[0][prompt_length:]
answer_token.shape[0]


answer_token


decoded_output = tokenizer.decode(answer_token, skip_special_tokens=True)
decoded_output


from IPython.display import display, Markdown

display(Markdown(decoded_output))


test_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
train_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')


len(train_data)


test_data.head(5)


train_data.head(5)


sample_submission.head(5)


def prompt(input: pd.Series):
    return """<start_of_turn>
You are a really experienced moderator for Reddit. You will be provided with
two positive examples and two negative examples of comments to help you determine if a 
violation occured or not. Your job is to determine if the following reported comments 
violates the rule: %s. 

"%s"\n
Decision: Does not violate

"%s"\n
Decision: Does not violate

"%s"\n
Decision: Violates

"%s"\n
Decision: Vialotes

<end_of_turn>
<start_of_turn>

Determine if the following example violates the rule: "%s". \n\nAnswer only with the phrase "Decision: does not violates" or "Decision: violates".
Do not add any explanations, extra information, or extra symbols such as न्ना or \n at the end of the answer.

<end_of_turn>
""" % (
    # input['subreddit'],
    input['rule'],
    input['positive_example_1'],
    input['positive_example_2'],
    input['negative_example_1'],
    input['negative_example_2'],
    input['body']    
)


for batch in more_itertools.batched(test_data.iterrows(), 4):
    prompts = [prompt(x) for _, x in batch]


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


token_ids = [tokenizer.get_vocab()[word] for word in ['True', 'False']]
if any(token_id == tokenizer.get_vocab()['<unk>'] for token_id in token_ids):
    raise ValueError('One of the target classes is not in the vocabulary.')


model_responses = []
prob_responses = []
count = 0
for batch in more_itertools.batched(test_data.iterrows(), 1):
# for batch in more_itertools.batched(test_data.iterrows(), len(test_data)):
    count += 1
    prompts = [prompt(x) for _, x in batch]
    prompt_encoded = tokenizer(text=prompts[0], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    prompt_length = prompt_encoded['input_ids'].shape[1]
    with torch.no_grad():
        output = model(**prompt_encoded)
        encoded_output = model.generate(**prompt_encoded)
        answer_output = encoded_output[0][prompt_length:]
        decoded_output = tokenizer.decode(answer_output, skip_special_tokens=True)
    model_responses.append(decoded_output)
    logits = output.logits[:, -1, token_ids]
    probabilities = torch.softmax(logits, dim=-1)
    prob_responses.extend(probabilities[:, 0].tolist())



test_data["model"] = "gemma-3-1b-it"
test_data["prob_responses"] = prob_responses
test_data["model_responses"] = model_responses


test_data.head(1)


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


test_data[["body", "rule", "subreddit", "model_responses"]]


# my_submission = pd.DataFrame({
#     'row_id': test_data['row_id'],
#     'rule_violation': responses
# })


# my_submission


# my_submission.to_csv('submission.csv', index=False)

