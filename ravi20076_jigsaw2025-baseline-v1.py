


import kagglehub
import more_itertools
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


%%time 

model_path    = f"/kaggle/input/phi-3/pytorch/phi-3.5-mini-instruct/2"
my_tokenizer  = AutoTokenizer.from_pretrained(model_path)
model         = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype       = torch.float16,
    device_map        = "auto" ,
    low_cpu_mem_usage = True,
)

test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')


def prompt(input: pd.Series):
    myprompt = """<start_of_turn>user
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

    return myprompt
             


%%time 

token_ids = [my_tokenizer.get_vocab()[word] for word in ['True', 'False']]
if any(token_id == my_tokenizer.get_vocab()['<unk>'] for token_id in token_ids):
      raise ValueError('One of the target classes is not in the vocabulary.')

responses = []
for batch in more_itertools.batched(test.iterrows(), 1):
    prompts = [prompt(x) for _, x in batch]
    pre = my_tokenizer(
        text=prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
      outputs = model(**pre)
        
    logits = outputs.logits[:, -1, token_ids]  
    probabilities = torch.softmax(logits, dim=-1)
    responses.extend(probabilities[:, 0].tolist())  

my_submission = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': responses
})
my_submission.to_csv('submission.csv', index=False)

print()
!head submission.csv
print()
!ls

