#!pip install -U  -q transformers timm


import kagglehub
import more_itertools
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')



GEMMA_PATH = kagglehub.model_download("google/gemma-2/transformers/gemma-2-2b-it")
#GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-1b-it")

#GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
processor = AutoTokenizer.from_pretrained(GEMMA_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH).to(device)
print(model)
model.eval()



#MODEL_NAME = '/kaggle/input/qwen-3/transformers/1.7b/1'


#processor = AutoTokenizer.from_pretrained(MODEL_NAME)
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model = AutoModelForCausalLM.from_pretrained(
 #   MODEL_NAME,
  #  torch_dtype="auto",
   # device_map="auto"  # or .to(device) for simple setups
#)
#print(model)
#model.eval()


test_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')


test_data


def prompt(input: pd.Series):
    return """<start_of_turn>user
You are acting as a Reddit moderator for the given subreddit. You are given a comment from reddit and a rule. Your task is to classify whether the comment violates the rule 
AS ENFORCED IN THAT SUBREDDIT. Only respond in True/False:
{rule}

{pos1}
Decision: True

{neg1}
Decision: False

{neg2}
Decision: False

{pos2}
Decision: True

{body}
<end_of_turn>
<start_of_turn>model
""".format(
        sub=input['subreddit'],
        rule=input['rule'],
        pos1="\n".join(["| " + x for x in input['positive_example_1'].split('\n')]),
        neg1="\n".join(["| " + x for x in input['negative_example_1'].split('\n')]),
        neg2="\n".join(["| " + x for x in input['negative_example_2'].split('\n')]),
        pos2="\n".join(["| " + x for x in input['positive_example_2'].split('\n')]),
        body="\n".join(["| " + x for x in input['body'].split('\n')])
    )


token_ids = processor.convert_tokens_to_ids(['True', 'False'])
if any(tid == processor.unk_token_id for tid in token_ids):
    raise ValueError("One of the target classes (True/False) is missing from the vocabulary.")



responses = []

# Batch prediction
for batch in more_itertools.batched(test_data.iterrows(), 2):
    prompts = [prompt(x) for _, x in batch]
    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits for the next predicted token
    logits = outputs.logits[:, -1, token_ids]
    probs = torch.softmax(logits, dim=-1)

    # Store probability of "True"
    responses.extend(probs[:, 0].tolist())


my_submission = pd.DataFrame({
    'row_id': test_data['row_id'].iloc[:len(responses)],
    'rule_violation': responses
})


my_submission.to_csv("submission.csv", index=False)



my_submission


