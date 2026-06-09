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


test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
test


from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

base_model_path = "/kaggle/input/qwen-3/transformers/1.7b-base/1"
adapter_path = "/kaggle/input/model/transformers/default/1/model/final_checkpoint"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    local_files_only=True,
    offload_folder="/kaggle/temp_offload"
)
model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
model.eval()




import torch.nn.functional as F

choices = [
    "Yes, this comment violates the rule.",
    "No, this comment does not violate the rule."
]

def get_choice_probs(row):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Reddit content moderator. "
                "Your job is to decide whether a comment violates the given subreddit rule. "
                "You will be given:\n"
                "- The subreddit name\n"
                "- The specific rule text\n"
                "- The comment text to evaluate\n"
                "- Two example comments that violate the rule\n"
                "- Two example comments that do not violate the rule\n\n"
                "Compare the comment with the examples and rule carefully. "
                "Respond only with one of the two options:\n"
                f"1. {choices[0]}\n"
                f"2. {choices[1]}"
            )
        },
        {
            "role": "user",
            "content": (
                f"Subreddit: r/{row['subreddit']}\n"
                f"Rule: {row['rule']}\n\n"
                f"Positive example 1 (violates): {row['positive_example_1']}\n"
                f"Positive example 2 (violates): {row['positive_example_2']}\n"
                f"Negative example 1 (does not violate): {row['negative_example_1']}\n"
                f"Negative example 2 (does not violate): {row['negative_example_2']}\n\n"
                f"Comment to evaluate: {row['body']}"
            )
        }
    ]

    prompt_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    probs = []
    with torch.no_grad():
        for choice in choices:
            choice_ids = tokenizer(choice, return_tensors="pt", add_special_tokens=False).to(model.device)
            input_ids = torch.cat([prompt_ids, choice_ids.input_ids], dim=1)
            attn_mask = torch.cat(
                [torch.ones_like(prompt_ids), torch.ones_like(choice_ids.input_ids)],
                dim=1
            )

            outputs = model(input_ids, attention_mask=attn_mask)
            logits = outputs.logits[:, :-1, :]  # skip last token
            labels = input_ids[:, 1:]           # shifted labels

            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            choice_log_prob = token_log_probs[:, -choice_ids.input_ids.size(1):].sum().item()
            probs.append(choice_log_prob)

    probs = torch.softmax(torch.tensor(probs), dim=0)
    return { probs[i].item() for i in range(len(choices))}



pred = []
for _, row in test.iterrows():
    pred.append(list(get_choice_probs(row))[0])



submission = pd.DataFrame({
    "row_id":test["row_id"],
    "rule_violation":pred
})
submission.to_csv("submission.csv" , index=False)


submission





































