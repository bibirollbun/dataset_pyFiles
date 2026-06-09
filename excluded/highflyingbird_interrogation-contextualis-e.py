!pip install --upgrade transformers accelerate bitsandbytes --quiet


import pandas as pd

train_df = pd.read_json('/kaggle/input/defi-text-mine-egc-2026/train_v2.jsonl', lines=True)
test_df = pd.read_json('/kaggle/input/defi-text-mine-egc-2026/test_v4.jsonl', lines=True)
train_df.head()


count = {}
for i, row in train_df.iterrows():
    options = row["options"]
    for k, v in options.items():
        if k not in count:
                count[k] = 0
        if v:
            count[k] += 1


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login; login(token=UserSecretsClient().get_secret("HF_TOKEN"))

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4",
                                             return_dict=True,
                                             quantization_config=quantization_config,
                                             dtype=torch.float16,
                                             device_map="auto")
pipe = pipeline(tokenizer=tokenizer, 
                model=model, 
                task='text-generation', 
                dtype=torch.float16, 
                device_map="auto")
pipe.tokenizer.pad_token_id = model.config.eos_token_id


def format_input(row_id):
    acronym, text, options = test_df.acronym.values[row_id], test_df.text.values[row_id], test_df.options.values[row_id]
    question = f'Parmi les définitions ci-dessous, laquelle ou lesquelles correspondent à l\'acronyme {acronym} dans le texte suivant : "{text}"\n'
    context = "Pour répondre, prends en compte les statistiques suivantes à propos de réponses passées : "
    N = 0
    for i, option in enumerate(options):
        N += count.get(option, 0)
    for i, option in enumerate(options):
        n = count.get(option, 0)
        if N == 0:
            apriori = round(1/len(options) * 100, 0)
        elif n > 0:
            apriori = round(n/N * 100, 0)
        else:
            apriori = 0
        question += f"\nDéfinition {i} (a priori {apriori}% de chances d'être juste) : {option}"
    system = "Tu es un assistant virtuel, spécialisé dans les QCM et expert du domaine ferroviaire. Lorsqu'on te pose une question, tu réponds strictement en listant le numéro ou les numéros des définitions correctes. "
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}]
    chat = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return chat

print(format_input(2))
inputs = [format_input(i) for i in test_df.index]


outputs = pipe(inputs, do_sample=False, batch_size=6)


import re

predicted_ids = [list(set([int(float(i)) for i in re.findall(r'\d+(?:\.\d+)?', output[0]["generated_text"].split("<|im_start|>assistant<|im_sep|>")[1])])) for output in outputs]
predicted_ids = [[i for i in pids if i < 15] for pids in predicted_ids]


submission = pd.DataFrame({"id": test_df.index, "prediction":predicted_ids})
submission.head()
submission.to_csv("/kaggle/working/submission.csv", index=False)

