import pandas as pd
from tqdm import tqdm


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = '/kaggle/input/phi-3.5-mini-instruct/pytorch/default/1'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)


model


tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)


tokenizer


test_path = '/kaggle/input/llms-you-cant-please-them-all/test.csv'
test_df = pd.read_csv(test_path)
test_df.head()


def get_prompt(topic):
    prompt = f"""<|system|>
You are a helpful assistant.<|end|>
<|user|>
Write an essay in 100 words on following topic: ```{topic}``` <|end|>
<|assistant|> 
"""
    return prompt

def get_input_tokens(topic):
    prompt = get_prompt(topic)
    inputs = tokenizer(prompt, return_tensors='pt')
    return inputs


def generate(inputs):
    """
    Do sample is set to false.
    """
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.3,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        num_return_sequences=1,
        do_sample=True,
        eos_token_id=32000,
    )
    decoded_text = None
    for output in outputs:
        decoded_text = tokenizer.decode(output, skip_special_tokens=False)
        decoded_text = decoded_text.split('<|assistant|>')[-1]
        decoded_text = decoded_text.split('<|endoftext|>')[0].strip()
        
    return decoded_text


ids = []
essays = []
for idx in tqdm(range(len(test_df))):
    essay_id = test_df.iloc[idx]['id']
    topic = test_df.iloc[idx]['topic']
    essay = f"this is essay {idx}"

    inputs = get_input_tokens(topic)    
    inputs = inputs.to('cuda')
    response = generate(inputs)
    
    ids.append(essay_id)
    essays.append(response)

    del inputs
    torch.cuda.empty_cache()

sub_df = pd.DataFrame({'id': ids, 'essay': essays})
sub_df.to_csv('submission.csv', index=False)


sub_df.head()


print(sub_df.iloc[0]['essay'])




