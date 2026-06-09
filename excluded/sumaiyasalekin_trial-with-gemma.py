import pandas as pd

df = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')
df.head()


# Prepare a function to generate text for a given input
def generate_output(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=32)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)




# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2")
model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Apply the model to each topic in the 'topic' column and store the outputs in a new column
df['essay'] = df['topic'].apply(lambda x: generate_output(f"Write me a short essay on the topic: {x}."))

#print(tokenizer.decode(outputs[0]))



df.drop(columns='topic', inplace=True)


df.head()


df.to_csv('/kaggle/working/submission.csv',index=False)




