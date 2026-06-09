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


import pandas as pd
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
# train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
# train = pd.concat([train, train_extra], axis=0, ignore_index=True)
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')
target = "Price"


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
print("Train shape:",train.shape)
train.head()
test.head()


CATS = [col for col in train.columns if col not in ["Price", "Weight Capacity (kg)"]]


numerical_feats = train.select_dtypes(include = ['float64','int64']).columns
cat_cols = train.select_dtypes(include = ['object','category']).columns


from sklearn.impute import KNNImputer

# Create the KNN imputer instance (e.g., with 5 neighbors)
knn_imputer = KNNImputer(n_neighbors=1000, weights="uniform")

# Apply the imputer only to the numerical features and update the train DataFrame
train[['Weight Capacity (kg)']] = knn_imputer.fit_transform(train[['Weight Capacity (kg)']])


for col in train[cat_cols].columns:
    train[col] = train[col].fillna("Missing").astype('category')


train.isnull().sum()


train["Price"][0]


numerical_featst = test.select_dtypes(include = ['float64','int64']).columns
cat_colst = test.select_dtypes(include = ['object','category']).columns


from sklearn.impute import KNNImputer

# Create the KNN imputer instance (e.g., with 5 neighbors)
knn_imputer = KNNImputer(n_neighbors=1000, weights="uniform")

# Apply the imputer only to the numerical features and update the train DataFrame
test[['Weight Capacity (kg)']] = knn_imputer.fit_transform(test[['Weight Capacity (kg)']])


for col in test[cat_colst].columns:
    test[col] = test[col].fillna("Missing").astype('category')


test.isnull().sum()


import pandas as pd

def dataset_for_llm(df):
    new_df = pd.DataFrame()
    instruction_text = (
        "Given the details of a bag, predict its price as a float or integer. "
        "Consider factors such as brand, material, size, number of compartments, "
        "presence of a laptop compartment, waterproofing, style, color, and weight capacity. "
        "The response should only be the numerical price窶馬o extra text."
    )
    new_df["instruction"] = [instruction_text] * len(df)
    new_df["input"] = df.apply(lambda row: (
        f"This is a {row['Brand']} bag made from {row['Material']}. "
        f"It has a size of {row['Size']} inches and comes with {row['Compartments']} compartments. "
        f"It {'includes' if row['Laptop Compartment'] == 'Yes' else 'does not include'} a laptop compartment. "
        f"The bag is {'waterproof' if row['Waterproof'] == 'Yes' else 'not waterproof'}. "
        f"It follows a {row['Style']} style, is available in {row['Color']} color, "
        f"and has a weight capacity of {row['Weight Capacity (kg)']} kg. "
        f"Based on these details, what is the expected price of this bag?"
    ), axis=1)
    new_df["output"] = df["Price"]  
    return new_df
new_df = dataset_for_llm(train)
new_df.to_csv("bag_price_dataset.csv", index=False)
print("New dataset created successfully!")



new_df


import pandas as pd

def dataset_for_llms(df):
    test_df = pd.DataFrame(index=df.index)
    instruction_text = (
        """Given the details of a bag, predict its price as a float or integer. 
        Consider factors such as brand, material, size, number of compartments, 
        presence of a laptop compartment, waterproofing, style, color, and weight capacity. 
        The response should only be the numerical price窶馬o extra text."""
    )
    test_df["instruction"] = instruction_text
    
    test_df["input"] = df.apply(
        lambda row: (
            f"This is a {row['Brand']} bag made from {row['Material']}. "
            f"It has a size of {row['Size']} inches and comes with {row['Compartments']} compartments. "
            f"It {'includes' if row['Laptop Compartment'] == 'Yes' else 'does not include'} a laptop compartment. "
            f"The bag is {'waterproof' if row['Waterproof'] == 'Yes' else 'not waterproof'}. "
            f"It follows a {row['Style']} style, is available in {row['Color']} color, "
            f"and has a weight capacity of {row['Weight Capacity (kg)']} kg. "
            f"Based on these details, what is the expected price of this bag?"
        ),
        axis=1
    )
    return test_df
test_df = dataset_for_llms(test)
test_df.to_csv("bag_price_dataset_test.csv", index=False)
print("New dataset created successfully!")
test_df.head()


test_df.isnull().sum()



!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth
!pip install trl


import torch
import os
os.environ["WANDB_DISABLED"] = "true"
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import pandas as pd
from trl import SFTTrainer

max_seq_length = 512
dtype = None
load_in_4bit = True


model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
    model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


tokenizer.chat_template = """\
{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}"""
def formatting_prompts_func(examples):
    texts = []
    for instruction, inp, output in zip(examples["instruction"], examples["input"], examples["output"]):
        instruction = str(instruction)
        inp = str(inp)
        output = str(output)
        
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{inp}"},
            {"role": "assistant", "content": output}
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}


df = pd.read_csv("/kaggle/working/bag_price_dataset.csv")
dataset = Dataset.from_pandas(df)
dataset = dataset.map(formatting_prompts_func, batched=True)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    max_steps=100,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    output_dir="outputs",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=True,
    args=training_args,
)


trainer.train()


from unsloth.chat_templates import get_chat_template

FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "Given the details of a bag, predict its price as a float or integer. Consider factors such as brand, material, size, number of compartments, presence of a laptop compartment, waterproofing, style, color, and weight capacity. The response should only be the numerical price窶馬o extra text."},
    {"role": "user", "content": "This is a Jansport bag made from Leather. It has a size of Medium inches and comes with 7.0 compartments. It includes a laptop compartment. The bag is not waterproof. It follows a Tote style, is available in Black color, and has a weight capacity of 11.611722805222309 kg. Based on these details, what is the expected price of this bag?"},
]


inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,  
    return_tensors = "pt",
).to("cuda")


outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 64,
    # pad_token_id = tokenizer.eos_token_id,
    do_sample = True,
    temperature = 0.7,
    top_p = 0.95,     
    use_cache = True,
)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(response.split("<|im_start|> assistant\n")[-1].strip())


sub=pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub


test_df=pd.read_csv("/kaggle/working/bag_price_dataset_test.csv")
test_df




import re
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

FastLanguageModel.for_inference(model)
def extract_price(text):
    matches = re.findall(r"[-+]?\d*\.?\d+|\d+", text)
    if matches:
        try:
            return float(matches[0])
        except:
            return None
    return None

def predict_prices(dataset, model, tokenizer, batch_size=100):
    model.eval()
    predictions = []
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.iloc[i:i+batch_size].to_dict(orient="records")
        messages_batch = [
            [
                {"role": "system", "content": example['instruction']},
                {"role": "user", "content":  example['input']}
            ] 
            for example in batch
        ]
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to("cuda")
        
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=16, 
            # pad_token_id=tokenizer.eos_token_id,
            do_sample=False,   
            temperature=0.01, 
            use_cache=True,
        )
        
        responses = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)
        batch_predictions = [extract_price(resp) for resp in responses]
        predictions.extend(batch_predictions)
    
    return predictions
if __name__ == "__main__":
    df = pd.read_csv("/kaggle/working/bag_price_dataset_test.csv")
    dataset = Dataset.from_pandas(df)
    prices = predict_prices(df, model, tokenizer)
    sub["Price"] = prices
    sub.to_csv("submission.csv", index=False)

