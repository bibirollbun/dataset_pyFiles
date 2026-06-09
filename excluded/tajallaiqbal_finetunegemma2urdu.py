%%capture
# !pip install pip3-autoremove
!pip install -q torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install -q 'unsloth==2025.1.1'
!pip uninstall -q transformers -y
!pip install -q 'transformers==4.47.1'


from unsloth import FastLanguageModel
import torch


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



max_seq_length = 512
dtype = None
load_in_4bit = True
model,tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-2b",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                     "gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


import pandas as pd
from datasets import Dataset

# Load your CSV dataset
csv_file = "/kaggle/input/urdu-news-dataset/Urdu-News-Dataset-1M.csv" 
df = pd.read_csv(csv_file)

# Select only the necessary columns (Headline and News Text)
df = df[["Headline", "News Text"]]




df.head()


alpaca_prompt = """You are an AI language model designed to generate detailed news articles based on provided titles.
Your task is to understand the given title and create a comprehensive and informative article in Urdu.

###Instruction:
Write a detailed news article based on the given title.

###Title:
{}

###Response:
{}
"""
EOS_TOKEN = tokenizer.eos_token
# Define the formatting function
def formatting_prompts_func(examples):
    # Extract data from the relevant columns
    headlines = examples["Headline"]
    articles = examples["News Text"]
    
    # Prepare a prompt for each row
    texts = []
    for inputs, output in zip(headlines, articles):
        # Example prompt structure (English instructions, Urdu content)
        text = alpaca_prompt.format(inputs,output) + EOS_TOKEN
        texts.append(text)
    
    # Return the formatted dataset as a dictionary
    return {"text": texts}

# Convert the Pandas DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Apply the formatting function
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

# Save the formatted dataset to a JSONL file
formatted_dataset.to_json("formatted_dataset.jsonl", force_ascii=False, orient="records", lines=True)

print("Formatted dataset saved as formatted_dataset.jsonl")


formatted_dataset[0]

    


# train_dataset = formatted_dataset.shuffle(seed=3407).select(range(int(0.1 * len(formatted_dataset))))


# import os
# os.environ['WANDB_DISABLED'] = 'true'


print("split dataset")
split_dataset = formatted_dataset.train_test_split(test_size=0.3, seed=3407)

# Get the training and testing datasets
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']





print('training args')
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    dataset_text_field = 'text',
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        #for 1 full training run
        #num_train_epochs=1
        # max_steps=60,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16= not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps=20,
        optim = 'adamw_8bit',
        weight_decay=0.01,
        lr_scheduler_type='linear',
        seed = 3407,
        report_to="none",
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 100,
    ),
)


print('trainer')
trainer_stats = trainer.train()


import os
print(os.listdir('/kaggle/input/checkpoint/other/default/1/checkpoint-10000'))


from unsloth import FastLanguageModel
import torch


model_test, tokenizer = FastLanguageModel.from_pretrained(
            model_name='/kaggle/input/checkpoint/other/default/1/checkpoint-10000',
            max_seq_length=512,
            dtype=None,  # Will use the dtype specified in config
            load_in_4bit=True,
            # local_files_only=True
        )
        


alpaca_prompt = """Write a detailed news article in Urdu based on the following title:

###Title:
{}

###Response:
"""
FastLanguageModel.for_inference(model_test)
inputs = tokenizer(
[
    alpaca_prompt.format(
       
        "اوپیک ممالک کا ااوپیک ممالک کا اجلاس تیل", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model_test.generate(**inputs,min_length=50, max_length=inputs['input_ids'].shape[1] + 500,
                              eos_token_id=tokenizer.eos_token_id, 
                              use_cache = True,no_repeat_ngram_size=3,
                              top_p=0.9,temperature=0.7,do_sample=True,)
tokenizer.batch_decode(outputs)





FastLanguageModel.for_inference(model_test)
inputs = tokenizer(
[
    alpaca_prompt.format(
       
        "سکیورٹی اداروں کو دھمکی", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model_test.generate(**inputs, max_new_tokens = 200, use_cache = True)
tokenizer.batch_decode(outputs)




