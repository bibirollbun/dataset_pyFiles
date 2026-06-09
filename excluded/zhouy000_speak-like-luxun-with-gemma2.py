import json
import re

file_path = r"luxun_dataset-master/data_dir/luxun.json"

with open(file_path, 'r', encoding='utf-8') as f:
    luxun_data = json.load(f)

print(luxun_data[0:3])


def content_cleaning(text):
    
    # Remove the content before \xa0\xa0\xa0\xa0, \t\t\t, \r\n, \n, 
    # such as the title and some special characters.
    try:
        if "\xa0\xa0\xa0\xa0" in text:
            cleaned_text = text.split("\xa0\xa0\xa0\xa0", 1)[1].strip()
        elif "\t\t\t" in text:
            cleaned_text = text.split("\t\t\t", 1)[1].strip()
        elif "\r\n" in text:
            cleaned_text = text.split("\r\n", 1)[1].strip()
        elif "\n" in text:
            cleaned_text = text.split("\n", 2)[2].strip()
        else:
            cleaned_text = text.strip()
    except IndexError:
        # IndexError handling, return the original text
        cleaned_text = text.strip()

    #Remove \r,\n,\s from the text
    cleaned_text = re.sub(r'[\r\n\s]', '', cleaned_text) 

    return cleaned_text


def data_collating(data):
    '''
    Collating data sets
    '''
    collated_data = []

    for record in data:

        # Process the longest data:
        # The longest data in this dataset be like 
        # ['1', '呐喊', '呐喊自序', '鲁迅', '小说', '晨报·文学旬刊', '1923/08/21',
        # '\n呐喊自序\t\t\t\n\t\t我在年青时候也曾经做过许多梦，后来大半忘...'],
        # we added fields to this data.

        if len(record) == 8:
            item = {
                "book": record[1],
                "title": record[2],
                "author": record[3],
                "type": record[4],
                "source": record[5],
                "date": record[6],
                "content": content_cleaning(record[7])
            }
            collated_data.append(item)
            continue
    
        # Process the dairy data:
        # The dairy data in this dataset be like 
        # ['甲寅日记(1914)', '三月', '\n\t\t\t\t\t\t\n\t\t三  月一日  晴。星期休息...'],
        # we added fields to this data.
      
        if len(record) == 3:
            item = {
                "book": None,
                "title": record[0],
                "author": "鲁迅",
                "type": "日记",
                "source": None,
                "date": record[1],
                "content": content_cleaning(record[2])
            }
            collated_data.append(item)
            continue

        # Process essays:
        # The essays in this dataset be like 
        # ['\n\t\t最艺术的国家\r\n我们中国的最伟大最永久，而且最普遍的“艺术”是男人\r\n扮女...'],
        # we added fields to this data.
    
        if len(record) == 1:
            for text in record:
                if "\r\n" in text:
                    text = text.split("\r\n", 1)
                    item = {
                        "book": None,
                        "title": re.sub(r'[\r\n\s]', '', text[0].strip()),
                        "author": "鲁迅",
                        "type": "杂文",
                        "source": None,
                        "date": None,
                        "content": re.sub(r'[\r\n\s]', '', text[1].strip())
                    }
                    collated_data.append(item)
                    continue

                elif "\n" in text:
                    text = text.split("\n", 2)
                    item = {
                        "book": None,
                        "title": re.sub(r'[\r\n\s]', '', text[1].strip()),
                        "author": "鲁迅",
                        "type": "杂文",
                        "source": None,
                        "date": None,
                        "content": re.sub(r'[\r\n\s]', '', text[2].strip())
                    }
                    collated_data.append(item)            
    
    return collated_data



import pandas as pd
data = pd.DataFrame(data_collating(luxun_data))
data
#data.to_csv('output/luxunV1.csv', index=False, encoding='utf-8-sig')


data = data[data['content'] != ''] # Remove the line with empty content
data = data[data['author'] != '']  # Remove the line with empty author

# Screen out the data whose author is not Lu Xun
#print(data['author'].unique())
name = ['未署名', '晏熬', '美子', '周建人乔峰', '周作人', '"老师"', '树人建人', 'ELEF.', 
 '鲁迅许广平', '培良', '它音', '荀继', '史癖', '史贲', '焉于', '石介译', '老师', '周乔峰', '白在宣', 
 '茹纯', '景宋', '周遐寿', '名知', '心印', '茅盾鲁迅', '鲁迅茅盾']
data = data[data['author'].isin(name) == False]


data_tc = data[data['type'].isin(['小说','日记','书信','杂文'])] # Select the type that best represents Lu Xun's speaking style
data_tc
# data_tc.to_csv('output/luxunV2.csv', index=False, encoding='utf-8-sig')


from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import torch
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

local_model_path = "LLM-Research/Meta-Llama-3-8B-Instruct"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)


# Create the HuggingFace pipeline for text generation
# Specify the model and tokenizer to be loaded locally

# The max_new_token parameter is typically set to 5000. However, for longer content such as novels and essays 
# with more than 5000 characters, this parameter has been set to 2000 to ensure sufficient GPU memory and to avoid 
# memory overflow issues.
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                model_kwargs={"pad_token_id": 128001},
                max_new_tokens=5000,
                )

hf_pipeline = HuggingFacePipeline(pipeline=pipe)

# Define semantic segmentation function
def semantic_segmentation_batch(texts, model):
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_msg}
                    <|start_header_id|>user<|end_header_id|>Text: {text}\nSegmentation:\n<|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>"""

    system_prompt = """您是一个语言模型助手，负责对鲁迅先生的文本进行语义分段。请仔细分析每段文字的主题、情感和风格，确保分段后的文本片段准确反映鲁迅先生的写作风格和语气。
分段时需保持原文的讽刺、批判和深刻的社会观察等特点。分段时根据内容自然断句，保证输出是中文，输出以换行符分隔的纯文本段落，不需要添加任何标题或标签。\n"""

    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(system_msg=system_prompt)
    
    try:
        # Remove input text and get result from generated text
        strip_output = RunnableLambda(lambda output: output.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1].strip())

        # Create a chain to segment text
        split_chain = {"text": lambda text: text} | prompt | hf_pipeline | strip_output | StrOutputParser()

        # Process all texts in batch
        results = []
        for text in texts:
            result = split_chain.invoke({"text": text})
            results.append(result)
        
        return results
    except Exception as e:
        return f"分段失败: {str(e)}"

# Read the dataset
data_tc = pd.read_csv("output/luxunV2.csv", encoding="utf-8")
data_tc["segment"] = None


# Batch processing
batch_size = 16  
for start_idx in range(0, len(data_tc), batch_size):
    end_idx = min(start_idx + batch_size, len(data_tc))
    batch_data = data_tc["content"].iloc[start_idx:end_idx].tolist()
    
    # Process the batch and segment the text
    segments = semantic_segmentation_batch(batch_data, hf_pipeline)

    # Update the DataFrame with the segments
    for i, segment in enumerate(segments):
        data_tc.loc[start_idx + i, "segment"] = segment


# Save the segmented data to CSV
data_tc.to_csv("output/luxunV2_seg.csv", encoding="utf-8-sig", index=False)


import pandas as pd
import langid

df = pd.read_csv('output/luxunV2_seg.csv')
df['segment'] = df['segment'].fillna('')  

# Define the language detection function
def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

# Determine whether it is a short text with little information 
# or has too many repeated characters 
def is_invalid_text(text):
    return len(set(text)) <= 30

dff = df[
    ~(df['segment'].apply(detect_language).isin(['en', 'ja']) |  # Remove English and Japanese
      df['segment'].apply(is_invalid_text))  
].drop([4489, 4537, 4542]) # Remove the rows that were not successfully segmented

dff.to_csv('output/luxunV2_fseg', encoding="utf-8-sig", index=False)


# An example of semantic segmentation results
ori_article = pd.read_csv("output/luxunV2_fseg.csv")["content"][0]
seg_article = pd.read_csv("output/luxunV2_fseg.csv")["segment"][0]
print(ori_article)
print("\n")
print(seg_article)


import pandas as pd 
import os
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.messages import HumanMessage, SystemMessage
import json
from tqdm import tqdm
import re
import time 

data_read = pd.read_csv("output/luxunV2_fseg.csv")
L = data_read.shape[0]

# Generate your api key from: https://platform.moonshot.cn/console/api-keys
os.environ["MOONSHOT_API_KEY"] = "***"

chat = MoonshotChat() 

def returnAIoutput(humanMessage,chat):
    messages = [
    SystemMessage(
        content= "请帮我把以下鲁迅的作品节选内容全部翻译成现代文,\n为分段符号, 请你对每一段进行翻译, 并输出成字典dict格式:每一段的具体格式如下: key为第几段, value同样是字典:格式为：{'input'：经过你翻译后的内容, 'output':我输入的鲁迅的原文} "
    ),
    HumanMessage(
        content= humanMessage
    ),
    ]

    output = chat.invoke(messages,max_tokens = 3000)
    return output


def returnCleanData(ai_output):
    pattern = re.compile(r'"(\d+)": {\s*"input": "(.*?)",\s*"output": "(.*?)"\s*}', re.S)

    
    result = {}
    # Match the data and fill in the dictionary
    for match in pattern.finditer(ai_output.content):
        entry_id = match.group(1)  # Extract entry ID
        input_text = match.group(2)  # Extract input content
        output_text = match.group(3)  # Extract output content
    
    # Add the results to the dictionary
        result[entry_id] = {
            "input": input_text,
            "output": output_text
        }
    return result


def returnjson_list(result):
    L = len(result)
    temp_list = []
    for i in range(1,L+1):
        temp_data = result[str(i)]
        new_dict = {}
        new_dict["instruction"] = "将下面口语化按照鲁迅风格输出"
        new_dict["input"] = temp_data["input"]
        new_dict["output"] = temp_data["output"]
        now_dict = new_dict.copy()
        temp_list.append(now_dict)
    return temp_list

error_list = []
all_json_list = []

for i in tqdm(range(259,L)):
    try:
        humanMessage = data_read.loc[i,"segment"] 
        ai_output = returnAIoutput(humanMessage,chat)
        clean_result = returnCleanData(ai_output)
        json_list = returnjson_list(clean_result)   
        all_json_list.extend(json_list)

    except Exception as e:
        error_list.append(i)
        print(f"Error occurred, skipped")
        time.sleep(60)


with open("output/luxunV2_gen.json", 'w', encoding='utf-8') as f:
    json.dump(all_json_list, f, ensure_ascii=False, indent=4)


# Examples of vernacular generation results

file_path = r"output/luxunV2_gen.json"

with open(file_path, 'r', encoding='utf-8') as f:
    merged_data = json.load(f)

for i in range(6):
    print('\n'.join([f"{k}: {v}" for k, v in merged_data[i].items()]))
    print("\n")


import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('output/luxunV2_gen.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)
    
json_data_filtered = [] 
for entry in json_data:
    input_text = entry['input']
    generate_text = entry['output']
    
    # Skip the empty text
    if not input_text or not generate_text:
        print(f"Skip the empty text: {input_text} | {generate_text}")
        continue 
        
    # The text is too short, may be meaningless text, skip
    if len(input_text) < 10:
        print(f"Skip too short text: {input_text} | {generate_text}")
        continue

    # Calculate text Similarity using TF-IDF + Cosine Similarity
    try:
        vectorizer = TfidfVectorizer().fit_transform([input_text, generate_text])
        similarity_score = cosine_similarity(vectorizer)[0, 1]
        if similarity_score  < 0.3:
            json_data_filtered.append(entry) # Remove data with high similarity

    except ValueError as e:
        print(f"Error processing pair: {input_text} | {generate_text} \n{e}")
        continue

print(f"Number of filtered data: {len(json_data_filtered)}")

with open('output/luxunV3.json', 'w', encoding='utf-8') as f:
    json.dump(json_data_filtered, f, ensure_ascii=False, indent=4)



from openai import OpenAI
import re, json
from tqdm import tqdm
import random

def return_translate_prompt(text):
    system_prompt = "你需要用不同风格的语言改写输入，尽可能给出多样化的输出。要求:\n"

    system_prompt += "1. 与输入保持相同的含义，不要过多扩展\n"
    system_prompt += "2. 尽量用不同的语言风格，例如：正式文本风格，日常聊天，散文等。\n"

    system_prompt += "Example 1:\n"
    system_prompt += "Input：又如看见兵士打车夫，在先也要愤愤的，但现在也就转念道，倘使这车夫当了兵，这兵拉了车，大抵也就这么打，便再也不放在心上了。\n"
    system_prompt += "正式文本：再比如，假如我们看到兵士打车夫，本能地会感到愤怒，但是如果思考一下，假如这个车夫成了兵，那么车夫还是会被兵打，这样的情况就再也不会让我们感到愤怒了。\n"
    system_prompt += "日常对话：嗨，听说你见过战士打马车夫？我以前也很生气，但现在我想，如果他们身份交换一下，他还会这样打的吗？也许我们就不会那么在意了。\n"

    system_prompt += f"Input：{text}\n"
    system_prompt += "请给出满足条件的5条数据,注意输出格式用1.，2.，3.，序号罗列:\n"
    
    return system_prompt


def handle_data_augmentation(text,style):
    client = OpenAI(api_key = "***", base_url = "https://api.moonshot.cn/v1")
    response=client.chat.completions.create(
        model = "moonshot-v1-8k",
        messages = [
        {"role": "user", "content": style(text)}
    ],
    temperature = 0.3,
    )
    
    output=response.choices[0].message.content
    print(output)
    if output is None:
        return []
    else:
        output_lst=output.split("\n\n")
        if output_lst[0].startswith("Output 1:"):
            output_list = [msg[11:] if msg.startswith("Output 10:") else msg[10:] for msg in output_lst]
            return output_list
        elif output_lst[0].startswith("1."):
            output_list = [msg[4:] if msg.startswith("10.") else msg[3:] for msg in output_lst]
            return output_list
        else:
            return "Unknown start number"

import json
import os

json_path = "output/luxunV3.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
new_data = []
for item in data:
    text = item["input"]
    output=handle_data_augmentation(text,return_translate_prompt)
    for output_item in output:
        new_data.append({"instruction":item["instruction"],"input":output_item,"output":item["output"]})

with open("output/luxunV3_aug.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
    print(output)
print("Amount of data after amplification：",len(new_data))


# Example of data augmentation results

file_path = r"output/luxunV3_aug.json"

with open(file_path, 'r', encoding='utf-8') as f:
    merged_data = json.load(f)

for i in range(5):
    print('\n'.join([f"{k}: {v}" for k, v in merged_data[i].items()]))
    print("\n")


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import deepspeed
from transformers import DataCollatorWithPadding

data_path = "output/luxunV3_aug.json"
dataset = load_dataset('json', data_files=data_path,split="train")
print("Data Length: ",len(dataset))
print("Data Example:",dataset[0])


eos_token = tokenizer.eos_token
pad_token = tokenizer.pad_token
tokenizer.padding_side = "right"
gemma_prompt = """<start_of_turn>user
{}: {}<end_of_turn>
<start_of_turn>model
{}<end_of_turn>"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = gemma_prompt.format(instruction, input, output) + eos_token
        texts.append(text)
    return { "text" : texts, }
pass
dataset = dataset.map(formatting_prompts_func, batched = True)

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    # Labels are identical to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

train_size = int(0.8 * len(dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))


# Set the CUDA device to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model and tokenizer
model_name_or_path = "models/gemma-2-9b-it"
base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


# Configure LoRA
lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj"],
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

# Apply LoRA to the model
model = get_peft_model(base_model, lora_config)

for param in model.parameters():
    param.requires_grad = False ## Freeze the entire model

for param in model.base_model.model.model.embed_tokens.parameters():
    param.requires_grad = True  # Unfreeze the embedding layer

for param in model.base_model.model.model.norm.parameters():
    param.requires_grad = True ## Unfreeze the normalization layer

for param in model.lm_head.parameters():
    param.requires_grad = True ## Unfreeze the output layer


from transformers import TrainingArguments

save_dictionary="models/saves_gemma2/lora/sft"
train_args = TrainingArguments(
    per_device_train_batch_size=2,  # Each GPU processes 4 examples per step.
    gradient_accumulation_steps=2,  # Gradients are accumulated over 4 steps before updating weights.
    warmup_steps=30,  # Learning rate warms up (gradually increases) for the first 30 steps.
    #max_steps=2500,  # Total number of optimization steps for training.
    num_train_epochs=3,  # Not used because `max_steps` defines the training duration.
    gradient_checkpointing=True,  # Saves memory by recomputing activations during backpropagation.
    learning_rate=1e-4,  # Base learning rate for the optimizer.
    fp16=False,  # FP16 precision is disabled (not used).
    bf16=True,  # Enables bfloat16 precision, optimized for RTX 4090 GPUs.
    logging_steps=20,  # Logs training metrics every 125 steps.
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    ddp_timeout=180000000,
    report_to="wandb",  # Disables logging to external tools like TensorBoard or WandB.
    output_dir=save_dictionary,  # Directory where model checkpoints and logs will be saved.
)

from transformers import DataCollatorForSeq2Seq
from transformers import Trainer

# Define a data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
    return_tensors="pt"
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
tokenizer.save_pretrained(save_dictionary)
model.save_pretrained(save_dictionary)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

def read_jsonl(file_path):
    log_data = []
    with open(file_path, 'r') as f:
        for line in f:
            log_data.append(json.loads(line.strip()))  # Read and parse each line of the jsonl file
    return log_data


file_path = 'models/saves_gemma2/lora/sft/trainer_log.jsonl'
log_data = read_jsonl(file_path)

# Extract training and evaluation loss from the log data
train_loss = [entry['loss'] for entry in log_data if 'loss' in entry]
eval_loss = [entry['eval_loss'] for entry in log_data if 'eval_loss' in entry]
epochs = [entry['epoch'] for entry in log_data if 'loss' in entry]

# Function to apply Simple Moving Average (SMA) for smoothing
def smooth_data_sma(y, window_size=5):
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


train_loss_sma = smooth_data_sma(train_loss, window_size=5)
eval_loss_sma = smooth_data_sma(eval_loss, window_size=5)


plt.figure(figsize=(12, 6))
# Plot training loss and the smoothed training loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', color='skyblue', marker='o')
plt.plot(epochs[2:-2], train_loss_sma, label='Smoothed Train Loss (SMA)', color='darkblue', linestyle='-', linewidth=3)  # Bold smoothed curve
plt.title('Training Loss Over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
# Plot evaluation loss and the smoothed evaluation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, eval_loss, label='Eval Loss', color='lightcoral', marker='x')
plt.plot(epochs[2:-2], eval_loss_sma, label='Smoothed Eval Loss (SMA)', color='darkred', linestyle='-', linewidth=3)  # Bold smoothed curve
plt.title('Evaluation Loss Over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()


from transformers import AutoModelForCausalLM, AutoTokenizer 
# Import the fine-tuned model and tokenizer 
lora_model_path = "models/gemma2_lora_sft" 
lora_model = AutoModelForCausalLM.from_pretrained(lora_model_path, torch_dtype=torch.bfloat16)
lora_tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

# Import the original model and tokenizer
base_model_path="models/gemma-2-9b-it"
base_model=AutoModelForCausalLM.from_pretrained(base_model_path)
base_tokenizer=AutoTokenizer.from_pretrained(base_model_path)

# Define a function to generate text with both models   
def generate_text_with_models(input_text, base_model=base_model, base_tokenizer=base_tokenizer, lora_model=lora_model, lora_tokenizer=lora_tokenizer):
    """
    Generate the output for the given input_text from both models (gemma_base and gemma_lora).

    Parameters:
    - input_text (str): The input text to be processed.
    - base_model: The original base model (not fine-tuned).
    - base_tokenizer: The tokenizer for the original base model.
    - lora_model: The LoRA fine-tuned model.
    - lora_tokenizer: The tokenizer for the LoRA fine-tuned model.

    Returns:
    - base_output_text (str): The output text from the base model (gemma_base).
    - lora_output_text (str): The output text from the LoRA fine-tuned model (gemma_lora).
    """
    # Generate output from the base model (gemma_base)
    base_input_ids = base_tokenizer(input_text, return_tensors="pt")
    base_output = base_model.generate(**base_input_ids, max_new_tokens=100)
    base_output_text = base_tokenizer.decode(base_output[0], skip_special_tokens=True)

    # Generate output from the LoRA fine-tuned model (gemma_lora)
    lora_input_ids = lora_tokenizer(input_text, return_tensors="pt")
    lora_output = lora_model.generate(**lora_input_ids, max_new_tokens=100)
    lora_output_text = lora_tokenizer.decode(lora_output[0], skip_special_tokens=True)

    result = {
        "gemma_base": print('Base Gemma:\n' + base_output_text+'\n\n'),
        "gemma_lora": print('LoRA Gemma:\n' + lora_output_text)
    }

    return result


# Try a few sentences
generate_text_with_models("把这段话按照鲁迅口吻输出：今天下雨了，我没带伞，结果被淋湿了。")


generate_text_with_models("把这段话按照鲁迅口吻输出：我整整一天都在打扫房间，虽然看起来很干净，但我敢打赌，明天它又会变得很乱。")


generate_text_with_models("把这段话按照鲁迅口吻输出：我站在古老的桥头，望着远方的山川，心中涌起了一股莫名的感情。")


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
lora_model_path = "/kaggle/input/lunxun_sft_ljy/transformers/default/1" 
lora_model = AutoModelForCausalLM.from_pretrained(lora_model_path, torch_dtype=torch.bfloat16)
lora_tokenizer = AutoTokenizer.from_pretrained(lora_model_path, use_fast=False)


def generate_text_with_lora(input_text, model=lora_model, tokenizer=lora_tokenizer, max_new_tokens=50, do_sample=True, top_p=0.7, temperature=0.7):
    # Force usage of CPU
    device = torch.device("cpu")
    
    # Load the model to CPU and use half precision
    model.to(device).half()
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)

    # Disable gradient calculation to save memory
    with torch.no_grad():
        output = model.generate(
            **input_ids, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample, 
            top_p=top_p, 
            temperature=temperature
        )

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return output_text


print(generate_text_with_lora("把这段话按照鲁迅口吻输出：我买了一个新手机，感觉很不错。",top_p=0.3))


print(generate_text_with_lora("把这段话按照鲁迅口吻输出：这个周末想去看电影，有什么好推荐的吗？", top_p=0.3))


print(generate_text_with_lora("把这段话按照鲁迅口吻输出：晚安，希望你有个美好的梦。"))

