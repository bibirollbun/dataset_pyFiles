import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '2'  # Faster HF downloads
os.environ['PYTHONIOENCODING'] = 'utf-8'       # Text encoding consistency
os.environ['PYTHONUTF8'] = '1'                 # Enable UTF-8 mode for Python

# GPU setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # for single gpu

import torch 
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")


from IPython.display import Markdown, FileLink, display, clear_output


%%capture

# Memory & performance optimization: Quantization, acceleration, efficient attention, GPU kernels
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 triton

# Unsloth fine-tuning ecosystem and parameter-efficient training
!pip install --no-deps unsloth unsloth_zoo peft trl cut_cross_entropy

# Data pipeline essentials
!pip install "datasets>=3.4.1" sentencepiece protobuf hf_transfer
!pip install -U "huggingface-hub>=0.34.0,<1.0"

# Computer vision model support (for multimodal capabilities)
!pip install --no-deps --upgrade timm

# Latest Transformers library from development branch
!pip install --no-deps git+https://github.com/huggingface/transformers.git

# Evaluation and logging tools
#!pip install evaluate sacrebleu jiwer wandb


from unsloth import FastModel 
import torch, gc

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4B-it",
    max_seq_length = 4096,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
    #max_memory={0: "6GB", "cpu": "14GB"}  
) 


# To Render response in Markdown 
from transformers import TextStreamer
from IPython.display import Markdown, display, clear_output
import torch, gc, time

class SimpleJupyterStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.generated_text = ""
        self.last_update = time.time()
    
    def put(self, value):
        if value.ndim > 1:
            if value.shape[0] > 1:
                raise ValueError("TextStreamer only supports batch size 1")
            value = value[0]
        
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        
        text = self.tokenizer.decode(value, **self.decode_kwargs)
        if text:
            self.generated_text += text
            if time.time() - self.last_update > 0.1:
                clear_output(wait=True)
                display(Markdown(f"ЁЯдЦ **Generating...**\n\n{self.generated_text}"))
                self.last_update = time.time()
    
def chat_inference(messages, model, tokenizer, max_new_tokens=2048):    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    streamer = SimpleJupyterStreamer(tokenizer, skip_prompt=True)
    
    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=64,
        top_p=0.95,
        streamer=streamer,
    )

    # Final output render
    clear_output(wait=True)
    display(Markdown(f"ЁЯдЦ **Response :**\n\n{streamer.generated_text.strip()}"))

    # Free memory
    del inputs
    torch.cuda.empty_cache()
    gc.collect()



import random
import numpy as np

# For reproducibility
set_all_seeds = lambda seed: seed is not None and [torch.manual_seed(seed), torch.cuda.manual_seed(seed), torch.cuda.manual_seed_all(seed), random.seed(seed), np.random.seed(seed)]

prompt="You are a expert assistant. Give response as clear and concise text."

# Simple utility to wrap user content in chat format
def create_message(content_list, role="user"):
    return [{"role": role, "content": content_list}]

# Adds system instruction and delegates to chat inference
def ask_multimodal(content_list, model, tokenizer, max_new_tokens=256, role="user", model_instruction=prompt, seed=73127):
    set_all_seeds(seed)
    messages = [{"role": "system",
                 "content": [{"type": "text", "text": model_instruction}]
               }] + create_message(content_list, role)
    chat_inference(messages, model, tokenizer, max_new_tokens=max_new_tokens)


# Add LoRA adapters to the model
model = FastModel.get_peft_model(
    model,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_cache = False,
    use_gradient_checkpointing=True,  # True or "unsloth" for very long context
    use_rslora=True,
    random_state=73
)


from unsloth.chat_templates import get_chat_template

# Set up the chat template for Gemma 3
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3", 
)


dataset_path = "/kaggle/input/tn-class-4-and-class-5-books-socratic-q-and-a/SynthResults"


from glob import glob
import os

page_files = glob(os.path.join(dataset_path, "**", "*.json"))


import json

data = []
for page_file in page_files:
    with open(page_file, "rb") as file:
        data.append(json.load(file))
        break


from pydantic import BaseModel
from typing import Optional

class SocraticPair(BaseModel):
    SocraticReplay: Optional[str]
    UserReplay: Optional[str]
    Answer: Optional[str] 

class SocraticConversation(BaseModel):
    Question: str
    SocraticPairs: list[SocraticPair]

class TutorData(BaseModel):
    ExtractedText: str
    ImageDescription: str
    SocraticConversations: list[SocraticConversation]


with open(page_files[0], "rb") as file:
    # print(json.load(file))
    load_data = TutorData.model_validate(json.load(file)["parsed"])


load_data.ExtractedText


load_data.ImageDescription


"""Book Content: 
7 роТро░рпБ ро╡ро┐ро╡роЪро╛ропро┐ роУро░рпН роЙро┤рпБро╡рпИ роЗропроирпНродро┐ро░родрпНродрпИ ро╡ро╛роЩрпНроХ ро╡ро┐ро░рпБроорпНрокро┐ройро╛ро░рпН. роЕроирпНрод роЙро┤рпБро╡рпИ роЗропроирпНродро┐ро░родрпНродро┐ройрпН ро╡ро┐ро▓рпИропро╛ройродрпБ тВ╣6,72,598 роЖроХрпБроорпН. роЖройро╛ро▓рпН роЕро╡ро░ро┐роЯроорпН тВ╣2,86,760 роороЯрпНроЯрпБроорпЗ роЗро░рпБроирпНродродрпБ роОройро┐ро▓рпН роЙро┤рпБро╡рпИ роЗропроирпНродро┐ро░родрпНродрпИ ро╡ро╛роЩрпНроХ роЕро╡ро░рпБроХрпНроХрпБ роОро╡рпНро╡ро│ро╡рпБ родрпКроХрпИ роХрпВроЯрпБродро▓ро╛роХ родрпЗро╡рпИрокрпНрокроЯрпБроорпН?\n8 роТро░рпБ роирокро░ро┐ройрпН роЪрпЗрооро┐рокрпНрокрпБроХрпН роХрогроХрпНроХро┐ро▓рпН тВ╣17,246 роЗро░рпБроирпНродродрпБ. роЕродро┐ро▓ро┐ро░рпБроирпНродрпБ роЕро╡ро░рпН ро╡рпАроЯрпНроЯрпБроХрпН роХроЯройрпН родро╡рогрпИроХрпНроХро╛роХ тВ╣8,891 роОроЯрпБроХрпНроХрокрпНрокроЯрпНроЯродрпБ роОройро┐ро▓рпН, роЕро╡ро░родрпБ роЪрпЗрооро┐рокрпНрокрпБроХрпН роХрогроХрпНроХро┐ро▓рпН роОро╡рпНро╡ро│ро╡рпБ родрпКроХрпИ роорпАродрооро┐ро░рпБроХрпНроХрпБроорпН?\nрокрогродрпНродро┐ро▓рпН рокрпЖро░рпБроХрпНроХро▓рпБроорпН ро╡роХрпБродрпНродро▓рпБроорпН\nроХрпВроЯро▓рпН 1\nрокро│рпНро│ро┐ рооро╛рогро╡ро░рпНроХро│рпБроХрпНроХро╛роХ роТро░рпБ рокрпБродрпНродроХ роиро┐ро▒рпБро╡ройроорпН роЕроХро░ро╛родро┐роХро│рпН роорпАродрпБ родро│рпНро│рпБрокроЯро┐ропрпИ роЕро│ро┐родрпНродродрпБ. родро│рпНро│рпБрокроЯро┐роХрпНроХрпБрокрпН рокро┐ро▒роХрпБ, роУро░рпН роЕроХро░ро╛родро┐ропро┐ройрпН ро╡ро┐ро▓рпИ тВ╣425 роЖроХрпБроорпН. роЗродройрпИ 25 рооро╛рогро╡ро░рпНроХро│рпН рокрпЖро▒ ро╡ро┐ро░рпБроорпНрокро┐ройро░рпН. роЕро╡ро░рпНроХро│рпН роЕродройрпИ ро╡ро╛роЩрпНроХ, роОро╡рпНро╡ро│ро╡рпБ рокрогроорпН родрпЗро╡рпИрокрпНрокроЯрпБроорпН?\nроЗродро▒рпНроХрпБ роиро╛роорпН, рооро╛рогро╡ро░рпНроХро│ро┐ройрпН роОрогрпНрогро┐роХрпНроХрпИропрпИропрпБроорпН роЕроХро░ро╛родро┐ропро┐ройрпН ро╡ро┐ро▓рпИропрпИропрпБроорпН рокрпЖро░рпБроХрпНроХ ро╡рпЗрогрпНроЯрпБроорпН.\nроУро╡рпНро╡рпКро░рпБ роЕроХро░ро╛родро┐ропро┐ройрпН ро╡ро┐ро▓рпИ = тВ╣425\nроЖроХро╡рпЗ, 25 роЕроХро░ро╛родро┐роХро│ро┐ройрпН ро╡ро┐ро▓рпИ = 25 ├Ч тВ╣425\n= тВ╣10,625\nроХрпВроЯро▓рпН 2\nроТро░рпБ рокройрпНройро╛роЯрпНроЯрпБрокрпН рокрпЛроЯрпНроЯро┐ропро┐ро▓рпН, роТро░рпБ рокро│рпНро│ро┐ропро┐ройрпН 8 рооро╛рогро╡ро░рпНроХро│рпН рокроЩрпНроХрпЗро▒рпНро▒рпБ тВ╣5,000роР ро░рпКроХрпНроХрокрпН рокро░ро┐роЪро╛роХ ро╡рпЖройрпНро▒ройро░рпН. роЗроирпНродродрпН родрпКроХрпИропро┐ройрпИ роЕро╡ро░рпНроХро│рпБроХрпНроХро┐роЯрпИропрпЗ рокроЩрпНроХро┐роЯрпНроЯрпБроХрпНроХрпКро│рпНро│ ро╡ро┐ро░рпБроорпНрокро┐ройро░рпН. роТро╡рпНро╡рпКро░рпБро╡ро░рпБроорпН роОро╡рпНро╡ро│ро╡рпБ рокроЩрпНроХро┐ройрпИрокрпН рокрпЖро▒рпБро╡ро░рпН?\nроЗроирпНрод роиро╛роорпН, роорпКродрпНродродрпН родрпКроХрпИропро┐ройрпИ рооро╛рогро╡ро░рпНроХро│ро┐ройрпН роОрогрпНрогро┐роХрпНроХрпИропро╛ро▓рпН ро╡роХрпБроХрпНроХ ро╡рпЗрогрпНроЯрпБроорпН.\nтВ╣5,000 ├╖ 8 = тВ╣625\nроЖроХро╡рпЗ, роТро╡рпНро╡рпКро░рпБро╡ро░ро┐ройрпН рокроЩрпНроХро╛ройродрпБ тВ╣625 роЖроХрпБроорпН.\nроОроЯрпБродрпНродрпБроХрпНроХро╛роЯрпНроЯрпБ 5.5\nроТро░рпБ роиро╛ро▒рпНроХро╛ро▓ро┐ропро┐ройрпН ро╡ро┐ро▓рпИ тВ╣520 роЖроХрпБроорпН роОройро┐ро▓рпН, 9 роиро╛ро▒рпНроХро╛ро▓ро┐роХро│ро┐ройрпН ро╡ро┐ро▓рпИ роОройрпНройро╡ро╛роХ роЗро░рпБроХрпНроХрпБроорпН?\nродрпАро░рпНро╡рпБ\nроТро░рпБ роиро╛ро▒рпНроХро╛ро▓ро┐ропро┐ройрпН ро╡ро┐ро▓рпИ = тВ╣520\n9 роиро╛ро▒рпНроХро╛ро▓ро┐роХро│ро┐ройрпН ро╡ро┐ро▓рпИ = тВ╣520 ├Ч 9\n= тВ╣4680\n46\n5th_Unit_05_Money_Term_3-TM.indd 46\n27-09-2023 14:27:10'

More Description:
роЗроирпНрод рокроЯроорпН роТро░рпБ роХрогро┐род рокро╛роЯрокрпНрокрпБродрпНродроХродрпНродро┐ройрпН роТро░рпБ рокроХрпНроХродрпНродрпИроХрпН роХро╛роЯрпНроЯрпБроХро┐ро▒родрпБ. роЗродро┐ро▓рпН ро░рпВрокро╛ропрпН роородро┐рокрпНрокрпБроХро│рпН родрпКроЯро░рпНрокро╛рой роХрогроХрпНроХрпБроХро│рпН рооро▒рпНро▒рпБроорпН роОроЯрпБродрпНродрпБроХрпНроХро╛роЯрпНроЯрпБроХро│рпН роЙро│рпНро│рой. роорпЗро▓рпЗ, роЗро░рогрпНроЯрпБ роХрпЗро│рпНро╡ро┐роХро│рпН (роХрпЗро│рпНро╡ро┐ 7 рооро▒рпНро▒рпБроорпН 8) роТро░рпБ роЪро┐ро╡рокрпНрокрпБрокрпН рокрпЖроЯрпНроЯро┐роХрпНроХрпБро│рпН роХрпКроЯрпБроХрпНроХрокрпНрокроЯрпНроЯрпБро│рпНро│рой, роЕро╡рпИ роХро┤ро┐родрпНродро▓рпН родрпКроЯро░рпНрокро╛ройро╡рпИ. рокро┐ройрпНройро░рпН, "рокрогродрпНродро┐ро▓рпН рокрпЖро░рпБроХрпНроХро▓рпБроорпН ро╡роХрпБродрпНродро▓рпБроорпН" роОройрпНро▒ родро▓рпИрокрпНрокро┐ройрпН роХрпАро┤рпН, роЕроХро░ро╛родро┐роХро│рпН ро╡ро╛роЩрпНроХрпБродро▓рпН рооро▒рпНро▒рпБроорпН рокрпЛроЯрпНроЯро┐рокрпН рокро░ро┐роЪрпБродрпН родрпКроХрпИропрпИрокрпН рокро┐ро░ро┐родрпНродро▓рпН родрпКроЯро░рпНрокро╛рой роЗро░рогрпНроЯрпБ роОроЯрпБродрпНродрпБроХрпНроХро╛роЯрпНроЯрпБроХро│рпБроЯройрпН (роХрпВроЯро▓рпН 1 рооро▒рпНро▒рпБроорпН роХрпВроЯро▓рпН 2) рокрпЖро░рпБроХрпНроХро▓рпН рооро▒рпНро▒рпБроорпН ро╡роХрпБродрпНродро▓рпН роХрогроХрпНроХрпБроХро│рпН ро╡ро┐ро│роХрпНроХрокрпНрокроЯрпНроЯрпБро│рпНро│рой. роТро░рпБ QR роХрпБро▒ро┐ропрпАроЯрпБроорпН роХрпВроЯро▓рпН 2 рокроХрпБродро┐роХрпНроХрпБ роЕро░рпБроХро┐ро▓рпН роЙро│рпНро│родрпБ. роХрпАро┤рпЗ, "роОроЯрпБродрпНродрпБроХрпНроХро╛роЯрпНроЯрпБ 5.5" роОройрпНро▒ родро▓рпИрокрпНрокро┐ро▓рпН роиро╛ро▒рпНроХро╛ро▓ро┐роХро│ро┐ройрпН ро╡ро┐ро▓рпИ родрпКроЯро░рпНрокро╛рой роТро░рпБ рокрпЖро░рпБроХрпНроХро▓рпН роХрогроХрпНроХрпБ роЕродройрпН родрпАро░рпНро╡рпБроЯройрпН роХрпКроЯрпБроХрпНроХрокрпНрокроЯрпНроЯрпБро│рпНро│родрпБ. рокроЯродрпНродро┐ройрпН роХрпАро┤рпН рокроХрпНроХ роОрогрпН 46 рооро▒рпНро▒рпБроорпН роХрпЛрокрпНрокрпБ ро╡ро┐ро╡ро░роЩрпНроХро│рпН роЕроЪрпНроЪро┐роЯрокрпНрокроЯрпНроЯрпБро│рпНро│рой.

Question:
5 рооро╛рогро╡ро░рпНроХро│рпБроХрпНроХрпБ роЕроХро░ро╛родро┐ ро╡ро╛роЩрпНроХ роОро╡рпНро╡ро│ро╡рпБ рокрогроорпН родрпЗро╡рпИрокрпНрокроЯрпБроорпН
"""


for SocraticConversation in load_data.SocraticConversations:
    print(f"Question: {SocraticConversation.Question}")
    print("-" * 20)
    for pair in SocraticConversation.SocraticPairs:
        if pair.Answer:
            print(f"Answer: {pair.Answer}")
            print("="* 20)
            continue
        print(f"SocraticReply: {pair.SocraticReplay}")
        print(f"UserReply: {pair.UserReplay}")
        print("-" * 20)


from datasets import Dataset

def transform_data(load_data):
    """
    Convert the dataset format into structured conversation format expected by Gemma-3.
    Applies Unsloth's tokenizer chat template and removes <bos> token (added later).
    """
    texts = []
    
    for SocraticConversation in load_data.SocraticConversations:
        try:
            conversation = []
            conversation.append(
                {"role": "user", "content": f"""Book Content:
{load_data.ExtractedText}

More description:
{load_data.ImageDescription}

Question:
{SocraticConversation.Question}"""
                }
            )
            for pair in SocraticConversation.SocraticPairs:
                if pair.SocraticReplay:
                    conversation.append(
                        {"role": "assistant", "content": f"{pair.SocraticReplay}"}
                    )
                if pair.UserReplay:
                    conversation.append(
                        {"role": "user", "content": f"{pair.UserReplay}"}
                    )
                if pair.Answer:
                    conversation.append(
                        {"role": "assistant", "content": f"{pair.Answer}"}
                    )
            
            # Apply chat template using tokenizer
            formatted_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            ).removeprefix('<bos>')  # BOS will be automatically handled during training
            
            texts.append(formatted_text)
        except Exception as e:
            RaiseException()
    
    return texts

dataset = []
files_with_loading_issues = []
for page_file in page_files:
    with open(page_file, "rb") as file:
        try:
            load_data = TutorData.model_validate(json.load(file)['parsed'])
            for template in transform_data(load_data):
                dataset.append({"text": template})
        except Exception as e:
            files_with_loading_issues.append(page_file)

print(f"Dataset: {len(dataset)}")
print(f"Loading Issues: {len(files_with_loading_issues)}")


dataset[0]


import json

json.dump(dataset, open("tn_books_dataset.json", "w"))


from datasets import load_dataset

whole_dataset = load_dataset("json", data_files="tn_books_dataset.json") 
print(whole_dataset)


# Get the dataset from the dict (usually the 'train' split)
dataset = whole_dataset["train"]

print(dataset)
print(type(dataset))  # This will show: <class 'datasets.arrow_dataset.Dataset'>


type(dataset)


from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)


# example
dataset[-1]["text"]


type(dataset)


# To Enable evaluation training 
use_eval_set = False 
patience = 7 


from transformers import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState
import torch
from typing import Dict, Any

class TrainingLossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int = 10, min_delta: float = 0.001, min_steps: int = 20):
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_step = 0
        
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        if logs is None or logs.get('loss') is None:
            return
            
        current_loss = logs.get('loss')
        
        if state.global_step < self.min_steps:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_step = state.global_step
                print(f"ЁЯОп New best training loss: {current_loss:.6f} at step {state.global_step} (warmup phase)")
            else:
                if state.global_step > 1:
                    print(f"No improvement at step {state.global_step} (warmup phase, < min_steps ({self.min_steps}))")
            return
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            self.best_step = state.global_step
            print(f"ЁЯОп New best training loss: {current_loss:.6f} at step {state.global_step}")
        else:
            self.patience_counter += 1
            if self.patience_counter <= 3:
                print(f"No improvement for {self.patience_counter}/{self.early_stopping_patience} steps")
                        
        if self.patience_counter >= self.early_stopping_patience:
            print(f"тП╣я╕П Early stopping at step {state.global_step}. Best loss: {self.best_loss:.6f}")
            control.should_training_stop = True

class FinalStepCallback(TrainerCallback):
    def __init__(self, use_eval_set: bool = False):
        self.use_eval_set = use_eval_set
        self.step_losses = []
        self.final_logged = False
    
    def on_step_end(self, args, state, control, **kwargs):
        # Force logging for final step if not already logged
        if (state.global_step == args.max_steps and 
            state.global_step % args.logging_steps != 0 and 
            not self.final_logged):
            control.should_log = True
            self.final_logged = True
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            step_loss = logs.get('loss')
            if step_loss is not None:
                self.step_losses.append({'step': state.global_step, 'loss': step_loss})
            
            print(f"\n=== Step {state.global_step} Results ===")
            for key, value in logs.items():
                if key == 'train_loss':  # Skip the average train_loss
                    continue
                if isinstance(value, float):
                    print(f"{key}: {value:.6f}")
                else:
                    print(f"{key}: {value}")
            print("-" * 40)
    
    def on_train_end(self, args, state, control, **kwargs):
        if not self.step_losses:
            return
            
        trainer = kwargs.get('trainer')
        first_loss = self.step_losses[0]['loss']
        final_loss = self.step_losses[-1]['loss']
        best_loss = min(entry['loss'] for entry in self.step_losses)
        improvement = first_loss - final_loss
        improvement_pct = (improvement / first_loss) * 100
        
        print("\n" + "="*50)
        print("ЁЯОп FINAL MODEL EVALUATION")
        print("="*50)
        print(f"ЁЯУИ Training Summary:")
        print(f"   Initial Loss: {first_loss:.6f}")
        print(f"   Last Step Loss: {final_loss:.6f}")
        print(f"   Best Loss: {best_loss:.6f}")
        print(f"   Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
        print(f"   Total Steps: {len(self.step_losses)}")
        
        if len(self.step_losses) >= 5:
            print(f"\nЁЯУК Loss Progression (Last 5 Steps):")
            for entry in self.step_losses[-5:]:
                print(f"   Step {entry['step']:3d}: {entry['loss']:.6f}")
        
        if trainer and self.use_eval_set and trainer.eval_dataset:
            try:
                eval_results = trainer.evaluate()
                print(f"\nЁЯФН Final Evaluation Results:")
                for key, value in eval_results.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.6f}")
            except:
                pass
        
        print("="*50)

def setup_callbacks(use_eval_set=use_eval_set, patience=patience):
    callbacks = []
    if use_eval_set:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    else:
        callbacks.append(TrainingLossEarlyStoppingCallback(early_stopping_patience=patience))
    callbacks.append(FinalStepCallback(use_eval_set=use_eval_set))
    return callbacks



from trl import SFTConfig, SFTTrainer
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback
import math 

# Dataset splitting logic
if use_eval_set:
    split_dataset = dataset.train_test_split(test_size=0.1, seed=73)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
else:
    train_dataset = dataset
    eval_dataset = None

# Auto-calculated training parameters
dataset_size = len(train_dataset)

# Hardware detection
gpu_stats = torch.cuda.get_device_properties(0)
available_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024 - torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 1)
size_factor, memory_factor = min(1.0, dataset_size / 100), min(1.0, available_memory / 8)

# Batch configuration
batch_size = max(1, min(int(1 + 3 * size_factor), int(1 + 7 * memory_factor)))
accumulation_steps = max(1, int(14 / batch_size)) # Maintain ~14 effective batch size
effective_batch_size = batch_size * accumulation_steps

# Training steps
steps_per_epoch = max(1, dataset_size // effective_batch_size)
epoch_scale = max(3, min(10, int(3 + 4 * size_factor)))
max_steps = max(20, min(1000, steps_per_epoch * epoch_scale))

# Learning rate
lr_scale = 0.3 + 0.9 * size_factor
base_lr = 5e-4 * lr_scale
dataset_scale = math.sqrt(min(dataset_size, 200) / 200)
adaptive_lr = max(1e-5, min(3e-3, base_lr * dataset_scale))

# Intervals and scheduling
log_interval, eval_interval = max(1, max_steps // 20), max(1, steps_per_epoch)
warmup_ratio = 0.2 - 0.1 * size_factor
warmup_steps = max(1, int(max_steps * warmup_ratio))
#patience = max(3, int(max_steps // (10 + 5 * size_factor)))

# Regularization
weight_decay = max(0.001, 0.01 - 0.009 * size_factor)
max_grad_norm = max(0.3, 1.0 - 0.7 * (1 - size_factor))
scheduler_type = "linear"

# Initialize the trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, 
    dataset_text_field = "text",
    packing = False,
    callbacks = setup_callbacks(use_eval_set=use_eval_set, patience=patience), 
    
    args = SFTConfig(
        # Training config
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = accumulation_steps,
        **{"max_steps": max_steps},
        
        # Learning rate scheduling
        learning_rate = adaptive_lr,
        warmup_steps = warmup_steps,
        optim = "adafactor", # More adaptive
        weight_decay = weight_decay,
        lr_scheduler_type = scheduler_type,
        
        # Performance
        #dataset_num_proc = 1, # good for limited hardware
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),        
        dataloader_pin_memory = True,
        max_grad_norm = max_grad_norm, 
        dataloader_drop_last = True,
        remove_unused_columns = True,

        # Checkpointing
        save_steps = log_interval,
        save_total_limit=patience + 1,        
        save_strategy = "steps",
        output_dir = "outputs",

        # Evaluation settings (conditional)
        **({
            "do_eval": True,            
            "eval_steps": eval_interval,
            "eval_strategy": "steps",            
            "per_device_eval_batch_size": 1,  # Smaller batch size for evaluation                               
            "eval_accumulation_steps": 1,       
            "greater_is_better": False,          
            "metric_for_best_model": "eval_loss",
            "load_best_model_at_end": True,
        } if use_eval_set else {
            "eval_strategy": "no",
        }),

        # Logging
        seed = 73,
        logging_steps = log_interval,
        logging_first_step = True,
        disable_tqdm = False,
        report_to = "none",  # Set this to "wandb" if using Weights & Biases
    ),
)

# Configuration summary
constraint = "Memory" if batch_size == int(1 + 7 * memory_factor) else "Dataset"
print(f"{'='*70}")
print(f"TRAINING CONFIGURATION SUMMARY")
print(f"Dataset: {dataset_size} samples | GPU: {available_memory}GB | Factors: size={size_factor:.2f}, memory={memory_factor:.2f}")
print(f"Batch: {batch_size} x {accumulation_steps} = {effective_batch_size} (limited by {constraint})")
print(f"Training: {max_steps} steps ({epoch_scale} epochs, {steps_per_epoch} steps/epoch)")
print(f"Learning: {adaptive_lr:.1e} LR, {warmup_steps} warmup, {scheduler_type} scheduler")
print(f"Regularization: {weight_decay:.4f} weight decay, {max_grad_norm:.1f} grad norm")
print(f"Monitoring: log every {log_interval}, eval every {eval_interval}, patience {patience}")
print(f"{'='*70}")



# Apply response-only training
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(    
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
    num_proc         = 1,
)


tokenizer.decode(trainer.train_dataset[8]["input_ids"])


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


from unsloth import unsloth_train
trainer_stats = unsloth_train(trainer) # trainer.train()


GB_CONVERSION = 1024 ** 3
SECONDS_TO_MINUTES = 60
    
# Memory calculations
used_memory_gb = torch.cuda.max_memory_reserved() / GB_CONVERSION
used_memory_for_training_gb = used_memory_gb - start_gpu_memory
used_percentage = (used_memory_gb / max_memory) * 100
training_percentage = (used_memory_for_training_gb / max_memory) * 100
    
# Time calculations
runtime_seconds = trainer_stats.metrics['train_runtime']
runtime_minutes = runtime_seconds / SECONDS_TO_MINUTES
    
print("TRAINING STATISTICS")
print("=" * 50)
print(f"Training time: {runtime_seconds:.1f} seconds ({runtime_minutes:.2f} minutes)")
print(f"Peak memory usage: {used_memory_gb:.3f} GB ({used_percentage:.1f}% of max)")
print(f"Memory for training: {used_memory_for_training_gb:.3f} GB ({training_percentage:.1f}% of max)")
print("=" * 50)


test_query = """Book Content: 
7 роТро░рпБ ро╡ро┐ро╡роЪро╛ропро┐ роУро░рпН роЙро┤рпБро╡рпИ роЗропроирпНродро┐ро░родрпНродрпИ ро╡ро╛роЩрпНроХ ро╡ро┐ро░рпБроорпНрокро┐ройро╛ро░рпН. роЕроирпНрод роЙро┤рпБро╡рпИ роЗропроирпНродро┐ро░родрпНродро┐ройрпН ро╡ро┐ро▓рпИропро╛ройродрпБ тВ╣6,72,598 роЖроХрпБроорпН. роЖройро╛ро▓рпН роЕро╡ро░ро┐роЯроорпН тВ╣2,86,760 роороЯрпНроЯрпБроорпЗ роЗро░рпБроирпНродродрпБ роОройро┐ро▓рпН роЙро┤рпБро╡рпИ роЗропроирпНродро┐ро░родрпНродрпИ ро╡ро╛роЩрпНроХ роЕро╡ро░рпБроХрпНроХрпБ роОро╡рпНро╡ро│ро╡рпБ родрпКроХрпИ роХрпВроЯрпБродро▓ро╛роХ родрпЗро╡рпИрокрпНрокроЯрпБроорпН?\n8 роТро░рпБ роирокро░ро┐ройрпН роЪрпЗрооро┐рокрпНрокрпБроХрпН роХрогроХрпНроХро┐ро▓рпН тВ╣17,246 роЗро░рпБроирпНродродрпБ. роЕродро┐ро▓ро┐ро░рпБроирпНродрпБ роЕро╡ро░рпН ро╡рпАроЯрпНроЯрпБроХрпН роХроЯройрпН родро╡рогрпИроХрпНроХро╛роХ тВ╣8,891 роОроЯрпБроХрпНроХрокрпНрокроЯрпНроЯродрпБ роОройро┐ро▓рпН, роЕро╡ро░родрпБ роЪрпЗрооро┐рокрпНрокрпБроХрпН роХрогроХрпНроХро┐ро▓рпН роОро╡рпНро╡ро│ро╡рпБ родрпКроХрпИ роорпАродрооро┐ро░рпБроХрпНроХрпБроорпН?\nрокрогродрпНродро┐ро▓рпН рокрпЖро░рпБроХрпНроХро▓рпБроорпН ро╡роХрпБродрпНродро▓рпБроорпН\nроХрпВроЯро▓рпН 1\nрокро│рпНро│ро┐ рооро╛рогро╡ро░рпНроХро│рпБроХрпНроХро╛роХ роТро░рпБ рокрпБродрпНродроХ роиро┐ро▒рпБро╡ройроорпН роЕроХро░ро╛родро┐роХро│рпН роорпАродрпБ родро│рпНро│рпБрокроЯро┐ропрпИ роЕро│ро┐родрпНродродрпБ. родро│рпНро│рпБрокроЯро┐роХрпНроХрпБрокрпН рокро┐ро▒роХрпБ, роУро░рпН роЕроХро░ро╛родро┐ропро┐ройрпН ро╡ро┐ро▓рпИ тВ╣425 роЖроХрпБроорпН. роЗродройрпИ 25 рооро╛рогро╡ро░рпНроХро│рпН рокрпЖро▒ ро╡ро┐ро░рпБроорпНрокро┐ройро░рпН. роЕро╡ро░рпНроХро│рпН роЕродройрпИ ро╡ро╛роЩрпНроХ, роОро╡рпНро╡ро│ро╡рпБ рокрогроорпН родрпЗро╡рпИрокрпНрокроЯрпБроорпН?\nроЗродро▒рпНроХрпБ роиро╛роорпН, рооро╛рогро╡ро░рпНроХро│ро┐ройрпН роОрогрпНрогро┐роХрпНроХрпИропрпИропрпБроорпН роЕроХро░ро╛родро┐ропро┐ройрпН ро╡ро┐ро▓рпИропрпИропрпБроорпН рокрпЖро░рпБроХрпНроХ ро╡рпЗрогрпНроЯрпБроорпН.\nроУро╡рпНро╡рпКро░рпБ роЕроХро░ро╛родро┐ропро┐ройрпН ро╡ро┐ро▓рпИ = тВ╣425\nроЖроХро╡рпЗ, 25 роЕроХро░ро╛родро┐роХро│ро┐ройрпН ро╡ро┐ро▓рпИ = 25 ├Ч тВ╣425\n= тВ╣10,625\nроХрпВроЯро▓рпН 2\nроТро░рпБ рокройрпНройро╛роЯрпНроЯрпБрокрпН рокрпЛроЯрпНроЯро┐ропро┐ро▓рпН, роТро░рпБ рокро│рпНро│ро┐ропро┐ройрпН 8 рооро╛рогро╡ро░рпНроХро│рпН рокроЩрпНроХрпЗро▒рпНро▒рпБ тВ╣5,000роР ро░рпКроХрпНроХрокрпН рокро░ро┐роЪро╛роХ ро╡рпЖройрпНро▒ройро░рпН. роЗроирпНродродрпН родрпКроХрпИропро┐ройрпИ роЕро╡ро░рпНроХро│рпБроХрпНроХро┐роЯрпИропрпЗ рокроЩрпНроХро┐роЯрпНроЯрпБроХрпНроХрпКро│рпНро│ ро╡ро┐ро░рпБроорпНрокро┐ройро░рпН. роТро╡рпНро╡рпКро░рпБро╡ро░рпБроорпН роОро╡рпНро╡ро│ро╡рпБ рокроЩрпНроХро┐ройрпИрокрпН рокрпЖро▒рпБро╡ро░рпН?\nроЗроирпНрод роиро╛роорпН, роорпКродрпНродродрпН родрпКроХрпИропро┐ройрпИ рооро╛рогро╡ро░рпНроХро│ро┐ройрпН роОрогрпНрогро┐роХрпНроХрпИропро╛ро▓рпН ро╡роХрпБроХрпНроХ ро╡рпЗрогрпНроЯрпБроорпН.\nтВ╣5,000 ├╖ 8 = тВ╣625\nроЖроХро╡рпЗ, роТро╡рпНро╡рпКро░рпБро╡ро░ро┐ройрпН рокроЩрпНроХро╛ройродрпБ тВ╣625 роЖроХрпБроорпН.\nроОроЯрпБродрпНродрпБроХрпНроХро╛роЯрпНроЯрпБ 5.5\nроТро░рпБ роиро╛ро▒рпНроХро╛ро▓ро┐ропро┐ройрпН ро╡ро┐ро▓рпИ тВ╣520 роЖроХрпБроорпН роОройро┐ро▓рпН, 9 роиро╛ро▒рпНроХро╛ро▓ро┐роХро│ро┐ройрпН ро╡ро┐ро▓рпИ роОройрпНройро╡ро╛роХ роЗро░рпБроХрпНроХрпБроорпН?\nродрпАро░рпНро╡рпБ\nроТро░рпБ роиро╛ро▒рпНроХро╛ро▓ро┐ропро┐ройрпН ро╡ро┐ро▓рпИ = тВ╣520\n9 роиро╛ро▒рпНроХро╛ро▓ро┐роХро│ро┐ройрпН ро╡ро┐ро▓рпИ = тВ╣520 ├Ч 9\n= тВ╣4680\n46\n5th_Unit_05_Money_Term_3-TM.indd 46\n27-09-2023 14:27:10'

More Description:
роЗроирпНрод рокроЯроорпН роТро░рпБ роХрогро┐род рокро╛роЯрокрпНрокрпБродрпНродроХродрпНродро┐ройрпН роТро░рпБ рокроХрпНроХродрпНродрпИроХрпН роХро╛роЯрпНроЯрпБроХро┐ро▒родрпБ. роЗродро┐ро▓рпН ро░рпВрокро╛ропрпН роородро┐рокрпНрокрпБроХро│рпН родрпКроЯро░рпНрокро╛рой роХрогроХрпНроХрпБроХро│рпН рооро▒рпНро▒рпБроорпН роОроЯрпБродрпНродрпБроХрпНроХро╛роЯрпНроЯрпБроХро│рпН роЙро│рпНро│рой. роорпЗро▓рпЗ, роЗро░рогрпНроЯрпБ роХрпЗро│рпНро╡ро┐роХро│рпН (роХрпЗро│рпНро╡ро┐ 7 рооро▒рпНро▒рпБроорпН 8) роТро░рпБ роЪро┐ро╡рокрпНрокрпБрокрпН рокрпЖроЯрпНроЯро┐роХрпНроХрпБро│рпН роХрпКроЯрпБроХрпНроХрокрпНрокроЯрпНроЯрпБро│рпНро│рой, роЕро╡рпИ роХро┤ро┐родрпНродро▓рпН родрпКроЯро░рпНрокро╛ройро╡рпИ. рокро┐ройрпНройро░рпН, "рокрогродрпНродро┐ро▓рпН рокрпЖро░рпБроХрпНроХро▓рпБроорпН ро╡роХрпБродрпНродро▓рпБроорпН" роОройрпНро▒ родро▓рпИрокрпНрокро┐ройрпН роХрпАро┤рпН, роЕроХро░ро╛родро┐роХро│рпН ро╡ро╛роЩрпНроХрпБродро▓рпН рооро▒рпНро▒рпБроорпН рокрпЛроЯрпНроЯро┐рокрпН рокро░ро┐роЪрпБродрпН родрпКроХрпИропрпИрокрпН рокро┐ро░ро┐родрпНродро▓рпН родрпКроЯро░рпНрокро╛рой роЗро░рогрпНроЯрпБ роОроЯрпБродрпНродрпБроХрпНроХро╛роЯрпНроЯрпБроХро│рпБроЯройрпН (роХрпВроЯро▓рпН 1 рооро▒рпНро▒рпБроорпН роХрпВроЯро▓рпН 2) рокрпЖро░рпБроХрпНроХро▓рпН рооро▒рпНро▒рпБроорпН ро╡роХрпБродрпНродро▓рпН роХрогроХрпНроХрпБроХро│рпН ро╡ро┐ро│роХрпНроХрокрпНрокроЯрпНроЯрпБро│рпНро│рой. роТро░рпБ QR роХрпБро▒ро┐ропрпАроЯрпБроорпН роХрпВроЯро▓рпН 2 рокроХрпБродро┐роХрпНроХрпБ роЕро░рпБроХро┐ро▓рпН роЙро│рпНро│родрпБ. роХрпАро┤рпЗ, "роОроЯрпБродрпНродрпБроХрпНроХро╛роЯрпНроЯрпБ 5.5" роОройрпНро▒ родро▓рпИрокрпНрокро┐ро▓рпН роиро╛ро▒рпНроХро╛ро▓ро┐роХро│ро┐ройрпН ро╡ро┐ро▓рпИ родрпКроЯро░рпНрокро╛рой роТро░рпБ рокрпЖро░рпБроХрпНроХро▓рпН роХрогроХрпНроХрпБ роЕродройрпН родрпАро░рпНро╡рпБроЯройрпН роХрпКроЯрпБроХрпНроХрокрпНрокроЯрпНроЯрпБро│рпНро│родрпБ. рокроЯродрпНродро┐ройрпН роХрпАро┤рпН рокроХрпНроХ роОрогрпН 46 рооро▒рпНро▒рпБроорпН роХрпЛрокрпНрокрпБ ро╡ро┐ро╡ро░роЩрпНроХро│рпН роЕроЪрпНроЪро┐роЯрокрпНрокроЯрпНроЯрпБро│рпНро│рой.

Question:
2 рооро╛рогро╡ро░рпНроХро│рпБроХрпНроХрпБ роЕроХро░ро╛родро┐ ро╡ро╛роЩрпНроХ роОро╡рпНро╡ро│ро╡рпБ рокрогроорпН родрпЗро╡рпИрокрпНрокроЯрпБроорпН?
"""


# calling for text generation
ask_multimodal([
    {"type": "text", "text": test_query}
], model, tokenizer, max_new_tokens=300)


# calling for text generation
ask_multimodal([
    {"type": "text", "text": "Who are you?"}
], model, tokenizer, max_new_tokens=300)





# Prevents tokenizer conflicts when running shell commands like !wget, !python
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# to save lora adapters (~100mb) 
model.save_pretrained("gemma-3-lora-adapters")
tokenizer.save_pretrained("gemma-3-lora-adapters")

import shutil
folder_path = "./gemma-3-lora-adapters"
zip_path = f"{folder_path}.zip"
shutil.make_archive(folder_path, 'zip', folder_path)

from IPython.display import FileLink
FileLink(zip_path)


import shutil

# To Remove outputs directory to free up disk space before merging
def cleanup_directory(output_dir="outputs"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"{output_dir} directory removed successfully")


# Merge to 16bit
model_dir = "gemma-3-finetune"
cleanup_directory(model_dir)
model.save_pretrained_merged(model_dir, tokenizer, save_method="merged_16bit")


import shutil, os
import urllib.request
from IPython.display import clear_output, FileLink

q_type = "Q4_K_M"

try:
    model.save_pretrained_gguf(model_dir, quantization_type=q_type)
    print("Model saved successfully using save_pretrained_gguf")
    
except Exception as e:
    print("Falling back to manual conversion...")
    
    # Download the llama.cpp zip file
    url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b5137.zip"
    zip_filename = "b5137.zip"
    urllib.request.urlretrieve(url, zip_filename)
    shutil.unpack_archive(zip_filename, extract_dir=".")
    os.remove(zip_filename)
    clear_output()
    
    # Configuration
    quant_type = q_type.lower()
    model_name = model_dir
    output_file = f"{model_name}.{quant_type.upper()}.gguf"
    converter_path = "./llama.cpp-b5137/convert_hf_to_gguf.py"
    
    print(f"Converting '{model_name}' to GGUF: {output_file} ...")
    !python "$converter_path" --outfile "$output_file" --outtype "$quant_type" "$model_name"
    
FileLink(f"./{model_dir}.{q_type}.gguf")


import shutil, os
import urllib.request
from IPython.display import clear_output, FileLink

q_type = "Q8_0"

try:
    model.save_pretrained_gguf(model_dir, quantization_type=q_type)
    print("Model saved successfully using save_pretrained_gguf")
    
except Exception as e:
    print("Falling back to manual conversion...")
    
    # Download the llama.cpp zip file
    url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b5137.zip"
    zip_filename = "b5137.zip"
    urllib.request.urlretrieve(url, zip_filename)
    shutil.unpack_archive(zip_filename, extract_dir=".")
    os.remove(zip_filename)
    clear_output()
    
    # Configuration
    quant_type = q_type.lower()
    model_name = model_dir
    output_file = f"{model_name}.{quant_type.upper()}.gguf"
    converter_path = "./llama.cpp-b5137/convert_hf_to_gguf.py"
    
    print(f"Converting '{model_name}' to GGUF: {output_file} ...")
    !python "$converter_path" --outfile "$output_file" --outtype "$quant_type" "$model_name"
    
FileLink(f"./{model_dir}.{q_type}.gguf")




