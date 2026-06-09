# # If you want to save 4-bit models, make sure to have `bitsandbytes>=0.41.3` installed
!pip install --no-index /kaggle/input/making-wheels-of-necessary-packages-for-hf-llms/bitsandbytes-0.42.0-py3-none-any.whl --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-hf-llms
!pip install --no-index /kaggle/input/making-wheels-of-necessary-packages-for-hf-llms/accelerate-0.27.2-py3-none-any.whl --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-hf-llms
!pip install --no-index /kaggle/input/making-wheels-of-necessary-packages-for-hf-llms/transformers-4.42.3-py3-none-any.whl --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-hf-llms
!pip install --no-index /kaggle/input/making-wheels-of-necessary-packages-for-hf-llms/optimum-1.17.1-py3-none-any.whl --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-hf-llms


import pandas as pd
import numpy as np
import os
import warnings
warnings.simplefilter("ignore")

# from string import Template
from pathlib import Path
import torch
from torch import nn
from transformers import (pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, 
AutoConfig, DataCollatorWithPadding, TrainingArguments)
from tqdm.notebook import tqdm
import bitsandbytes
from accelerate import Accelerator
import optimum
from datasets import Dataset
import random
random.seed(42)
from sklearn.model_selection import train_test_split


import random

# Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)


qwen_path = '/kaggle/input/qwen2.5/transformers/7b-instruct/1'


data_path = Path('/kaggle/input/llm-prompt-recovery')

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    test = pd.read_csv(data_path / 'test.csv', index_col='id')
    test["rewrite_prompt"] = "-"
else:
    test = pd.read_csv(data_path / 'train.csv', index_col='id')
test.head()


# 4-bit 양자화 셋업
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,)


import re

def generate_output(model_path):
    # Set the random seed for reproducibility
    seed_everything(42)

    MODEL_PATH = model_path
    
    # 토크나이저 셋업
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map = "auto",
        # trust_remote_code = True,
        quantization_config = quantization_config,)
    
    predictions = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for _, row in test.iterrows():
        prompt = f"""
    Instruction: Generate a 'rewrite_prompt' that effectively transforms the given 'original_text' into the provided 'rewritten_text'.

    Here are the examples:
    - original_text: Hey there! Just a heads up: our friendly dog may bark a bit, but don't worry, he's all bark and no bite!
    - rewritten_text: Warning: Protective dog on premises. May exhibit aggressive behavior. Ensure personal safety by maintaining distance and avoiding direct contact.
    - rewrite_prompt: Improve this text to be a warning.
    
    - original_text: BIGBANG is one of those musical entities that transcends language. It’s one of those rare groups that both innovates and defines the direction a genre takes. Covering a sound that includes hip hop, R&B and electronic dance, BIGBANG and its solo acts (G-Dragon, T.O.P, Taeyang, Seungri and Daesung) have left a musical imprint that has affected the global music market. In fact, even Diplo, a household name in EDM, worked with G-Dragon and T.O.P for their rap album. So when the band announced its world tour to promote the release of its third full-length studio album MADE after a 3
    - rewritten_text: **BIGBANG's impact = (Hip Hop + R&B + EDM) x Global Influence = Global Stardom** This equation perfectly summarizes the impact of BIGBANG on the global music market. The group's sound, which incorporates elements of hip hop, R&B, and electronic dance music, has been widely influential. This unique blend of genres has not only captivated a large fan base but also inspired other artists, including EDM superstar Diplo, to collaborate with members G-Dragon and T.O.P.
    - rewrite_prompt: Imagine this as a mathematician's equation.
    
    - original_text: Please note that this page has not been updated since early 2015. I now recommend stack Rationale I’m trying to learn the Haskell programming language, and I’m tripping over the cabal-install package management system sufficiently often that it’s putting me off putting time into learning the language. Here are my requirements for a development environment I’m willing to learn in: don’t leave Debian Wheezy 32-bit; Debian Jessie will be out soon, but its versions of GHC and the Haskell platform are still very old so upgrading early is not a way out. minimise how often I have to delete all
    - rewritten_text: **Panel 1:**\n\n **Haskell Learner (in distress)**: "I'm trying to learn Haskell, but cabal-install is driving me nuts!"\n\n **Narrator:** "The problem is, the latest version of GHC and the Haskell platform for Debian Jessie are ancient. Upgrading early is a no-go."\n\n **Panel 2:**\n\n **Haskell Learner (with a frustrated look)**: "Every time I try to use cabal-install, I have to delete everything and start from scratch!"\n\n **Narrator:** "The repeated deletion is driving the learner bonkers. They need a solution."
    - rewrite_prompt: Turn this into a superhero comic strip dialogue.
    
    - original_text: December 4, 2013 - TF2 Team We've been busy these last few months, trying to settle a longstanding debate as to what "bimonthly" actually means. Some members of the team are convinced it means "once every two months." Others maintain even trying to define it would ruin the enchanting mystery of reading. Anyway, three months later we're no closer to an answer. However, we are close to releasing the second issue of Team Fortress Comics, this time as a tri-monthly publication. How close? It's here. When we last left our gripping ongoing tale, Saxton Hale had lost Mann Co. to
    - rewritten_text: ## The Enigmatic "Bimonthly"  **Good morning, esteemed team.** I know, I know, we've been stuck in the throes of a debate for the past few months, a debate as profound as the meaning of life itself - the enigmatic term "bimonthly."  Some of you believe it's a clear-cut "once every two months." Others, like myself, hold the belief that even defining it would be to strip away the enchanting mystery that surrounds its presence. Three months later, we're no closer to an answer, but I'm happy to report that we are close to releasing the second issue of Team Fortress Comics, a publication that will be released... tri-monthly. It's a bit like a bridge between the two opposing camps. We're bridging the gap between the practical and the fantastical, the concrete and the abstract. I'm not saying we've cracked the code of "bimonthly," but I am saying we're making progress. So, come join me in celebrating this milestone, the second issue of Team Fortress Comics, a testament to the power of storytelling and the enduring spirit of the red team. And who knows, maybe one day we'll finally unravel the mystery of "bimonthly."
    - rewrite_prompt: Present this as a TED talk.    

    Now, analyze the following pair of original_text and rewritten_text to generate the most suitable rewrite_prompt. Please output only the single line of generated rewrite_prompt.
    - original_text: {row['original_text']}
    - rewritten_text: {row['rewritten_text']}
    - Output:"""

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        # 모델 출력 생성
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        
        # 디코딩된 출력
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("Decoded Output:", decoded_output)
        
        # 모든 - rewrite_prompt: 추출
        matches = re.findall(r'- Output:\s*(.+)', decoded_output, re.DOTALL)
        # matches = re.findall(r'Rewrite_prompt:\s*(.+)', decoded_output, re.DOTALL) 
        if matches:
            # 마지막으로 나타난 rewrite_prompt 이후의 모든 텍스트를 선택
            rewrite_prompt = matches[-1].strip()
            # rewrite_prompt = matches.join("").split('\n\n**Prompt:**\n\n')[-1]
        else:
            # rewrite_prompt가 없는 경우 default_prompt를 사용
            rewrite_prompt = decoded_output.strip()
        
        # predictions 리스트에 추가
        predictions.append(rewrite_prompt)
        
    return predictions


preds = generate_output(qwen_path)


submission = pd.read_csv(data_path / 'sample_submission.csv')
submission["rewrite_prompt"] = preds


def preprocess_text(text):
    if not isinstance(text, str):
        return text  # Skip non-string values

    # Capture text after specific patterns
    patterns = [
        r"Rewrite prompt:\s*", r"\*\*Rewrite prompt:\*\*\s*",
        r"\*\*Rewrite_prompt:\*\*\\n\\n", r"\*\*rewrite_prompt:\*\*",
        r"\*\*Rewrite prompt:", r"rewrite_prompt:\s*", 
        r"Rewrite Prompt:\s*", r"\*\*Prompt:\*\*", 
        r"\*\*Rewrite prompt:\*\*", r"\*\*Rewrite prompt:\*\*"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[match.end():]  # Extract text after the match
            break

    # Remove special characters (only keep alphanumeric characters and spaces)
    text = re.sub(r"[^a-zA-Z0-9\s.,]", "", text)
    return text.strip()

# Apply preprocessing to each column
for col in submission.columns:
    submission[col] = submission[col].apply(preprocess_text)


submission.loc[0, 'rewrite_prompt']


submission.head()


submission.to_csv('submission.csv', index=False)




