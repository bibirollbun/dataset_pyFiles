# install tools to perform training
!pip install -q -U transformers=="4.47.1" # allows to load Gemma model 
!pip install -q -U tokenizers==0.21.0
!pip install -q accelerate==0.34.2
!pip install -q bitsandbytes==0.45.0 # run the models in 4-bit precision to fit the model on GPU memory
!pip install -q peft==0.14.0


# !apt-get install espeak-ng -y
!sudo apt install espeak -y


#!apt-get install espeak libespeak1 libespeak-dev


!pip install -q kagglehub # to upload model to Kaggle Models


# install nlp and poetry tools
!pip install -q gTTS==2.5.4 # synthesize sound from text
!pip install -q wordcloud==1.9.3 # generate a word cloud
!pip install -q rhymetagger==0.2.9 # rhyme recognition
!pip install -q ipa-rhyming==1.1.0 # to evaluate text rhyme


# install NLP tools
!pip install -q -U nameparser==1.1.3
!pip install -q pycountry==24.6.1
!pip install -q transliterate==1.10.2
!pip install -q prosodic==2.1.2
!pip install -q -U nltk


import os
# tells PyTorch which GPUs to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# tells the Hugging Face Transformers library whether to parallelize the tokenization process
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

from enum import Enum
import string
import shutil
import datetime
import json
import random
import time
import uuid
from pathlib import Path

import nltk
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

import wandb

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import pycountry
import seaborn as sns

from tqdm import tqdm
import bitsandbytes as bnb

import torch
import torch.nn as nn


import transformers
from datasets import Dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging, Trainer)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)

# different NLP tools
import prosodic
from transliterate import translit
from nameparser import HumanName
from rhymetagger import RhymeTagger
import re

from IPython.core.display import display, HTML

print(f"pytorch version {torch.__version__}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"working on {device}")


# configure wandb to log training
# make sure you have the WANDB secret value attached to the notebook
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_value = user_secrets.get_secret("WANDB")
    wandb.login(key=secret_value)
    anony = None
except:
    anony = "must"
    print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


# type of training prompt
class PromptType(str, Enum):
    FullPrompt = 'full_prompt'
    ShortPrompt = 'short_summary'
    NoSummaryPrompt = 'no_summary'
    InstructionInpResponse = 'instruction_inp_response'
    InstructionInpResponseNoSummary = 'instruction_inp_response_no_summary'

# use training/inference configuration to control the fine-tuning process:
class cfg:
    seed = 42 # to reproduce the results on every run

    # parameters for wandb logging
    wandb_project = "gemma2-lora-peft-poem-writer_KAGGLE"
    wandb_run_name = "gemma2-lora-peft-poem-writer_KAGGLE"

    dataset_percentage = 1.0 # 1.0 means we train the full dataset
    dataset_name = 'poetree_poetryfoundation_v1'
    poem_lang = 'en'  # poem dataset language (en, ru)
    
    max_body_token_len = 224 # max token length of the poem text (make it lower for for memory purposes)
    max_seq_length = 512 # max token length of the prompt sequence
    logging_steps = 10 # logging frequency while training
    save_strategy = "epoch"

    run_train = False # should we train OR just use the fine-tuned model?
    prompt_type = PromptType.FullPrompt # short or long prompt for training

    # In LoRA, we kept original weights of model frozen and inject the small new trainable parameters with low dimensions matrices.
    lr_value = 2e-4  # 5e-5  # 5e-5 # controls the learning rate for the AdamW optimizer
    weight_decay = 0.01  # controls the weight decay for the AdamW optimizer

    epoch = 3  # number of training epochs
    batch_size = 2 # batch size for training
    gradient_accumulation_steps = 2   # global batch size is 2x2 = 4 (kaggle)

    # gemma2 pre-trained model (version 2)
    base_model_name = '/kaggle/input/gemma-2/transformers/gemma-2-2b/2'
    
    # fine-tuned model
    ft_base_model_dir = '/kaggle/input/gemma2-poem-rhyme-synthesizer/transformers'
    ft_ru_model_dir = f'{ft_base_model_dir}/gemma2_2b_ru_2025_01_13_10_48_01_970119/1'
    ft_ru_model_name = f'{ft_ru_model_dir}/peft_model_1736811878.0543547'

    # folders to keep the generated data
    uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    work_dir = '/kaggle/working'
    out_dir = f'{work_dir}/weights_{uniq_filename}'  # directory for saving the fine-tuning model
    log_dir = f'{out_dir}/logs'

    # use 4-bit quantization to address memory issues
    bnb_quant_type = 'nf4'
    bnb_compute_quant_type = getattr(torch, "float16")


# utility functions
def save_config(save_dir, cls_to_write):
    """Save training/inference parameters to config file"""
    cfg_file_output = os.path.join(save_dir, 'config.json')
    print('Save config', cfg_file_output)
    with open(cfg_file_output, 'w') as fp:
        json.dump({x: dict(cls_to_write.__dict__)[x] for x in dict(cls_to_write.__dict__) if
                   not x.startswith('_') and x != 'device' and x != 'bnb_compute_quant_type'}, fp, indent=4)


def random_seed(SEED):
    # set seed for all random processes involved in the training,
    # ensuring that our fine-tuning results are reproducible.
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True


import ctypes, gc, torch
libc = ctypes.CDLL("libc.so.6")
def clear_memory():
    """Clear RAM/GPU memory"""
    libc.malloc_trim(0)
    torch.cuda.empty_cache()
    gc.collect()


# utilities
def clean_title_col(s):
    # Removes all types of quotes from a string
    s = re.sub(r'[\'\"â€œâ€�â€˜â€™Â«Â»]', '', s)
    s = s.lower()
    s = s.strip()  # remove new lines start/end
    s = s.capitalize()  # capitalize for cosistency
    # remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    s = s.translate(translator)
    return s


def clean_poem_text(txt, author=None):
    txt = txt.strip()
    # Replace triple newlines with double newlines
    txt = re.sub(r'\n{3,}', '\n\n', txt)

    if author is not None:
        txt_lines = [line.strip() for line in txt.split("\n")]
        # remove author name if it is in the poem text
        full_name1 = HumanName(author)
        full_name2 = HumanName(txt_lines[0])
        is_first_line_author_name = full_name1 == full_name2
        if is_first_line_author_name:
            txt_lines = txt_lines[1:]
        txt = '\n'.join(txt_lines).strip()  # poem text
    return txt


def cut_off_summary(summary, max_sentences=3):
    # Tokenize into sentences
    sentences = nltk.tokenize.sent_tokenize(summary)
    # Cut off after the second sentence
    cutoff_sentences = sentences[:max_sentences]  # max 3 sentences
    # Join the sentences back together
    summary_cutoff = " ".join(cutoff_sentences)
    return summary_cutoff


def format_prompt_func(example, prompt_template):
    title = example["title"]
    author = example["author"]
    text = example["body"]
    summary = example["summary"]
    summary = cut_off_summary(summary, max_sentences=3)
    title = clean_title_col(title)
    text = clean_poem_text(text, author=author)
    return prompt_template.format(title=title, author=author, summary=summary, text=text)


def format_no_summary_prompt_func(example, prompt_template):
    title = example["title"]
    author = example["author"]
    text = example["body"]
    title = clean_title_col(title)
    text = clean_poem_text(text, author=author)
    return prompt_template.format(title=title, author=author, text=text)


def format_short_prompt_func(example, prompt_template):
    text = example["body"]
    summary = example["summary"]
    summary = cut_off_summary(summary, max_sentences=3)
    text = clean_poem_text(text, author=None)
    return prompt_template.format(summary=summary, text=text)


def get_prompt_template():
    # prompt
    if cfg.prompt_type is PromptType.ShortPrompt:
        poem_template_params = """
### Summary:
{summary}

### Poem:
{text}"""
        sent1 = "Write a poem inspired by the summary provided below, ensuring it aligns closely with the outlined details."
        prompt_template = (
            f"{sent1}\n{poem_template_params}".strip()
        )
    elif cfg.prompt_type is PromptType.FullPrompt:
        poem_template_params = """
### Title:
{title}

### Poet:
{author}

### Summary:
{summary}

### Poem:
{text}"""
        sent1 = "Your task is to compose a poem based on the title, aligning with the style of the poet and the brief summary provided below."
        sent2 = "The poem should reflect the outlined summary while capturing the poet's unique tone, language, and themes."
        prompt_template = (
            f"{sent1} {sent2}\n{poem_template_params}".strip()
        )
    elif cfg.prompt_type is PromptType.NoSummaryPrompt:
        poem_template_params = """
### Title:
{title}

### Poet:
{author}

### Poem:
{text}"""
        sent1 = "Your task is to compose a poem based on the title, aligning with the style of the poet provided below."
        sent2 = "The poem should reflect the poet's unique tone, language, and themes."
        prompt_template = (
            f"{sent1} {sent2}\n{poem_template_params}".strip()
        )
    else:
        raise Exception('Invalid type', cfg.prompt_type)
    return prompt_template


def get_inference_template():
    poem_template_params = """
        ### Title:
        {title}

        ### Poet:
        {author}

        ### Summary:
        {summary}

        ### Poem:
        {text}"""
    sent1 = "Write a poem based on the title, aligning with the style of the poet and the brief summary without adding any comments or external information."
    sent2 = "The poem should reflect the outlined summary provided below while capturing the poet's unique tone, language, and themes."
    prompt_template = (
        f"{sent1} {sent2}\n{poem_template_params}".strip()
    )
    return prompt_template


tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)


# load pre-trained Gemma2 model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type=cfg.bnb_quant_type,
    bnb_4bit_compute_dtype=cfg.bnb_compute_quant_type
)

model = AutoModelForCausalLM.from_pretrained(
    cfg.base_model_name,
    quantization_config=bnb_config,
    device_map=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model = model.to(device)


def generate_text(model, tokenizer, prompt, device=None, cut_off_prompt=True, max_length=100):
    """Generate text using the LLM model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=False)

    # calculate the encodings
    input_ids = inputs.input_ids
    input_token_len = input_ids.shape[-1]
    # print(f'{input_token_len=}')

    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
    output = outputs[0]
    # cut off template
    if cut_off_prompt:
        output = output[input_token_len:]
    #Decode to get the text
    return tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)


# title = "Ode to mercy."
# author = "William Collins"
# text = """O Thou, who sitt'st a smiling bride
# By Valour's arm'd and awful side,
# Gentlest of sky-born forms, and best adored;
# Who oft with songs, divine to hear,
# Winn'st from his fatal grasp the spear,
# And hidest in wreaths of flowers his bloodless sword!
# Thou who, amidst the deathful field,
# By godlike chiefs alone beheld,
# Oft with thy bosom bare art found,
# Pleading for him the youth who sinks to ground:
# See, Mercy, see, with pure and loaded hands,
# Before thy shrine my country's genius stands,
# And decks thy altar still, though pierced with many a wound."""
# summary = """The poem describes the goddess of mercy, who is depicted as a smiling bride by the side of a warrior. She is praised for her divine songs that can calm the warrior's anger and make him lay down his sword. The goddess is also seen pleading for the lives of soldiers on the battlefield, and her altar is still decorated despite the many wounds she has suffered."""

title = "As seamen on the seas..."
author = "Robert Louis Stevenson"
text = """As seamen on the seas
With song and dance descry
Adown the morning breeze
An islet in the sky:
In Araby the dry,
As o'er the sandy plain
The panting camels cry
To smell the coming rain:
So all things over earth
A common law obey,
And rarity and worth
Pass, arm in arm, away;
And even so, to-day,
The printer and the bard,
In pressless Davos, pray
Their sixpenny reward."""
summary = """The poem describes the beauty of nature and the importance of rarity and worth. It also highlights the struggles of the printer and the bard in a pressless Davos, who pray for their sixpenny reward."""

prompt_template = get_inference_template()
prompt = prompt_template.format(title=title, author=author, summary=summary, text="")

model.eval()
resp = generate_text(model, tokenizer, prompt, max_length=256)
print('Generated poem:')
print(resp) # print generated poem


from nltk.tokenize import SyllableTokenizer

def print_poem_statistics(text):
    sonnet = prosodic.Text(text)
    rhyming_lines = sonnet.get_rhyming_lines()
    print(f'{rhyming_lines=}')
    
    print(f'''Poem has:
      * {len(sonnet.stanzas):,} "stanzas"        (in this text, each one a sonnet)
      * {len(sonnet.lines):,} lines
      * {len(sonnet.wordtokens):,} wordtokens    (including punctuation)
      * {len(sonnet.wordtypes):,} wordtypes     (each token has one wordtype object)
      * {len(sonnet.wordforms):,} wordforms     (a word + IPA pronunciation; no punctuation)
      * {len(sonnet.syllables):,} syllables
      * {len(sonnet.phonemes):,} phonemes
    ''')

print_poem_statistics(text)


def count_rhymes_with_rhyme_tagger(text, rt):
    lines = text.splitlines()
    clean_text = [re.sub("[^a-zA-Z\ \']", "", text) for text in lines]
    #         :output_format  = [int] 1: returns list of indices for each line
    #                                 2: returns list of indices for each rhyme-
    #                                 3: returns classic ABBA list where ints instead of letters
    return rt.tag(clean_text,
                  output_format=2,
                  same_words=True,
                  window=2, #  how many lines forward to look for rhymes
    ) # e.g. [(0, 2)]

# measure meter
rt = RhymeTagger()
rt.load_model(model='en', verbose=False)
rhymes_list = count_rhymes_with_rhyme_tagger(text, rt)
print(f'{rhymes_list=}')
rhymes_cnt = len(rhymes_list)
print('Rhymes count', rhymes_cnt)


from collections import defaultdict

def resolve_line_rhyme_from_text(text, rhymes_list):
    # sort ascending
    rhymes_list = sorted(rhymes_list, key=lambda x: (x[0], x[1]))
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    idx_letter = 0
    line_rhyme_dict = defaultdict(list)

    # 2+ lines may create rhyme
    for rh_lines in rhymes_list:
        for i in range(len(rh_lines)):
            for j in range(i + 1, len(rh_lines)):
                rh_line1 = rh_lines[i]
                rh_line2 = rh_lines[j]
                line_rhyme_dict[rh_line1].append(rh_line2)
                line_rhyme_dict[rh_line2].append(rh_line1)
    
    idx_letter = 0
    text_lines = text.splitlines()
    dict_letter = {}
    rhyme_scheme = []
    for i in range(len(text_lines)):
        if i in dict_letter:
            # get letter from cache
            f_letter = dict_letter[i]
        else:
            # get next letter
            f_letter = letters[idx_letter]
            dict_letter[i] = f_letter
            # assign same letter to other lines
            for j in line_rhyme_dict[i]:
                dict_letter[j] = f_letter
            idx_letter += 1
        rhyme_scheme.append(f_letter)

    ret_list = []
    for idx, line in enumerate(text_lines):
        ret_list.append((line, dict_letter[idx]))
    return ret_list


rhymed_list = resolve_line_rhyme_from_text(text, rhymes_list)
rhyme_scheme = "".join([x[1] for x in rhymed_list])
    
display(HTML(f'Rhyme scheme <span style="font-weight: bold">{"".join(rhyme_scheme)}</span>'))


color_palette = [
    'bisque',
    'darkorange',
    'red',
    'pink',
    'goldenrod',
    'gold',
    'khaki',
    'darkseagreen',
    'mediumaquamarine',
    'palegreen',
    'yellowgreen',
    'cornflowerblue',
    'lightblue',
    'lightsteelblue',
    'mediumturquoise',
    'skyblue',
    'plum',
    'thistle',
    'blanchedalmond',
    'burlywood',
]

def wrap_span_color(txt, color):
    return f"<span style='background-color: {color};'>{txt}</span>"

def gen_html_poem_from_rhymed_list(rhymed_list):
    html_rows_list = []
    # cherry pick from https://austingil.com/css-named-colors/
    color_idx = 0
    colors_cache = {}
    for idx, (txt_line, letter) in enumerate(rhymed_list):
        if letter not in colors_cache:
            colors_cache[letter] = color_palette[color_idx]
            color_idx += 1
        txt = f"""<tr style='background: {colors_cache[letter]};'>
    <th style='font-weight: normal; text-align: left;'>{wrap_span_color(txt_line, colors_cache[letter])}</th>
    <th style='font-weight: bold; text-align: center;'>{wrap_span_color(letter, colors_cache[letter])}</th></tr>"""
        html_rows_list.append(txt)

    html_txt = f"""<table>
  <thead>
    <tr>
      <th style='text-align: left;'>Poem</th>
      <th style='font-weight: bold; text-align: center;'>Rhyme</th>
    </tr>
  </thead>
  <tbody>{''.join(html_rows_list)}</tbody>
</table>"""
    return html_txt


# visualize rhyme scheme via HTML
html_txt = gen_html_poem_from_rhymed_list(rhymed_list)
display(HTML(html_txt))


def tokenize_rhyme_text(line, stop_words):
    word_tokens = word_tokenize(line)
    # converts the words in word_tokens to lower case and then checks whether
    # they are present in stop_words or not
    return [w.lower() for w in word_tokens if not w.lower() in stop_words and w.isalpha()]


def do_they_rhyme(rhymer, word1, word2):
    # do these words rhyme?
    if word1 == word2:
        return True
    rhyme_info = rhymer.get_rhyme_type(word1, word2)
    rhyme = rhyme_info['rhyme_type']
    #is_rhyme = not rhyme.startswith('not')
    is_rhyme = rhyme.startswith('exact_perfect')
    return is_rhyme
    

def compute_rhyme_params(text, rhymer, stop_words, MAX_LINE_DIST=1, verbose=False):
    rhyme_lines_list = []
    word_list = []
    lines = text.splitlines()
    for idx in range(0, len(lines)):
        words = tokenize_rhyme_text(lines[idx], stop_words)
        if len(words) > 0:
            word_list.append(words)
    total_lines = len(word_list)

    # iterate lines
    rhymed_word_pairs = [] # list of tuples (line1, line2, word1, word2)
    for idx1 in range(0, len(lines)):
        line1 = lines[idx1]
        words1 = tokenize_rhyme_text(line1, stop_words)
        for idx2 in range(idx1 + 1, len(lines)):
            if idx2 - idx1 > MAX_LINE_DIST:
                break
            line2 = lines[idx2]
            words2 = tokenize_rhyme_text(line2, stop_words)

            # iterate words
            rhyme_w_score = 0
            for w_idx1 in range(0, len(words1)):
                word1 = words1[w_idx1]
                for w_idx2 in range(w_idx1 + 1, len(words2)):
                    word2 = words2[w_idx2]
                    if verbose:
                        print(f'CHECK {word1=} {word2=}')
                    is_rhyme = do_they_rhyme(rhymer, word1, word2)
                    rhyme_w_score += 1 if is_rhyme else 0
                    if is_rhyme:
                        rhymed_word_pairs.append((idx1, idx2, word1, word2))
            if rhyme_w_score > 0:
                rhyme_lines_list.extend([idx1, idx2])

    # remove duplicates
    rhyme_lines_list = list(set(rhyme_lines_list))
    rhyme_lines_cnt = len(rhyme_lines_list)
    rhyme_score = rhyme_lines_cnt / total_lines
    return rhyme_lines_cnt, total_lines, rhyme_score, rhymed_word_pairs


import ipa_rhyming

def compute_rhyme_features(text):
    rhymer = ipa_rhyming.Rhymer('en', 'en')
    stop_words = set(stopwords.words('english'))
    rhyme_lines, total_lines, rhyme_score, rhymed_word_pairs = compute_rhyme_params(
        text, rhymer, stop_words, MAX_LINE_DIST=1, verbose=False)
    return rhyme_lines, total_lines, rhyme_score, rhymed_word_pairs


rhyme_lines, total_lines, rhyme_score, rhymed_word_pairs = compute_rhyme_features(text)
print(f'{rhyme_lines=} {total_lines=} {rhyme_score=} {len(rhymed_word_pairs)=}')


rhymed_word_pairs


def replace_all(pattern, repl, string) -> str:
   occurences = re.findall(pattern, string, re.IGNORECASE)
   for occurence in occurences:
       string = string.replace(occurence, repl)
   return string

def get_html_rhyme_wordy(text, rhymed_word_pairs):
    # draw rhyme for every word in poem
    text_lines = text.splitlines() # poem lines
    html_text_lines = [] # keep html lines
    colors_cache = {} # remember color used for painting
    w_key_cache = {} # same word is in different pairs
    color_idx = 0 # current color index
    
    for idx, txt_line in enumerate(text_lines):
        new_line = txt_line
        
        for idx2, (l1, l2, w1, w2) in enumerate(rhymed_word_pairs):
            if idx == l1 or idx == l2:
                w = w1 if idx == l1 else w2 # word1 or word2 for current line
            else:
                continue # no matched line
            #print('word', w1, w2)
    
            if (l1, w1) in w_key_cache:
                colors_cache[idx2] = w_key_cache[(l1, w1)]
            if (l2, w2) in w_key_cache:
                colors_cache[idx2] = w_key_cache[(l2, w2)]
            if idx2 not in colors_cache:
                colors_cache[idx2] = color_palette[color_idx]
                w_key_cache[(l1, w1)] = color_palette[color_idx]
                w_key_cache[(l2, w2)] = color_palette[color_idx]
                color_idx += 1
            new_line = replace_all(w, f'<span style="background: {colors_cache[idx2]}">{w}</span>', new_line)
        html_text_lines.append(new_line)
    html_rhyme_word_poem = "<br />".join(html_text_lines)
    return html_rhyme_word_poem


html_rhyme_word_poem = get_html_rhyme_wordy(text, rhymed_word_pairs)
display(HTML(html_rhyme_word_poem))


# utilities to clean up the poem content
def clean_text_col(s):
    return s.strip() # remove new lines start/end


# read train set
poems_pf_dir = '/kaggle/input/poetryfoundation-hf-jnb666poems/poems'
pf_train_file = f'{poems_pf_dir}/poems-train.jsonl'
pf_val_file = f'{poems_pf_dir}/poems-val.jsonl'

# train split
pf_train_df = pd.read_json(pf_train_file, lines=True)
pf_train_df['title'] = pf_train_df['title'].apply(clean_title_col)
pf_train_df['body'] = pf_train_df['body'].apply(clean_text_col)
pf_train_df.drop('url', axis=1, inplace=True)
pf_train_df = pf_train_df.sort_values(by=['id'], ascending=True)
pf_train_df = pf_train_df.reset_index(drop=True)
print(pf_train_df.shape)
pf_train_df.head(3)


# read validation split
pf_val_df = pd.read_json(pf_val_file, lines=True)
pf_val_df['title'] = pf_val_df['title'].apply(clean_title_col)
pf_val_df['body'] = pf_val_df['body'].apply(clean_text_col)
pf_val_df.drop('url', axis=1, inplace=True)
pf_val_df = pf_val_df.sort_values(by=['id'], ascending=True)
pf_val_df = pf_val_df.reset_index(drop=True)
print(pf_val_df.shape)
pf_val_df.head(3)


# check if any common rows exist
common_values = pf_train_df[pf_train_df['id'].isin(pf_val_df['id'])]['id']
print(len(common_values), common_values)


# merge train and validation split
df_pf_all_poems = pd.concat([pf_train_df, pf_val_df], ignore_index=True, sort=False)
df_pf_all_poems = df_pf_all_poems.sort_values(by=['id'], ascending=True)
df_pf_all_poems = df_pf_all_poems.reset_index(drop=True)
print(df_pf_all_poems.shape)
df_pf_all_poems.head()


print('Original', df_pf_all_poems.shape)
df_pf_all_poems.dropna(inplace=True)
print('New shape after n/a removal', df_pf_all_poems.shape)


# save to csv file
pf_csv_file = f'{cfg.work_dir}/pf_merge_poems.csv'
df_pf_all_poems.to_csv(pf_csv_file, header=True, index=False)
print('Saved csv to', pf_csv_file)


# load english/russian poems only
lang_str = cfg.poem_lang # en, ru
src_poems_file = f'/kaggle/input/poetree-poetry-dataset/{lang_str}.zipp' # language specific poems
dst_poems_file = f'{cfg.work_dir}/{lang_str}.zip' # english poems
shutil.copy(src_poems_file, dst_poems_file)
!ls {cfg.work_dir}


# unpack poems
!unzip -o -q {dst_poems_file} # force unpack poem file, overwrite
!rm {dst_poems_file} # clean-up
!ls {cfg.work_dir}


lang_poems_dir = f'{cfg.work_dir}/{lang_str}'
# path joining version for other paths
poems_list = [os.path.join(lang_poems_dir, name) for name in os.listdir(lang_poems_dir) if os.path.isfile(os.path.join(lang_poems_dir, name))]
print('Sample', poems_list[0])
print(f'Poems count {len(poems_list)}')


# convert poems from json files to pandas dataframe
# follow this scheme https://versologie.cz/poetree/json-schema/
df_rows = []
multiple_authors = []
duplicate_poems = []
for poem_file in tqdm(poems_list, total=len(poems_list)):
    with open(poem_file, "r") as f:
        f_data = f.read()
    data = json.loads(f_data)

    # ignore poems written by multiple authors
    if type(data["author"]) is list:
        multiple_authors.append(data["id"])
        continue

    # ignore duplicate poems
    if type(data['duplicate']) is str:
        duplicate_poems.append(data["id"])
        continue

    # read neccessary fields; remove leading and trailing spaces
    txt_lines = [line['text'].strip() for line in data['body']] # poem text lines
    p_content = '\n'.join(txt_lines) # poem text
    p_content = clean_poem_text(p_content, author=None) # text cleaning
    
    p_id = data["id"] # id
    p_y_created = data['year_created'] # year creation
    p_title = data["title"] # title of poem
    if p_title is not None:
        p_title = p_title.strip()
    a_name = data["author"]['name']
    a_born, a_died = data["author"]['born'], data["author"]['died']
    a_name = HumanName(a_name).full_name
    a_country = data["author"]['country'] # country the author born in
    df_rows.append((p_id, p_title, p_content, a_name, a_country, a_born, a_died))

print('Skip multiple authors', len(multiple_authors))
print('Skip duplicate poems', len(duplicate_poems))

# concat all rows
cols = ['id', 'title', 'body', 'author', 'country', 'born', 'died']
df_poems = pd.DataFrame(df_rows, columns=cols)
print(f'{df_poems.shape=}')
df_poems.head(3)


print('Original', df_poems.shape)
df_poems = df_poems.dropna()
print('New shape after n/a removal', df_poems.shape)


df_vis = df_poems[df_poems['born'].notnull()].copy()
df_vis.born = df_vis.born.astype('int32')
df_vis.died = df_vis.died.astype('int32')
df_vis.head(3)


df_vis.to_csv(f'{cfg.work_dir}/df_poetree_poems_vis.csv', header=True, index=False)


unique_values = df_vis['country'].unique()
print('Unique poem authors', len(unique_values))

max_length = df_vis['body'].str.len().max()
print('Poem max length', max_length)

min_value = df_vis['born'].min()
max_value = df_vis['died'].max()
print("Author period", int(min_value), 'to', int(max_value))


def get_country_name(code):
    code = code.lower().replace('uk', 'gb')
    try:
        country = pycountry.countries.get(alpha_2=code)
        return country.name
    except AttributeError:
        return "Country not found"


df_vis['country_name'] = df_vis['country'].apply(get_country_name)
fig = px.choropleth(
    #df,
    df_vis.groupby(["country_name"]).size().reset_index(name="count"),
    locations="country_name", # name of column country
    hover_name="country_name",
    projection="natural earth",
    locationmode="country names",
    title="Countries of number of poems",
    #color="country_name",
    color='count',
    template="plotly",
    color_continuous_scale="agsunset",
    labels={"country_name": "Country"}
)
fig.show(renderer='iframe')


country_by_poem = df_vis.groupby(["country_name"]).size().reset_index(name="count")
fig = px.bar(country_by_poem, x='country_name', y='count', title='Countries by number of poems')
fig.show(renderer='iframe')


# drop columns we don't need
df_poems.drop(['born', 'died', 'country'], axis=1, inplace=True, errors='ignore')
df_poems.head(3)


poetree_csv_file = f'{cfg.work_dir}/poetree_{lang_str}_poems.csv'
df_poems.to_csv(poetree_csv_file, header=True, index=False)
print('Saved csv to', poetree_csv_file)


!rm -rf {lang_poems_dir} # delete poem directory
!ls


print('Reading', poetree_csv_file)
df_poetree_poems = pd.read_csv(poetree_csv_file)
print(f'{df_poetree_poems.shape=}')
df_poetree_poems['id'] = range(1, len(df_poetree_poems) + 1) # temp
uniq_authors = df_poetree_poems['author'].unique()
print('Unique poem authors', len(uniq_authors))

print('Reading', pf_csv_file)
df_pf_poems = pd.read_csv(pf_csv_file)
print(f'{df_pf_poems.shape=}')
df_pf_poems['id'] = range(1, len(df_pf_poems) + 1)
uniq_authors = df_pf_poems['author'].unique()
print('Unique poem authors', len(uniq_authors))

# drop the rows where at least one element is missing
df_poetree_poems.dropna(inplace=True)
df_pf_poems.dropna(inplace=True)
print('New shape after n/a removal', df_poetree_poems.shape)
print('New shape after n/a removal', df_pf_poems.shape)

df_poetree_sub = df_poetree_poems
df_pf_sub = df_pf_poems


dict_poetree = defaultdict(list)
for idx, row in tqdm(df_poetree_sub.iterrows()):
    author = row.author
    title = clean_title_col(row.title)
    dict_poetree[author].append((row.id, title))

dict_pf = defaultdict(list)
for idx, row in tqdm(df_pf_sub.iterrows()):
    author = row.author
    title = clean_title_col(row.title)
    dict_pf[author].append((row.id, title))

# identify duplicates in dict_pf
duplicate_pf_ids_list = []
for k_author, v_titles in dict_poetree.items():
    titles1 = [x[1] for x in v_titles]
    titles2 = [x[1] for x in dict_pf[k_author]]
    set1 = set(titles1)
    set2 = set(titles2)
    common_titles = list(set1 & set2)

    pf_duplicate_ids = [x[0] for x in dict_pf[k_author] if x[1] in set1]
    if len(pf_duplicate_ids) > 0:
        duplicate_pf_ids_list.extend(pf_duplicate_ids)
duplicate_pf_ids_list = set(duplicate_pf_ids_list)
print(f'{len(duplicate_pf_ids_list)=}')

# remove duplicate titles by id
print('Before removing duplicates', df_pf_poems.shape)
df_pf_poems = df_pf_poems[~df_pf_poems["id"].isin(duplicate_pf_ids_list)]
print('After removing duplicates', df_pf_poems.shape)


# merge
df_merged = pd.concat([df_poetree_poems, df_pf_poems], ignore_index=True, sort=False)
uniq_authors = df_merged['author'].unique()
print('Unique poem authors', len(uniq_authors))
df_merged.drop('id', axis=1, inplace=True)

df_merged = df_merged.reset_index(drop=True)
for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc='Capitalize titles'):
    df_merged.at[idx, 'title'] = row['title'].capitalize()


# remove duplicates by title; keep poems with shorter size
df_merged = df_merged.sort_values(by="body", key=lambda x: x.str.len(), ascending=True)
df_merged = df_merged.drop_duplicates(subset=['title'], keep='first')
df_merged = df_merged.reset_index(drop=True)
print('After removing duplicates', df_merged.shape)


# save a final poems dataframe
merge_csv_file = f'{cfg.work_dir}/merge_all_poems.csv'
df_merged.to_csv(merge_csv_file, header=True, index=False)
print(f'{df_merged.shape=}')
print('Saved csv to', merge_csv_file)
df_merged.head(3)


# Calculate the top 5 values
top_n_authors = 5
top_5 = df_merged['author'].value_counts().nlargest(top_n_authors)
# Create the barplot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_5.index, y=top_5.values)
plt.title(f'Top {top_n_authors} popular authors in poem dataset')
plt.xlabel('Author')
plt.ylabel('Poems')
plt.show()


def token_len(tokenizer, text):
    # count number of tokens in text
    tokenized = tokenizer(text, return_length=True)
    return tokenized['length'][0]

total_tokens = 0
tokens_count = []
for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc='Computing number of tokens'):
    n_tokens = token_len(tokenizer, row['body']) 
    total_tokens += n_tokens
    tokens_count.append(n_tokens)
print(f'Total tokens available: {total_tokens} (or {total_tokens / 1e6} million)')

# Plot the cumulative distribution
max_body_len = cfg.max_body_token_len
plt.hist(tokens_count, bins=max_body_len, cumulative=True, density=True)
plt.xlabel(f'#tokens ({max_body_len=})')
plt.ylabel('Cumulative density')
plt.title('CDF of tokens')
plt.axvline(x=max_body_len, color='r')
plt.show()


def tokenize_function(examples):
    # convert text to tokens
    tokens = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=cfg.max_seq_length,
    )
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens
    

def token_len(tokenizer, text):
    # count number of tokens in text
    tokenized = tokenizer(text, return_length=True)
    length = tokenized['length'][0]
    return length


def filter_df_by_prompts(df, prompt_template, tokenizer):
    # create prompts for training
    filtered_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Creating prompts'):
        # prompt type
        if cfg.prompt_type is PromptType.ShortPrompt:
            prompted_text = format_short_prompt_func(row, prompt_template).strip()  # + tokenizer.eos_token
        elif cfg.prompt_type is PromptType.FullPrompt:
            prompted_text = format_prompt_func(row, prompt_template).strip()  # + tokenizer.eos_token
        elif cfg.prompt_type is PromptType.NoSummaryPrompt:
            prompted_text = format_no_summary_prompt_func(row, prompt_template).strip()
        else:
            raise Exception('Invalid type', cfg.prompt_type)

        # Tokenize and check the length
        prompt_length = token_len(tokenizer, prompted_text)
        body_length = token_len(tokenizer, row['body'])
        
        # Skip data if the token length is longer than our limit
        if prompt_length < cfg.max_seq_length and body_length < cfg.max_body_token_len:
            df.at[idx, 'prompt_text'] = prompted_text
            filtered_rows.append(idx)
    print(f'{len(filtered_rows)=}')

    # keep only prompted rows
    df_new = df.loc[df.index.isin(filtered_rows)]
    df_new = df_new.reset_index(drop=True)
    return df_new


def resolve_train_dataset(poem_csv_file, tokenizer):
    # read merged dataset
    print('Reading', poem_csv_file)
    df = pd.read_csv(poem_csv_file)
    df.drop('prompt', axis=1, inplace=True)  # delete prompt used to generate summary
    print(f'Original {df.shape=}')

    # drop the rows where at least one element is missing
    df.dropna(inplace=True)
    print('New shape after n/a removal', df.shape)

    if cfg.prompt_type is not PromptType.ShortPrompt:
        # remove duplicates by title; keep poems with shorter size
        df = df.sort_values(by="body", key=lambda x: x.str.len(), ascending=True)
        df = df.drop_duplicates(subset=['title'], keep='first')
        df = df.reset_index(drop=True)
        print('After removing duplicates', df.shape)

    # shuffle rows
    df = df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    if cfg.dataset_percentage < 1.0:
        df = df.head(int(len(df) * cfg.dataset_percentage))
        print(f'Dataset percentage {cfg.dataset_percentage} = {df.shape=}')
    uniq_authors = df['author'].unique()
    print('Unique poem authors', len(uniq_authors))
    df = df.reset_index(drop=True)

    def translit_en_2_ru(s):
        return translit(s, 'ru')

    if cfg.poem_lang == 'ru':
        df['author'] = df['author'].apply(translit_en_2_ru)

    # get prompt template
    prompt_template = get_prompt_template()
    print(f'{prompt_template=}')

    # filter by length
    df_all_poems_f = filter_df_by_prompts(df, prompt_template, tokenizer)
    print(f'After filtering {df_all_poems_f.shape}')

    # split train/val/test
    df_train, df_val, df_test = \
        np.split(df_all_poems_f.sample(frac=1, random_state=cfg.seed),
                 [int(.9 * len(df_all_poems_f)), int(.95 * len(df_all_poems_f))])
    print('Total', df_all_poems_f.shape)
    print(f'Splits {df_train.shape} {df_val.shape} {df_test.shape}')


    # SAVE dataframes
    # Convert the resulting arrays back to DataFrames
    df_pd_train = pd.DataFrame(df_train)
    df_pd_val = pd.DataFrame(df_val)
    df_pd_test = pd.DataFrame(df_test)

    train_df_path = os.path.join(cfg.out_dir, cfg.poem_lang + '_split_train_df.csv')
    df_pd_train.to_csv(train_df_path, index=False)
    val_df_path = os.path.join(cfg.out_dir, cfg.poem_lang + '_split_val_df.csv')
    df_pd_val.to_csv(val_df_path, index=False)
    test_df_path = os.path.join(cfg.out_dir, cfg.poem_lang + '_split_test_df.csv')
    df_pd_test.to_csv(test_df_path, index=False)
    

    # prepare datasets for beining used by transformers
    df_train = Dataset.from_pandas(df_train)
    df_val = Dataset.from_pandas(df_val)
    df_test = Dataset.from_pandas(df_test)

    def formatting_prompts_func(examples):
        return {"text": [example for example in examples["prompt_text"]]}

    train_dataset = df_train.map(formatting_prompts_func, batched=True)
    val_dataset = df_val.map(formatting_prompts_func, batched=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True,
                                                remove_columns=train_dataset.column_names)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
    return tokenized_train_dataset, tokenized_val_dataset


# create output directory to write logs, checkpoints, weights, etc
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
print(f'{cfg.out_dir=}\n{cfg.log_dir=}')


# save config parameters to keep tracking the experiments
save_config(cfg.out_dir, cfg)
# set seed for all random processes involved in the training, ensuring that our fine-tuning results are reproducible.
random_seed(cfg.seed)


# memory clean-up to avoid OOM
try:
    del model
except:
    pass
clear_memory()


# load poems data from csv file
poem_csv_file = '/kaggle/input/poetry-compiled-datasets/merge_all_poems_prompted_summary_v2.csv'
train_data, eval_data = resolve_train_dataset(poem_csv_file, tokenizer)


if cfg.run_train:
    # configure wandb for logging
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config={
        "base_model_name": cfg.base_model_name, # pretrained gemma2
        "ft_model_name": cfg.ft_ru_model_name, # path of fine-tuned gemma2 model
        "dataset_name": cfg.dataset_name, # dataset name
        "max_length": cfg.max_seq_length, # max sequence length to use for training LLM
        "batch_size": cfg.batch_size, # batch size for training
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.lr_value, # learning rate for training
        "epochs": cfg.epoch, # how many epochs to train
        "logging_steps": cfg.logging_steps,
        "save_strategy": cfg.save_strategy,
        "dataset_percentage": cfg.dataset_percentage, # how much data we want to use for training
        "seed": cfg.seed, # to reproduce the results
    })


if cfg.run_train:
    print(f"Loading tokenizer: {cfg.base_model_name}")
    
    # Load model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    # set the padding token to be the end-of-sequence (EOS) token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_eos_token = True  # add <eos> at the end
    tokenizer.padding_side = "right"  # side on which the model should have padding applied


# load model for training
if cfg.run_train:
    print(f"Loading Gemma2 model: {cfg.base_model_name}")
    # to enable 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type=cfg.bnb_quant_type,
        bnb_4bit_compute_dtype=cfg.bnb_compute_quant_type
    )

    train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load pre-trained Gemma2 model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        quantization_config=bnb_config,
        # load_in_4bit=True,
        device_map=train_device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        # attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    model = prepare_model_for_kbit_training(model)


if cfg.run_train:
    # PEFT is used for efficient adaptation of general-purpose models to specific applications.
    lora_config = LoraConfig(
        r=64,
        lora_alpha=4,  # 16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # only target self-attention
        target_modules=["q_proj", "k_proj", "v_proj",
                        "down_proj", "up_proj", "o_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora_config)
    print("Model configured with PEFT.")

    # first gpu or fallback to cpu
    model.to(train_device) # move to gpu if available
    model.train() # enable training model

    # print the number of trainable parameters in the model.
    model.print_trainable_parameters()


if cfg.run_train:
    # define training arguments first
    training_arguments = TrainingArguments(
            output_dir=cfg.out_dir, # where to save finetuned weights
            num_train_epochs=cfg.epoch,
            # number of training examples processed concurrently on each device (GPU, CPU) during training step
            per_device_train_batch_size=cfg.batch_size,
            # accumulates gradients over multiple training steps before applying an optimizer update
            gradient_accumulation_steps=cfg.gradient_accumulation_steps, # number of updates steps to accumulate the gradients for
            per_device_eval_batch_size=4, # batch size per GPU/CPU for evaluation
            gradient_checkpointing=True, # to save memory at the expense of slower backward pass
            warmup_ratio=0.1,  # Linear warmup over warmup_ratio fraction of total steps
            optim="adamw_8bit",  # optimizer
            save_steps=0,
            save_strategy=cfg.save_strategy,
            logging_steps=cfg.logging_steps,
            learning_rate=cfg.lr_value,
            weight_decay=cfg.weight_decay,
            fp16=True, # supported -> enable
            bf16=False, # not natively support BF16 -> disable
            lr_scheduler_type="cosine",
            # report_to="tensorboard",
            report_to="wandb",
            logging_dir=cfg.log_dir,
            seed=cfg.seed, # to reproduce the results
    )
    
    trainer = Trainer(
        model=model,
        args=training_arguments, # training arguments
        train_dataset=train_data, # training data
        eval_dataset=eval_data, # evaluation data
        tokenizer=tokenizer, # tokenizer
        # callbacks=[eval_callback],
    )


if cfg.run_train:
    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")


# start training process, may take a long time
if cfg.run_train:
    print("Starting model training...")
    start_time = time.time()
    trainResult = trainer.train()
    training_time = time.time() - start_time
    print(f"Model training finished in {training_time:.2f} seconds.")
    wandb.log({"training_time": training_time})  # Log training time


if cfg.run_train:
    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


if cfg.run_train:
    # Save model
    print(f"Saving model to: {cfg.out_dir}")
    time_now = time.time()

    peft_model_path = os.path.join(cfg.out_dir, f"peft_model_{time_now}")
    trainer.model.save_pretrained(peft_model_path)
    
    out_model_path = os.path.join(cfg.out_dir, f"saved_model_{time_now}")
    trainer.save_model(out_model_path)
    
    tokenizer_files = tokenizer.save_pretrained(cfg.tokenizer_dir)
    print(f'{tokenizer_files=}')


# read train log csv
if cfg.run_train:
    train_hist_df = pd.DataFrame(trainer.state.log_history)
    train_df_path = cfg.weights_dir + '/train_state.csv'
    train_hist_df.to_csv(train_df_path, index=False)
    print('Saved', train_df_path)
else:
    train_hist_df = pd.read_csv(os.path.join(cfg.ft_ru_model_dir, 'train_state.csv'))
train_hist_df.head(3)


# plot the training loss
trainloss = train_hist_df[~train_hist_df["loss"].isnull()]
plt.plot(trainloss["loss"], label="Train")
plt.title("Training Loss")
plt.legend()
plt.show()


# marks the completion of a W&B run and ensures all data is synced to the server.
wandb.finish()


if cfg.run_train:
    # upload the model if we're in the train mode
    import kagglehub
    from kaggle_secrets import UserSecretsClient

    # get kaggle username
    user_secrets = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = user_secrets.get_secret("kaggle_username")
    os.environ["KAGGLE_KEY"] = user_secrets.get_secret("kaggle_key")
    KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
    
    # define parameters for the model
    local_model_dir = cfg.out_dir
    folder_basename = os.path.basename(local_model_dir)
    folder_name = folder_basename.replace('weights_', '')
    model_name = folder_name
    model_name = model_name.replace('-', '_').replace('.', '_')
    print(f'{model_name=}')
    
    MODEL = "gemma2-poem-rhyme-synthesizer"
    FRAMEWORK = "transformers"
    VARIATION = model_name
    handle = f'{KAGGLE_USERNAME}/{MODEL}/{FRAMEWORK}/{VARIATION}'
    ignore_patterns = ["checkpoint*/"] # ignore certain folders

    # upload the model to Kaggle Models
    kagglehub.model_upload(
      handle=handle,
      local_model_dir=local_model_dir,
      ignore_patterns=ignore_patterns,
    )


if cfg.run_train:
    # memory clean-up to avoid OOM
    try:
        del model
    except:
        pass
    clear_memory()


print(f"Loading tokenizer: {cfg.base_model_name}")

# Load model tokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
# set the padding token to be the end-of-sequence (EOS) token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.add_eos_token = True  # new
# tokenizer.padding_side = "right"  # new


def load_peft_ft_model(base_model_name, ft_model_name):
    """
    Load fine-tuned model on GPU. Fallback to CPU if GPU is not available.
    :param base_model_name: path to the pre-trained Gemma2
    :param ft_model_name:  path to the fine-tuned Gemma2
    :return: fine-tuned LLM model
    """
    print('Loading gemma2 pre-trained model', base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        use_cache=False,
        trust_remote_code=True)
    
    print('Loading peft model', ft_model_name)
    peft_model_path = ft_model_name
    ft_model = PeftModel.from_pretrained(base_model, peft_model_path, use_cache=False, is_trainable=False)
    ft_model = ft_model.merge_and_unload()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft_model = ft_model.to(device)
    ft_model = ft_model.eval()
    return ft_model


# load fine-tuned model
ft_model = load_peft_ft_model(cfg.base_model_name, cfg.ft_ru_model_name)
print('Model loaded!')


# load pre-trained gemma2
# due to limited GPU memory, use second T4 to host pre-trained
device_gpu2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print('Loading base gemma2 model', cfg.base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    cfg.base_model_name,
    use_cache=False,
    device_map=device_gpu2,
    trust_remote_code=True)


# define prompt in Russian language
prompt_template_params = """
### Ğ¡Ñ‚Ğ¸Ñ…Ğ¾Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ¸Ğµ:
{text}

### Ğ�Ğ²Ñ‚Ğ¾Ñ€:
{response}"""
sent1 = "Ğ’Ğ°ÑˆĞ° Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ° â€“ Ğ¾Ğ¿Ñ€ĞµĞ´ĞµĞ»Ğ¸Ñ‚ÑŒ Ğ°Ğ²Ñ‚Ğ¾Ñ€Ğ° ÑƒĞºĞ°Ğ·Ğ°Ğ½Ğ½Ğ¾Ğ³Ğ¾ Ñ�Ñ‚Ğ¸Ñ…Ğ¾Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ¸Ñ�."

prompt_author_template = (
    f"{sent1}\n{prompt_template_params}".strip()
)
prompt_author_template


%%time
# test
def run_test1(model, device=None):
    print('Ğ�Ğ²Ñ‚Ğ¾Ñ€:')
    body = """ĞŸĞ¾ Ğ²ĞµÑ‡ĞµÑ€Ğ°Ğ¼ Ğ½Ğ°Ğ´ Ñ€ĞµÑ�Ñ‚Ğ¾Ñ€Ğ°Ğ½Ğ°Ğ¼Ğ¸
    Ğ“Ğ¾Ñ€Ñ�Ñ‡Ğ¸Ğ¹ Ğ²Ğ¾Ğ·Ğ´ÑƒÑ… Ğ´Ğ¸Ğº Ğ¸ Ğ³Ğ»ÑƒÑ…,
    Ğ˜ Ğ¿Ñ€Ğ°Ğ²Ğ¸Ñ‚ Ğ¾ĞºÑ€Ğ¸ĞºĞ°Ğ¼Ğ¸ Ğ¿ÑŒÑ�Ğ½Ñ‹Ğ¼Ğ¸
    Ğ’ĞµÑ�ĞµĞ½Ğ½Ğ¸Ğ¹ Ğ¸ Ñ‚Ğ»ĞµÑ‚Ğ²Ğ¾Ñ€Ğ½Ñ‹Ğ¹ Ğ´ÑƒÑ….
    Ğ’Ğ´Ğ°Ğ»Ğ¸ Ğ½Ğ°Ğ´ Ğ¿Ñ‹Ğ»ÑŒÑ� Ğ¿ĞµÑ€ĞµÑƒĞ»Ğ¾Ñ‡Ğ½Ğ¾Ğ¹,
    Ğ�Ğ°Ğ´ Ñ�ĞºÑƒĞºĞ¾Ğ¹ Ğ·Ğ°Ğ³Ğ¾Ñ€Ğ¾Ğ´Ğ½Ñ‹Ñ… Ğ´Ğ°Ñ‡,
    Ğ§ÑƒÑ‚ÑŒ Ğ·Ğ¾Ğ»Ğ¾Ñ‚Ğ¸Ñ‚Ñ�Ñ� ĞºÑ€ĞµĞ½Ğ´ĞµĞ»ÑŒ Ğ±ÑƒĞ»Ğ¾Ñ‡Ğ½Ğ¾Ğ¹,
    Ğ˜ Ñ€Ğ°Ğ·Ğ´Ğ°ĞµÑ‚Ñ�Ñ� Ğ´ĞµÑ‚Ñ�ĞºĞ¸Ğ¹ Ğ¿Ğ»Ğ°Ñ‡.
    Ğ˜ ĞºĞ°Ğ¶Ğ´Ñ‹Ğ¹ Ğ²ĞµÑ‡ĞµÑ€, Ğ·Ğ° ÑˆĞ»Ğ°Ğ³Ğ±Ğ°ÑƒĞ¼Ğ°Ğ¼Ğ¸,
    Ğ—Ğ°Ğ»Ğ°Ğ¼Ñ‹Ğ²Ğ°Ñ� ĞºĞ¾Ñ‚ĞµĞ»ĞºĞ¸,
    Ğ¡Ñ€ĞµĞ´Ğ¸ ĞºĞ°Ğ½Ğ°Ğ² Ğ³ÑƒĞ»Ñ�Ñ�Ñ‚ Ñ� Ğ´Ğ°Ğ¼Ğ°Ğ¼Ğ¸
    Ğ˜Ñ�Ğ¿Ñ‹Ñ‚Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ¾Ñ�Ñ‚Ñ€Ñ�ĞºĞ¸."""
    
    prompt = prompt_author_template.format(text=body, response="")
    resp = generate_text(model, tokenizer, prompt, device=device, cut_off_prompt=True, max_length=256)
    return resp

resp = run_test1(ft_model)
print(f'{resp}') # print author


# use pretrained Gemma2
resp = run_test1(base_model, device=device_gpu2)
print(f'{resp}')


prompt_template_params = """
### Ğ�Ğ²Ñ‚Ğ¾Ñ€:
{author}

### Ğ�Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸Ğµ:
{title}

### Ğ¡Ñ‚Ğ¸Ñ…Ğ¾Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ¸Ğµ:
{response}"""
sent1 = "ĞšĞ°ĞºĞ¾Ğµ Ñ�Ñ‚Ğ¸Ñ…Ğ¾Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ¸Ğµ Ñ�Ğ¾Ğ¾Ñ‚Ğ²ĞµÑ‚Ñ�Ñ‚Ğ²ÑƒĞµÑ‚ ÑƒĞºĞ°Ğ·Ğ°Ğ½Ğ½Ğ¾Ğ¼Ñƒ Ğ°Ğ²Ñ‚Ğ¾Ñ€Ñƒ Ğ¸ Ğ·Ğ°Ğ´Ğ°Ğ½Ğ½Ğ¾Ğ¼Ñƒ Ğ½Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸Ñ�?"
prompt_poem_template = (
    f"{sent1}\n{prompt_template_params}".strip()
)
prompt_poem_template


%%time
# test
def run_test3(model, device=None):
    print('Ğ¡Ñ‚Ğ¸Ñ…Ğ¾Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ¸Ğµ:')

    author = 'Ğ�.Ğ�.Ğ‘Ğ»Ğ¾Ğº'
    title = 'Ğ�ĞµĞ·Ğ½Ğ°ĞºĞ¾Ğ¼ĞºĞ°'
    prompt = prompt_poem_template.format(author=author, title=title, response="")
    resp = generate_text(model, tokenizer, prompt, device=device, cut_off_prompt=True, max_length=256)
    return resp

resp = run_test3(ft_model)
print(f'{resp}') # print out a poem


# use pretrained Gemma2
resp = run_test3(base_model, device=device_gpu2)
print(f'{resp}') # print out a poem


prompt_template_params = """
### Ğ¢ĞµĞ¼Ğ°:
{topic}

### Ğ¡Ñ‚Ğ¸Ñ…Ğ¾Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ¸Ğµ:
{response}"""
sent1 = "Ğ�Ğ°Ğ¿Ğ¸Ñ�Ğ°Ñ‚ÑŒ ĞºĞ¾Ñ€Ğ¾Ñ‚ĞºĞ¾Ğµ Ñ�Ñ‚Ğ¸Ñ…Ğ¾Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ¸Ğµ Ğ½Ğµ Ğ±Ğ¾Ğ»ĞµĞµ 8 Ñ�Ñ‚Ñ€Ğ¾Ğº Ğ½Ğ° Ğ·Ğ°Ğ´Ğ°Ğ½Ğ½ÑƒÑ� Ñ‚ĞµĞ¼Ñƒ."
prompt_poem_write_template = (
    f"{sent1}\n{prompt_template_params}".strip()
)
prompt_poem_write_template


%%time
# test
def run_test4(model, tokenizer, device=None):
    print('Ğ¡Ñ‚Ğ¸Ñ…Ğ¾Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ¸Ğµ:')
    topic = "Ğ¡Ñ‚Ğ¸Ñ…Ğ¾Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ¸Ğµ Ğ¾ Ğ²ĞµÑ‡ĞµÑ€Ğ½ĞµĞ¼ Ğ½ĞµĞ±Ğµ, Ğ¿Ğ¾Ğ»Ğ½Ğ¾Ğ¼ Ğ·Ğ²Ñ‘Ğ·Ğ´, Ğ¸ Ğ¾ Ñ‚Ğ¾Ğ¼, ĞºĞ°Ğº Ğ¾Ğ½Ğ¸ Ğ²Ğ´Ğ¾Ñ…Ğ½Ğ¾Ğ²Ğ»Ñ�Ñ�Ñ‚ Ğ½Ğ° Ğ¼ĞµÑ‡Ñ‚Ñ‹ Ğ¸ Ñ‚Ğ²Ğ¾Ñ€Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾."
    prompt = prompt_poem_write_template.format(topic=topic, response="")
    resp = generate_text(model, tokenizer, prompt, device=device, cut_off_prompt=True, max_length=160)
    return resp

resp = run_test4(ft_model, tokenizer)
print(f'{resp}') # print out the generated poem


# use pretrained Gemma2
resp = run_test4(base_model, tokenizer=tokenizer, device=device_gpu2)
print(f'{resp}')  # print out the generated poem


# side by side comparison of the poems
def display_poems_side_by_side(prompt, poem_items):
    style = """
    <style>
    .results-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;  /* Controls space between rows and columns */
    }
    .poem-result {
        flex: 1 0 48%; /* Adjusts so each box takes up roughly half the space, less the gap */
        border: 1px solid #ddd;
        padding: 10px;
        background-color: #f5f0e8;
        box-sizing: border-box; /* Ensures padding is included in width calculation */
        min-width: 280px; /* Ensures that flex items have a minimum width */
        max-width: 48%; /* Ensures that items do not exceed the set percentage */
        margin-bottom: 15px; /* Maintains spacing for the last row if less than two items */
    }
    .content-wrapper {
        background-color: #ffffff; /* Calm white tone */
        padding: 10px;
        border-radius: 5px; /* Soft rounded corners for the inner content */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Subtle shadow for depth */
    }
    </style>
    """
    html_content = style + f""" 
    <h3>Prompt: {prompt}</h3>
    <div class="results-container">
    """
    for poem_idx, poem_txt in enumerate(poem_items):
        html_content += f"""
        <div class="poem-result">
            <div class="content-wrapper">
                <h4>Poem {poem_idx+1}:</h4>
                <p>{poem_txt}</p>
            </div>
        </div>
        """
    html_content += "</div>"  # Close the container div
    display(HTML(html_content))


poem_prompt = "Generate a poem on a topic"

ft_model_poem = """Ğ’ĞµÑ‡ĞµÑ€Ğ½ĞµĞµ Ğ½ĞµĞ±Ğ¾, Ğ¿Ğ¾Ğ»Ğ½Ğ¾Ğµ Ğ·Ğ²Ñ‘Ğ·Ğ´,
Ğ˜ Ğ²Ğ´Ğ¾Ñ…Ğ½Ğ¾Ğ²Ğ»Ñ�ĞµÑ‚ Ğ½Ğ° Ğ¼ĞµÑ‡Ñ‚Ñ‹ Ğ¸ Ñ‚Ğ²Ğ¾Ñ€Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾.
Ğ˜Ñ… Ñ�Ğ¸Ñ�Ğ½Ğ¸Ğµ, ĞºĞ°Ğº Ñ�Ğ²ĞµÑ‚,
Ğ’ Ñ�ĞµÑ€Ğ´Ñ†Ğµ Ğ·Ğ°Ğ¶Ğ³Ñ‘Ñ‚,
Ğ˜ Ğ·Ğ°Ñ�Ñ‚Ğ°Ğ²Ğ¸Ñ‚ Ğ¼ĞµÑ‡Ñ‚Ğ°Ñ‚ÑŒ,
Ğ˜ Ñ‚Ğ²Ğ¾Ñ€Ğ¸Ñ‚ÑŒ.
Ğ˜Ñ… ĞºÑ€Ğ°Ñ�Ğ¾Ñ‚Ğ°, ĞºĞ°Ğº ĞºĞ°Ñ€Ñ‚Ğ¸Ğ½Ğ°,
Ğ’ Ğ´ÑƒÑˆĞµ Ğ·Ğ°Ğ¿ĞµÑ‡Ğ°Ñ‚Ğ»ĞµĞµÑ‚,
Ğ˜ Ğ·Ğ°Ñ�Ñ‚Ğ°Ğ²Ğ¸Ñ‚ Ğ¼ĞµÑ‡Ñ‚Ğ°Ñ‚ÑŒ,
Ğ˜ Ñ‚Ğ²Ğ¾Ñ€Ğ¸Ñ‚ÑŒ.
Ğ˜Ñ… Ñ�Ğ²ĞµÑ‚, ĞºĞ°Ğº Ğ»ÑƒĞ½Ğ°,
Ğ’ Ñ�ĞµÑ€Ğ´Ñ†Ğµ Ğ·Ğ°Ğ¶Ğ³Ñ‘Ñ‚,
Ğ˜ Ğ·Ğ°Ñ�Ñ‚Ğ°Ğ²Ğ¸Ñ‚ Ğ¼ĞµÑ‡Ñ‚Ğ°Ñ‚ÑŒ"""
ft_model_poem = ft_model_poem.replace("\n", "<br />")

base_poem = """Ğ—Ğ²Ñ‘Ğ·Ğ´Ñ‹, ĞºĞ°Ğº Ğ·Ğ²Ñ‘Ğ·Ğ´Ñ‹,
Ğ’ Ğ½ĞµĞ±Ğµ Ñ�Ğ¸Ñ�Ñ�Ñ‚,
Ğ˜ Ğ²Ğ´Ğ¾Ñ…Ğ½Ğ¾Ğ²Ğ»Ñ�Ñ�Ñ‚
Ğ�Ğ° Ğ¼ĞµÑ‡Ñ‚Ñ‹ Ğ¸ Ñ‚Ğ²Ğ¾Ñ€Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾.

Ğ—Ğ²Ñ‘Ğ·Ğ´Ñ‹, ĞºĞ°Ğº Ğ·Ğ²Ñ‘Ğ·Ğ´Ñ‹,
Ğ’ Ğ½ĞµĞ±Ğµ Ñ�Ğ¸Ñ�Ñ�Ñ‚,
Ğ˜ Ğ²Ğ´Ğ¾Ñ…Ğ½Ğ¾Ğ²Ğ»Ñ�Ñ�Ñ‚
Ğ�Ğ° Ğ¼ĞµÑ‡Ñ‚Ñ‹ Ğ¸ Ñ‚Ğ²Ğ¾Ñ€Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾.

Ğ—Ğ²Ñ‘Ğ·Ğ´Ñ‹, ĞºĞ°Ğº Ğ·Ğ²Ñ‘Ğ·Ğ´Ñ‹,
Ğ’ Ğ½ĞµĞ±Ğµ Ñ�Ğ¸Ñ�Ñ�Ñ‚,
Ğ˜ Ğ²Ğ´Ğ¾Ñ…Ğ½Ğ¾Ğ²Ğ»Ñ�Ñ�Ñ‚
Ğ�Ğ° Ğ¼ĞµÑ‡Ñ‚Ñ‹ Ğ¸ Ñ‚Ğ²Ğ¾Ñ€Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾.

Ğ—Ğ²Ñ‘Ğ·Ğ´Ñ‹, ĞºĞ°Ğº Ğ·Ğ²Ñ‘Ğ·Ğ´Ñ‹,
Ğ’ Ğ½ĞµĞ±Ğµ Ñ�Ğ¸Ñ�Ñ�Ñ‚,"""
base_poem = base_poem.replace("\n", "<br />")

poem_items = [ft_model_poem, base_poem]
display_poems_side_by_side(poem_prompt, poem_items)


# poem should be generated
poem_text = """O Thou, who sitt'st a smiling bride
By Valour's arm'd and awful side,
Gentlest of sky-born forms, and best adored;
Who oft with songs, divine to hear,
Winn'st from his fatal grasp the spear,
And hidest in wreaths of flowers his bloodless sword!
Thou who, amidst the deathful field,
By godlike chiefs alone beheld,
Oft with thy bosom bare art found,
Pleading for him the youth who sinks to ground:
See, Mercy, see, with pure and loaded hands,
Before thy shrine my country's genius stands,
And decks thy altar still, though pierced with many a wound."""


# print out poem details
print_poem_statistics(poem_text)


rhyme_lines, total_lines, rhyme_score, rhymed_word_pairs = compute_rhyme_features(poem_text)
print(f'{rhyme_lines=} {total_lines=} {rhyme_score=} {len(rhymed_word_pairs)=}')


html_rhyme_word_poem = get_html_rhyme_wordy(poem_text, rhymed_word_pairs)
display(HTML(html_rhyme_word_poem))


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image

# Create a word cloud object of the most common words in the poem
wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=STOPWORDS).generate(poem_text)

# Display the generated image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig(os.path.join(cfg.work_dir, "poem_word_cloud.jpg"))
plt.show()


from gtts import gTTS

short_poem_text = """
Roses are red, violets are blue,
Sugar is sweet, and so are you.
"""

# convert text to sound (TTS - text-to-speech) and save as a file
poem_tts_path = f'{cfg.work_dir}/poem.mp3'
g_tts = gTTS(short_poem_text, lang="en")
g_tts.save(poem_tts_path)


import IPython

# play a poem audio to check soundness of the text
IPython.display.Audio(poem_tts_path)


from gtts import gTTS

# convert text to sound (TTS - text-to-speech) and save as a file
poem_ru_tts_path = f'{cfg.work_dir}/poem_ru.mp3'
g_tts = gTTS(ft_model_poem, lang="ru")
g_tts.save(poem_ru_tts_path)


import IPython

# play a poem audio to check soundness of the text
IPython.display.Audio(poem_ru_tts_path)


# quiz utils
# credits to notebook https://www.kaggle.com/code/sitaberete/build-ai-agents-with-google-s-llm-gemma
# questions are composed by myself

from ipywidgets import widgets
from IPython.display import display, HTML
from dataclasses import dataclass, asdict
import json

# print() function style
GREEN = '\033[92m'
RED = '\033[91m'
BOLD = '\033[1m'

# CSS style
PRIMARY_BUTTON_STYLE = "background: #1080e3;color:white;border:none;border-radius: 4px;padding: 1px 24px ;text-decoration: none;font-weight:500"
SECONDARY_BUTTON_STYLE = "background: #fcba03;color:white;border:none;border-radius: 4px;padding: 1px 24px ;text-decoration: none;font-weight:500"

@dataclass
class QuizEntry:
    label: str
    message: str
    message_color: str = 'red'
        
    def __str__(self):
        return self.label
    
    def to_json(self):
        return json.dumps(asdict(self))
    
@dataclass
class Quiz:
    id: str
    question: str
    entries: QuizEntry
    
    def to_json(self):
        return json.dumps(asdict(self))
    
    def entries_json(self):
        return json.dumps({"quiz_entries": entries },default=asdict)

    
def _render_quiz(quizModel: Quiz):

    # This allow the quiz to be interactive even when no kernel is running by executing the quiz logic in javascript on browser-side
    # So the the quiz is usable in kaggle notebook viewer mode, github or somewhere else where the notebook is converted to HTML.
    html = f"""
        <h4>{quizModel.question}</h4>
        <form id="{quizModel.id}" style="margin-bottom: 5px"></form>
        <button type="button" id="{quizModel.id}-button" style="{PRIMARY_BUTTON_STYLE}">
            Submit
        </button>
        <div id="{quizModel.id}-output"><p></p></div>
        
        <script>
            function {quizModel.id}createQuiz(){{
                const quizJSON = {quizModel.to_json()}
                let quizHTML = ""
                for ( const[index, quizEntry] of quizJSON.entries.entries()) {{
                    radioInput = `<input type="radio" id="{quizModel.id}${{quizEntry.label}}" name="{quizModel.id}" value="${{index}}" style="margin-right: 3px;margin-top: 3px">`
                    quizHTML += `<label for="{quizModel.id}${{quizEntry.label}}">${{radioInput}}${{quizEntry.label}}</label><br>`
                }}
                const form = document.getElementById("{quizModel.id}")
                form.innerHTML = quizHTML
                form.onchange = _ => document.getElementById("{quizModel.id}-output").innerHTML = "<p></p>"
            }}
            
            function {quizModel.id}printOutput(message, color){{
                output = `<p style="color: ${{color}}; font-weight: bold;">${{message}}</p>`
                document.getElementById("{quizModel.id}-output").innerHTML = output
            }}
            
            function {quizModel.id}submitQuiz() {{
                const quizJSON = {quizModel.to_json()}
                const quizEntries = quizJSON.entries
                selectedIndex = document.getElementById("{quizModel.id}").querySelector('input[type="radio"]:checked')?.value
                if(selectedIndex) {{
                    selectedEntry = quizEntries[selectedIndex]
                    {quizModel.id}printOutput(selectedEntry.message, selectedEntry.message_color)
                }} else {{
                    {quizModel.id}printOutput("Please select one option", "red")
                }}
            }}
            document.getElementById("{quizModel.id}-button").onclick = {quizModel.id}submitQuiz
            {quizModel.id}createQuiz()
        </script>
    """
    return HTML(html)

    
def display_quiz(): 
    quizes = [
        Quiz(
            id='quiz_1_1',
            question='Which one of these options is true about the Gemma models?',
            entries=[
                QuizEntry('Gemma models require complete retraining for every new task.', 'Incorrect'),
                QuizEntry('Gemma models specialize in domain-specific tasks with efficient fine-tuning.', 'Correct!', 'green'),
                QuizEntry('Gemma models are primarily used for processing audio data.', 'Incorrect'),
            ],
        ),
        Quiz(
            id='quiz_1_2',
            question='Which of the following is a key advantage of fine-tuning large language models (LLMs) for specific tasks?',
            entries=[
                QuizEntry('It reduces the size of the model, making it faster to train from scratch.', 'Incorrect'),
                QuizEntry('It completely removes the need for labeled training data.', 'Incorrect'),
                QuizEntry('It ensures that the model will never generate incorrect or biased outputs.', 'Incorrect'),
                QuizEntry('It allows the model to retain its general knowledge while adapting to a specific domain.', 'Correct', 'green'),
            ],
        ),
        Quiz(
            id='quiz_1_3',
            question='How many syllables are there in the poem titled "As seamen on the seas..." written by Robert Louis Stevenson? (use this notebook)',
            entries=[
                QuizEntry('73', 'Incorrect'),
                QuizEntry('91', 'Incorrect'),
                QuizEntry('101', 'Correct', 'green'),
            ],
        ),
        Quiz(
            id='quiz_1_4',
            question='What is the primary benefit of using Parameter-Efficient Fine-Tuning (PEFT) techniques in training large language models?',
            entries=[
                QuizEntry('It eliminates the need for pre-trained models entirely.', 'Incorrect'),
                QuizEntry('It guarantees that the model achieves state-of-the-art performance on all tasks without additional data.', 'Incorrect'),
                QuizEntry('It allows fine-tuning of large models by updating only a small subset of parameters, reducing computational and storage costs.', 'Correct!', 'green'),
                QuizEntry('It enables the fine-tuning process to use significantly smaller training datasets than traditional methods.', 'Incorrect'),
            ],
        ),
    
    ]
    
    rendered_quizes = [_render_quiz(quiz) for quiz in quizes]
    display(*rendered_quizes)


display_quiz()

